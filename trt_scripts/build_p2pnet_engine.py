"""Build a TensorRT engine from the P2PNet ONNX (no trtexec needed).

The ONNX is fixed-shape so no optimization profile is required. Run on the GPU box:

    # FP16
    uv run python build_p2pnet_engine.py \
        --onnx weights/p2pnet/cutout_8k.onnx \
        --engine weights/p2pnet/cutout_8k.engine --fp16

    # INT8 (+FP16 fallback per layer) with calibration data
    uv run python build_p2pnet_engine.py \
        --onnx weights/p2pnet/cutout_8k_fused.onnx \
        --engine weights/p2pnet/cutout_8k_int8.engine \
        --fp16 --int8 \
        --calib-data /homes/SHARE/MLDatasets/Hanabi --calib-frames 128 --calib-cache weights/p2pnet/cutout_8k_int8_hanabi.calib

Calibration sources may be image files, video files, or directories
containing either. Frames are resized to the engine's input resolution and
fed through the same layout the engine expects (raw uint8 BGR HWC for
--fuse-preprocess graphs, normalized float32 CHW for legacy graphs).
A calibration cache is written next to the engine so rebuilds skip the
calibration pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# ImageNet stats used by albumentations' A.Normalize() defaults.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def collect_sources(paths: list[str]) -> tuple[list[Path], list[Path]]:
    """Expand files/directories into (image_paths, video_paths)."""
    images, videos = [], []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            candidates = sorted(p.rglob("*"))
        else:
            candidates = [p]
        for c in candidates:
            suffix = c.suffix.lower()
            if suffix in _IMAGE_EXTS:
                images.append(c)
            elif suffix in _VIDEO_EXTS:
                videos.append(c)
    return images, videos


def iter_calibration_frames(paths: list[str], num_frames: int, hw: tuple[int, int]):
    """Yield up to num_frames BGR uint8 frames resized to (h, w)."""
    import cv2

    h, w = hw
    images, videos = collect_sources(paths)
    assert images or videos, f"No calibration images/videos found under: {paths}"
    print(f"[calib] sources: {len(images)} images, {len(videos)} videos")

    yielded = 0

    def emit(frame):
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(frame)

    for img_path in images:
        if yielded >= num_frames:
            return
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        yield emit(frame)
        yielded += 1

    remaining = num_frames - yielded
    if remaining <= 0 or not videos:
        return
    per_video = max(1, remaining // len(videos))
    for vid_path in videos:
        if yielded >= num_frames:
            return
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        # sample frames evenly across the video
        indices = np.linspace(0, total - 1, per_video, dtype=int)
        for idx in indices:
            if yielded >= num_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            yield emit(frame)
            yielded += 1
        cap.release()


def _resize_size(raw_hw: tuple[int, int]) -> tuple[int, int]:
    """Match P2PNetDetector: snap each side down to a multiple of 128."""
    return (raw_hw[0] // 128 * 128, raw_hw[1] // 128 * 128)


def _bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) -> packed NV12 uint8 (H*W*3//2,)."""
    import cv2

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr.shape[:2]
    y = yuv[:h, :].reshape(-1)
    u = yuv[h : h + h // 4].reshape(h // 2, w // 2)
    v = yuv[h + h // 4 :].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv[..., 0] = u
    uv[..., 1] = v
    return np.concatenate([y, uv.reshape(-1)])


class _FrameCalibratorImpl:
    """Feeds calibration frames matching the network's input layout.

    - uint8 (1,H,W,3) input (fused-preprocess graph): raw BGR frames
    - uint8 (1, H*W*3//2) input (fused-nv12 graph): packed NV12 frames
    - float32 (1,3,h,w) input (legacy graph): resize + ImageNet-normalized RGB
    """

    def _init_impl(
        self,
        calib_paths: list[str],
        num_frames: int,
        input_shape: tuple[int, ...],
        input_is_uint8: bool,
        cache_path: Path,
    ):
        import torch

        self._torch = torch
        self.cache_path = cache_path
        self.input_shape = tuple(input_shape)
        self.input_is_uint8 = input_is_uint8
        self.input_is_nv12 = False

        if input_is_uint8 and len(self.input_shape) == 2:
            # (1, H*W*3//2) packed NV12
            self.input_is_nv12 = True
            nv12_len = int(self.input_shape[1])
            # solve H*W*3//2 = nv12_len with typical 16:9; prefer exact factors
            # store as None and infer from first frame path via common sizes
            self._nv12_hw = _infer_hw_from_nv12_len(nv12_len)
            raw_hw = self._nv12_hw
            buf_dtype = torch.uint8
        elif input_is_uint8:
            # (1, H, W, 3) raw frame input
            raw_hw = (self.input_shape[1], self.input_shape[2])
            buf_dtype = torch.uint8
        else:
            # (1, 3, h, w) preprocessed input; calibration frames are raw-sized
            resized_hw = (self.input_shape[2], self.input_shape[3])
            # feed frames at the resized resolution directly (scale-equivalent)
            raw_hw = resized_hw
            buf_dtype = torch.float32
            self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

        self._frames = iter_calibration_frames(calib_paths, num_frames, raw_hw)
        self._buf = torch.empty(self.input_shape, dtype=buf_dtype, device="cuda:0")
        self._count = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        torch = self._torch
        frame = next(self._frames, None)
        if frame is None:
            print(f"[calib] done: {self._count} frames")
            return None
        if self.input_is_nv12:
            nv12 = _bgr_to_nv12(frame)
            t = torch.from_numpy(nv12).unsqueeze(0)  # (1, H*W*3//2)
        elif self.input_is_uint8:
            t = torch.from_numpy(frame).unsqueeze(0)  # (1,H,W,3) uint8 BGR
        else:
            t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
            t = t.flip(1)  # BGR -> RGB
            t = t.div_(255.0).sub_(self._mean).div_(self._std)
        self._buf.copy_(t.to(self._buf.dtype))
        torch.cuda.synchronize()
        self._count += 1
        if self._count % 10 == 0:
            print(f"[calib] fed {self._count} frames")
        return [int(self._buf.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            print(f"[calib] using cache: {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        print(f"[calib] wrote cache: {self.cache_path}")


def _infer_hw_from_nv12_len(nv12_len: int) -> tuple[int, int]:
    """Recover (H, W) from packed NV12 length (= H*W*3//2)."""
    pixels = nv12_len * 2 // 3
    # try common resolutions first
    for h, w in (
        (1080, 1920),
        (2160, 3840),
        (4320, 7680),
        (720, 1280),
        (1440, 2560),
    ):
        if h * w == pixels:
            return (h, w)
    # generic search: prefer even dims, 16:9-ish
    for h in range(2, int(pixels**0.5) + 1, 2):
        if pixels % h == 0:
            w = pixels // h
            if w % 2 == 0:
                return (h, w)
    raise ValueError(f"Cannot infer H,W from NV12 length {nv12_len}")


class EntropyFrameCalibrator(_FrameCalibratorImpl, trt.IInt8EntropyCalibrator2):
    def __init__(self, *args, **kwargs):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self._init_impl(*args, **kwargs)


class MinMaxFrameCalibrator(_FrameCalibratorImpl, trt.IInt8MinMaxCalibrator):
    """MinMax keeps the full activation range; often better than entropy for
    detection heads whose rare large activations carry the signal."""

    def __init__(self, *args, **kwargs):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._init_impl(*args, **kwargs)


def apply_int8_exclusions(network, config, pattern: str):
    """Force layers matching the regex to FP16 so INT8 never touches them.

    P2PNet's classification/regression heads produce per-anchor logits whose
    quantization collapses detection scores; keeping the heads (and softmax)
    in FP16 retains most of the INT8 speedup on the VGG backbone.
    """
    import re

    regex = re.compile(pattern, re.IGNORECASE)
    n_excluded = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not regex.search(layer.name):
            continue
        # constants / shape ops cannot take precision constraints
        if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
            continue
        layer.precision = trt.float16
        n_excluded += 1
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    print(f"[build] kept {n_excluded} layers matching '{pattern}' in FP16")


def build(
    onnx_path: str,
    engine_path: str,
    fp16: bool,
    int8: bool,
    workspace_gb: float,
    calib_data: list[str] | None,
    calib_frames: int,
    calib_cache: str | None,
    calib_algo: str = "entropy",
    int8_exclude: str | None = None,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[build] WARNING: platform has no fast fp16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")

    inp = network.get_input(0)
    print(f"[build] input: {inp.name} {inp.shape} {inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"[build] output: {out.name} {out.shape} {out.dtype}")

    calibrator = None  # keep alive until build finishes
    if int8:
        if not builder.platform_has_fast_int8:
            print("[build] WARNING: platform has no fast int8; building anyway")
        assert calib_data, "--int8 requires --calib-data (images/videos/dirs)"
        cache_path = (
            Path(calib_cache)
            if calib_cache
            else Path(engine_path).with_suffix(".calib")
        )
        calibrator_cls = (
            MinMaxFrameCalibrator
            if calib_algo == "minmax"
            else EntropyFrameCalibrator
        )
        calibrator = calibrator_cls(
            calib_paths=calib_data,
            num_frames=calib_frames,
            input_shape=tuple(inp.shape),
            input_is_uint8=inp.dtype == trt.uint8,
            cache_path=cache_path,
        )
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        print(
            f"[build] INT8 enabled (calibrator: {calib_algo}, "
            f"calibration frames: {calib_frames})"
        )
        if int8_exclude:
            apply_int8_exclusions(network, config, int8_exclude)

    print("[build] building serialized engine (this can take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    out_path = Path(engine_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = memoryview(serialized)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"[build] wrote engine: {out_path} ({data.nbytes / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="weights/p2pnet/cutout_8k.onnx")
    parser.add_argument("--engine", default="weights/p2pnet/cutout_8k.engine")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument(
        "--calib-data",
        nargs="+",
        default=None,
        help="images / videos / directories for INT8 calibration",
    )
    parser.add_argument("--calib-frames", type=int, default=64)
    parser.add_argument(
        "--calib-cache",
        default=None,
        help="calibration cache path (default: <engine>.calib)",
    )
    parser.add_argument(
        "--calib-algo",
        choices=["entropy", "minmax"],
        default="entropy",
        help="INT8 calibration algorithm",
    )
    parser.add_argument(
        "--int8-exclude",
        default=None,
        help="regex of layer names to keep in FP16 under INT8 "
        "(e.g. 'regression|classification|Softmax')",
    )
    args = parser.parse_args()

    assert Path(args.onnx).exists(), f"onnx not found: {args.onnx}"
    build(
        args.onnx,
        args.engine,
        args.fp16,
        args.int8,
        args.workspace_gb,
        args.calib_data,
        args.calib_frames,
        args.calib_cache,
        args.calib_algo,
        args.int8_exclude,
    )


if __name__ == "__main__":
    main()
