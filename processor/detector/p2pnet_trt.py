"""TensorRT inference backend for P2PNet.

Loads a serialized TensorRT engine (built from the ONNX produced by
``export_p2pnet_onnx.py``) and runs inference with torch CUDA tensors as I/O
buffers (no pycuda needed). Pre/post-processing mirrors
``processor.detector.p2pnet.P2PNetDetector`` so results stay consistent.

The engine is fixed-shape: it only accepts the resolution it was built for.
``img_size`` here must match the ``--img-size`` used at export time.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ImageNet stats used by albumentations' A.Normalize() defaults.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_TRT_TO_TORCH_DTYPE = None


def _trt_dtype_map():
    """Build the TensorRT->torch dtype map lazily (tensorrt import is GPU-only)."""
    global _TRT_TO_TORCH_DTYPE
    if _TRT_TO_TORCH_DTYPE is not None:
        return _TRT_TO_TORCH_DTYPE
    import tensorrt as trt

    _TRT_TO_TORCH_DTYPE = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.uint8: torch.uint8,
    }
    if hasattr(trt, "bool"):
        _TRT_TO_TORCH_DTYPE[trt.bool] = torch.bool
    return _TRT_TO_TORCH_DTYPE


class P2PNetTRTDetector:
    def __init__(
        self,
        engine_path,
        device,
        threshold,
        raw_img_size,
        trt_img_size,
    ):
        assert Path(engine_path).exists(), (
            f"Engine file does not exist: {engine_path} "
            f"Current working directory: {Path.cwd()}"
        )
        assert trt_img_size is not None, "trt_img_size is required for the TRT backend"

        self.engine_path = engine_path
        self.device = torch.device(device)
        self.threshold = threshold

        self.image_size = (int(trt_img_size[0]), int(trt_img_size[1]))
        self.raw_img_size = (int(raw_img_size[0]), int(raw_img_size[1]))
        self.resize_size = (
            int(trt_img_size[0]) // 128 * 128,
            int(trt_img_size[1]) // 128 * 128,
        )
        self._mean = torch.tensor(
            _IMAGENET_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            _IMAGENET_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._scale = torch.tensor(
            [
                raw_img_size[1] / self.resize_size[1],
                raw_img_size[0] / self.resize_size[0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self._load_engine()

    def _load_engine(self):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, "Failed to deserialize TensorRT engine"
        self.context = self.engine.create_execution_context()

        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError(
                "This backend requires TensorRT >= 8.5 (name-based I/O API)."
            )

        dtype_map = _trt_dtype_map()
        self.input_names = []
        self.output_names = []
        self.buffers = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.context.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = dtype_map.get(trt_dtype, torch.float32)
            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        assert len(self.input_names) == 1, (
            f"Expected exactly 1 input, got {self.input_names}"
        )
        self.input_name = self.input_names[0]
        self.stream = torch.cuda.Stream(device=self.device)
        # pinned staging buffer reused across frames for fast H2D copies.
        self._pinned: torch.Tensor | None = None

        # Engines exported with --fuse-preprocess take the raw uint8 BGR HWC
        # frame and do cast/resize/normalize in-graph: the Python side then
        # only performs a single H2D copy.
        in_buf = self.buffers[self.input_name]
        self.fused_preprocess = in_buf.dtype == torch.uint8
        if self.fused_preprocess:
            expected = (1, self.image_size[0], self.image_size[1], 3)
            assert tuple(in_buf.shape) == expected, (
                f"Fused-preprocess engine input shape {tuple(in_buf.shape)} does "
                f"not match img_size {self.image_size} (expected {expected})"
            )

    def _upload(self, img: np.ndarray) -> torch.Tensor:
        """Host->device copy through a reusable pinned staging buffer."""
        src = torch.from_numpy(
            img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)
        )
        if self._pinned is None or self._pinned.shape != src.shape:
            self._pinned = torch.empty_like(src, pin_memory=True)
        self._pinned.copy_(src)
        return self._pinned.to(self.device, non_blocking=True)

    def preprocess(self, img: np.ndarray):
        """GPU pre-processing identical to P2PNetDetector (BGR uint8 -> norm CHW)."""
        t = self._upload(img)
        # BGR -> RGB while still uint8 (4x less memory traffic than after float()).
        t = t.flip(-1)
        t = t.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W) RGB
        t = F.interpolate(t, size=self.resize_size, mode="area")
        t = t.div_(255.0).sub_(self._mean).div_(self._std)
        return t

    def infer(self, image: np.ndarray):
        # Run preprocessing, the input copy and the engine on the same dedicated
        # stream: this guarantees ordering (the old code enqueued the preprocess
        # on the default stream with no sync before execute) and lets this
        # detector overlap with others running concurrently on the GPU.
        with torch.cuda.stream(self.stream):
            if self.fused_preprocess:
                # single H2D copy straight into the engine input buffer.
                src = torch.from_numpy(
                    image
                    if image.flags["C_CONTIGUOUS"]
                    else np.ascontiguousarray(image)
                )
                if self._pinned is None or self._pinned.shape != src.shape:
                    self._pinned = torch.empty_like(src, pin_memory=True)
                self._pinned.copy_(src)
                self.buffers[self.input_name][0].copy_(self._pinned, non_blocking=True)
            else:
                input_tensor = self.preprocess(image)
                self.buffers[self.input_name].copy_(input_tensor)
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        scores = self.buffers["scores"][0]
        points = self.buffers["points"][0]
        return self.post_process(scores, points)

    def post_process(self, scores: torch.Tensor, points: torch.Tensor):
        """Threshold-filter on GPU, scale to raw image coords, copy survivors to CPU."""
        keep = scores > self.threshold
        points = points[keep].float()
        scores = scores[keep].float()
        if points.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)

        points = points * self._scale
        h, w = self.raw_img_size
        inbounds = (
            (points[:, 0] >= 0)
            & (points[:, 0] < w)
            & (points[:, 1] >= 0)
            & (points[:, 1] < h)
        )
        points = points[inbounds]
        scores = scores[inbounds]
        if points.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)
        result = torch.cat([points, scores.unsqueeze(-1)], dim=1)
        return result.cpu().numpy()

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for x, y, score in result:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        return image
