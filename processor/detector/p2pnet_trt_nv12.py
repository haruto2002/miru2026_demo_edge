"""TensorRT backend for NV12-fused P2PNet engines.

Expects an engine built with ``export_p2pnet_onnx.py --fuse-nv12``:
input is packed uint8 NV12 of shape ``(1, H*W*3//2)``; NV12->RGB, resize and
ImageNet normalize are already in-graph. The Python side only does H2D + execute.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


_TRT_TO_TORCH_DTYPE = None


def _trt_dtype_map():
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


class P2PNetTRTNV12Detector:
    def __init__(
        self,
        engine_path,
        device,
        threshold,
        img_size,
    ):
        """
        Args:
            engine_path: path to NV12-fused TensorRT engine
            device: e.g. "cuda:0"
            threshold: score threshold
            img_size: (H, W) of the packed NV12 frame (= GStreamer output size)
        """
        assert Path(engine_path).exists(), (
            f"Engine file does not exist: {engine_path} "
            f"Current working directory: {Path.cwd()}"
        )
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.threshold = threshold
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.nv12_nbytes = self.img_size[0] * self.img_size[1] * 3 // 2
        # model resize snaps to multiple of 128 (same as export)
        self.resize_size = (
            self.img_size[0] // 128 * 128,
            self.img_size[1] // 128 * 128,
        )
        self._scale = torch.tensor(
            [
                self.img_size[1] / self.resize_size[1],
                self.img_size[0] / self.resize_size[0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._pinned: torch.Tensor | None = None
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
        self.buffers = {}
        self.input_names = []
        self.output_names = []

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

        in_buf = self.buffers[self.input_name]
        expected = (1, self.nv12_nbytes)
        assert in_buf.dtype == torch.uint8, (
            f"NV12-fused engine must take uint8 input, got {in_buf.dtype}"
        )
        assert tuple(in_buf.shape) == expected, (
            f"Engine input shape {tuple(in_buf.shape)} != expected {expected} "
            f"for img_size={self.img_size}"
        )

    def infer(self, nv12: np.ndarray) -> np.ndarray:
        """Run inference on packed NV12 uint8 of length H*W*3//2."""
        src = np.asarray(nv12).reshape(-1)
        assert src.size == self.nv12_nbytes, (
            f"NV12 size {src.size} != expected {self.nv12_nbytes}"
        )
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)
        src_t = torch.from_numpy(src)

        with torch.cuda.stream(self.stream):
            if self._pinned is None or self._pinned.numel() != src_t.numel():
                self._pinned = torch.empty_like(src_t, pin_memory=True)
            self._pinned.copy_(src_t)
            self.buffers[self.input_name][0].copy_(self._pinned, non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        scores = self.buffers["scores"][0]
        points = self.buffers["points"][0]
        return self.post_process(scores, points)

    def post_process(self, scores: torch.Tensor, points: torch.Tensor) -> np.ndarray:
        keep = scores > self.threshold
        points = points[keep].float()
        scores = scores[keep].float()
        if points.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)
        points = points * self._scale
        h, w = self.img_size
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

    def display_result(self, image: np.ndarray, result: np.ndarray) -> np.ndarray:
        for x, y, score in result:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        return image
