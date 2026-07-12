"""P2PNet detector that accepts packed NV12 frames from GStreamer.

Input format matches ``bench_video_reader.Nv12Grabber`` / appsink output:
packed NV12 uint8 of size ``H * W * 3 // 2`` (Y plane followed by interleaved UV).

Pre-processing (GPU):
  NV12 -> RGB (BT.709 full-range) -> area resize -> ImageNet normalize
yielding the same ``(1, 3, h, w)`` float tensor that ``P2PNetDetector`` expects.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from processor.modules.detector.p2pnet.network.backbone import Backbone_VGG
from processor.modules.detector.p2pnet.network.p2pnet import P2PNet

# ImageNet stats used by albumentations' A.Normalize() defaults.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# BT.709 full-range YUV(0-255) -> RGB (source is typically yuvj420p).
_A_V_R = 1.5748
_A_U_G = 0.1873
_A_V_G = 0.4681
_A_U_B = 1.8556


class P2PNetNV12Detector:
    def __init__(
        self,
        cfg_path,
        weight_path,
        device,
        dtype,
        threshold,
        img_size,
        input_size,
    ):
        assert Path(cfg_path).exists(), (
            f"Config file does not exist: {cfg_path} Current working directory: {Path.cwd()}"
        )
        assert Path(weight_path).exists(), (
            f"Weight file does not exist: {weight_path} Current working directory: {Path.cwd()}"
        )
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.detector = self.build_detector()
        # pinned staging buffer reused across frames for fast H2D copies.
        self._pinned: torch.Tensor | None = None
        # dedicated stream so this detector can overlap with others on the GPU.
        self._stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.threshold = threshold
        # mean/std as (1, 3, 1, 1) GPU tensors for fused normalization.
        self._mean = torch.tensor(
            _IMAGENET_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            _IMAGENET_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        # img_size = NV12 frame size (H, W) from GStreamer / appsink.
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.input_size = input_size
        self.resize_size = (
            int(input_size[0] // 128 * 128),
            int(input_size[1] // 128 * 128),
        )
        self._nv12_nbytes = self.img_size[0] * self.img_size[1] * 3 // 2

    def build_detector(self):
        cfg = OmegaConf.load(self.cfg_path)
        backbone = Backbone_VGG(cfg.network.backbone, True, False)
        model = P2PNet(backbone, cfg.network.row, cfg.network.line, self.device)
        state_dict = torch.load(
            self.weight_path, weights_only=True, map_location=self.device
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _upload_nv12(self, nv12: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Host/device NV12 -> contiguous 1D uint8 tensor on ``self.device``."""
        if isinstance(nv12, torch.Tensor):
            t = nv12.reshape(-1).contiguous()
            if t.device != self.device:
                t = t.to(self.device, non_blocking=True)
            return t

        src = np.asarray(nv12)
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)
        src = src.reshape(-1)
        assert src.size == self._nv12_nbytes, (
            f"NV12 size {src.size} != expected {self._nv12_nbytes} "
            f"for img_size={self.img_size}"
        )
        src_t = torch.from_numpy(src)
        if self.device.type != "cuda":
            return src_t.to(self.device)
        if self._pinned is None or self._pinned.numel() != src_t.numel():
            self._pinned = torch.empty_like(src_t, pin_memory=True)
        self._pinned.copy_(src_t)
        return self._pinned.to(self.device, non_blocking=True)

    def _nv12_to_rgb(self, nv12_1d: torch.Tensor) -> torch.Tensor:
        """Packed NV12 uint8 1D -> (1, 3, H, W) float RGB in [0, 255]."""
        h, w = self.img_size
        y_size = h * w
        y = nv12_1d[:y_size].view(h, w).float()
        uv = nv12_1d[y_size:].view(h // 2, w // 2, 2).float()
        u = uv[..., 0] - 128.0
        v = uv[..., 1] - 128.0
        # nearest-neighbor chroma upsample (matches bench_video_reader)
        u = u.repeat_interleave(2, 0).repeat_interleave(2, 1)
        v = v.repeat_interleave(2, 0).repeat_interleave(2, 1)
        r = y + _A_V_R * v
        g = y - _A_U_G * u - _A_V_G * v
        b = y + _A_U_B * u
        rgb = torch.stack([r, g, b], dim=0).unsqueeze(0).clamp_(0, 255)
        return rgb  # (1, 3, H, W)

    def preprocess(self, nv12: np.ndarray | torch.Tensor):
        """NV12 uint8 -> normalized (1, 3, h, w) float tensor for P2PNet."""
        t = self._upload_nv12(nv12)
        t = self._nv12_to_rgb(t)
        t = F.interpolate(t, size=self.resize_size, mode="area")
        t = t.div_(255.0).sub_(self._mean).div_(self._std)
        return t

    def infer(self, image: np.ndarray | torch.Tensor):
        if self._stream is not None:
            # blocking .cpu() in post_process syncs this stream before returning.
            with torch.cuda.stream(self._stream):
                return self._infer(image)
        return self._infer(image)

    def _infer(self, image: np.ndarray | torch.Tensor):
        input = self.preprocess(image)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype):
            outputs = self.detector(input)
            scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
            points = outputs["pred_points"][0]
            return self.post_process(scores, points)

    def post_process(self, scores: torch.Tensor, points: torch.Tensor):
        """Filter by threshold on the GPU, then move only survivors to CPU."""
        keep = scores > self.threshold
        points = points[keep].float()
        scores = scores[keep].float()
        if points.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)
        ratio_h = self.img_size[0] / self.resize_size[0]
        ratio_w = self.img_size[1] / self.resize_size[1]
        scale = torch.tensor(
            [ratio_w, ratio_h], device=points.device, dtype=points.dtype
        )
        points = points * scale
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

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for x, y, score in result:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        return image
