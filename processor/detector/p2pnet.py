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


class P2PNetDetector:
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

        self.img_size = img_size
        self.input_size = input_size
        self.resize_size = (
            int(input_size[0] // 128 * 128),
            int(input_size[1] // 128 * 128),
        )

    def build_detector(self):
        cfg = OmegaConf.load(self.cfg_path)
        backbone = Backbone_VGG(cfg.network.backbone, True, False)
        model = P2PNet(backbone, cfg.network.row, cfg.network.line, self.device)
        state_dict = torch.load(
            self.weight_path, weights_only=True, map_location=self.device
        )
        model.load_state_dict(state_dict)
        # model.compile()
        model.to(self.device)
        model.eval()
        return model

    def _upload(self, img: np.ndarray) -> torch.Tensor:
        """Host->device copy through a reusable pinned staging buffer.

        Pageable H2D transfers of an 8K frame (~95 MB) dominate pre-processing;
        staging through pinned memory roughly halves the upload cost and lets the
        copy run asynchronously on the current stream.
        """
        src = torch.from_numpy(
            img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)
        )
        if self.device.type != "cuda":
            return src.to(self.device)
        if self._pinned is None or self._pinned.shape != src.shape:
            self._pinned = torch.empty_like(src, pin_memory=True)
        self._pinned.copy_(src)
        return self._pinned.to(self.device, non_blocking=True)

    def preprocess(self, img: np.ndarray):
        """GPU pre-processing: BGR uint8 (H,W,3) -> normalized (1,3,h,w) tensor.

        Replaces the old CPU albumentations path (cv2 BGR2RGB + INTER_AREA resize
        + Normalize) which was the dominant cost at 8K. Everything runs on the GPU
        after a single uint8 host->device copy.
        """

        t = self._upload(img)
        # BGR -> RGB while still uint8 (4x less memory traffic than after float()).
        t = t.flip(-1)
        t = t.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W) RGB
        # area resampling matches cv2.INTER_AREA for downsampling.
        t = F.interpolate(t, size=self.resize_size, mode="area")
        t = t.div_(255.0).sub_(self._mean).div_(self._std)
        return t

    def infer(self, image: np.ndarray):
        if self._stream is not None:
            # blocking .cpu() in post_process syncs this stream before returning.
            with torch.cuda.stream(self._stream):
                return self._infer(image)
        return self._infer(image)

    def _infer(self, image: np.ndarray):
        input = self.preprocess(image)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype):
            outputs = self.detector(input)
            scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
            points = outputs["pred_points"][0]
            return self.post_process(scores, points)

    def post_process(self, scores: torch.Tensor, points: torch.Tensor):
        """Filter by threshold on the GPU, then move only survivors to CPU.

        Avoids transferring the full ~2M anchor points every frame.
        Drops points that fall outside the original image bounds.
        """
        keep = scores > self.threshold
        points = points[keep].float()
        scores = scores[keep].float()
        if points.numel() == 0:
            return np.empty((0, 3), dtype=np.float32)
        # invert the resize on the GPU before the (small) copy to host.
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
