"""GPU perspective warp for packed NV12 using PyTorch CUDA.

Applies homography separately to the Y and interleaved UV planes so the
packed NV12 contract for the TRT detector is preserved without a BGR round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_homography(path: str | Path) -> np.ndarray:
    """Load a 3x3 homography matrix from a whitespace-separated text file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Homography file not found: {p}")
    H = np.loadtxt(p, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape} from {p}")
    if abs(H[2, 2]) < 1e-12:
        raise ValueError(f"Homography bottom-right element is zero in {p}")
    return H


class Nv12HomographyWarper:
    """CUDA perspective warp for packed NV12 frames."""

    def __init__(
        self,
        homography: np.ndarray,
        width: int,
        height: int,
        device: str = "cuda",
    ):
        self.w = int(width)
        self.h = int(height)
        self.device = torch.device(device)
        if self.w <= 0 or self.h <= 0 or self.h % 2 != 0:
            raise ValueError(f"NV12 size must be positive even height, got {self.w}x{self.h}")

        H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        H_inv = np.linalg.inv(H).astype(np.float32)
        self._H_inv = torch.from_numpy(H_inv).to(self.device)

        self._grid_y = self._make_grid(self.w, self.h, chroma=False)
        self._grid_uv = self._make_grid(self.w, self.h, chroma=True)
        self._nv12_nbytes = self.w * self.h * 3 // 2
        self._out_buf = torch.empty(self._nv12_nbytes, dtype=torch.uint8, device=self.device)

    def _make_grid(self, width: int, height: int, chroma: bool) -> torch.Tensor:
        if chroma:
            grid_h, grid_w = height // 2, width // 2
            ys, xs = torch.meshgrid(
                torch.arange(grid_h, device=self.device, dtype=torch.float32),
                torch.arange(grid_w, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            # Map each chroma sample to the center of its 2x2 luma block.
            full_x = xs * 2.0 + 1.0
            full_y = ys * 2.0 + 1.0
            src_w = max(width // 2 - 1, 1)
            src_h = max(height // 2 - 1, 1)
        else:
            grid_h, grid_w = height, width
            ys, xs = torch.meshgrid(
                torch.arange(grid_h, device=self.device, dtype=torch.float32),
                torch.arange(grid_w, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            full_x = xs
            full_y = ys
            src_w = max(width - 1, 1)
            src_h = max(height - 1, 1)

        ones = torch.ones_like(full_x)
        pts = torch.stack([full_x, full_y, ones], dim=-1)
        src = pts @ self._H_inv.T
        src_x = src[..., 0] / src[..., 2]
        src_y = src[..., 1] / src[..., 2]
        if chroma:
            src_x = src_x / 2.0
            src_y = src_y / 2.0

        grid = torch.stack(
            [
                2.0 * src_x / src_w - 1.0,
                2.0 * src_y / src_h - 1.0,
            ],
            dim=-1,
        )
        return grid.unsqueeze(0)

    @property
    def nv12_nbytes(self) -> int:
        return self._nv12_nbytes

    def warp(self, nv12: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Warp one packed NV12 frame on GPU and return a host copy."""
        if nv12.nbytes < self._nv12_nbytes:
            raise ValueError(
                f"NV12 buffer too small: {nv12.nbytes} < {self._nv12_nbytes}"
            )

        src = torch.as_tensor(nv12, device=self.device, dtype=torch.uint8)
        y_size = self.w * self.h
        y_plane = src[:y_size].view(1, 1, self.h, self.w).float()
        uv_plane = (
            src[y_size : y_size + self.w * (self.h // 2)]
            .view(self.h // 2, self.w)
            .view(self.h // 2, self.w // 2, 2)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        y_out = F.grid_sample(
            y_plane,
            self._grid_y,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        uv_out = F.grid_sample(
            uv_plane,
            self._grid_uv,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        y_u8 = y_out.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).reshape(-1)
        uv_u8 = (
            uv_out.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 255)
            .to(torch.uint8)
            .reshape(-1)
        )
        self._out_buf[:y_size] = y_u8
        self._out_buf[y_size : y_size + uv_u8.numel()] = uv_u8

        if out is None:
            return self._out_buf.detach().cpu().numpy().copy()
        if out.nbytes < self._nv12_nbytes:
            raise ValueError(
                f"Output buffer too small: {out.nbytes} < {self._nv12_nbytes}"
            )
        out[: self._nv12_nbytes] = self._out_buf.detach().cpu().numpy()
        return out
