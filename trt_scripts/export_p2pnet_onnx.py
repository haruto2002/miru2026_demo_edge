"""Export the P2PNet crowd-counting model to ONNX for TensorRT.

The exported graph is fixed-shape and inference-optimized:
  - FPN dead branches (P3, P5_2) are skipped via forward_inference
  - Anchor points are baked as constants for the given resolution
  - with --fuse-preprocess, the whole pre-processing pipeline (uint8 cast,
    BGR->RGB, resize, ImageNet normalization) is baked into the graph: the
    engine then takes the raw uint8 BGR HWC frame and the Python side only
    does a single H2D copy.
  - with --fuse-nv12, packed NV12 uint8 (GStreamer appsink layout) is the
    engine input; NV12->RGB + resize + normalize are in-graph.

Outputs:
    - scores: (1, N) person probability (softmax already applied)
    - points: (1, N, 2) predicted (x, y) in the resized space

Example:
    uv run python export_p2pnet_onnx.py \
        --cfg processor/modules/detector/p2pnet/conf/p2p.yaml \
        --weight weights/p2pnet/cutout.pth \
        --img-size 1080 1920 \
        --fuse-nv12 \
        --out weights/p2pnet/cutout_fhd_nv12.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from processor.modules.detector.p2pnet.network.backbone import Backbone_VGG
from processor.modules.detector.p2pnet.network.p2pnet import P2PNet

# ImageNet stats used by albumentations' A.Normalize() defaults.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# BT.709 full-range YUV(0-255) -> RGB (matches P2PNetNV12Detector / yuvj420p).
_A_V_R = 1.5748
_A_U_G = 0.1873
_A_V_G = 0.4681
_A_U_B = 1.8556


class P2PNetONNX(nn.Module):
    """Wrap P2PNet for ONNX: slim forward, baked anchors, softmax in-graph."""

    def __init__(self, model: nn.Module, anchors: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        out = self.model.forward_inference(x, self.anchors)
        scores = torch.softmax(out["pred_logits"], dim=-1)[..., 1]
        points = out["pred_points"]
        return scores, points


class P2PNetONNXFused(P2PNetONNX):
    """Same as P2PNetONNX but with pre-processing fused into the graph.

    Input: (1, H, W, 3) uint8 BGR raw frame (H, W = raw image size).
    In-graph: cast -> NCHW -> bilinear resize to (h, w) -> BGR->RGB ->
    fused normalize ``x * 1/(255*std) - mean/std``.

    Note: the resize uses ``bilinear`` because ONNX Resize has no ``area``
    mode. The downscale here is tiny (e.g. 4320 -> 4224 in H only) so the
    numeric difference vs the PyTorch detector is negligible, but validate
    with validate_trt.py after building an engine.
    """

    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        resize_size: tuple[int, int],
    ):
        super().__init__(model, anchors)
        self.resize_size = list(resize_size)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        # (x/255 - mean)/std == x * scale + bias
        self.register_buffer("pre_scale", 1.0 / (255.0 * std))
        self.register_buffer("pre_bias", -mean / std)

    def forward(self, x):
        x = x.float().permute(0, 3, 1, 2)  # (1,3,H,W) BGR (cast first: TRT rejects UINT8 intermediates)
        x = F.interpolate(
            x, size=self.resize_size, mode="bilinear", align_corners=False
        )
        x = x[:, [2, 1, 0]]  # BGR -> RGB (on the resized tensor)
        x = x * self.pre_scale + self.pre_bias
        return super().forward(x)


class P2PNetONNXFusedNV12(P2PNetONNX):
    """Fused pre-processing for packed NV12 from GStreamer appsink.

    Input: (1, H*W*3//2) uint8 packed NV12 (Y plane then interleaved UV).
    In-graph: NV12->RGB (BT.709 full-range) -> bilinear resize -> ImageNet normalize.
    """

    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        raw_size: tuple[int, int],
        resize_size: tuple[int, int],
    ):
        super().__init__(model, anchors)
        self.raw_h = int(raw_size[0])
        self.raw_w = int(raw_size[1])
        self.resize_size = list(resize_size)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pre_scale", 1.0 / (255.0 * std))
        self.register_buffer("pre_bias", -mean / std)

    def forward(self, x):
        # cast first: TRT rejects UINT8 intermediates
        x = x.float()
        h, w = self.raw_h, self.raw_w
        y_size = h * w
        y = x[:, :y_size].view(1, 1, h, w)
        uv = x[:, y_size:].view(1, h // 2, w // 2, 2)
        u = uv[..., 0:1].permute(0, 3, 1, 2) - 128.0  # (1,1,H/2,W/2)
        v = uv[..., 1:2].permute(0, 3, 1, 2) - 128.0
        # nearest upsample (ONNX-friendly; matches P2PNetNV12Detector)
        u = F.interpolate(u, size=(h, w), mode="nearest")
        v = F.interpolate(v, size=(h, w), mode="nearest")
        r = y + _A_V_R * v
        g = y - _A_U_G * u - _A_V_G * v
        b = y + _A_U_B * u
        rgb = torch.cat([r, g, b], dim=1).clamp(0, 255)  # (1,3,H,W)
        rgb = F.interpolate(
            rgb, size=self.resize_size, mode="bilinear", align_corners=False
        )
        rgb = rgb * self.pre_scale + self.pre_bias
        return super().forward(rgb)


def build_model(cfg_path: str, weight_path: str, device: str) -> P2PNet:
    cfg = OmegaConf.load(cfg_path)
    backbone = Backbone_VGG(cfg.network.backbone, True, False)
    model = P2PNet(backbone, cfg.network.row, cfg.network.line, device)
    state_dict = torch.load(weight_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resize_size_from_img_size(img_size) -> tuple[int, int]:
    """Match P2PNetDetector: snap each side down to a multiple of 128."""
    return (int(img_size[0]) // 128 * 128, int(img_size[1]) // 128 * 128)


def bake_anchors(model: P2PNet, h: int, w: int, device: str) -> torch.Tensor:
    dummy = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad():
        return model.anchor_points(dummy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        default="processor/modules/detector/p2pnet/conf/p2p.yaml",
    )
    parser.add_argument("--weight", default="weights/p2pnet/cutout.pth")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[4320, 7680],
        metavar=("H", "W"),
        help="Raw image size (height width); resized down to a multiple of 128.",
    )
    parser.add_argument("--out", default="weights/p2pnet/cutout.onnx")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fuse-preprocess",
        action="store_true",
        help="bake uint8 cast + BGR->RGB + resize + normalize into the graph "
        "(engine input becomes the raw uint8 BGR HWC frame)",
    )
    parser.add_argument(
        "--fuse-nv12",
        action="store_true",
        help="bake NV12->RGB + resize + normalize into the graph "
        "(engine input becomes packed uint8 NV12 of size H*W*3//2)",
    )
    args = parser.parse_args()

    assert Path(args.cfg).exists(), f"cfg not found: {args.cfg}"
    assert Path(args.weight).exists(), f"weight not found: {args.weight}"
    assert not (args.fuse_preprocess and args.fuse_nv12), (
        "--fuse-preprocess and --fuse-nv12 are mutually exclusive"
    )

    device = args.device
    raw_h, raw_w = int(args.img_size[0]), int(args.img_size[1])
    h, w = resize_size_from_img_size(args.img_size)
    if args.fuse_nv12:
        nv12_len = raw_h * raw_w * 3 // 2
        print(
            f"[export] raw img-size=({raw_h},{raw_w}) -> input (1,{nv12_len}) uint8 NV12"
            f" -> in-graph RGB resize to (1,3,{h},{w})"
        )
    elif args.fuse_preprocess:
        print(
            f"[export] raw img-size=({raw_h},{raw_w}) -> input (1,{raw_h},{raw_w},3) uint8"
            f" -> in-graph resize to (1,3,{h},{w})"
        )
    else:
        print(f"[export] raw img-size={tuple(args.img_size)} -> input (1,3,{h},{w})")
    print(f"[export] device={device}")

    model = build_model(args.cfg, args.weight, device)
    anchors = bake_anchors(model, h, w, device)
    print(f"[export] baked anchors: {tuple(anchors.shape)}")

    if args.fuse_nv12:
        wrapper = P2PNetONNXFusedNV12(
            model, anchors, (raw_h, raw_w), (h, w)
        ).eval()
        dummy = torch.randint(
            0, 256, (1, raw_h * raw_w * 3 // 2), dtype=torch.uint8, device=device
        )
    elif args.fuse_preprocess:
        wrapper = P2PNetONNXFused(model, anchors, (h, w)).eval()
        dummy = torch.randint(
            0, 256, (1, raw_h, raw_w, 3), dtype=torch.uint8, device=device
        )
    else:
        wrapper = P2PNetONNX(model, anchors).eval()
        dummy = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        scores, points = wrapper(dummy)
    print(
        f"[export] sanity forward: scores={tuple(scores.shape)} "
        f"points={tuple(points.shape)}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["scores", "points"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export] wrote ONNX: {out_path}")


if __name__ == "__main__":
    main()
