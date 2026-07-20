# TensorRT エンジンビルド

エッジ本番は **NV12 融合** エンジンを使う。GStreamer appsink の packed NV12 をそのまま入力し、NV12→RGB・resize・ImageNet 正規化はグラフ内で行う。

## フロー

```mermaid
flowchart LR
  ckpt[cutout.pth] --> onnx[export_fuse_nv12]
  onnx --> engine[build_fp16]
  engine --> yaml[base.yaml_engine_path]
```

| 段階 | スクリプト | 成果物 |
|------|------------|--------|
| 1. ONNX エクスポート | [`trt_scripts/export_p2pnet_onnx.py`](../../trt_scripts/export_p2pnet_onnx.py) | `*_nv12.onnx` |
| 2. エンジンビルド | [`trt_scripts/build_p2pnet_engine.py`](../../trt_scripts/build_p2pnet_engine.py) | `*_nv12.engine` |
| 3. 設定反映 | [`pipeline_jetson/config/base.yaml`](../../pipeline_jetson/config/base.yaml) | `detector.engine_path` / `img_size` / `size` |

TensorRT は JetPack 付属を使う（pip の `tensorrt` は入れない）。venv は `uv venv --system-site-packages` が前提。

## FHD（1920×1080）— 本番推奨

```bash
# 1. ONNX（NV12→RGB・resize・正規化を融合）
uv run python trt_scripts/export_p2pnet_onnx.py \
  --img-size 1080 1920 \
  --fuse-nv12 \
  --yuv-matrix bt601-limited \
  --out weights/p2pnet/cutout_fhd_nv12_bt601lim.onnx \
  --device cpu

# 2. TensorRT エンジン（FP16）
uv run python trt_scripts/build_p2pnet_engine.py \
  --onnx weights/p2pnet/cutout_fhd_nv12_bt601lim.onnx \
  --engine weights/p2pnet/cutout_fhd_nv12_bt601lim.engine \
  --fp16 \
  --workspace-gb 16
```

`base.yaml` 側の一致例:

```yaml
size: [1920, 1080]              # W, H
detector:
  engine_path: weights/p2pnet/cutout_fhd_nv12_bt601lim.engine
  img_size: [1080, 1920]        # H, W
```

## エクスポートの要点（`--fuse-nv12`）

- 入力: packed uint8 NV12、形状 `(1, H*W*3//2)`
- グラフ内: NV12→RGB（`--yuv-matrix`: `bt601-limited` または従来の `bt709-full`）、bilinear resize（128 倍数）、ImageNet normalize、P2PNet、softmax 済み scores
- 出力: `scores` `(1, N)`、`points` `(1, N, 2)`（リサイズ空間）
- アンカーは解像度固定で定数化、FPN の推論用スリム forward を使用

Python 推論側（`P2PNetTRTNV12Detector`）は H2D + `execute_async_v3` + 後処理のみ。詳細は [04_detection.md](../stages/04_detection.md)。

## ベンチマーク

```bash
uv run python trt_scripts/bench_p2pnet_trt.py \
  --engine weights/p2pnet/cutout_fhd_nv12_bt601lim.engine \
  --input_img_size 1080 1920

uv run python trt_scripts/bench_p2pnet_ckpt.py \
  --input_img_size 1080 1920
```

## 参考: BGR 融合エンジン

`--fuse-preprocess` は uint8 BGR HWC 入力のエンジンを作る。エッジ本番の GStreamer NV12 経路では使わない。

```bash
uv run python trt_scripts/export_p2pnet_onnx.py \
  --img-size 1080 1920 \
  --fuse-preprocess \
  --out weights/p2pnet/cutout_fhd_fused.onnx \
  --device cpu

uv run python trt_scripts/build_p2pnet_engine.py \
  --onnx weights/p2pnet/cutout_fhd_fused.onnx \
  --engine weights/p2pnet/cutout_fhd_fused.engine \
  --fp16 \
  --workspace-gb 16
```

対応する Python バックエンドは `P2PNetTRTDetector`（[`p2pnet_trt.py`](../../processor/detector/p2pnet_trt.py)）。

## INT8（任意）

`build_p2pnet_engine.py` は `--int8` とキャリブレーションデータ指定をサポートする。本番の既定パスは FP16 NV12 融合。INT8 を使う場合は入力レイアウト（NV12 融合か BGR 融合か）に合わせたキャリブレーションが必要。

## 関連文書

- [configuration.md](../reference/configuration.md)
- [04_detection.md](../stages/04_detection.md)
- [data_contracts.md](../reference/data_contracts.md)
