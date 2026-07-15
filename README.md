# 混雑状況把握システム — Jetson エッジ

Jetson AGX Orin 上でカメラ映像をリアルタイム解析し、人物検出結果を MQTT で集約側へ配信するエッジ処理です。

追跡・マップ描画・ダッシュボードは集約側で行い、エッジは検出に計算資源を集中します。

## 処理の流れ

```
RTSP / 動画
  → GStreamer (NV12, フレームドロップなし)
  → TensorRT P2PNet（NV12 融合エンジン）
  → MQTT publish（検出点のみ）
```

- 入力は GStreamer で NV12 にデコードし、appsink の backpressure でフレームを落とさない
- 検出は NV12→RGB・正規化込みの TensorRT エンジン（`--fuse-nv12`）を使用
- MQTT トピック `camera/<camera_name>` に検出点 `(x, y, score)` を JSON で配信
- 任意で検出オーバーレイを `nveglglessink` 等に表示可能

実装レベルのフロー・工程別詳細・MQTT 契約などは [docs/](docs/README.md) を参照。

## 前提環境

| 項目 | 内容 |
|------|------|
| 端末 | Jetson AGX Orin（aarch64） |
| JetPack | 5.1.2 |
| Python | 3.8 |
| TensorRT | JetPack 付属 8.5.2（pip では入れない） |
| MQTT | 集約 PC 上の mosquitto 等 |

## 環境構築

```bash
uv venv --system-site-packages
uv sync
```

`--system-site-packages` が必須です。TensorRT は JetPack のシステムパッケージ（`/usr/lib/python3.8/dist-packages/tensorrt`）を利用します。PyPI の `tensorrt` を入れると aarch64 で失敗します。

依存の torch は Jetson 向け wheel（JP 5.1.2 / PyTorch 2.1）を `pyproject.toml` で指定しています。

## 実行方法

集約 PC 側で MQTT ブローカが起動していることを確認してから実行します。

```bash
# 推奨
./run.sh

# または
uv run python run.py --cfg pipeline_jetson/config/edge.yaml
```

GUI 表示を使う場合は X11 の DISPLAY を設定します。

```bash
export DISPLAY=:1
```

### 主な設定（`pipeline_jetson/config/edge.yaml`）

| キー | 説明 |
|------|------|
| `source` | RTSP URL または動画パス |
| `size` | `[W, H]`（エンジン入力と一致させる） |
| `transport` | RTSP は `tcp` 推奨 |
| `capture_fps` | カメラ 30fps から意図的に間引く場合（例: `15`） |
| `prefetch` | 次フレームの NV12 ホストコピーと推論を重ねる |
| `display` | 検出オーバーレイ表示の ON/OFF |
| `detector.engine_path` | NV12 融合 TRT エンジン |
| `publisher.broker_host` | 集約側 mosquitto のホスト |

## MQTT ペイロード

トピック: `camera/<camera_name>`

```json
{
  "camera_name": "worldporter",
  "pc_name": "jetson01",
  "timestamp": 1710000000.0,
  "frame_id": 123,
  "detections": {
    "point": [{"x": 100.0, "y": 200.0, "score": 0.9}],
    "bbox": []
  }
}
```

`timestamp` は Unix epoch（秒, float）です。最初のフレームをエッジが受け取った壁時計を起点に、以降は GStreamer PTS の差分で進めます（処理遅延で間隔が揺れない）。カメラの真の撮影時刻（RTCP/NTP）ではありません。

座標はエッジ側のセンサ解像度（例: FHD）です。追跡 ID は含めません。

## ディレクトリ構成

```
realtime/
├── run.py / run.sh              エントリポイント
├── pyproject.toml               依存定義（uv）
├── pipeline_jetson/             Jetson エッジアプリ
│   ├── edge_app.py              キャプチャ → 検出 → 配信のループ
│   ├── config/edge.yaml         パイプライン設定
│   └── components/edge/
│       ├── gst_io.py            GStreamer NV12 キャプチャ / 表示
│       ├── publisher.py         MQTT 配信
│       └── timer.py             処理時間計測
├── processor/                   検出・追跡の実装
│   ├── detector/
│   │   ├── p2pnet_trt_nv12.py   ★ 本番: NV12 融合 TRT
│   │   ├── p2pnet_trt.py        BGR 融合 TRT
│   │   ├── p2pnet_nv12.py       NV12 の PyTorch 参照実装
│   │   └── p2pnet.py            通常の P2PNet
│   ├── tracker/                 ByteTrack 系（集約側向け）
│   └── modules/                 外部実装本体・設定 YAML
├── trt_scripts/                 ONNX エクスポート / TRT ビルド / ベンチ
└── weights/p2pnet/              重み・ONNX・engine
```

## TensorRT エンジンのビルド

本番パスは **NV12 融合**（GStreamer appsink の packed NV12 をそのまま入力）です。

### FHD（1920×1080）— 推奨

```bash
# 1. ONNX エクスポート（NV12→RGB・resize・正規化をグラフ内に融合）
uv run python trt_scripts/export_p2pnet_onnx.py \
  --img-size 1080 1920 \
  --fuse-nv12 \
  --out weights/p2pnet/cutout_fhd_nv12.onnx \
  --device cpu

# 2. TensorRT エンジンビルド（FP16）
uv run python trt_scripts/build_p2pnet_engine.py \
  --onnx weights/p2pnet/cutout_fhd_nv12.onnx \
  --engine weights/p2pnet/cutout_fhd_nv12.engine \
  --fp16 \
  --workspace-gb 16
```

`edge.yaml` の `detector.engine_path` / `img_size` / `size` は、この解像度と一致させてください。

### （参考）BGR 融合エンジン

`--fuse-preprocess` で uint8 BGR HWC 入力のエンジンも作れます。エッジ本番の GStreamer NV12 経路では使いません。

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

### ベンチマーク

```bash
uv run python trt_scripts/bench_p2pnet_trt.py \
  --engine weights/p2pnet/cutout_fhd_nv12.engine \
  --input_img_size 1080 1920

uv run python trt_scripts/bench_p2pnet_ckpt.py \
  --input_img_size 1080 1920
```

## カメラ接続確認（GStreamer）

RTSP 形式: `rtsp://<user>:<pass>@<ip>:554/...`

Jetson で映像まで確認:

```bash
gst-launch-1.0 -v \
  rtspsrc location='rtsp://USER:PASS@192.168.0.11:554/ONVIF/MediaInput?profile=def_profile1' \
  protocols=tcp latency=0 ! \
  rtph265depay ! h265parse ! nvv4l2decoder ! \
  nvvidconv ! xvimagesink sync=false
```

接続のみ確認（表示なし）:

```bash
gst-launch-1.0 -v \
  rtspsrc location='rtsp://USER:PASS@192.168.0.11:554/ONVIF/MediaInput?profile=def_profile1' \
  protocols=tcp latency=0 ! \
  rtph265depay ! h265parse ! nvv4l2decoder ! \
  fakesink sync=false
```

## 詳細ドキュメント

実装レベルのパイプライン説明は [docs/](docs/README.md) にあります。

| 文書 | 内容 |
|------|------|
| [docs/01_pipeline_overview.md](docs/01_pipeline_overview.md) | 全体フローと設計原則 |
| [docs/stages/](docs/stages/) | 工程別詳細（キャプチャ・検出・MQTT など） |
| [docs/reference/](docs/reference/) | 設定・MQTT 契約・データ契約 |
| [docs/build/tensorrt_engine.md](docs/build/tensorrt_engine.md) | ONNX / TensorRT エンジンビルド |
