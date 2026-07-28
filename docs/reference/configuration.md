# 設定リファレンス（base.yaml + jetsonN.yaml）

本番設定は次の2層:

| ファイル | 役割 |
|----------|------|
| [`pipeline_jetson/config/base.yaml`](../../pipeline_jetson/config/base.yaml) | 艦隊共通（解像度・検出器・broker・prefetch など） |
| [`pipeline_jetson/config/jetsonN.yaml`](../../pipeline_jetson/config/) | 機器固有（`source` / キャリブ / `camera_name` / `pc_name` / `display` など） |

`run.py` が機種 YAML の `extends: base.yaml` を解決し、`OmegaConf.merge(base, override)` してから `instantiate` する。起動例: `./run.sh jetson1` または `uv run python run.py --cfg pipeline_jetson/config/jetson1.yaml`。

Hydra の `instantiate` 用。トップレベル `_target_` が `EdgeApp`。`_recursive_: false` のため、ネストした `detector` / `publisher` は `EdgeApp` 内で改めてインスタンス化される。

RTSP の認証情報は実ファイルに書かれることがあるが、文書ではプレースホルダのみを示す。

## どこに書くか

| 置き場 | キー例 |
|--------|--------|
| 機種 YAML（高） | `extends`、`source`、`transport`、`calibration_*`、`display`、`publisher.camera_name` / `pc_name` |
| base（中〜低、必要なら機種で上書き） | `size`、`capture_fps`、`detector.*`、`publisher.broker_*`、`prefetch*`、`max_buffers`、描画スタイル |

## トップレベル

| キー | 型 / 例 | 説明 |
|------|---------|------|
| `extends` | `base.yaml` | 機種 YAML のみ。マージ後に削除される |
| `_target_` | `pipeline_jetson.edge_app.EdgeApp` | アプリクラス |
| `_recursive_` | `false` | ネストの自動 instantiate を無効化 |
| `source` | `rtsp://USER:PASS@HOST:554/...` / 動画パス / `/dev/video0` | 入力ソース（機種 YAML） |
| `size` | `[1920, 1080]` | **(W, H)**。NV12 出力・表示サイズ。エンジンと一致必須（`base.yaml`） |
| `transport` | `tcp` | RTSP のプロトコル。RTSP 時は必須（機種 YAML） |
| `max_buffers` | `4` | appsink 深さ |
| `capture_fps` | `15` または `null` | PTS スキップによる間引き。`null` で全フレーム |
| `drop_frames_when_lagging` | `true` / `false` | `true` で appsink `drop=true`。満杯時に古いフレームを捨ててリアルタイム性を優先 |
| `calibration_enabled` | `true` | `true` でホモグラフィ射影変換を有効化（機種 YAML） |
| `calibration_homography_path` | `calib_data/homography.txt` | 3x3 ホモグラフィ行列ファイル（機種 YAML） |
| `prefetch` | `true` | 次フレーム host コピーと推論のオーバーラップ |
| `prefetch_queue_size` | `10` | prefetch キュー深さ（`base.yaml`） |
| `report_every` | `30` | N フレームごとに wait/det/pub/e2e をログ |
| `max_wall_seconds` | `null` または秒 | 壁時計で停止 |
| `max_frames` | `null` または整数 | 処理フレーム数で停止 |
| `display` | `false` | 検出オーバーレイ表示（既定は base。必要機種のみ上書き） |
| `display_sink` | `nveglglessink` | 表示シンク名 |
| `display_point_size` | `10` | 描画円の半径 |
| `display_threshold` | `0.5` | 描画用スコア閾値（検出 `threshold` とは別） |
| `display_color` | `[0, 0, 255]` | BGR |

## detector

| キー | 型 / 例 | 説明 |
|------|---------|------|
| `_target_` | `processor.detector.p2pnet_trt_nv12.P2PNetTRTNV12Detector` | 本番検出器 |
| `engine_path` | `weights/p2pnet/cutout_fhd_nv12_bt601lim.engine` | NV12 融合 TRT エンジン（BT.601 limited） |
| `device` | `cuda:0` | Torch / CUDA デバイス |
| `threshold` | `0.1` | 推論後処理のスコア閾値 |
| `img_size` | `[1080, 1920]` | **(H, W)**。packed NV12 の高さ・幅 |

## publisher

| キー | 型 / 例 | 説明 |
|------|---------|------|
| `_target_` | `pipeline_jetson.components.edge.publisher.Publisher` | MQTT 配信 |
| `broker_host` | `192.168.0.100` | 集約側ブローカ |
| `broker_port` | `1883` | ポート |
| `camera_name` | `worldporter_partial_01` | トピック `camera/<name>` とペイロード |
| `pc_name` | `jetson_01` | ペイロードと MQTT client_id |

`publisher` ブロックを省略すると配信なしで動作する。

## 解像度の一致契約

次の3つは同じ画素サイズを指す必要がある。

| 設定 | 表記 | 例（FHD） |
|------|------|-----------|
| `size` | `[W, H]` | `[1920, 1080]` |
| `detector.img_size` | `[H, W]` | `[1080, 1920]` |
| エンジン入力 | `(1, H*W*3//2)` uint8 | `(1, 3110400)` |

不一致はエンジンロード時または `infer` 時に失敗する。エンジンの作り方は [tensorrt_engine.md](../build/tensorrt_engine.md)。

## 環境変数

| 変数 | 用途 |
|------|------|
| `DISPLAY` | `display: true` のとき X11 ディスプレイ（例: `:1`） |

アプリは `.env` を読まない。設定の本体は YAML。

## 関連文書

- [01_entry_and_wiring.md](../stages/01_entry_and_wiring.md)
- [data_contracts.md](data_contracts.md)
- [mqtt_contract.md](mqtt_contract.md)
