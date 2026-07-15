# 実装ドキュメント

このディレクトリは、Jetson エッジ本番パイプラインの**実装レベルの説明**です。起動手順・環境構築はリポジトリ直下の [README.md](../README.md) を参照してください。

## 読む順番

1. **[01_pipeline_overview.md](01_pipeline_overview.md)** — 全体フローと設計原則
2. **[stages/](stages/)** — 処理工程ごとの詳細（起動 → キャプチャ → 時刻 → 検出 → MQTT → 表示）
3. 必要に応じて **[reference/](reference/)**（設定・契約）と **[build/](build/)**（TRT エンジン）

## 目次

| 文書 | 内容 |
|------|------|
| [01_pipeline_overview.md](01_pipeline_overview.md) | システム境界、フレーム単位の流れ、設計原則 |
| [stages/01_entry_and_wiring.md](stages/01_entry_and_wiring.md) | `run.py` / Hydra / `EdgeApp` の配線 |
| [stages/02_capture.md](stages/02_capture.md) | GStreamer NV12 キャプチャと prefetch |
| [stages/03_timestamp.md](stages/03_timestamp.md) | PTS → Unix epoch 変換 |
| [stages/04_detection.md](stages/04_detection.md) | NV12 融合 TensorRT 検出 |
| [stages/05_mqtt_publish.md](stages/05_mqtt_publish.md) | 検出結果の MQTT 配信 |
| [stages/06_display.md](stages/06_display.md) | デバッグ用オーバーレイ表示 |
| [reference/configuration.md](reference/configuration.md) | `edge.yaml` 全キー解説 |
| [reference/mqtt_contract.md](reference/mqtt_contract.md) | 集約側向け MQTT 契約 |
| [reference/data_contracts.md](reference/data_contracts.md) | フレーム・検出・解像度の約束 |
| [build/tensorrt_engine.md](build/tensorrt_engine.md) | ONNX エクスポート〜エンジンビルド |

## スコープ

- **対象**: エッジ本番経路（キャプチャ → 検出 → MQTT）
- **対象外（簡潔な言及のみ）**: `processor/tracker/`、YOLO / DEIMV2、未使用の `timer.py`
