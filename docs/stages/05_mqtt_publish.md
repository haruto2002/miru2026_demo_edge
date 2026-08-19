# 工程: MQTT 配信

## 目的

検出結果を JSON にして集約側の MQTT ブローカへ送る。エッジは検出点のみを載せ、追跡 ID は付けない。

## 入出力

| | 内容 |
|--|------|
| 入力 | `frame_id`（キャプチャ `seq`）、`timestamp`（Unix epoch）、`detections` 辞書 |
| 出力 | トピック `camera/<camera_name>` への publish（QoS 0, retain false） |

### ペイロード化

`dets_to_payload`（[`edge_app.py`](../../pipeline_jetson/edge_app.py)）が `Nx3` 配列を次の形に変換する:

```json
{
  "point": [{"x": 100.0, "y": 200.0, "score": 0.9}, ...],
  "bbox": []
}
```

`bbox` は常に空配列（点検出パイプラインのため）。

### 完全なメッセージ例

```json
{
  "camera_name": "worldporter_partial_01",
  "pc_name": "jetson_01",
  "timestamp": 1710000000.0,
  "frame_id": 123,
  "detections": {
    "point": [{"x": 100.0, "y": 200.0, "score": 0.9}],
    "bbox": []
  }
}
```

座標はエッジのセンサ解像度（例: FHD）。詳細契約は [mqtt_contract.md](../reference/mqtt_contract.md)。

## 実装

| 要素 | 場所 |
|------|------|
| `dets_to_payload` | [`edge_app.py`](../../pipeline_jetson/edge_app.py) |
| `Publisher` | [`publisher.py`](../../pipeline_jetson/components/edge/publisher.py) |
| クライアント | `paho.mqtt.client` |

### ライフサイクル

1. `start()` — `connect(broker_host, broker_port, keepalive=60)` と `loop_start()`
2. フレームごと `publish_detections(frame_id, timestamp, detections)`
3. `stop()` — `loop_stop()` と `disconnect()`

`client_id` は `{pc_name}_{camera_name}_publisher`。トピックは `camera/{camera_name}`。

### レガシー API

`publish_result(frame_id, timestamp, objects)` は追跡済み objects を載せる旧形式。`EdgeApp` 本番ループでは呼ばれない。互換のため残っている。

## 設定キー

`broker_*` / `_target_` は [`base.yaml`](../../pipeline_jetson/config/base.yaml)、`camera_name` / `pc_name` は機種 YAML（`jetsonN.yaml`）。

| キー | 説明 |
|------|------|
| `publisher._target_` | `pipeline_jetson.components.edge.publisher.Publisher` |
| `publisher.broker_host` | 集約側 mosquitto 等のホスト |
| `publisher.broker_port` | 例: `1883` |
| `publisher.camera_name` | トピック末尾とペイロード |
| `publisher.pc_name` | ペイロードと client_id |

`publisher` 自体を YAML から外すと `EdgeApp` は配信なしで動く（デバッグ用）。

## 失敗・タイムアウト時の挙動

- 接続失敗は `start()` 時に paho の例外・挙動に依存（アプリ側の専用リトライはない）
- publish は QoS 0 のため到達保証なし。ネットワーク瞬断時の再送はブローカ／クライアント設定次第
- 検出 0 件でも空の `point` 配列で publish する（フレーム欠落ではなく「検出なし」を伝える）

## 関連コード

- [`publisher.py`](../../pipeline_jetson/components/edge/publisher.py) — `publish_detections`, `start`, `stop`
- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — `dets_to_payload`, `publish_detections` 呼び出し
- [mqtt_contract.md](../reference/mqtt_contract.md)
