# MQTT 契約（集約側向け）

エッジが集約側へ送るメッセージの契約。実装は [`Publisher.publish_detections`](../../pipeline_jetson/components/edge/publisher.py)。

## トピック

```text
camera/<camera_name>
```

`camera_name` は機種 YAML（`jetsonN.yaml`）の `publisher.camera_name`（例: `worldporter_partial_01`）。

## QoS / retain

| 項目 | 値 |
|------|-----|
| QoS | `0`（到達保証なし） |
| retain | `false` |

## JSON スキーマ

```json
{
  "camera_name": "string",
  "pc_name": "string",
  "timestamp": 1710000000.0,
  "frame_id": 123,
  "detections": {
    "point": [
      {"x": 100.0, "y": 200.0, "score": 0.9}
    ],
    "bbox": []
  }
}
```

| フィールド | 型 | 意味 |
|------------|-----|------|
| `camera_name` | string | カメラ識別子（トピック末尾と同一）。機種 YAML の `publisher.camera_name` |
| `pc_name` | string | エッジ端末名。機種 YAML の `publisher.pc_name` |
| `timestamp` | number (float) | Unix epoch 秒。詳細は下記 |
| `frame_id` | number (int) | エッジが採用したフレームの通番（`seq`）。1 始まり |
| `detections.point` | array | 検出点のリスト。0 件もあり得る |
| `detections.point[].x` | number | センサ座標系の X（画素） |
| `detections.point[].y` | number | センサ座標系の Y（画素） |
| `detections.point[].score` | number | 検出スコア（閾値通過済み） |
| `detections.bbox` | array | 常に `[]`（点検出パイプライン） |

## 座標系

- 原点は画像左上、単位は画素
- 解像度はエッジの `size` / `img_size`（例: 1920×1080）
- 集約側でクロップやマップ変換する場合は、エッジ解像度を前提に変換すること

## timestamp の意味

- 最初のフレームをエッジが受け取った壁時計をアンカーし、以降は GStreamer PTS の差分で進める
- 同一ラン内のフレーム間隔は処理ジッタの影響を受けにくい
- **カメラの真の撮影時刻（RTCP / NTP）ではない**
- 詳細: [03_timestamp.md](../stages/03_timestamp.md)

## 追跡 ID

エッジは追跡しない。ペイロードに track id / object id は含まれない。追跡は集約側で行う。

## レガシー形式（非本番）

`publish_result` は `objects` キーを持つ旧形式。現行 `EdgeApp` は使用しない。新規の集約実装は `detections` 形式のみを前提にすればよい。

## クライアント ID（参考）

エッジ側 MQTT client_id: `{pc_name}_{camera_name}_publisher`

## 関連文書

- [05_mqtt_publish.md](../stages/05_mqtt_publish.md)
- [configuration.md](configuration.md)
