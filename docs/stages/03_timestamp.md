# 工程: タイムスタンプ変換

## 目的

GStreamer の PTS（メディア相対時刻）を、MQTT に載せる Unix epoch（秒, float）に変換する。フレーム間隔は処理ジッタではなくメディア時間に追従させる。

## 入出力

| | 内容 |
|--|------|
| 入力 | `pts`（秒）。`GstNv12Capture.pull` がバッファ PTS から算出 |
| 出力 | Unix epoch 秒（float）。`Publisher.publish_detections` の `timestamp` になる |

## 実装

| 要素 | 場所 |
|------|------|
| `PtsUnixEpochClock` | [`pipeline_jetson/edge_app.py`](../../pipeline_jetson/edge_app.py) |
| 呼び出し | `EdgeApp.run()` 内、`pull` 直後 |

### 変換式

最初の有効サンプルで壁時計と PTS をアンカーし、以降は差分で進める。

```text
timestamp = base_wall + (pts - base_pts)
```

- `base_wall = time.time()`（最初のフレームをエッジが受け取った瞬間）
- `base_pts =` そのフレームの PTS
- 以後のフレームは PTS が進んだ分だけ epoch が進む

これにより、検出や MQTT が遅くなっても **タイムスタンプ間隔は PTS 間隔のまま** です。処理遅延で「時刻が伸びる」ことはありません。

### アンカーの寿命

`EdgeApp.run()` の開始時に `PtsUnixEpochClock()` を1つ作り、そのラン全体で使い回します。コメントどおり、PTS がリセットされる場合は新しいランが必要です（`reset()` はあるがループ内では呼ばれない）。

## 設定キー

この工程専用の YAML キーはない。入力の PTS 品質はキャプチャ側（ソースと GStreamer）に依存する。

## 意味と限界（集約側向け）

| 事実 | 説明 |
|------|------|
| 相対間隔は安定 | 同一ラン内でフレーム間の Δt は PTS に忠実 |
| 絶対時刻は近似 | カメラの真の撮影時刻（RTCP / NTP）ではない |
| 起点はエッジ受信 | 最初のフレーム到着時の壁時計が原点 |
| 追跡 ID とは無関係 | 時刻フィールドのみ。ID はエッジでは付けない |

詳細な MQTT 契約は [mqtt_contract.md](../reference/mqtt_contract.md) を参照。

## 失敗・タイムアウト時の挙動

- PTS が `Gst.CLOCK_TIME_NONE` の場合、キャプチャ側は `pts_s = 0.0` を返す。以降の相対計算は 0 起点になる
- 時計クラス自体は例外を投げない

## 関連コード

- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — `PtsUnixEpochClock`, `EdgeApp.run` 内の `epoch_clock.to_unix(pts)`
- [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) — `_pull_one` での PTS → 秒変換
- [`publisher.py`](../../pipeline_jetson/components/edge/publisher.py) — `timestamp` フィールドの意味を docstring で記載
