# 工程: デバッグ表示（任意）

## 目的

検出点を映像に重ねてローカル表示し、動作確認する。本番運用では通常オフし、NV12→BGR 変換と描画のコストを避ける。

表示が有効でも、変換・描画・push は **別スレッド** で行い、検出・MQTT のメインループをブロックしない。

## 入出力

| | 内容 |
|--|------|
| 入力 | 推論に使った同一 `nv12` と `dets`（メインから非ブロッキング投入） |
| 中間 | BGR `uint8` `(H, W, 3)`（表示ワーカー内） |
| 出力 | GStreamer シンク（`nveglglessink` 等）への push |

`display: false` のとき、投入・変換・描画・`push` は実行されない（計測の `t_disp` もほぼゼロ）。

## 実装

| 要素 | 場所 |
|------|------|
| `AsyncDetectionDisplay` | [`edge_app.py`](../../pipeline_jetson/edge_app.py) — 深さ1キュー＋ワーカー |
| `nv12_to_bgr` | 同上 — OpenCV `COLOR_YUV2BGR_NV12`（ワーカー側） |
| `draw_detections` | 同上 — 閾値以上の点を円で描画 |
| `GstBgrDisplay` | [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) |

### 非同期・最新優先

```
main: infer → publish → async_display.submit(nv12, dets)  # 待たない
worker: queue(get) → nv12_to_bgr → draw_detections → display.push
```

- キュー深さは **1**。満杯時は未処理ジョブを捨てて新しいフレームで上書きする
- 表示が推論より遅いと画面側のフレームは間引かれるが、検出・MQTT は止まらない
- ログの `disp=` は **submit の所要時間**（ほぼ 0）。BGR 変換時間はメインの e2e に含まれない

### 表示パイプライン（`nveglglessink`）

```
appsrc (BGR) ! videoconvert ! RGBA
  ! nvvidconv ! NVMM RGBA ! nvegltransform ! nveglglessink sync=false
```

`nvegltransform` は Jetson で NVMM surface-array を EGL が直接扱えない問題への対処。`nv3dsink` は transform なしの NVMM 経路。`ximagesink` / `autovideosink` / `fakesink` は CPU 経路。

appsrc は `block=True`（表示が遅いとワーカー側で待つ。メインは影響を受けない）。

### 描画パラメータ

`draw_detections` はスコアが `display_threshold` 未満の点を描かない。検出器の `threshold`（推論後処理）とは別パラメータである点に注意。

## 設定キー

描画スタイルは [`base.yaml`](../../pipeline_jetson/config/base.yaml)。`display` の ON/OFF は機種 YAML で上書き（例: jetson1 のみ `true`）。

| キー | 説明 |
|------|------|
| `display` | `true` でオーバーレイ表示（常に非同期） |
| `display_sink` | 例: `nveglglessink` / `nv3dsink` / `ximagesink` |
| `display_threshold` | 描画用スコア閾値（例: `0.5`） |
| `display_point_size` | 円半径（例: `10`） |
| `display_color` | BGR 色（例: `[0, 0, 255]`） |

GUI 利用時は X11 の `DISPLAY` が必要（例: `export DISPLAY=:1`）。

## 失敗・タイムアウト時の挙動

- `display.push` が `False`（appsrc 消失や flush 失敗）→ 表示ワーカーが終了しログを出す。**検出・MQTT ループは継続**
- `display: false` のときは `AsyncDetectionDisplay` を作らず、変換・描画は一切走らない

## 本番推奨

- `display: false` のまま運用し、必要時のみオン
- 表示オンでもメインの検出/MQTT は守られるが、CPU/GPU の争合は残る（負荷を下げたいならオフ）

## 関連コード

- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — `AsyncDetectionDisplay`, `nv12_to_bgr`, `draw_detections`
- [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) — `GstBgrDisplay._build_desc`, `push`
