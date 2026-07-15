# 工程: デバッグ表示（任意）

## 目的

検出点を映像に重ねてローカル表示し、動作確認する。本番運用では通常オフし、NV12→BGR 変換と描画のコストを避ける。

## 入出力

| | 内容 |
|--|------|
| 入力 | 推論に使った同一 `nv12` と `dets` |
| 中間 | BGR `uint8` `(H, W, 3)` |
| 出力 | GStreamer シンク（`nveglglessink` 等）への push |

`display: false` のとき、ループ内の変換・描画・`push` は実行されない（計測の `t_disp` もほぼゼロ）。

## 実装

| 要素 | 場所 |
|------|------|
| `nv12_to_bgr` | [`edge_app.py`](../../pipeline_jetson/edge_app.py) — OpenCV `COLOR_YUV2BGR_NV12` |
| `draw_detections` | 同上 — 閾値以上の点を円で描画 |
| `GstBgrDisplay` | [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) |

### 表示パイプライン（`nveglglessink`）

```
appsrc (BGR) ! videoconvert ! RGBA
  ! nvvidconv ! NVMM RGBA ! nvegltransform ! nveglglessink sync=false
```

`nvegltransform` は Jetson で NVMM surface-array を EGL が直接扱えない問題への対処。`nv3dsink` は transform なしの NVMM 経路。`ximagesink` / `autovideosink` / `fakesink` は CPU 経路。

appsrc は `block=True`（表示が遅いと backpressure）。

### 描画パラメータ

`draw_detections` はスコアが `display_threshold` 未満の点を描かない。検出器の `threshold`（推論後処理）とは別パラメータである点に注意。

## 設定キー

| キー | 説明 |
|------|------|
| `display` | `true` でオーバーレイ表示 |
| `display_sink` | 例: `nveglglessink` / `nv3dsink` / `ximagesink` |
| `display_threshold` | 描画用スコア閾値（例: `0.5`） |
| `display_point_size` | 円半径 |
| `display_color` | BGR 色（例: `[0, 0, 255]`） |

GUI 利用時は X11 の `DISPLAY` が必要（例: `export DISPLAY=:1`）。

## 失敗・タイムアウト時の挙動

- `display.push` が `False`（appsrc 消失や flush 失敗）→ `[edge] display push failed; stopping` でループ終了
- `enabled=False` で構築した場合、内部シンクは `fakesink`（ウィンドウなし）。ただし `EdgeApp` は `display_enabled` が偽なら `push` 自体を呼ばない

## 本番推奨

- `display: false` のまま運用し、必要時のみオン
- 表示オンは e2e レイテンシと CPU/GPU 負荷を増やす（ログの `disp=` と `e2e=` で確認可能）

## 関連コード

- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — `nv12_to_bgr`, `draw_detections`, display 分岐
- [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) — `GstBgrDisplay._build_desc`, `push`
