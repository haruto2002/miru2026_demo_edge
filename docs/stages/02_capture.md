# 工程: キャプチャ（GStreamer NV12）

## 目的

RTSP / 動画ファイル / USB（V4L2）をデコードし、必要ならホモグラフィ射影変換を適用した packed NV12（ホスト上の `uint8` 配列）を順序どおり供給する。負荷時にフレームを捨てず、必要なら PTS ベースで意図的に間引く。

## 入出力

| | 内容 |
|--|------|
| 入力 | `source`（`rtsp://...` / ファイルパス / `/dev/video*`）、解像度 `size` = `(W, H)` |
| 出力 | `(nv12, seq, pts)` — `nv12` は長さ `W*H*3//2` の `uint8`、`seq` は 1 始まりの通番、`pts` は秒（float） |
| 終端 | EOS / 致命エラーで `None`。待ち時間超過は `PullTimeout` |

## 実装

| クラス | ファイル | 役割 |
|--------|----------|------|
| `GstNv12Capture` | [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) | パイプライン構築と `pull` |
| `Nv12HomographyWarper` | [`nv12_homography.py`](../../pipeline_jetson/components/edge/nv12_homography.py) | CUDA 上で NV12 射影変換 |
| `PrefetchNv12Capture` | 同上 | 次フレームの host コピーを別スレッドで先行 |
| `PullTimeout` | 同上 | タイムアウト例外 |

### GStreamer パイプライン

**RTSP**

```
rtspsrc location=... protocols=<tcp|udp> latency=200 ! parsebin
  ! nvv4l2decoder ! nvvidconv
  ! video/x-raw,format=NV12,width=W,height=H
  ! appsink name=sink emit-signals=false max-buffers=N drop=false sync=false
```

**ファイル**

```
filesrc location=... ! qtdemux ! parsebin
  ! nvv4l2decoder ! nvvidconv
  ! video/x-raw,format=NV12,width=W,height=H
  ! appsink ...（同上）
```

**USB（V4L2）**

```
v4l2src device=/dev/video0
  ! videoconvert
  ! video/x-raw,format=NV12,width=W,height=H
  ! appsink name=sink emit-signals=false max-buffers=N drop=false sync=false
```

**calibration ON（RTSP / ファイル）**

```
rtspsrc/filesrc ... ! parsebin
  ! nvv4l2decoder ! nvvidconv
  ! video/x-raw(memory:NVMM),format=NV12,width=W,height=H
  ! nvvidconv
  ! video/x-raw,format=NV12,width=W,height=H
  ! appsink ...（同上）
  -> pull() 内で CUDA 射影変換（BGR 変換なし）
```

**calibration ON（USB）**

```
v4l2src device=/dev/video0
  ! videoconvert
  ! video/x-raw,format=NV12,width=W,height=H
  ! appsink ...（同上）
  -> pull() 内で CUDA 射影変換（BGR 変換なし）
```

要点:

- `parsebin` が H.264 / H.265 と depay を自動選択
- `drop_frames_when_lagging: false` のとき、`drop=false` + 有限 `max-buffers` → 満杯時はデコーダ側がブロック（backpressure）。負荷でフレームを捨てない
- `drop_frames_when_lagging: true` のとき、`drop=true` → appsink 満杯時は古いフレームを捨てて最新寄りを維持する
- `sync=false` → コンシューマの `pull` ペースで進む（パイプラインクロックに縛られない）
- RTSP の `latency=200` は一部カメラ（i-PRO 等）で最初のフレーム到着を安定させるため
- `calibration_enabled: true` のときは `calibration_homography_path` の 3x3 行列を読み込み、Y/UV 平面を CUDA 上で射影変換してから検出器へ渡す

### `pull()` と `capture_fps`

`capture_fps` は **videorate 要素ではなく** `pull()` 内の PTS スキップで実現しています（この RTSP+NV12 経路では Jetson 上で videorate が詰まりやすいため）。

- `_min_frame_dt = 1 / capture_fps`
- 前回採用 PTS からの経過が `_min_frame_dt * 0.85` 未満ならスキップして次を取る（PTS を先に見て、捨てるフレームは map/copy しない）
- PTS が欠落（`None`）のフレームは間引きせず常に採用
- `capture_fps: null`（または未設定）なら全フレームを採用

返り値の `nv12` は appsink バッファから 1 回 `.copy()` した独立配列です（呼び出し側が保持しても次の pull で壊れない）。calibration 有効時は射影変換後のバッファをさらに `.copy()` して返す。

### Prefetch

`PrefetchNv12Capture` は内部で `GstNv12Capture.pull` をワーカーが実行し、結果を有限キューに載せます。

- GStreamer のデコード自体はコンシューマが忙しい間も進む
- 加えて **appsink → host memcpy** を `detector.infer()` と時間的に重ねる
- キュー満杯時はワーカーが `put` で待つ → 上流 backpressure が維持される
- EOS（`None`）もキューに入れ、コンシューマが終了できるようにする

### `EdgeApp` 側のタイムアウト扱い

```text
PullTimeout:
  RTSP / USB → 「waiting for live frame...」を出して継続
  ファイル → 停止
None (EOS):
  ループ終了
```

## 設定キー

`source` / `transport` / `calibration_*` は機種 YAML（`jetsonN.yaml`）、それ以外は主に [`base.yaml`](../../pipeline_jetson/config/base.yaml)。詳細は [configuration.md](../reference/configuration.md)。

| キー | 説明 |
|------|------|
| `source` | RTSP URL / 動画パス / `/dev/video*`（USB） |
| `size` | `[W, H]`。エンジン入力と一致必須 |
| `transport` | RTSP 必須。本番は `tcp` 推奨 |
| `max_buffers` | appsink 深さ（例: `4`） |
| `drop_frames_when_lagging` | `true` で古いフレームを捨ててリアルタイム性を優先 |
| `capture_fps` | 意図的間引き（例: `15`）。`null` で全フレーム |
| `calibration_enabled` | ホモグラフィ射影変換の ON/OFF |
| `calibration_homography_path` | 3x3 ホモグラフィ行列ファイル |
| `prefetch` | prefetch ラップの ON/OFF |
| `prefetch_queue_size` | prefetch キュー深さ（例: `10`） |

## 失敗・タイムアウト時の挙動

| 状況 | 挙動 |
|------|------|
| RTSP で `transport` なし | `ValueError`（構築時） |
| ファイル不存在 / USB デバイス不存在 | `FileNotFoundError` |
| バッファ map 失敗 / 短すぎるバッファ | `PullTimeout` |
| サンプルなし（タイムアウト内） | bus を確認し、EOS/ERROR なら `None`、それ以外は `PullTimeout` |
| GST ERROR | ログ出力後 `_eos=True`、以降 `None` |

## 関連コード

- [`gst_io.py`](../../pipeline_jetson/components/edge/gst_io.py) — `GstNv12Capture._build_desc`, `pull`, `_try_pull_sample`, `_copy_mapped_nv12`, `PrefetchNv12Capture`
- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — キャプチャ生成と `PullTimeout` 分岐
- データ形状の約束: [data_contracts.md](../reference/data_contracts.md)
