# 工程: 検出（P2PNet TensorRT NV12）

## 目的

packed NV12 フレームから人物の点検出 `(x, y, score)` を得る。前処理（NV12→RGB・resize・ImageNet 正規化）はエンジングラフ内に融合済みで、Python 側はホスト→デバイス転送と推論実行が中心。

## 入出力

| | 内容 |
|--|------|
| 入力 | packed NV12 `uint8`、長さ `H*W*3//2`（`img_size` = `(H, W)`） |
| エンジン出力 | `scores` `(1, N)`、`points` `(1, N, 2)`（リサイズ空間の座標） |
| 後処理出力 | `np.ndarray` shape `(M, 3)` — `(x, y, score)`。センサ解像度（例: 1920×1080） |
| 検出なし | shape `(0, 3)` の空配列 |

## 実装

| 要素 | 場所 |
|------|------|
| `P2PNetTRTNV12Detector` | [`processor/detector/p2pnet_trt_nv12.py`](../../processor/detector/p2pnet_trt_nv12.py) |
| エンジン前提 | [`trt_scripts/export_p2pnet_onnx.py`](../../trt_scripts/export_p2pnet_onnx.py) の `--fuse-nv12` |

### 推論パス（`infer`）

```text
nv12 (host numpy)
  → contiguous uint8
  → pinned memory へ copy
  → H2D (non_blocking) へ input buffer
  → context.execute_async_v3(stream)
  → synchronize
  → post_process(scores, points)
```

エンジン入力形状の期待値は `(1, H*W*3//2)`、dtype `uint8`。不一致は `_load_engine` または `infer` で assert する。

### 後処理（`post_process`）

1. `scores > threshold` でフィルタ
2. 点座標にスケールを掛ける — モデル内部は 128 の倍数にスナップした `resize_size` 上の座標のため、センサ `(H, W)` へ戻す
3. 画面内フィルタ: `0 <= x < W` かつ `0 <= y < H`
4. `(x, y, score)` を CPU numpy で返す

スケール:

```text
resize_size = (H // 128 * 128, W // 128 * 128)
scale = (W / resize_W, H / resize_H)
```

例: FHD `1080×1920` → resize `1024×1920`（高さのみ 128 倍数に切り捨て）。

### エンジン内前処理（参考）

`--fuse-nv12` エクスポート時、グラフ内でおおむね次が行われる（詳細は [tensorrt_engine.md](../build/tensorrt_engine.md)）:

- packed NV12 → RGB（`--yuv-matrix`: `bt601-limited` / `bt709-full`）
- bilinear resize → ImageNet normalize
- P2PNet 推論 + softmax 済み scores

## 設定キー

検出器設定は [`base.yaml`](../../pipeline_jetson/config/base.yaml)（必要なら機種 YAML で上書き）。

| キー | 説明 |
|------|------|
| `detector._target_` | `processor.detector.p2pnet_trt_nv12.P2PNetTRTNV12Detector` |
| `detector.engine_path` | NV12 融合 `.engine` のパス |
| `detector.device` | 例: `cuda:0` |
| `detector.threshold` | スコア閾値（本番例: `0.1`） |
| `detector.img_size` | `[H, W]`。キャプチャの `size` `[W, H]` と一致必須 |

## 失敗・タイムアウト時の挙動

| 状況 | 挙動 |
|------|------|
| `engine_path` 不存在 | 初期化時 assert |
| エンジン deserialize 失敗 | assert |
| TensorRT &lt; 8.5（name-based I/O なし） | `RuntimeError` |
| 入力 shape / dtype 不一致 | assert |
| NV12 バイト数不一致 | `infer` で assert |
| 閾値通過ゼロ | 空の `(0, 3)` を返し、ループは継続 |

検出器はタイムアウトを持たない。遅い場合はキャプチャ側の backpressure / prefetch キューが詰まる。

## 関連コード

- [`p2pnet_trt_nv12.py`](../../processor/detector/p2pnet_trt_nv12.py) — `__init__`, `_load_engine`, `infer`, `post_process`
- [`edge_app.py`](../../pipeline_jetson/edge_app.py) — `detector.infer(nv12)`
- エンジンビルド手順: [tensorrt_engine.md](../build/tensorrt_engine.md)
- 座標・配列契約: [data_contracts.md](../reference/data_contracts.md)
