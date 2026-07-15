# 工程: エントリと配線

## 目的

YAML 設定から `EdgeApp` とその依存コンポーネント（検出器・Publisher）を組み立て、メインループを開始する。

## 入出力

| | 内容 |
|--|------|
| 入力 | `--cfg` で指定する YAML（本番は `pipeline_jetson/config/edge.yaml`） |
| 出力 | インスタンス化された `EdgeApp` が `run()` でパイプラインを駆動 |

## 実装

| 要素 | 場所 |
|------|------|
| CLI エントリ | [`run.py`](../../run.py) |
| 起動シェル | [`run.sh`](../../run.sh) |
| 設定 | [`pipeline_jetson/config/edge.yaml`](../../pipeline_jetson/config/edge.yaml) |
| アプリ本体 | [`pipeline_jetson/edge_app.py`](../../pipeline_jetson/edge_app.py) の `EdgeApp` |

### 起動チェーン

```
./run.sh
  └─ uv run python run.py --cfg pipeline_jetson/config/edge.yaml
       ├─ OmegaConf.load(cfg_path)
       ├─ hydra.utils.instantiate(cfg)   # _target_: EdgeApp
       │     ├─ detector  → P2PNetTRTNV12Detector(...)
       │     └─ publisher → Publisher(...)
       └─ app.run()
```

`run.py` は Hydra のフルアプリではなく、OmegaConf で YAML を読み `instantiate` する薄いローダです。`_recursive_: false` のため、ネストした `detector` / `publisher` は `EdgeApp.__init__` 内の `_maybe_instantiate` で改めてインスタンス化されます。

### `EdgeApp.__init__` での配線

1. `GstNv12Capture(source, size, transport, max_buffers, capture_fps)` を生成
2. `prefetch` が真なら `PrefetchNv12Capture` でラップ、偽なら生のキャプチャを使用
3. `detector` / `publisher` を `instantiate`（または既にオブジェクトならそのまま）
4. `GstBgrDisplay` を常に生成（`enabled=display`）。`display: false` でもオブジェクトはあるが、ループ内の変換・描画はスキップされる

### `run()` 開始・終了順

**開始**

1. `capture.start()`
2. `display` が有効なら `display.start()`
3. `publisher` があれば `publisher.start()`
4. `max_wall_seconds` が設定されていればタイマーで `request_stop()`

**終了（`finally`）**

1. 有効なら `display.stop()`
2. `capture.stop()`
3. `publisher` があれば `publisher.stop()`

## 設定キー

トップレベルの `_target_` とアプリ引数は [configuration.md](../reference/configuration.md) を参照。配線に直結するもの:

| キー | 役割 |
|------|------|
| `_target_` | `pipeline_jetson.edge_app.EdgeApp` |
| `_recursive_` | `false`（ネストは手動 instantiate） |
| `detector` | 検出器の `_target_` と引数 |
| `publisher` | MQTT Publisher の `_target_` と引数 |
| `prefetch` / `prefetch_queue_size` | キャプチャのラップ有無 |

## 失敗・タイムアウト時の挙動

- YAML のパス不正や `_target_` 解決失敗は起動時に例外
- 検出器の `engine_path` 不在は `P2PNetTRTNV12Detector` 初期化時に assert
- RTSP で `transport` 未指定は `GstNv12Capture` が `ValueError`
- ファイル入力でパス不存在は `FileNotFoundError`

ループ内のタイムアウト・EOS は [02_capture.md](02_capture.md) を参照。

## 関連コード

- [`run.py`](../../run.py) — `main`, `get_args`
- [`pipeline_jetson/edge_app.py`](../../pipeline_jetson/edge_app.py) — `_maybe_instantiate`, `EdgeApp.__init__`, `EdgeApp.run`
- [`pipeline_jetson/config/edge.yaml`](../../pipeline_jetson/config/edge.yaml)
