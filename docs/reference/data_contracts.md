# データ契約

パイプライン各段で受け渡すデータの形と、解像度・座標系の約束。

## フレーム（NV12）

| 項目 | 内容 |
|------|------|
| レイアウト | packed NV12（GStreamer appsink と同じ） |
| dtype | `numpy.uint8` |
| 長さ | `H * W * 3 // 2` |
| 取得元 | `GstNv12Capture.pull` / `PrefetchNv12Capture.pull` |
| タプル | `(nv12, seq, pts)` |

`seq` は採用フレームの通番（1 始まり）。`capture_fps` でスキップされたフレームは番号に入らない。`pts` は秒（float）。

## 検出結果

| 項目 | 内容 |
|------|------|
| 型 | `numpy.ndarray` |
| shape | `(N, 3)`。検出なしは `(0, 3)` |
| 列 | `x`, `y`, `score`（いずれも float） |
| 座標空間 | センサ解像度（`img_size` の W×H） |
| 閾値 | `detector.threshold` 通過済み |

MQTT 化後は [mqtt_contract.md](mqtt_contract.md) の `detections.point` になる。

## 解像度の表記差

同じ画素サイズを、場所によって `(W, H)` と `(H, W)` で書いている。

| 場所 | 表記 | FHD の例 |
|------|------|----------|
| `base.yaml` の `size` | `[W, H]` | `[1920, 1080]` |
| `GstNv12Capture` / `EdgeApp` | `w, h` | 1920, 1080 |
| `detector.img_size` | `[H, W]` | `[1080, 1920]` |
| TRT export `--img-size` | `H W` | `1080 1920` |
| エンジン入力 | `(1, H*W*3//2)` | `(1, 3110400)` |

モデル内部のリサイズは 128 の倍数にスナップする（例: 1080 → 1024）。後処理でセンサ座標へスケールバックする。詳細は [04_detection.md](../stages/04_detection.md)。

## メインループ内の時間計測フィールド

`report_every` ごとのログ（ミリ秒）:

| ラベル | 区間 |
|--------|------|
| `wait` | `pull` 待ち |
| `det` | `infer` |
| `pub` | MQTT publish |
| `disp` | 表示（有効時のみ） |
| `e2e` | 1 フレーム全体 |
| `avg` / fps | ラン開始からの平均 |

## 本番外モジュール（契約対象外）

次はエッジ本番のデータ経路に含まれない。

| モジュール | 備考 |
|------------|------|
| `processor/tracker/` | 集約側向け追跡 |
| `yolo26` / DEIMV2 重み | 未配線 |
| `P2PNetTRTDetector` 等 | BGR / PyTorch 参照経路 |
| `timer.py` | `EdgeApp` 未使用 |
| `Publisher.publish_result` | 旧 `objects` 形式 |

## 関連文書

- [01_pipeline_overview.md](../01_pipeline_overview.md)
- [configuration.md](configuration.md)
- [mqtt_contract.md](mqtt_contract.md)
