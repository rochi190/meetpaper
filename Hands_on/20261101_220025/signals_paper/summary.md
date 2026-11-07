# 訊號結果圖摘要：`20261101_220025`

參數：CI = 30.00 ms、預傾斜 0 ppm、論文原版（穩定期 α = β = 0）、RSE 真實延遲 = 正弦擬合。由 `st_signals.py` 產生。

## 每塊板子

| 節點 | 封包 | 掉包 | 樞紐點 | 有輸出 | α (ppm) | β (ms) | σ100 平均 (ms) | ε P50 / P99 (ms) | > 1 CI / > 2 CI (%) | Vpp (V) | SINAD (dB) | 同步殘差 std (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NODE_A | 36433 | 0 | 515 | 0.0 % | -43.8（下緣擬合） | 10.30 | 14.67 | 14.5 / 39.8 | 1.46 / 0.047 | 1.003 | 43.8 | nan |
| NODE_B | 36333 | 0 | 479 | 0.0 % | -47.4（下緣擬合） | 72.77 | 14.68 | 14.5 / 37.2 | 1.56 / 0.017 | 0.997 | 41.3 | nan |
| NODE_C | 36451 | 0 | 427 | 0.7 % | -34.4（下緣擬合） | 120.02 | 14.86 | 14.6 / 43.2 | 1.92 / 0.036 | 1.001 | 45.1 | 3.198 |

同步殘差（`<node>_4_waveform.png` 的 (f)）= 每秒在 t_P 軸上量到的真實延遲（正弦相位）減掉 SharkTooth 的 δ̂T，再扣掉中位數（中位數包含 ADC 取樣時刻與時間戳的固定差，量不到）。它是單一節點的同步誤差，不需要另一塊板子。原始殘差還有一條所有板子共用的斜線 = 函數產生器時基相對 PC 時鐘的漂移，這次是 +0.00 ppm（各板斜率的中位數），已扣掉；它在 RSE 裡本來就會相消。

## 每一對節點（RSE，Eq. 26）

| 節點對 | 有效秒數 | mean abs (ms) | std (ms) | P95 abs (ms) | 跳過前 600 s 的 mean abs (ms) |
|---|---|---|---|---|---|
| NODE_A−NODE_B | 0 | – | – | – | – |
| NODE_A−NODE_C | 0 | – | – | – | – |
| NODE_B−NODE_C | 0 | – | – | – | – |

### 每段（600 s）：Eq. 27 只取最差的一對 vs 所有節點對平均

| 段 | 最差的一對（P95 最大） | 最差那對 mean abs (ms) | 所有對平均 mean abs (ms) | 最差那對 P95 (ms) |
|---|---|---|---|---|

論文的 Table II 只報最差的一對（Eq. 27）。節點越多、節點對越多，「最差的一對」本身就會越大，所以節點數的效應有一部分是統計上的取最大值，不全是網路壅塞。

## 圖

- [`NODE_A_1_timestamp_model.png`](NODE_A_1_timestamp_model.png)
- [`NODE_A_2_algorithm_signals.png`](NODE_A_2_algorithm_signals.png)
- [`NODE_A_3_delay.png`](NODE_A_3_delay.png)
- [`NODE_A_4_waveform.png`](NODE_A_4_waveform.png)
- [`NODE_B_1_timestamp_model.png`](NODE_B_1_timestamp_model.png)
- [`NODE_B_2_algorithm_signals.png`](NODE_B_2_algorithm_signals.png)
- [`NODE_B_3_delay.png`](NODE_B_3_delay.png)
- [`NODE_B_4_waveform.png`](NODE_B_4_waveform.png)
- [`NODE_C_1_timestamp_model.png`](NODE_C_1_timestamp_model.png)
- [`NODE_C_2_algorithm_signals.png`](NODE_C_2_algorithm_signals.png)
- [`NODE_C_3_delay.png`](NODE_C_3_delay.png)
- [`NODE_C_4_waveform.png`](NODE_C_4_waveform.png)
- [`link_quality.png`](link_quality.png)
- [`pairs_1_rse.png`](pairs_1_rse.png)
- [`pairs_2_segments.png`](pairs_2_segments.png)
- [`sync_overlay.png`](sync_overlay.png)
