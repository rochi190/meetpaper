# 訊號結果圖摘要：`20261101_220025`

參數：CI = 30.00 ms、預傾斜 100 ppm、穩定期保留上一組 α、β、RSE 真實延遲 = 正弦擬合。由 `st_signals.py` 產生。

## 每塊板子

| 節點 | 封包 | 掉包 | 樞紐點 | 有輸出 | α (ppm) | β (ms) | σ100 平均 (ms) | ε P50 / P99 (ms) | > 1 CI / > 2 CI (%) | Vpp (V) | SINAD (dB) | 同步殘差 std (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NODE_A | 36433 | 0 | 6 | 99.0 % | -43.9 | 9.92 | 14.67 | 15.1 / 40.4 | 1.65 / 0.047 | 1.003 | 43.8 | 0.123 |
| NODE_B | 36333 | 0 | 6 | 98.8 % | -47.4 | 72.32 | 14.68 | 15.2 / 37.8 | 1.83 / 0.028 | 0.997 | 41.3 | 0.137 |
| NODE_C | 36451 | 0 | 3 | 99.0 % | -34.5 | 119.80 | 14.86 | 15.2 / 43.6 | 2.13 / 0.039 | 1.001 | 45.1 | 0.183 |

同步殘差（`<node>_4_waveform.png` 的 (f)）= 每秒在 t_P 軸上量到的真實延遲（正弦相位）減掉 SharkTooth 的 δ̂T，再扣掉中位數（中位數包含 ADC 取樣時刻與時間戳的固定差，量不到）。它是單一節點的同步誤差，不需要另一塊板子。原始殘差還有一條所有板子共用的斜線 = 函數產生器時基相對 PC 時鐘的漂移，這次是 +2.90 ppm（各板斜率的中位數），已扣掉；它在 RSE 裡本來就會相消。

## 每一對節點（RSE，Eq. 26）

| 節點對 | 有效秒數 | mean abs (ms) | std (ms) | P95 abs (ms) | 跳過前 600 s 的 mean abs (ms) |
|---|---|---|---|---|---|
| NODE_A−NODE_B | 3558 | 0.077 | 0.090 | 0.193 | 0.072 |
| NODE_A−NODE_C | 3568 | 0.131 | 0.130 | 0.307 | 0.107 |
| NODE_B−NODE_C | 3558 | 0.095 | 0.115 | 0.260 | 0.072 |

### 每段（600 s）：Eq. 27 只取最差的一對 vs 所有節點對平均

| 段 | 最差的一對（P95 最大） | 最差那對 mean abs (ms) | 所有對平均 mean abs (ms) | 最差那對 P95 (ms) |
|---|---|---|---|---|
| 1 | NODE_A−NODE_C | 0.255 | 0.193 | 0.581 |
| 2 | NODE_A−NODE_C | 0.144 | 0.108 | 0.245 |
| 3 | NODE_A−NODE_C | 0.163 | 0.114 | 0.239 |
| 4 | NODE_A−NODE_C | 0.089 | 0.072 | 0.177 |
| 5 | NODE_B−NODE_C | 0.075 | 0.057 | 0.146 |
| 6 | NODE_A−NODE_C | 0.087 | 0.068 | 0.170 |

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
