# SharkTooth 復現模擬（依《SharkTooth 復現模擬：延遲來源整理與測試規格》）

論文：Landra et al., *SharkTooth: A Scalable Real-Time Algorithm for BLE-Based WBSN Synchronization*, IEEE IoT-J 12(22), 2025。
規格書：`docs/SharkTooth_模擬測試規格.md`；上一版模擬：`docs/ble_sim_v1.py`。

**先看這兩個檔**：`results/README.md`（所有測試的結論總表）、`results/fig_compare.png`（論文 Fig.3 / 4 / 6 與模擬並排）。

## 執行

```bash
pip install numpy scipy matplotlib numba pyyaml pymupdf pytest
python run_test.py all                       # 全部（T0 → T9 → 交付物），4 核心約 25 分鐘
python run_test.py config/T4_fig6.yaml       # 單一測試
python -m pytest tests -q                    # 正確性測試（SharkTooth 兩種實作一致、下凸包 = LP、快速互相關 = 逐 lag 互相關）
```

T5–T9 會讀 `results/T4_fig6/summary.json` 的擬合參數，所以要先跑 T4。

## 目錄（規格 §4.1）

| 檔案 | 內容 |
|---|---|
| `config/*.yaml` | 每個測試一個設定檔；`base:` 可繼承另一個設定檔 |
| `sim/config.py` | 預設值。每個延遲來源（規格 §2 的 ID）都有 `enabled` 開關；數值可寫成分布（每節點各抽一次） |
| `sim/clocks.py` | A 類：晶振 ρ、溫度係數與暖機、隨機漫步、PC 時鐘與 NTP、tick 量化、換算誤差、溢位 |
| `sim/node.py` | B 類：ADC 計時器取樣、ISR 抖動、buffer 讀錯、打包（count / timer）、RTOS 延遲、重開機；F 類參考訊號 |
| `sim/link.py` | C 類：connection event、Gilbert–Elliott 重傳、韌體流量控制（window / ack）、跳頻、斷線、TX 佇列、起始延遲 |
| `sim/central.py` | C5 / C6：每支 dongle 的多連線排程（anchor 錯開、事件重疊略過） |
| `sim/host.py` | D、E 類：dongle 處理、序列化佇列、USB frame、OS 延遲、共同模式停頓、GIL、t_C 解析度 |
| `sim/simulate.py` | 串起整條流程（§4.2）；輸出 §4.3 格式（另附真值欄位） |
| `sim/algorithms/` | SharkTooth（純 Python 參考版 + numba 版）與 §6 所有基準方法；`integrity.py` = §7 完整性檢查 |
| `sim/evaluate.py` | δT 描述統計、真值誤差、論文式 RSE（Eq.23–27，含 10 kHz 互相關的等價快速實作） |
| `sim/digitize.py` | T0：直接讀 PDF 向量繪圖指令數位化 Fig.3 / 4 / 6 / 10 / 12 |
| `sim/figures.py` | 與論文相同座標範圍的圖 |
| `sim/experiments/tN.py` | 每個測試的流程與報告 |
| `data/` | 數位化結果（`digitized/*.csv`）、`targets.json`、Table II 轉錄（`paper_table2.csv`） |
| `results/<test>/` | 圖、`summary.json`、`README.md`（結論） |

## 讀實測資料

`sim/simulate.py` 輸出的每節點欄位是 `node_id, pkt_id, n_samples, t_P, t_C, t_true_last_sample, n_retx, dropped_flag, buf_err_flag`
（模擬另有 `t_pc_true`、`first_last_dt` 等）。演算法只用 `t_P`、`t_C`（完整性檢查另用 `pkt_id`、`n_samples`、`first_last_dt`），
所以開發板資料只要整理成同樣欄位，就能直接呼叫 `sim.algorithms.run(name, t_P, t_C)` 或 `sim.algorithms.integrity.run(...)`。
