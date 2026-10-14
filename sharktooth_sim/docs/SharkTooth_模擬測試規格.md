# SharkTooth 復現模擬：延遲來源整理與測試規格

> 給 Claude Code 的工作說明書。目標：用模擬產生與論文 Fig.3、Fig.4、Fig.6（以及 Fig.10、Table II 的趨勢）**看起來與統計上都相近**的資料，並用同一套模擬比較 SharkTooth 與先前方法。
>
> 論文：Landra et al., *SharkTooth: A Scalable Real-Time Algorithm for BLE-Based Wireless Body Sensor Network Synchronization*, IEEE IoT-J 12(22), 2025。以下「§」「Eq.」「Fig.」皆指這篇。
>
> 標記：**【原文】** 論文寫的；**【規格】** 晶片／協定文件的事實；**【推測】** 合理但未經證實，需要用模擬或實驗檢驗。

---

## 0. 為什麼上一版模擬「不像」（先修正的問題）

上一版（`ble_sim.py`）與論文 Fig.6 的差異：

| 論文 Fig.6 的特徵（目測） | 上一版模擬 | 推測原因 |
|---|---|---|
| δT 是一條**連續、厚約 30–40 ms 的帶狀**，帶內點分布很均勻 | 只有幾條**離散的水平線**（間距 10 ms） | 封包週期 100 ms 與 CI 30 ms 的公因數 g = 10 ms，等待時間只落在固定幾個相位；上一版的 PC 端抖動只有約 1 ms，填不滿 10 ms 的間隙。真實系統一定有更大的「連續型」延遲（多節點排程、序列化佇列、OS） |
| 下緣是平滑斜線（約 0.29 → 0.32 s / 140 s ≈ 210 ppm） | 下緣是**階梯**（每 50 s 跳 10 ms） | 同上；連續抖動夠大時階梯會被抹平 |
| 帶的上方有**稀疏離群點**，高出下緣 40–90 ms | 有，但分布不像 | 重傳次數分布、OS 尖峰比例未校準 |
| **起始處有一根垂直尖峰**（t_P ≈ 5 s，δT 從約 0.41 掉到 0.29） | 沒有 | 【推測】連線建立／啟用 notification 前累積的封包一次送出（backlog flush），前幾包延遲遞減 |
| Fig.6 可能來自多節點配置（論文沒說是哪一個） | 單節點 | 需要模擬多節點共用 dongle |

**結論**：不能再靠手調參數「看起來像」。本規格要求先把論文圖**數位化成數字目標**，再讓模擬去擬合（§5 T0、T4）。

---

## 1. 符號與時間定義

| 符號 | 定義 |
|---|---|
| `t` | 真實時間（模擬內的上帝時鐘，單位 s） |
| `t_P` | 節點本地時間戳（封包最後一筆樣本的取樣時刻，以節點時鐘讀數表示）【原文 §III-B】 |
| `t_C` | PC 端 callback 打的時間戳（以 PC 時鐘讀數表示）【原文 §III-B】 |
| `t_S` | 共同參考時間；模擬中定義為 PC 時鐘 |
| `δT = t_C − t_P` | Eq.(3) |
| `ρ_n(t)` | 節點 n 的時鐘頻率誤差（無因次，ppm）；**允許隨時間變化** |
| `ρ_PC(t)` | PC 時鐘頻率誤差 |
| `CI` | connection interval（論文 30 ms）|
| `P` | 封包週期（論文 100 ms，以節點時鐘計）|

時鐘模型（取代論文 Eq.(1) 的常數 α）：

```
clock_n(t) = θ_n + ∫_0^t (1 + ρ_n(τ)) dτ        # 節點時鐘讀數
clock_PC(t) = θ_PC + ∫_0^t (1 + ρ_PC(τ)) dτ
```

當 ρ 為常數時退化成論文的 Eq.(1)：δT 的斜率 α ≈ ρ_PC − ρ_n。

---

## 2. 延遲與誤差來源總表

每一項都要在模擬中實作成**可開關**的模組（`enabled: true/false` + 參數），以便做消融實驗。

### A. 時鐘類（造成「漂移」，也就是 δT 的斜率與彎曲）

| ID | 來源 | 物理 / 數值 | 在 δT 圖上的樣子 | 建模方式（建議預設） |
|---|---|---|---|---|
| A1 | 晶振出廠容差 | 32.768 kHz 音叉晶振常見 ±20 ppm【規格：Raltron R38】；論文節點用 ±20 ppm【原文 §II-A】 | 固定斜率 | `ρ_n ~ U(−20, 20) ppm` |
| A2 | **溫度係數（拋物線）** | Δf/f = −0.034 ppm/°C² × (T − 25 °C)²【規格：R38，turnover 25±5 °C】。貼身 33–35 °C → 約 −2.2 ~ −3.4 ppm | 斜率會隨溫度改變 | `ρ_temp(T) = k·(T − T0)²`，k = −0.034e-6 |
| A3 | **升溫暖機（使用者特別提到）** | 開機後板子自熱、貼上皮膚後升溫，溫度約以時間常數 τ 指數上升到穩態 → ρ 從初值漸變到穩態值 | **前期 δT 有彎曲**（斜率在變），穩態後變直線；OLS 早期估不準 | `T(t) = T_end − (T_end − T_start)·e^(−t/τ)`，T_start 22–25 °C、T_end 30–35 °C、τ = 2–10 min【推測：τ 依外殼與貼附方式】 |
| A4 | 老化 | 第一年 ±3 ppm【規格：R38】 | 1 小時內可忽略 | 不實作（列為說明） |
| A5 | 供電電壓變化 | 晶振頻率對電壓敏感度小；電池放電、無線發射瞬間壓降 | 很小的擾動 | 選配：小振幅隨機漫步 |
| A6 | **PC 端時鐘誤差** | `t_C` 用的是 PC 的時鐘，它本身也有 ppm 誤差；若用 wall-clock，NTP 會「慢慢調速」或跳時 | 所有節點共同的斜率；NTP 跳時 → 全部節點同時一階跳 | `ρ_PC ~ U(−50, 50) ppm`【推測量級】；選配 NTP step / slew 事件 |
| A7 | 計時器量化 | 32.768 kHz 一格 30.52 μs | 微小鋸齒 | `floor(clock·32768)/32768` |
| A8 | **tick→時間換算誤差** | 韌體若用近似常數把 tick 換成 ms（例如把 32768 當成 32000 或用整數除法），會產生固定的「假 ppm」 | 額外的固定斜率 | `scale_err_ppm` 參數。【推測】這是論文 Fig.3/Fig.6 斜率約 200 ppm（遠超 ±20 ppm 規格）的候選解釋之一 |
| A9 | 時鐘源選錯 | 若時間戳用內部 RC 振盪器而不是 32.768 kHz 晶振，誤差可達數百 ppm 以上且隨溫度大幅變化 | 大斜率＋彎曲 | 選配：`clock_source: rc`，ρ 大、溫度係數大【推測】 |
| A10 | 計時器溢位 | t_P 欄位 4 B【原文 §III-B】，若單位是 μs，約 71.6 分鐘繞回 | 突然大幅下跳 | `tp_bits=32, tp_unit` |

### B. 節點取樣與韌體（造成 t_P 本身的錯）

| ID | 來源 | 說明 | 樣子 | 建模 |
|---|---|---|---|---|
| B1 | ADC 中斷延遲 | 中斷被更高優先權（BLE radio 中斷通常最高）擋住 → 打時間戳的時刻晚了 | t_P 抖動，μs 級 | `isr_jitter ~ U(0, 50 μs)` |
| B2 | RTOS 排程 | 打包 task 被搶佔 → 打包晚 | 封包進佇列延遲，ms 級 | `fw_delay ~ U(0, 2 ms)` |
| B3 | **buffer 讀取競爭（學長）** | 讀環形 buffer 時少抓／多抓一筆、或與上一包重疊 | t_P 對到錯的樣本 → δT 錯 ±k/f_s | `buf_err_prob`, `k ∈ {±1}` |
| B4 | 打包觸發方式 | 計時器觸發（每包樣本數 99/100/101 浮動）vs 湊滿 N 筆觸發 | 封包間隔不等 | `pack_mode: timer|count` |
| B5 | 重開機／重連 | t_P 從 0 重來 | δT 大跳 | `reset_events: [(t, …)]` |

### C. BLE 連線層（造成「延遲」：鋸齒、尖峰、帶寬）

| ID | 來源 | 說明 | 樣子 | 建模 |
|---|---|---|---|---|
| C1 | **等待 connection event** | 封包產生時刻與下一個 event 不對齊，等待 0 ~ CI【原文 §III-A】 | 鋸齒；只有等待時 σ(δ²T) ≤ CI/2【原文：「around half of the CI」】 | event 時刻 = anchor_n + j·CI·(1+ρ_dongle) |
| C2 | **相位鎖定（gcd）** | P 與 CI 的公因數 g 讓等待只落在 g 的倍數相位 → δT 呈離散層、下緣成階梯 | 階梯 | 由 C1 自然產生；測試時改 P/CI 組合 |
| C3 | **重傳** | CRC 錯誤或沒收到 ACK → 下一個 event 重送，延遲多 k·CI【原文 §III-A】。原因：2.4 GHz Wi-Fi 干擾、其他 BLE、人體遮蔽與吸收、距離、多路徑 | 往上的尖峰；k 越大越高 | 每次嘗試錯誤率 PER；**Gilbert–Elliott 兩態**（好／壞）模擬突發 |
| C4 | 每個 event 能送的包數 | 244 B 需要 7.5 ms event length【原文 §IV-A】；送不完的排到下一個 event | 重傳後的「追趕」：後續封包一起延遲 | `max_pkts_per_event`，FIFO |
| C5 | **多連線排程（central 端）** | central 依序服務各 peripheral 的 event，各連線在 CI 內有不同時間偏移；事件重疊時會被**延後或丟棄**【規格：Nordic SoftDevice Controller scheduling：「its timing-event will be rescheduled for a later time」「…is dropped because it collides with…」】 | 每個節點有不同的固定偏移；碰撞時多一個 CI | 每 dongle 一個排程器：連線 n 的 event 偏移 = n·event_len；碰撞機率參數 |
| C6 | 多個 piconet 並存 | 論文用 3 支 dongle 各 4 節點；鄰近網路互相干擾【原文 §IV-A、§V-A：「more than two piconets operating in parallel」】 | 節點多時 PER 上升 | `per_good` 隨 dongle 數增加 |
| C7 | 頻道跳頻 / AFH | 37 個資料頻道，部分頻道受 Wi-Fi 干擾 | 週期性的錯誤率變化 | 選配：每個 event 的頻道 = 跳頻序列；壞頻道 PER 高 |
| C8 | slave latency | peripheral 可略過 event 省電 | 額外延遲 | 選配（論文未提，預設 0） |
| C9 | 斷線（supervision timeout） | 連續失敗超過 timeout → 斷線，佇列資料遺失，重連後參數可能改變 | 資料缺口；重連後 δT 可能偏移 | `disconnect_events` 或由連續失敗觸發 |
| C10 | **TX 佇列滿 → 真正掉包** | notification 產生速度 > 送出速度（BLE 4.0 一包僅 20 B 時特別嚴重；4.2 Data Length Extension 才到 244 B） | 封包編號跳號 | `tx_queue_len` |
| C11 | 起始 backlog | 【推測】連線建立、啟用 notification 前已累積的封包一次送出 | **Fig.6 起始的垂直尖峰** | `startup_backlog: n_pkts`，前 n 包 t_C 擠在一起 |
| C12 | window widening | 收發兩端睡眠時鐘誤差造成接收窗擴張 | μs 級，可忽略 | 不實作（說明） |

### D. Dongle 與序列化（論文用 nRF52840 dongle + Blatann）

| ID | 來源 | 說明 | 樣子 | 建模 |
|---|---|---|---|---|
| D1 | dongle 韌體處理 | connectivity firmware 把 BLE 事件序列化送給 PC【推測：Blatann 透過 Nordic pc-ble-driver 的 serialization 傳輸；需確認版本】 | 小延遲 | 常數 + 小抖動 |
| D2 | **序列化佇列** | 同一支 dongle 的多個節點事件共用一條序列／USB 通道，**依序處理** → 節點多時排隊 | **連續型延遲**，節點越多越大；是填滿 Fig.6 帶狀的主要候選【推測】 | 單一伺服器佇列（M/G/1）：服務時間 0.3–2 ms／事件 |
| D3 | USB full-speed 輪詢 | 每 1 ms 一個 frame；一次 transfer 可帶多包 | 到達時間量化；多包同時到 | `ceil(t/1ms)`；batch |

### E. PC 端（論文用 Python）

| ID | 來源 | 說明 | 樣子 | 建模 |
|---|---|---|---|---|
| E1 | OS 排程 | callback 執行緒被延後喚醒 | 連續型延遲，長尾 | lognormal |
| E2 | Python GIL、garbage collection | 偶發數十 ms 停頓 | 稀疏尖峰（同時影響所有節點） | `spike_prob`, `spike_size`；**同一時刻影響所有節點**（共同模式） |
| E3 | 執行緒數量 | 論文：「the thread-based control software may introduce unpredictable delays … as the number of concurrent threads increases」【原文 §V-A】 | 節點越多延遲越大 | E1 的尺度 ∝ 節點數 |
| E4 | 時間戳函式解析度 | 不同作業系統／函式解析度不同（例如 wall-clock vs 單調時鐘）；解析度粗會量化 t_C | t_C 階梯 | `tc_resolution`（預設 1 μs；測 1 ms、15.6 ms） |
| E5 | 背景負載 | 存檔、畫圖、其他程式 | 不定期大延遲 | 選配 |

### F. 量測（評估）本身的誤差

| ID | 來源 | 說明 |
|---|---|---|
| F1 | 互相關解析度 | 上取樣到 10 kHz → 0.1 ms【原文 §IV-B】 |
| F2 | 正弦週期 200 ms → 只能判定 ±100 ms【原文 §IV-B】 |
| F3 | ADC 雜訊、類比前端群延遲差異 | 各節點類比濾波器相位不同 → 固定偏差【推測】 |
| F4 | 取樣時間未對齊的內插誤差 | 重新取樣到共同 t_P 格點【原文 §IV-B】 |

模擬要能同時輸出「**真值誤差**」（直接和上帝時鐘比）與「**論文式誤差**」（模擬 5 Hz 正弦 → 互相關，§IV-B 的流程），兩者差異就是 F 類誤差。

---

## 3. 為什麼 MAC 層時間戳難取得（簡報用素材）

1. **協定堆疊不開放**：論文指出 MAC 層時間戳「generally inaccessible in most commercial BLE platforms due to restrictions imposed by the protocol stack and firmware architecture」【原文 §I-A】。以 Nordic 為例，BLE 鏈路層在 SoftDevice（預編譯二進位）裡，應用程式拿不到每個封包實際上空的時刻。
2. **重傳對應用程式不可見**：Nordic 的技術文章：「The application generally has no idea of how many tries it takes to transmit a packet, which is a problem when it comes to time synchronization.」
3. **要自己繞過堆疊**：該文章的做法是用 SoftDevice 的 **timeslot API** 在 BLE 活動空檔借用 radio，用硬體計時器 + **PPI** 在封包發送／接收時硬體捕捉時間，避免 CPU 中斷造成的抖動；兩塊板子可同步到約 1–2 個 16 MHz 週期（數十 ns）。代價：要改兩端韌體、占用 radio 時間、和一般 BLE 連線並存的排程問題，而且 **central 端（dongle）也要跑自訂韌體**——SharkTooth 刻意避免的就是這些【原文 §III-B：central 跑 default firmware】。
4. **PC 端更拿不到**：即使 dongle 知道上空時刻，經過序列化、USB、OS 後，PC 只看到 callback 時刻。

---

## 4. 模擬器架構（請重構 `ble_sim.py`）

### 4.1 模組

```
sim/
  config/            # YAML，每個測試一個檔
  clocks.py          # A 類：ClockModel（常數 ρ、溫度暖機、PC 時鐘、量化、換算誤差、溢位）
  node.py            # B 類：取樣、buffer、打包（timer/count）、TX 佇列、重開機
  link.py            # C 類：connection event、PER（Gilbert–Elliott／頻道）、重傳、每 event 包數、斷線、起始 backlog
  central.py         # C5/C6：每支 dongle 的多連線排程（event 偏移、碰撞延後/丟棄）
  host.py            # D、E 類：序列化佇列、USB frame、OS 延遲、共同模式尖峰、時間戳解析度
  evaluate.py        # 真值誤差、論文式 RSE（Eq.23–27）、分段指標、Table II 格式輸出
  algorithms/        # SharkTooth 與基準方法（§6）
  digitize.py        # 從論文 PDF 數位化 Fig.3/4/6/10（§5 T0）
  figures.py         # 畫成與論文相同座標範圍的圖
  run_test.py        # python run_test.py config/T4_fig6.yaml
```

### 4.2 事件流程（每個封包）

```
取樣（節點時鐘）→ B1 打 t_P → B3 buffer 讀取 → B4 打包 → B2 進 TX 佇列（C10 可能丟）
→ C1 等 event（central 排程 C5 決定該節點在 CI 內的時間）→ C3 每次嘗試成功/失敗（C4 每 event 上限）
→ D1/D2 dongle 序列化佇列 → D3 USB frame → E1/E2/E3 OS → E4 以 PC 時鐘打 t_C
```

### 4.3 輸出格式（每節點一個 parquet/csv）

`node_id, pkt_id, n_samples, t_P, t_C, t_true_last_sample, n_retx, dropped_flag, buf_err_flag`

**同一份格式要能直接讀實測資料**（實測沒有真值欄位），讓同一份演算法程式碼同時跑模擬與開發板資料。

### 4.4 原則
- 所有隨機性用 `numpy.random.Generator(seed)`；每個測試至少 20 個 seed（對應論文 20 次重複）。
- 預設時間長度 3600 s；畫 Fig.6 類圖時只取前 145 s。
- 圖的座標範圍、刻度與論文相同，方便並排比較。

---

## 5. 測試清單

每個測試都要產出：設定檔、圖（與論文並排）、數字摘要（`results/<test>/summary.json`）、一段文字結論。

### T0　數位化論文圖，建立數字目標（**先做**）
- 從論文 PDF 擷取 Fig.3（p.46179）、Fig.4（p.46179）、Fig.6（p.46181）、Fig.10（p.46185）的點。
- 方法：PyMuPDF 取頁面影像（≥ 600 dpi）→ 依軸刻度建立像素↔座標轉換 → 依顏色門檻抓藍點（Fig.6 的 δT）與紅線（δT_*）。無法自動時用 WebPlotDigitizer 手動，存成 csv。
- 由 Fig.6 計算目標統計量：
  - 下緣斜率（用 5% 分位數迴歸或下凸包）
  - 殘差 r = δT − 下緣 的分位數：P10、P50、P90、P99、最大值
  - r > 45 ms 的點比例（離群比例）
  - 起始尖峰：前 10 個點的 δT 與其後下緣的差
- 由 Fig.3 計算：鋸齒週期、層數、峰對峰、σ_100 平均。由 Fig.4 計算：尖峰高度分布、σ_100 範圍。
- **通過標準**：數位化的點在疊回原圖時肉眼對齊；輸出 `targets.json`。

### T1　復現 Fig.3（最小延遲）
- 設定：單節點，A1（固定 ρ 使斜率約 200 ppm）+ C1 + D3 + E1（小）；關閉 C3、E2。
- 掃描：E1 尺度 0.2、1、2、5 ms。
- 指標：鋸齒週期（應為 0.3 s）、σ_100（目標 14.3–14.7 ms；只等待的理論值 √(r(CI−r)) = 14.14 ms）。
- 通過：σ_100 平均落在 14.3–14.7 ms；鋸齒層數與峰對峰和 T0 目標差 < 20%。

### T2　相位鎖定（gcd）與「階梯 vs 平滑」
- 設定：T1 基礎上，掃描 P/CI = 100/30、100/31.25、100/37.5、100/45、100/50 ms；E1 尺度 0.5、5、10 ms。
- 指標：下緣的「階梯度」（下緣一階差分的最大跳躍 / 平均斜率）、σ_100。
- 目的：說明什麼條件下下緣是階梯、什麼條件下是平滑斜線；推論論文系統至少有多大的連續型延遲。
- 也驗證 SharkTooth 在階梯情況下 α 估計是否仍無偏。

### T3　復現 Fig.4（雜訊）
- 設定：T1 + C3（Gilbert–Elliott）+ E2。
- 掃描：per_good ∈ {0.01, 0.03, 0.1}、per_bad ∈ {0.3, 0.6, 0.9}、壞狀態平均長度 ∈ {1, 3, 10} events、spike_prob ∈ {0.001, 0.01, 0.03}。
- 指標：σ_100 範圍（目標 30–50 ms）、尖峰高度分布（目標由 T0）。

### T4　復現 Fig.6（**主要目標**）
- 設定：假設 Fig.6 來自多節點配置；預設 4 節點 / 1 dongle（與論文每 dongle 4 節點一致），244 B。
- 開啟：A1、A7、B1、B2、C1、C3、C4、C5、C11、D2、D3、E1、E2、E3。
- **參數擬合**：以 T0 的殘差分位數與離群比例為目標，用貝氏最佳化或格點搜尋擬合 D2 服務時間、E1 尺度、PER、spike 參數、C11 backlog 長度；目標函數 = 分位數誤差平方和 + 離群比例誤差 + 起始尖峰誤差。
- 產出：與 Fig.6 相同座標的圖（藍點 δT、紅線 SharkTooth δT_*、綠星樞紐點），並排原圖；QQ 圖（模擬殘差 vs 數位化殘差）。
- **消融**：每次關掉一個模組，記錄哪個特徵消失（帶寬、離群、起始尖峰、平滑下緣）。這張表就是簡報「延遲從何而來」的核心證據。
- 通過：殘差 P10/P50/P90 與目標差 < 3 ms、P99 差 < 10 ms、離群比例差 < 2 個百分點；肉眼並排相似。

### T5　時鐘暖機（使用者提出）
- 設定：T4 最佳參數 + A2/A3。掃描 T_start ∈ {22, 25}、T_end ∈ {30, 34} °C、τ ∈ {2, 5, 10} min；另加 A8 換算誤差 0/200 ppm。
- 指標：
  - δT 下緣的曲率（前 20 分鐘斜率 vs 後 40 分鐘斜率）
  - SharkTooth 在 600 s 分段下的誤差（模仿 Table II）是否呈「第 1 段大、之後收斂」
- 目的：檢驗「論文 Table II 第 1 段誤差大」能不能部分用晶振暖機解釋；並對照只用累積 OLS 時，前期的彎曲會被永久留在累加量裡。
- 加做：論文的遞迴 OLS vs 加遺忘因子（λ = 0.999、0.9999）的版本。

### T6　多節點與 Table II / Fig.10 / Fig.11
- 設定：節點數 2、4、6、8、10、12；封包 17、127、244 B；1 dongle 每 4 節點（論文配置）；每配置 20 seeds × 3600 s。
- 每節點加入**固定偏差**來源：C5 的 CI 內偏移、F3 類比延遲差、D 類路徑不同（不同 dongle）。
- 指標：照論文 §IV-B 算 RSE（Eq.23–27）：600 s 分段、每段取 P95 最大的那一對、取平均|RSE|、標準差、P95；20 次取中位數與 IQR。同時輸出真值誤差版本。
- 比對：
  - Table II 的趨勢（節點多、封包長 → 誤差大；隨時段下降）
  - 數值量級：第 6 段 A2 ≈ 0.21 ms、C12 ≈ 0.62 ms（mean |RSE|）
  - Fig.10 的「收斂後各對有固定偏移（約 ±0.3 ms）」
- 這個測試回答：論文的 0.2–0.6 ms 主要來自演算法估計誤差，還是節點間固定偏差與量測誤差？

### T7　資料完整性錯誤（學長提出）
- 子測試：
  - T7a 掉包（C10、C9）：掉包率 0.1%、1%、5%；斷線 1 次 × 5 s
  - T7b buffer 讀錯（B3）：0.1%、0.5%；f_s = 50 Hz 與 1 kHz
  - T7c 重複封包、t_P 重複（觸發 ∞×0 情境）
  - T7d 節點重開機（B5）、計時器溢位（A10）
  - T7e 負漂移：ρ 使 α = −20、−5 ppm（已知會讓樞紐點不斷重置）
- 每個子測試比較：(1) 原版 SharkTooth；(2) 先做完整性檢查（封包編號連續性、樣本數、頭尾時間銜接，§7）再進 SharkTooth。
- 指標：真值誤差 P95、樞紐點重設次數、無效輸出比例。

### T8　與先前方法比較（簡報「以前怎麼解延遲」的模擬佐證）
- 方法見 §6。同一份資料（T4、T6 的設定）跑所有方法。
- 指標：真值誤差的平均、P95，收斂時間（誤差首次持續 < 1 ms 的時間），每封包運算量。
- 情境：各節點延遲分布相同 vs **不同**（一個節點加重 D2/E1 負載或 PER）。【推測】直接 OLS 的截距會吸收平均延遲；各節點延遲相同時互相抵消，不同時才會看出 SharkTooth 的優勢。

### T9　SharkTooth 參數敏感度
- M（Eq.7）∈ {20, 50, 100, 200}；最大權重 0.5 ∈ {0.2, 0.5, 0.8}；γ_1 增量 10⁻⁶ ∈ {10⁻⁷, 10⁻⁶, 10⁻⁵}；穩定期 30 s ∈ {5, 10, 30, 60}；封包率 5、10、20 Hz（檢驗 10⁻⁶ 是否綁定 10 Hz）。
- 穩定期輸出：論文字面（α = β = 0）vs 保留上一組 α、β。

---

## 6. 要實作的演算法（基準方法）

| 名稱 | 做法 | 對應文獻（論文編號） | 備註 |
|---|---|---|---|
| `tp_only` | t̂_S = t_P + 首包 offset | — | 只修 offset，不修 drift |
| `tc_only` | t̂_S = t_C | — | 學長之前的「主機打時間」；含全部延遲抖動 |
| `ols_raw` | 對全部 (t_P, δT) 做（遞迴）OLS | 應用層配對＋回歸 [25] | 不處理尖峰 |
| `ols_window` | 最近 W 秒的 OLS | FTSP 風格回歸 [9] | 回歸視窗 |
| `ols_trimmed` | OLS 後剔除殘差 > kσ 再擬合 | 一般離群處理 | |
| `min_filter` | 每 W 秒取 δT 最小值，再對最小值做 OLS | 「最小延遲」濾波 | |
| `lower_hull` | 對 (t_P, δT) 取下凸包，用線性規劃求「在所有點下方、且最貼近」的直線 | 網路量測的 clock skew 估計（Moon, Skelly, Towsley；convex hull 法） | **與 SharkTooth 下緣想法同源，簡報可介紹** |
| `sharktooth` | Algorithm 1+2（Eq.3–22） | 本文 | 已在 `ble_sim.py` |
| `sharktooth_ff` | SharkTooth + 遺忘因子 | 本規格新增 | T5 |

> 無法在此模擬中公平比較、只在簡報文字說明的：RBS [18]、TPSN [19]、FTSP 的 MAC 層時間戳、CheepSync [10]、BlueSync [14]（需要 MAC/硬體時間戳或 advertising 模式）、神經網路法 [8]（需訓練資料）、離線法 [6]。若要，可另外加「假設有 MAC 時間戳」的理想化版本作為上界。

---

## 7. 完整性檢查（T7 用，SharkTooth 前處理）

每收到一包：
1. **封包編號**：與上一包差 1 → 正常；> 1 → 掉包，記錄缺口；≤ 0 → 重複或重開機。
2. **樣本數**：等於預期 N？
3. **時間銜接**：本包第一筆時間 = t_P − (n−1)/f_s，應等於上一包最後一筆 + 1/f_s（容差 0.5/f_s）。早 → 重疊（丟掉重疊樣本）；晚 → 缺樣本（標 NaN）。論文驗證封包含「首末樣本時間差 2 B + 封包編號 1 B」【原文 §IV-A】，正是做這件事需要的欄位。
4. **t_P 倒退或大跳** → 重開機或溢位 → 重置 SharkTooth 狀態（或做 unwrap）。
5. 同一 t_P 重複 → 丟棄後者（避免 Eq.9 的 ∞×0）。

---

## 8. 評估指標定義

- **真值誤差**（模擬專用）：e_n(i) = t̂_S,n(i) − clock_PC(t_true,n(i))。報告時扣除全段中位數（常數偏差另外列）。
- **節點對誤差**：e_nm = e_n − e_m（在共同的 1 s epoch 上平均）。
- **論文式 RSE**：完整照 §IV-B：50 Hz 重取樣 → Eq.(23)(24) → 模擬 5 Hz 參考訊號的互相關 → Eq.(25)(26) → 600 s 分段、Eq.(27) 最差一對 → 平均、標準差、P95 → 20 seeds 中位數與 IQR。
- 收斂時間、樞紐點次數、每封包運算時間。

---

## 9. 交付物

1. `results/<test>/`：圖、summary.json、結論 README。
2. `results/fig_compare.png`：Fig.3、Fig.4、Fig.6 與模擬並排。
3. `results/ablation_T4.md`：每個延遲來源 → 產生的圖形特徵（簡報用表）。
4. `results/baselines_T8.md`：方法比較表。
5. 可直接放進簡報的圖：與論文相同座標、字體 ≥ 12 pt。

---

## 10. 待確認（請問學長／負責硬體的同學）

- 我們的板子 BLE 版本（4.0/4.2/5.x）、是否支援 Data Length Extension、2M PHY。
- 打包觸發方式（計時器 vs 湊滿 N 筆）、時間戳用哪個時鐘源與單位。
- PC 端接收方式（dongle + 序列化 / 內建藍牙 / 手機）。
- 實測掉包率、典型距離與佩戴位置。

---

## 11. 參考

- 論文本身（Fig.3、4、6、10、11、Table I–III；§III–V）。
- Nordic SoftDevice Controller — Scheduling（多連線排程、事件延後與丟棄）：https://developer.nordicsemi.com/nRF_Connect_SDK/doc/latest/nrfxlib/softdevice_controller/doc/scheduling.html
- Nordic DevZone — Wireless timer synchronization among nRF5 devices（timeslot API、PPI、重傳對應用不可見）：https://devzone.nordicsemi.com/guides/short-range-guides/b/bluetooth-low-energy/posts/wireless-timer-synchronization-among-nrf5-devices
- Raltron R38 32.768 kHz 晶振規格（±20 ppm、−0.034 ppm/°C²、turnover 25 °C、老化 ±3 ppm/年）：https://www.raltron.com/webproducts/specs/CRYSTAL/R38.pdf
- Moon, Skelly, Towsley, *Estimation and Removal of Clock Skew from Network Delay Measurements*（下界／線性規劃法）：https://web.cs.umass.edu/publication/docs/1998/UM-CS-1998-043.pdf
- 既有模擬起點：`ble_sim.py`（上一版，需依 §4 重構）。
