# Meeting Paper 論文導讀教材與原論文整理

本儲存庫整理了論文原始 PDF 以及由 Claude 與 Gemini 分別生成的論文導讀教材（簡報/教材 PDF），適合用於學術討論、Meeting 報告與實驗重現。

---

## 包含論文清單

1. **A Generic Non-invasive Neuromotor Interface (Meta, Nature 2025)**
   - 原文：`A generic non-invasive neuromotor .pdf`
   - Claude 導讀：`claude/通用神經運動介面導讀教材.pdf`
   - Gemini 導讀：`gemini/Generic Neuromotor Interface (Meta Nature 2025) 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf`
   - [雲端最新文件（Google Docs）](https://docs.google.com/document/d/1Tpaliy4jbEZedTQtej2nYgUAmJwSMZ2qK0tx03xjQNQ/edit?usp=drivesdk&ouid=109413389150100365786)

2. **SASG-DA: Sparse-Aware Semantic-Guided Diffusion Augmentation for Myoelectric Gesture Recognition**
   - 原文：`SASG-DA_Sparse-Aware_Semantic-Guided_Diffusion_Augmentation_for_Myoelectric_Gesture_Recognition.pdf`
   - Claude 導讀：`claude/SASG-DA 導讀教材.pdf`
   - Gemini 導讀：`gemini/SASG-DA 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf`
   - [雲端最新文件（Google Docs）](https://docs.google.com/document/d/1jSaFXDZzkhpgf2QVRmZuMQV7e0WgX7qfn-Ulyewb06A/edit?usp=drivesdk&ouid=109413389150100365786)

3. **SharkTooth: A Scalable Real-Time Algorithm for BLE-Based Wireless Body Sensor Networks Synchronization (IEEE IoT-J 2025)**
   - 原文：`SharkTooth_A_Scalable_Real-Time_Algorithm_for_BLE-Based_Wireless_Body_Sensor_Networks_Synchronization.pdf`
   - Claude 導讀：`claude/SharkTooth 導讀教材.pdf`
   - Gemini 導讀：`gemini/SharkTooth (IEEE IoT-J 2025) 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf`
   - [雲端最新文件（Google Docs）](https://docs.google.com/document/d/1jih2iMagblxgZbkoUX6KayA835-cIG58xrNfm3fW7hw/edit?usp=drivesdk&ouid=109413389150100365786)

---

## 重要勘誤與校正紀錄

針對 Gemini 導讀教材中與原文對照發現的 6 項核心數學公式、狀態機與數據偏差（如 SharkTooth Eq. 11–12 幾何探針、Nature Fig. 2 冪律指數、LR 單位換算、SASG-DA Table II 欄位等），已詳細整理於：
- [gemini/勘誤與核心修正對照.md](gemini/勘誤與核心修正對照.md)

---

## SharkTooth 復現模擬與測試

依《SharkTooth 復現模擬：延遲來源整理與測試規格》完成的模擬器與 T0–T13 測試，放在 [`sharktooth_sim/`](sharktooth_sim/)。
結論總表：[`sharktooth_sim/results/README.md`](sharktooth_sim/results/README.md)；論文圖與模擬並排：`sharktooth_sim/results/fig_compare.png`。
每個參數的出處與合理性：[`sharktooth_sim/results/provenance_paper_match.md`](sharktooth_sim/results/provenance_paper_match.md)。

---

## 目錄結構

```
meetpaper/
├── README.md
├── A generic non-invasive neuromotor .pdf
├── SASG-DA_Sparse-Aware_Semantic-Guided_Diffusion_Augmentation_for_Myoelectric_Gesture_Recognition.pdf
├── SharkTooth_A_Scalable_Real-Time_Algorithm_for_BLE-Based_Wireless_Body_Sensor_Networks_Synchronization.pdf
├── sharktooth_sim/          # SharkTooth 復現模擬（模擬器、設定檔、T0–T13 結果）
├── claude/
│   ├── SASG-DA 導讀教材.pdf
│   ├── SharkTooth 導讀教材.pdf
│   └── 通用神經運動介面導讀教材.pdf
└── gemini/
    ├── 勘誤與核心修正對照.md
    ├── Generic Neuromotor Interface (Meta Nature 2025) 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf
    ├── SASG-DA 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf
    └── SharkTooth (IEEE IoT-J 2025) 論文完整導讀教材（Meeting報告與實驗重現專用）.pdf
```
