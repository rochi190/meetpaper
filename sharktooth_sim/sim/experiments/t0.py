"""T0：數位化論文圖、建立數字目標 targets.json。"""
import numpy as np

from .. import digitize as dg
from .. import figures as F
from ..runner import ROOT, outdir, write_json, write_text


def run(cfg_path):
    od = outdir("T0_digitize")
    o, c = dg.digitize_all()
    dg.save(o, c)
    T = dg.build_targets()
    T["calibration_max_fit_error"] = {k: dict(x=v["x_fit_maxerr"], y=v["y_fit_maxerr"]) for k, v in c.items()}
    write_json(ROOT / "data" / "targets.json", T)
    write_json(od / "summary.json", T)
    dg.overlay_check(od / "overlay_check.png")
    # 殘差直方圖（Fig.6）
    fig, ax = F.plt.subplots(1, 2, figsize=(12, 4.2))
    d = dg.load("fig6", "dots")
    d = d[np.argsort(d[:, 0])]
    a, b = T["fig6"]["lower_edge_lp"]["slope_ppm"] * 1e-6, T["fig6"]["lower_edge_lp"]["intercept"]
    r = (d[:, 1] - (a * d[:, 0] + b)) * 1e3
    ax[0].plot(d[:, 0], d[:, 1], ".", ms=2.5, color=F.BLUE)
    ax[0].plot([0, 145], [b, b + a * 145], "k--", lw=1, label=f"下緣（LP）{a*1e6:.0f} ppm")
    ax[0].set_xlim(0, 145)
    ax[0].set_ylim(0.28, 0.42)
    ax[0].set_xlabel("t_P (s)")
    ax[0].set_ylabel("δT (s)")
    ax[0].legend()
    ax[0].set_title("Fig.6 數位化（1383 點，無掉包）")
    ax[1].hist(r[d[:, 0] > d[0, 0] + 1], bins=np.arange(0, 105, 2.5), color=F.BLUE)
    ax[1].axvline(30, color="r", ls="--", lw=1)
    ax[1].axvline(60, color="r", ls="--", lw=1)
    ax[1].set_xlabel("殘差 r = δT − 下緣 (ms)")
    ax[1].set_ylabel("點數")
    ax[1].set_title("0–30 ms 均勻（等 event）＋ 30–60 ms 平台（重傳 1 次）")
    F.save(fig, od / "fig6_residual_hist.png")
    f6, f3, f4 = T["fig6"], T["fig3"], T["fig4"]
    txt = f"""# T0　數位化論文圖、建立數字目標

**方法**：論文的圖是 pgfplots 向量圖。直接讀 PDF 繪圖指令（PyMuPDF `get_drawings()`），以灰色刻度線最小平方擬合像素↔座標（最大殘差見 `summary.json` 的 `calibration_max_fit_error`，皆 < 0.1 % 刻度間距）。
疊回原圖的檢查：`overlay_check.png`（綠點 / 黑點與原圖完全重合）。原始點存在 `data/digitized/*.csv`，目標統計量在 `data/targets.json`。
Table II 以 220 dpi 影像人工轉錄成 `data/paper_table2.csv`。

## 主要數字

| 項目 | Fig.3 | Fig.4 | Fig.6 |
|---|---|---|---|
| 封包間隔（以 t_P 計） | {f3['pkt_period_ms_zoom']:.2f} ms | {f4['pkt_period_ms_zoom']:.2f} ms | {f6['pkt_period_mean_ms']:.2f} ms |
| δT 下緣斜率 | {f3['slope_ppm']:.0f} ppm | {f4['slope_ppm']:.0f} ppm | {f6['lower_edge_lp']['slope_ppm']:.0f} ppm（LP）/ {f6['lower_edge_q05']['slope_ppm']:.0f} ppm（5 % 分位） |
| 每包 δT 下降（鋸齒） | {f3['step_down_ms']:.2f} ms | {f4['step_down_ms']:.2f} ms | — |
| 每個鋸齒週期的封包數 | {f3['packets_per_cycle_zoom']:.2f} | {f4['packets_per_cycle_zoom']:.2f} | — |
| σ100 平均（範圍） | {f3['sigma100_ms']['mean']:.2f}（{f3['sigma100_ms']['min']:.2f}–{f3['sigma100_ms']['max']:.2f}）ms | {f4['sigma100_ms']['mean']:.1f}（{f4['sigma100_ms']['min']:.1f}–{f4['sigma100_ms']['max']:.1f}）ms | — |
| 殘差 P10 / P50 / P90 / P99 | {f3['residual_ms']['P10']:.1f} / {f3['residual_ms']['P50']:.1f} / {f3['residual_ms']['P90']:.1f} / {f3['residual_ms']['P99']:.1f} ms | {f4['residual_ms']['P10']:.1f} / {f4['residual_ms']['P50']:.1f} / {f4['residual_ms']['P90']:.1f} / {f4['residual_ms']['P99']:.1f} ms | {f6['residual_ms']['P10']:.1f} / {f6['residual_ms']['P50']:.1f} / {f6['residual_ms']['P90']:.1f} / {f6['residual_ms']['P99']:.1f} ms |
| 殘差 > 1 CI / > 2 CI 的比例 | {f3['frac_gt_CI']*100:.1f} % / {f3['frac_gt_2CI']*100:.1f} % | {f4['frac_gt_CI']*100:.1f} % / {f4['frac_gt_2CI']*100:.1f} % | {f6['frac_gt_CI']*100:.1f} % / {f6['frac_gt_2CI']*100:.1f} % |
| 離群（> 45 ms） | — | — | {f6['outlier_frac_gt45ms']*100:.1f} % |
| 尖峰後 δT 每包下降 | — | {f4['drain_step_ms']:.1f} ms | 起始：{f6['startup_drain_ms_per_pkt']:.1f} ms |

Fig.6 起始尖峰：前 4 點高出下緣 {', '.join(f'{v:.0f}' for v in f6['startup']['excess_ms'][:4])} ms；樞紐點 (t_P, δT) = ({f6['pivot']['tp']:.3f} s, {f6['pivot']['dT']:.4f} s)。
Fig.10 收斂後各對的固定偏移（第 6 段中位數）：{f"{T['fig10']['steady_offset_range_ms'][0]:.2f}"} ～ {f"{T['fig10']['steady_offset_range_ms'][1]:.2f}"} ms。

## 結論（這幾點改變了模擬的設定）

1. **封包間隔不是 100 ms，而是 100.64–100.93 ms**（三張圖都是，且彼此不同）。t_P 由 ±20 ppm 晶振量測，所以這 0.6–0.9 % 的差是「取樣計時器本身比較慢」【推測：ADC 計時器由 HFRC 衍生】。
   後果：P mod CI = 10.6–10.9 ms（不是 10 ms），每 3 包相位再偏 2–3 ms，**等待時間在約 3–4 s 內掃過整個 CI** → δT 是**連續**的 30 ms 帶、下緣平滑。
   上一版模擬用剛好 100 ms，P/CI 的公因數 10 ms 把等待鎖在 3 個相位 → 才會出現「離散水平線」與「階梯下緣」。不需要假設很大的連續型延遲。
2. **σ100 的理論值**：純等待時 δ²T 只有兩個值（−r、CI − r），σ = √(r(CI − r))；r = 10.93 ms → 14.44 ms。Fig.3c 平均 14.53 ms → 其餘連續抖動只需 σ ≈ 1 ms。
3. **Fig.6 殘差 = 0–30 ms 均勻 + 30–60 ms 平台**（`fig6_residual_hist.png`）：均勻部分是等 connection event；平台是「重傳一次」（+1 CI），約 9 %；> 60 ms 的極少（重傳兩次 ≈ 0.09²）。→ 單節點、每次嘗試 PER ≈ 9 % 就能解釋 Fig.6，不一定要多節點。
4. **起始尖峰與 Fig.4 尖峰後都是每包 −40 ms 的斜坡**：100.7 − 40 ≈ 60 ms = 2 × CI。表示背壓時**每 2 個 event 才送得出一包**——符合「stack 只有一個 notification 緩衝，要等下一個 event 的 ack 回來才放下一包」的韌體流量控制（論文 §IV-A 也寫「封包週期至少要是 CI 的兩倍」）。
5. **Table II 的穩態底線**：A2 第 2–6 段 std ≈ 0.25 ms、mean|RSE| ≈ 0.21 ms ≈ 0.8 σ（零均值常態的 E|x|）→ 穩態幾乎全是零均值雜訊；C12 的 0.62 / 0.41 則需要約 0.5 ms 的固定偏移。這是 T6 要檢驗的。
"""
    write_text(od / "README.md", txt)
