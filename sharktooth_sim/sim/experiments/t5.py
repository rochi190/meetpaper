"""T5：晶振暖機（A2/A3）、A8 換算誤差、遺忘因子。"""
import itertools
import json

import numpy as np

from .. import algorithms
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import RESULTS, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate


def make_cfg(cfg0, c):
    cfg = cfg0
    tau = c["tau_min"] * 60
    sp = cfg0["node_spread"]
    cfg = set_path(cfg, "clock_node.temp.T_start", c["T_start"])
    cfg = set_path(cfg, "clock_node.temp.T_end", {"normal": [c["T_end"], sp["T_end_sd"]]})
    cfg = set_path(cfg, "clock_node.temp.tau_s", {"normal": [tau, tau * sp["tau_rel_sd"]]})
    cfg = set_path(cfg, "clock_node.scale_err_ppm", c["scale_err_ppm"])
    return cfg


def seg_err(t, e, valid, bias):
    out = []
    for k in range(6):
        m = valid & (t >= 600 * k) & (t < 600 * (k + 1)) & np.isfinite(e)
        v = (e[m] - bias) * 1e3
        out.append([float(np.mean(np.abs(v))) if v.size else np.nan, float(np.percentile(np.abs(v), 95)) if v.size else np.nan])
    return out


def _job(args):
    cfg, seed, forgets = args
    sim = simulate(cfg, seed)
    res = {}
    curv = []
    errs = {f: [] for f in forgets}
    pair_e = {f: [] for f in forgets}
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        tp = p["t_P"]
        dtrue = p["t_pc_true"] - tp                       # 無延遲的 δT（真曲線）
        tt = p["t_true_last_sample"]
        a1 = np.polyfit(tp[tt < 1200], dtrue[tt < 1200], 1)[0]
        a2 = np.polyfit(tp[tt >= 1200], dtrue[tt >= 1200], 1)[0]
        lin = np.polyval(np.polyfit(tp, dtrue, 1), tp)
        a_end = np.polyfit(tp[tt >= cfg["T"] - 600], dtrue[tt >= cfg["T"] - 600], 1)[0]
        r0 = algorithms.run("sharktooth", tp, p["t_C"], CI=cfg["CI"])
        curv.append(dict(slope_first20_ppm=a1 * 1e6, slope_last40_ppm=a2 * 1e6, dslope_ppm=(a2 - a1) * 1e6,
                         max_dev_from_line_ms=float(np.max(np.abs(dtrue - lin)) * 1e3),
                         alpha_negative=float(a2 < 0), n_pivot=float(r0["n_pivot"]),
                         invalid_frac=float(1 - r0["valid"].mean()),
                         end_slope_err_ppm=float(abs(r0["alpha"][-1] - a_end) * 1e6)))
        grid = np.arange(0, cfg["T"], 1.0) + 0.5
        for f in forgets:
            r = algorithms.run("sharktooth", tp, p["t_C"], CI=cfg["CI"], keep_last=f < 1.0, forget=f)
            e = r["ts"] - p["t_pc_true"]
            v = r["valid"]
            steady = v & (tt > 2400)
            bias = np.median(e[steady])
            errs[f].append(seg_err(tt, e, v, bias))
            pair_e[f].append(np.interp(grid, tt[v], e[v], left=np.nan, right=np.nan))
    pairs = {}
    for f in forgets:
        d = pair_e[f][0] - pair_e[f][1]
        pairs[f] = seg_err(np.arange(len(d)) + 0.5, d, np.isfinite(d), 0.0)   # 節點對不扣偏差（固定偏移也算誤差）
    return dict(curv=curv, errs={str(f): v for f, v in errs.items()}, pairs={str(f): v for f, v in pairs.items()})


def grid_mask(n):
    m = np.zeros(n, bool)
    m[2400:] = True
    return m


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    names = list(cfg0["sweep"])
    combos = [dict(zip(names, v)) for v in itertools.product(*cfg0["sweep"].values())]
    # 對照組：沒有溫度效應
    base = set_path(cfg0, "clock_node.temp.enabled", False)
    forgets = cfg0["forget_values"]
    jobs = [(make_cfg(cfg0, c), s, forgets) for c in combos for s in cfg0["seeds"]]
    jobs += [(base, s, forgets) for s in cfg0["seeds"]]
    res = pmap(_job, jobs)
    ns = len(cfg0["seeds"])
    rows = []
    for i, c in enumerate(combos + [dict(T_start=None, T_end=None, tau_min=None, scale_err_ppm=None)]):
        rr = res[i * ns:(i + 1) * ns]
        row = dict(c)
        cv = [x for r in rr for x in r["curv"]]
        for k in cv[0]:
            row[k] = float(np.mean([x[k] for x in cv]))
        for f in forgets:
            se = np.array([x for r in rr for x in r["errs"][str(f)]])     # (seeds*nodes, 6, 2)
            pe = np.array([r["pairs"][str(f)] for r in rr])
            row[f"node_absavg_f{f}"] = np.nanmedian(se[:, :, 0], axis=0).tolist()
            row[f"node_p95_f{f}"] = np.nanmedian(se[:, :, 1], axis=0).tolist()
            row[f"pair_absavg_f{f}"] = np.nanmedian(pe[:, :, 0], axis=0).tolist()
        rows.append(row)
    write_json(od / "summary.json", rows)
    # ---- 圖 1：一個例子（T 22→34 °C、τ = 10 min）的真 δT 彎曲與誤差 ----
    c_ex = dict(T_start=22.0, T_end=34.0, tau_min=10.0, scale_err_ppm=-248.0)
    cfg = make_cfg(cfg0, c_ex)
    sim = simulate(cfg, 3)
    p = node_series(sim, 0)
    tp = p["t_P"]
    tt = p["t_true_last_sample"]
    dtrue = p["t_pc_true"] - tp
    lin = np.polyval(np.polyfit(tp[tt > 1200], dtrue[tt > 1200], 1), tp)
    fig, axs = F.plt.subplots(1, 3, figsize=(17, 4.6))
    g, Tc = sim["nodes"][0]["meta"].get("temperature", (None, None)) if False else (None, None)
    temp = cfg["clock_node"]["temp"]
    tg = np.linspace(0, 3600, 400)
    axs[0].plot(tg / 60, 34.0 - (34.0 - 22.0) * np.exp(-tg / 600), label="溫度 (°C)")
    ax2 = axs[0].twinx()
    ax2.plot(tg / 60, -0.034 * ((34.0 - (34.0 - 22.0) * np.exp(-tg / 600)) - 25) ** 2, "r-", label="ρ_temp (ppm)")
    axs[0].set_xlabel("時間 (min)")
    axs[0].set_ylabel("溫度 (°C)")
    ax2.set_ylabel("ρ_temp (ppm)", color="r")
    axs[0].set_title("A3 暖機：22 → 34 °C，τ = 10 min")
    axs[1].plot(tt / 60, (dtrue - lin) * 1e3, color=F.BLUE)
    axs[1].set_xlabel("時間 (min)")
    axs[1].set_ylabel("真 δT − 後 40 min 直線 (ms)")
    axs[1].set_title("δT 前期彎曲（相對穩態直線）")
    for f, col in zip(forgets, ["k", "tab:orange", "tab:green"]):
        r = algorithms.run("sharktooth", tp, p["t_C"], CI=cfg["CI"], keep_last=f < 1.0, forget=f)
        e = (r["ts"] - p["t_pc_true"])
        v = r["valid"]
        bias = np.median(e[v & (tt > 2400)])
        axs[2].plot(tt[v] / 60, (e[v] - bias) * 1e3, "-", lw=0.8, color=col,
                    label="論文（累積 OLS）" if f == 1.0 else f"遺忘因子 λ = {f}")
    axs[2].set_ylim(-3, 3)
    axs[2].set_xlabel("時間 (min)")
    axs[2].set_ylabel("真值誤差 − 穩態偏差 (ms)")
    axs[2].legend(fontsize=10)
    axs[2].set_title("單一節點 SharkTooth 誤差")
    for x in range(10, 60, 10):
        axs[2].axvline(x, color="b", ls="--", lw=0.6)
    F.save(fig, od / "warmup_example.png")
    # ---- 圖 2：分段誤差（單節點、節點對）vs 有無暖機 ----
    fig, axs = F.plt.subplots(1, 2, figsize=(13, 4.6))
    sel = [r for r in rows if r["T_start"] in (22.0, None) and r["T_end"] in (34.0, None) and r["scale_err_ppm"] in (0.0, None)]
    for r in sel:
        lab = "無溫度效應" if r["T_start"] is None else f"22→34 °C, τ={r['tau_min']:g} min"
        axs[0].plot(range(1, 7), r["node_absavg_f1.0"], "o-", label=lab)
        axs[1].plot(range(1, 7), r["pair_absavg_f1.0"], "o-", label=lab)
    axs[0].set_title("單一節點 mean|e|（論文累積 OLS）")
    axs[1].set_title("節點對 mean|e_nm|（論文累積 OLS）")
    for ax in axs:
        ax.set_xlabel("600 s 分段")
        ax.set_ylabel("ms")
        ax.legend(fontsize=9)
    F.save(fig, od / "segment_errors.png")

    def fmt(v):
        return " / ".join(f"{x:.2f}" for x in v)

    def line(r):
        return (f"| {r['T_start'] if r['T_start'] is not None else '—'} | {r['T_end'] if r['T_end'] is not None else '—'} | "
                f"{r['tau_min'] if r['tau_min'] is not None else '—'} | {r['dslope_ppm']:+.2f} | {r['max_dev_from_line_ms']:.2f} | "
                f"{fmt(r['node_absavg_f1.0'])} | {fmt(r['pair_absavg_f1.0'])} | {r['end_slope_err_ppm']:.2f} | "
                f"{r['node_absavg_f0.9999'][5]:.2f} / {r['node_absavg_f0.999'][5]:.2f} | {r['alpha_negative']*100:.0f} % / {r['n_pivot']:.0f} / {r['invalid_frac']*100:.1f} % |")
    hdr = ("| T_start | T_end | τ (min) | 斜率變化 (ppm) | 真 δT 偏離直線 (ms) | 單節點 mean\\|e\\| 第 1–6 段 (ms) | 節點對 mean\\|e_nm\\| 第 1–6 段 (ms) | "
           "結束時 α 誤差 (ppm) | 第 6 段 λ=0.9999 / 0.999 | α<0 比例 / 樞紐點數 / 無效輸出 |\n|---|---|---|---|---|---|---|---|---|---|")
    ctrl = rows[-1]
    rA = [r for r in rows[:-1] if r["scale_err_ppm"] == -248.0] + [ctrl]
    r0 = [r for r in rows[:-1] if r["scale_err_ppm"] == 0.0]
    worst = max(rA[:-1], key=lambda r: r["node_absavg_f1.0"][0])
    write_text(od / "README.md", f"""# T5　時鐘暖機（A2/A3）與 A8 換算誤差

**設定**（`config/T5_warmup.yaml`）：T4 擬合出的延遲參數（2 節點 / 1 dongle）＋ 晶振溫度係數 −0.034 ppm/°C²（turnover 25 °C）＋ 暖機
T(t) = T_end − (T_end − T_start)e^(−t/τ)。每個節點的 T_end 與 τ 各自抽樣（sd 1 °C、20 %），不然節點對會完全抵消。
掃 T_start ∈ {{22, 25}}、T_end ∈ {{30, 34}} °C、τ ∈ {{2, 5, 10}} min、A8 ∈ {{0, −248}} ppm；每格 20 seeds × 3600 s；晶振 ρ ~ U(−20, 20) ppm。
單節點誤差 = 真值誤差扣掉穩態（後 20 min）中位數（該常數是最小路徑延遲）；節點對誤差**不扣**偏差。表中是 20 seeds 的中位數。

## A8 = −248 ppm（δT 斜率 ≈ +248 ppm，與論文圖相同）

{hdr}
{chr(10).join(line(r) for r in rA)}

（最後一列 — = 沒有溫度效應的對照組。）

## A8 = 0（δT 斜率 = 晶振誤差，−20～+20 ppm，正負各半）

{hdr}
{chr(10).join(line(r) for r in r0)}

## 結論

1. **暖機讓 δT 前期彎曲**：22 → 34 °C、τ = 10 min 時，斜率前後差 {worst['dslope_ppm']:+.1f} ppm，真 δT 偏離直線最多 {worst['max_dev_from_line_ms']:.1f} ms。
2. **單一節點**（A8 = −248）：第 1–3 段誤差明顯變大（最差組合 {fmt(worst['node_absavg_f1.0'][:3])} ms，對照 {fmt(ctrl['node_absavg_f1.0'][:3])} ms），
   τ 越長、溫差越大越嚴重；第 5–6 段已回到對照水準。累積 OLS 在結束時的 α 誤差 {worst['end_slope_err_ppm']:.2f} ppm（對照 {ctrl['end_slope_err_ppm']:.2f}）
   ——**彎曲確實被留在累加量裡，表現為斜率偏差**；在 1 小時內換算成時間誤差不大，但記錄越長、外插越遠越明顯。
   遺忘因子：λ = 0.9999（記憶約 1000 s）第 6 段 {worst['node_absavg_f0.9999'][5]:.2f} ms，與累積 OLS 相當；λ = 0.999（約 100 s）反而變差（{worst['node_absavg_f0.999'][5]:.2f} ms，δT* 的雜訊漏進來）。
3. **節點對**（論文 RSE 量的東西）：大家一起升溫，ρ_temp 大部分抵消，第 1 段幾乎不變（{worst['pair_absavg_f1.0'][0]:.2f} vs 對照 {ctrl['pair_absavg_f1.0'][0]:.2f} ms）；
   但各節點升溫曲線不同時，**彎曲留在累積 OLS 裡造成的斜率偏差會讓第 5–6 段的節點對誤差變大**
   （22→34 °C、τ = 10 min：{fmt(worst['pair_absavg_f1.0'][4:])} ms，對照 {fmt(ctrl['pair_absavg_f1.0'][4:])} ms）。
   遺忘因子對節點對：λ = 0.9999 把這組第 5–6 段降到 {fmt(worst['pair_absavg_f0.9999'][4:])} ms（對照組則從 {fmt(ctrl['pair_absavg_f1.0'][4:])} 稍升到 {fmt(ctrl['pair_absavg_f0.9999'][4:])}）；
   λ = 0.999 各段都在 {np.mean(worst['pair_absavg_f0.999'][2:]):.2f} ms 左右，太短。→ **溫度會變的長時間記錄，λ ≈ 0.9999 是合理折衷**。
   → 暖機**不能**解釋 Table II「第 1 段大、之後收斂」：它造成的是「後段慢慢變大」。第 1 段大主要是 SharkTooth 起始收斂（對照組第 1 段 {ctrl['pair_absavg_f1.0'][0]:.2f} vs 第 6 段 {ctrl['pair_absavg_f1.0'][5]:.2f} ms），見 T6。
4. **A8 的意外發現：大的正斜率對 SharkTooth 有利。** A8 = 0 時約 {np.mean([r['alpha_negative'] for r in r0])*100:.0f} % 的節點 α < 0（δT 隨時間下降）：
   每個新低點都成為新樞紐點 → 迴歸矩陣歸零 → 論文字面版在 30 s 穩定期內輸出 α = β = 0（無效）。
   結果無效輸出 {np.mean([r['invalid_frac'] for r in r0])*100:.0f} %（A8 = −248 時 {ctrl['invalid_frac']*100:.1f} %）、樞紐點 {np.mean([r['n_pivot'] for r in r0]):.0f} 個（{ctrl['n_pivot']:.0f} 個），
   節點對誤差各段約 {np.mean([np.mean(r['pair_absavg_f1.0'][3:]) for r in r0]):.2f} ms（A8 = −248 時約 {np.mean(ctrl['pair_absavg_f1.0'][3:]):.2f} ms）。詳見 T7e。
   論文三張圖的斜率都是 235–245 ppm（遠超 ±20 ppm 晶振）——**很像韌體共同的 tick 換算常數**（例如 1/32768 s 寫成 30.51 µs → −248 ppm），
   恰好讓所有節點 α 都 > 0，SharkTooth 才一直在「正常」模式下運作。換成換算正確的韌體，就會碰到負漂移問題。
5. 圖：`warmup_example.png`（溫度與 ρ、真 δT 的彎曲、三種 OLS 的誤差）、`segment_errors.png`。
""")
