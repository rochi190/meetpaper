"""T3：復現 Fig.4（雜訊條件）。"""
import itertools
import json

import numpy as np

from .. import digitize as dg
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import ROOT, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate

KEYS = ["sigma100_mean", "P50", "P75", "P90", "P99", "frac_1to2CI", "frac_gt_2CI", "spikes_per_100s", "spike_med",
        "drain_frac", "gap_frac"]


def apply(cfg, combo):
    for k, v in combo.items():
        if k == "spike_prob":
            cfg = set_path(cfg, "pauses.rate_hz", v / cfg["packet"]["period_nom"])
        else:
            cfg = set_path(cfg, k, v)
    return cfg


def series_metrics(tp, dt):
    bs, r = ev.band_stats(tp, dt, skip_start_s=2.0)
    sig = ev.sigma_M(dt, 100)
    is_pk = (r[1:-1] > 45) & (r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:])
    pk = r[1:-1][is_pk]
    d2 = np.diff(dt) * 1e3
    dtp = np.diff(tp)
    gaps = np.round(dtp / np.median(dtp)) - 1
    return dict(sigma100_mean=float(np.nanmean(sig) * 1e3), P75=bs["P75"],
                frac_1to2CI=bs["frac_gt_CI"] - bs["frac_gt_2CI"],
                gap_frac=float(max(gaps.sum(), 0) / (len(tp) + gaps.sum())) + 1e-3, sigma100_p5=float(np.nanpercentile(sig, 5) * 1e3),
                sigma100_p95=float(np.nanpercentile(sig, 95) * 1e3),
                P50=bs["P50"], P90=bs["P90"], P99=bs["P99"], frac_gt_CI=bs["frac_gt_CI"], frac_gt_2CI=bs["frac_gt_2CI"],
                spikes_per_100s=float(len(pk) / (tp[-1] - tp[0]) * 100),
                spike_med=float(np.median(pk)) if len(pk) else 0.0,
                spike_p90=float(np.percentile(pk, 90)) if len(pk) else 0.0,
                drain_frac=float(np.mean((d2 < -30) & (d2 > -50))))


def target_metrics():
    la = dg.load("fig4a", "line")
    return series_metrics(la[:, 0], la[:, 1])


def loss(m, t):
    s = 0.0
    for k in KEYS:
        s += ((m[k] - t[k]) / max(abs(t[k]), 1e-3)) ** 2
    return s


def _job(args):
    cfg, seed = args
    p = node_series(simulate(cfg, seed))
    return series_metrics(p["t_P"], p["t_C"] - p["t_P"])


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    od = outdir(cfg0["name"])
    tgt = target_metrics()
    names = list(cfg0["sweep"])
    combos = [dict(zip(names, v)) for v in itertools.product(*cfg0["sweep"].values())]
    jobs = [(apply(cfg0, c), s) for c in combos for s in cfg0["seeds"]]
    res = pmap(_job, jobs)
    rows = []
    ns = len(cfg0["seeds"])
    for i, c in enumerate(combos):
        rr = res[i * ns:(i + 1) * ns]
        m = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
        rows.append(dict(**c, **m, loss=loss(m, tgt)))
    rows.sort(key=lambda r: r["loss"])
    grid_best = dict(rows[0])
    # 精修：隨機搜尋
    ref = cfg0.get("refine")
    refined = []
    if ref:
        rng = np.random.default_rng(123)
        rc = []
        for _ in range(ref["n"]):
            rc.append({k: float(rng.uniform(*v)) for k, v in ref["ranges"].items()})
        rjobs = [(apply(cfg0, c), s) for c in rc for s in cfg0["seeds"]]
        rres = pmap(_job, rjobs)
        for i, c in enumerate(rc):
            rr = rres[i * ns:(i + 1) * ns]
            m = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
            refined.append(dict(**c, **m, loss=loss(m, tgt)))
        refined.sort(key=lambda r: r["loss"])
    best = refined[0] if refined and refined[0]["loss"] < grid_best["loss"] else grid_best
    combo = {k: best[k] for k in names}
    cfgb = set_path(apply(cfg0, combo), "T", 3600.0)
    p = node_series(simulate(cfgb, 7))
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    la, lb, lc = dg.load("fig4a", "line"), dg.load("fig4b", "xmark"), dg.load("fig4c", "line")
    a_p, b_p = ev.lower_line_lp(la[:, 0], la[:, 1])
    m = (tp > 3366) & (tp < 3494)
    a_s, b_s = ev.lower_line_lp(tp[m], dt[m])
    dt_s = dt + (a_p * 3366 + b_p) - (a_s * 3366 + b_s)
    ptp = np.r_[la[:, 0], lb[:, 0]]
    o = np.argsort(ptp)
    paper = dict(tp=ptp[o], dt=np.r_[la[:, 1], lb[:, 1]][o], sig=np.interp(ptp[o], lc[:, 0], lc[:, 1]) / 1e3)
    F.paper_vs_sim_threepanel(od / "fig4_compare.png", paper, dict(tp=tp, dt=dt_s, sig=ev.sigma_M(dt, 100)), "fig4",
                              "模擬（最佳參數）")
    full = series_metrics(tp[(tp > 60)], dt[(tp > 60)])
    # 敏感度：每個參數的邊際效果（其餘固定在最佳值）
    sens = {}
    for k in names:
        sens[k] = [dict(value=r[k], **{q: r[q] for q in ["sigma100_mean", "P90", "frac_gt_2CI", "spikes_per_100s"]})
                   for r in rows if all(r[j] == best[j] for j in names if j != k)]
        sens[k].sort(key=lambda d: d["value"])
    write_json(od / "summary.json", dict(target=tgt, best=best, grid_best=grid_best, refined_top10=refined[:10],
                                         best_full_3600s=full, top10=rows[:10],
                                         sensitivity=sens, n_combos=len(rows), all_rows=rows))
    sig_ok = [r for r in rows if 30 <= r["sigma100_mean"] <= 50]
    fig, axs = F.plt.subplots(1, 2, figsize=(12, 4.3))
    allr = [r["sigma100_mean"] for r in rows]
    axs[0].hist(allr, bins=30, color=F.BLUE)
    axs[0].axvspan(30, 50, color="y", alpha=0.3, label="目標 30–50 ms")
    axs[0].set_xlabel("σ100 平均 (ms)")
    axs[0].set_ylabel("參數組合數")
    axs[0].legend()
    _, r_p = ev.band_stats(la[:, 0], la[:, 1], skip_start_s=0)
    _, r_s = ev.band_stats(tp[m], dt[m], skip_start_s=0)
    bins = np.arange(0, 260, 10)
    axs[1].hist(r_p, bins=bins, histtype="step", lw=2, density=True, label="論文 Fig.4a")
    axs[1].hist(r_s, bins=bins, histtype="step", lw=2, density=True, label="模擬（同時間窗）")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("殘差 δT − 下緣 (ms)")
    axs[1].legend()
    F.save(fig, od / "sigma_and_residual.png")
    lines = "\n".join(
        f"| {r['per.good']:.2f} | {r['per.bad']:.1f} | {r['per.mean_bad_events']:g} | {r['per.p_g2b']:.3f} | {r['spike_prob']:.3f} | "
        f"{r['sigma100_mean']:.1f} | {r['P90']:.0f} | {r['P99']:.0f} | {r['frac_gt_2CI']*100:.1f} | {r['spikes_per_100s']:.0f} | {r['spike_med']:.0f} | {r['loss']:.2f} |"
        for r in rows[:5])
    lines_ref = "\n".join(
        f"| {r['per.good']:.3f} | {r['per.bad']:.2f} | {r['per.mean_bad_events']:.1f} | {r['per.p_g2b']:.3f} | {r['spike_prob']:.3f} | "
        f"{r['sigma100_mean']:.1f} | {r['P90']:.0f} | {r['P99']:.0f} | {r['frac_gt_2CI']*100:.1f} | {r['spikes_per_100s']:.0f} | {r['spike_med']:.0f} | {r['loss']:.2f} |"
        for r in refined[:5])
    sens_txt = "\n".join(f"- `{k}`：" + "；".join(f"{d['value']:g} → σ100 {d['sigma100_mean']:.1f} ms、>2CI {d['frac_gt_2CI']*100:.1f} %"
                                                  for d in v) for k, v in sens.items())
    write_text(od / "README.md", f"""# T3　復現 Fig.4（雜訊條件）

**設定**（`config/T3_fig4.yaml`）：T1 的單節點（斜率 235 ppm、封包間隔 100.64 ms，T0 量到的 Fig.4 值）＋ C3 Gilbert–Elliott 重傳 ＋ E2 共同模式停頓（5–80 ms）。
掃描 {len(rows)} 組（per_good × per_bad × 壞狀態平均長度 × spike_prob × 進入壞狀態機率），每組 2 seeds × 600 s。
目標（Fig.4a 數位化）：σ100 平均 {tgt['sigma100_mean']:.1f} ms，殘差 P50/P90/P99 = {tgt['P50']:.0f}/{tgt['P90']:.0f}/{tgt['P99']:.0f} ms，
> 2 CI 的比例 {tgt['frac_gt_2CI']*100:.1f} %，尖峰（殘差 > 45 ms 的局部最大）{tgt['spikes_per_100s']:.0f} 個/100 s、高度中位數 {tgt['spike_med']:.0f} ms，
δT 以 −30～−50 ms/包遞減（背壓排空）的比例 {tgt['drain_frac']*100:.0f} %。

## 規格格點中最接近的 5 組（loss = 各指標相對誤差平方和）

| per_good | per_bad | 壞狀態長度 (events) | p_g2b | spike_prob | σ100 (ms) | P90 | P99 | >2CI (%) | 尖峰/100 s | 尖峰中位 (ms) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
{lines}

## 精修（最佳點附近隨機搜尋 {len(refined)} 組）最接近的 5 組

| per_good | per_bad | 壞狀態長度 (events) | p_g2b | spike_prob | σ100 (ms) | P90 | P99 | >2CI (%) | 尖峰/100 s | 尖峰中位 (ms) | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
{lines_ref}
| **論文 Fig.4a** | | | | | {tgt['sigma100_mean']:.1f} | {tgt['P90']:.0f} | {tgt['P99']:.0f} | {tgt['frac_gt_2CI']*100:.1f} | {tgt['spikes_per_100s']:.0f} | {tgt['spike_med']:.0f} | |

最佳組合（{'精修' if best is not grid_best else '格點'}）在 3600 s 長跑（去掉前 60 s）：σ100 {full['sigma100_mean']:.1f} ms（P5–P95 {full['sigma100_p5']:.1f}–{full['sigma100_p95']:.1f}），
P90 {full['P90']:.0f} ms、P99 {full['P99']:.0f} ms、>2CI {full['frac_gt_2CI']*100:.1f} %、尖峰 {full['spikes_per_100s']:.0f}/100 s、排空比例 {full['drain_frac']*100:.0f} %。
σ100 落在 30–50 ms 的組合：{len(sig_ok)}/{len(rows)}。

## 各參數的邊際效果（其餘固定在最佳值）

{sens_txt}

## 結論

- **Fig.4 需要「突發」而不是獨立重傳**：要同時得到 σ100 ≈ 40 ms、約 19 % 的點高於 2 CI、和每秒 1 個以上的尖峰，需要壞狀態 PER ≈ {best['per.bad']:.2f}、平均持續 {best['per.mean_bad_events']:.0f} 個 event、進入壞狀態機率 {best['per.p_g2b']:.3f}/event（壞狀態約占 {best['per.p_g2b']/(best['per.p_g2b']+1/best['per.mean_bad_events'])*100:.0f} % 的時間）。
  獨立重傳（per_good 單獨調高）只會讓殘差在 0–60 ms 均勻加厚，做不出 100–200 ms 的尖峰。
- **尖峰後的「−40 ms/包」斜坡**由 C4 流量控制自然產生（壞狀態結束後，佇列以每 2 個 event 一包排空）；這是 PC 端停頓（E2）做不到的——PC 停頓結束時排隊的 callback 會一次處理完，δT 每包掉約 100 ms。
- E2 停頓在 spike_prob ≤ 0.01 時影響很小；Fig.4 的尖峰主要來自連線層。
- 圖：`fig4_compare.png`（左論文、右模擬，同座標）、`sigma_and_residual.png`（σ100 分布、殘差分布對數圖）。
""")
