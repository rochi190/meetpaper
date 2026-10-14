"""T1：復現 Fig.3（最小延遲條件）。"""
import json

import numpy as np

from .. import digitize as dg
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import ROOT, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate


def metrics(cfg, seed, win=(2888.0, 3017.0)):
    sim = simulate(cfg, seed)
    p = node_series(sim)
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    sig = ev.sigma_M(dt, 100)
    m = (tp >= win[0]) & (tp <= win[1])
    mz = (tp >= 2951.2) & (tp <= 2957.3)
    st = ev.sawtooth_metrics(tp[m], dt[m], cfg["CI"])
    stz = ev.sawtooth_metrics(tp[mz], dt[mz], cfg["CI"])
    return dict(sigma100_mean=float(np.nanmean(sig[m]) * 1e3), sigma100_min=float(np.nanmin(sig[m]) * 1e3),
                sigma100_max=float(np.nanmax(sig[m]) * 1e3), sigma100_all=float(np.nanmean(sig) * 1e3),
                packets_per_cycle=stz["packets_per_cycle"], period_s=stz["period_s"], p2p_ms=st["p2p_ms"],
                p2p_zoom_ms=stz["p2p_ms"], step_down_ms=st["step_down_ms"],
                staircase=ev.staircase_index(tp[m], dt[m]),
                theory_sigma_ms=float(np.sqrt(((cfg["packet"]["period_nom"] * (1 + cfg["clock_adc"]["period_err_ppm"] * 1e-6)) % cfg["CI"]) * 1e3 * (cfg["CI"] * 1e3 - ((cfg["packet"]["period_nom"] * (1 + cfg["clock_adc"]["period_err_ppm"] * 1e-6)) % cfg["CI"]) * 1e3)))), (tp, dt, sig)


def _job(args):
    cfg, seed = args
    return metrics(cfg, seed)[0]


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    od = outdir(cfg0["name"])
    tg = json.loads((ROOT / "data" / "targets.json").read_text())["fig3"]
    key, vals = next(iter(cfg0["sweep"].items()))
    rows = []
    for v in vals:
        cfg = set_path(cfg0, key, v)
        res = pmap(_job, [(cfg, s) for s in cfg0["seeds"]])
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        agg[key] = v
        agg["pass_sigma"] = bool(14.3 <= agg["sigma100_mean"] <= 14.7)
        agg["err_levels_pct"] = abs(agg["packets_per_cycle"] / tg["packets_per_cycle_zoom"] - 1) * 100
        agg["err_p2p_pct"] = abs(agg["p2p_ms"] / tg["p2p_ms"] - 1) * 100
        agg["pass_shape"] = bool(agg["err_levels_pct"] < 20 and agg["err_p2p_pct"] < 20)
        rows.append(agg)
    best = min(rows, key=lambda r: abs(r["sigma100_mean"] - tg["sigma100_ms"]["mean"]))
    cfgb = set_path(cfg0, key, best[key])
    _, (tp, dt, sig) = metrics(cfgb, cfg0["seeds"][0])
    la = dg.load("fig3a", "line")
    # 對齊：讓模擬的下緣在 t_P = 2888 s 時與論文相同（β 是任意常數）
    a_p, b_p = ev.lower_line_lp(la[:, 0], la[:, 1])
    m = (tp > 2888) & (tp < 3017)
    a_s, b_s = ev.lower_line_lp(tp[m], dt[m])
    dt_s = dt + (a_p * 2888 + b_p) - (a_s * 2888 + b_s)
    lc = dg.load("fig3c", "line")
    lb = dg.load("fig3b", "xmark")
    # 放大圖用叉號資料（比折線精確）
    paper_tp = np.r_[la[:, 0], lb[:, 0]]
    o = np.argsort(paper_tp)
    paper = dict(tp=paper_tp[o], dt=np.r_[la[:, 1], lb[:, 1]][o],
                 sig=np.interp(paper_tp[o], lc[:, 0], lc[:, 1]) / 1e3)
    F.paper_vs_sim_threepanel(od / "fig3_compare.png", paper, dict(tp=tp, dt=dt_s, sig=sig), "fig3",
                              f"模擬（E1 中位數 {best[key]*1e3:.1f} ms）")
    # σ100 vs E1 尺度
    fig, ax = F.plt.subplots(figsize=(7, 4.2))
    xs = [r[key] * 1e3 for r in rows]
    ax.plot(xs, [r["sigma100_mean"] for r in rows], "o-", label="模擬 σ100 平均")
    ax.axhspan(14.3, 14.7, color="y", alpha=0.3, label="目標 14.3–14.7 ms")
    ax.axhline(rows[0]["theory_sigma_ms"], color="k", ls="--", lw=1, label=f"只有等待：√(r(CI−r)) = {rows[0]['theory_sigma_ms']:.2f} ms")
    ax.axhline(14.14, color="gray", ls=":", lw=1, label="r = 10 ms（P = 100 ms）時 14.14 ms")
    ax.set_xlabel("E1 OS 延遲中位數 (ms)")
    ax.set_ylabel("σ100 (ms)")
    ax.legend(fontsize=10)
    F.save(fig, od / "sigma100_vs_E1.png")
    write_json(od / "summary.json", dict(target=tg, sweep_key=key, rows=rows, best=best))
    tbl = "\n".join(
        f"| {r[key]*1e3:.1f} | {r['sigma100_mean']:.2f}（{r['sigma100_min']:.2f}–{r['sigma100_max']:.2f}） | "
        f"{r['packets_per_cycle']:.2f} | {r['period_s']:.3f} | {r['p2p_ms']:.1f} | {r['staircase']:.1f} | "
        f"{'✔' if r['pass_sigma'] else '✘'} / {'✔' if r['pass_shape'] else '✘'} |" for r in rows)
    write_text(od / "README.md", f"""# T1　復現 Fig.3（最小延遲）

**設定**（`config/T1_fig3.yaml`）：單節點；A1 固定 ρ（δT 斜率 245 ppm）、封包間隔 100.93 ms（T0 量到的 Fig.3 值）；
C1 等待 event、D3 USB 1 ms、E1 lognormal（掃描中位數）、B1/B2/D1/D2 小延遲；關閉 C3 重傳、E2 尖峰。5 個 seed 平均，看 t_P = 2888–3017 s（與 Fig.3 同範圍）。

| E1 中位數 (ms) | σ100 平均（範圍）ms | 每週期封包數 | 鋸齒週期 (s) | 峰對峰 (ms) | 下緣階梯度 | 通過 σ / 形狀 |
|---|---|---|---|---|---|---|
{tbl}
| **論文 Fig.3** | {tg['sigma100_ms']['mean']:.2f}（{tg['sigma100_ms']['min']:.2f}–{tg['sigma100_ms']['max']:.2f}） | {tg['packets_per_cycle_zoom']:.2f} | {tg['sawtooth_period_s_zoom']:.3f} | {tg['p2p_ms']:.1f} | — | |

只有等待時的理論值：r = P mod CI = {(0.1*1.0093 % 0.03)*1e3:.2f} ms → σ = √(r(CI−r)) = {rows[0]['theory_sigma_ms']:.2f} ms。

## 結論

- **通過**：E1 中位數 {', '.join(f'{r[key]*1e3:.1f}' for r in rows if r['pass_sigma'] and r['pass_shape'])} ms 時 σ100 落在 14.3–14.7 ms，鋸齒層數與峰對峰和 T0 目標差 < 20 %。
- 規格原本寫「鋸齒週期應為 0.3 s、理論 σ = 14.14 ms」，那是假設封包週期剛好 100 ms（r = 10 ms）。T0 量到 Fig.3 的封包間隔是 100.93 ms，
  r = 10.93 ms → 每 CI/r = 2.75 包才 wrap 一次（週期 0.277 s，論文放大圖數位化 0.278 s），理論 σ = 14.44 ms。
  所以 Fig.3c 的 14.3–14.7 ms 中，只有約 1 ms 的連續抖動（B1/B2/D1/D3/E1 合計），**PC 端延遲在最小延遲條件下很小**。
- E1 中位數 ≥ 2 ms 時 σ100 開始明顯高於目標（抖動變異以 2σ² 加到 δ²T 上）。
- 圖：`fig3_compare.png`（左論文、右模擬，同座標）、`sigma100_vs_E1.png`。
""")
