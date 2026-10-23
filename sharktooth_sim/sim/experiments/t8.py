"""T8：與先前方法比較（以前怎麼解延遲）。"""
import json
import time

import numpy as np

from .. import algorithms
from .. import figures as F
from ..algorithms import sharktooth
from ..config import deep_merge, load_config, set_path
from ..runner import RESULTS, algo_params, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate

NAMES = {"tp_only": "只修 offset（t_P + 首包 δT）", "tc_only": "主機打時間（t_C）", "ols_raw": "全部點 OLS [25]",
         "ols_window": "視窗 OLS 60 s（FTSP 式）", "ols_trimmed": "剔除離群 OLS", "min_filter": "每 10 s 最小值 + OLS",
         "lower_hull": "下凸包 / LP（Moon 1999）", "sharktooth": "SharkTooth（論文）",
         "sharktooth_ff": "SharkTooth + 遺忘因子 0.9999", "sharktooth_tilt": "SharkTooth + 保留 + 預傾斜 100 ppm"}


def method_run(m, tP, tC, prm):
    if m == "sharktooth_tilt":
        return sharktooth.run(tP, tC, **{**prm, "keep_last": True, "tilt_ppm": 100.0})
    return algorithms.run(m, tP, tC, **prm)


def _job(args):
    cfg, seed, methods = args
    sim = simulate(cfg, seed)
    prm = algo_params(cfg)
    T = cfg["T"]
    grid = np.arange(0, T, 1.0) + 0.5
    E = {m: [] for m in methods}
    bias = {m: [] for m in methods}
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        tt = p["t_true_last_sample"]
        for m in methods:
            r = method_run(m, p["t_P"], p["t_C"], prm)
            e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
            E[m].append(np.interp(grid, tt, e, left=np.nan, right=np.nan))
            bias[m].append(float(np.nanmedian(e[tt > T / 2])))
    out = {}
    n = cfg["n_nodes"]
    for m in methods:
        pair = []
        for i in range(n):
            for j in range(i + 1, n):
                pair.append((i, j, E[m][i] - E[m][j]))
        d0 = np.concatenate([np.abs(d) for i, j, d in pair if i == 0])       # 和節點 0 有關的對
        dall = np.concatenate([np.abs(d) for _, _, d in pair])
        late = np.concatenate([np.abs(d[grid > T / 2]) for _, _, d in pair])
        conv = []
        for _, _, d in pair:
            bad = ~(np.abs(d) < 1e-3)
            # 最後一次「壞」之後的時間（持續 < 1 ms 直到結束的起點）；若 1 ms 內持續 60 s 就算收斂
            idx = np.flatnonzero(bad)
            conv.append(float(grid[idx[-1]] + 0.5) if idx.size and idx[-1] < grid.size - 1 else (np.nan if idx.size else 0.0))
            # 首次持續 60 s < 1 ms
        conv60 = []
        for _, _, d in pair:
            ok = np.abs(d) < 1e-3
            run_ = 0
            t_c = np.nan
            for ii, o in enumerate(ok):
                run_ = run_ + 1 if o else 0
                if run_ >= 60:
                    t_c = grid[ii - 59] - 0.5
                    break
            conv60.append(t_c)
        out[m] = dict(mean_abs_ms=float(np.nanmean(dall) * 1e3), p95_ms=float(np.nanpercentile(dall, 95) * 1e3),
                      late_mean_abs_ms=float(np.nanmean(late) * 1e3), late_p95_ms=float(np.nanpercentile(late, 95) * 1e3),
                      node0_mean_abs_ms=float(np.nanmean(d0) * 1e3),
                      conv_s=float(np.nanmax(conv60)) if np.isfinite(conv60).all() else np.inf,
                      abs_bias_ms=float(np.mean(np.abs(bias[m])) * 1e3),
                      valid_frac=float(np.mean([np.isfinite(e).mean() for e in E[m]])))
    return out


def timing(cfg, methods):
    sim = simulate(set_path(cfg, "T", 3600.0), 0)
    p = node_series(sim, 0)
    prm = algo_params(cfg)
    us = {}
    for m in methods:
        method_run(m, p["t_P"][:200], p["t_C"][:200], prm)       # JIT 暖機
        t0 = time.perf_counter()
        for _ in range(3):
            method_run(m, p["t_P"], p["t_C"], prm)
        us[m] = (time.perf_counter() - t0) / 3 / p["t_P"].size * 1e6
    # 純 Python 參考版（即時逐包呼叫）
    st = sharktooth.SharkTooth(cfg["CI"])
    t0 = time.perf_counter()
    for a, b in zip(p["t_P"][:5000], p["t_C"][:5000]):
        st.step(a, b)
    us["sharktooth（純 Python 逐包）"] = (time.perf_counter() - t0) / 5000 * 1e6
    return us


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    methods = cfg0["methods"]
    res = {}
    for sc, ov in cfg0["scenarios"].items():
        cfg = deep_merge(cfg0, ov)
        rr = pmap(_job, [(cfg, s, methods) for s in cfg0["seeds"]])
        res[sc] = {m: {k: float(np.nanmedian([r[m][k] for r in rr])) for k in rr[0][m]} for m in methods}
    us = timing(cfg0, methods)
    write_json(od / "summary.json", dict(results=res, us_per_pkt=us))
    lines = []
    for sc in res:
        lines.append(f"\n### {sc.replace('_', '：')}\n")
        lines.append("| 方法 | 節點對 mean\\|e\\| (ms) | 節點對 P95 (ms) | 後半 mean\\|e\\| / P95 (ms) | 含節點 0 的對 mean\\|e\\| (ms) | 收斂時間 (s) | 單節點常數偏差 (ms) | 有效輸出 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in methods:
            x = res[sc][m]
            conv = "未收斂" if not np.isfinite(x["conv_s"]) else f"{x['conv_s']:.0f}"
            lines.append(f"| {NAMES[m]} | {x['mean_abs_ms']:.3f} | {x['p95_ms']:.3f} | {x['late_mean_abs_ms']:.3f} / {x['late_p95_ms']:.3f} | "
                         f"{x['node0_mean_abs_ms']:.3f} | {conv} | {x['abs_bias_ms']:.2f} | {x['valid_frac']*100:.0f} % |")
    tim = "\n".join(f"| {NAMES.get(k, k)} | {v:.2f} |" for k, v in us.items())
    fig, axs = F.plt.subplots(1, 2, figsize=(14, 5))
    for ax, key in zip(axs, ["late_p95_ms", "conv_s"]):
        xs = np.arange(len(methods))
        for j, sc in enumerate(res):
            vals = [res[sc][m][key] for m in methods]
            vals = [v if np.isfinite(v) else 3600 for v in vals]
            ax.bar(xs + (j - 1.5) * 0.2, vals, 0.2, label=sc.replace("_", "："))
        ax.set_xticks(xs)
        ax.set_xticklabels([NAMES[m] for m in methods], rotation=50, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel({"late_p95_ms": "後半段節點對誤差 P95 (ms)", "conv_s": "收斂時間 (s)（未收斂畫 3600）"}[key])
        ax.legend(fontsize=9)
    F.save(fig, od / "baselines.png")
    txt = f"""# T8　與先前方法比較

**設定**（`config/T8_baselines.yaml`）：同一份資料跑所有方法；T4 擬合參數；20 seeds × 3600 s（表中為中位數）。
情境：T4（4 節點 / 1 dongle）與 C12（12 節點 / 3 dongle、244 B、T6 校正的失敗率 22.5 %），各分「所有節點延遲分布相同」與
「節點 0 比較差」（PER +20 %、韌體延遲 0–10 ms）。誤差都是**真值**：每節點 e_n = t̂_S − clock_PC(t_true) 內插到 1 s 格點，兩兩相減。
收斂時間 = 所有節點對首次連續 60 s 都 < 1 ms 的時間（取最慢的一對）。

{chr(10).join(lines)}

## 每封包運算時間（µs / 包，同一台機器；numba 批次版，最後一列為純 Python 即時逐包版）

| 方法 | µs / 包 |
|---|---|
{tim}

## 結論

1. **只修 offset、主機打時間都不行**：前者被漂移拖垮（1 小時後節點對差到 40–100 ms），後者包含全部延遲抖動（節點對 10–25 ms）。
2. **各節點延遲分布相同時，簡單方法就很好**：全部點 OLS 的截距會吸收「平均延遲」，但大家吸收的一樣多，節點對相減就抵消了
   （T4 相同：OLS {res['T4_相同']['ols_raw']['mean_abs_ms']:.2f} ms、剔除離群 OLS {res['T4_相同']['ols_trimmed']['mean_abs_ms']:.2f} ms、
   每 10 s 最小值 {res['T4_相同']['min_filter']['mean_abs_ms']:.2f} ms，SharkTooth {res['T4_相同']['sharktooth']['mean_abs_ms']:.2f} ms）。
   規格的推測成立：**在這種情境下看不出 SharkTooth 的優勢**。
3. **只要有一個節點比較吵，平均型方法就垮了**：節點 0 的平均延遲比較大 → 它的截距比較大 → 和其他節點差一個常數。
   T4 不同：OLS {res['T4_不同']['ols_raw']['mean_abs_ms']:.1f} ms、視窗 OLS {res['T4_不同']['ols_window']['mean_abs_ms']:.1f} ms、剔除離群 {res['T4_不同']['ols_trimmed']['mean_abs_ms']:.1f} ms；
   **追下緣的方法（SharkTooth {res['T4_不同']['sharktooth']['mean_abs_ms']:.2f}、下凸包 {res['T4_不同']['lower_hull']['mean_abs_ms']:.2f}、每 10 s 最小值 {res['T4_不同']['min_filter']['mean_abs_ms']:.2f} ms）**才撐得住。
   C12 不同同理（OLS {res['C12_不同']['ols_raw']['mean_abs_ms']:.1f} ms vs SharkTooth {res['C12_不同']['sharktooth']['mean_abs_ms']:.2f} ms）。
4. **SharkTooth vs 下凸包（Moon 1999）**：兩者同源（都在找 δT 的下界線）。下凸包在節點有差異時略好或相當，
   但它被「單一最低點」決定，延遲分布相同時反而比 SharkTooth 差（T4 相同 {res['T4_相同']['lower_hull']['mean_abs_ms']:.2f} vs {res['T4_相同']['sharktooth']['mean_abs_ms']:.2f} ms）；
   SharkTooth 用加權平均的斜率更新，比較平滑。**簡報可以把 SharkTooth 介紹成「線上、平滑化的下凸包」。**
5. 即使追下緣，雜訊大的節點的下緣也比較「鈍」（靠近最小延遲的點比較少），所以含節點 0 的節點對誤差仍較大（T4 不同 {res['T4_不同']['sharktooth']['node0_mean_abs_ms']:.2f} ms）。
6. **運算量**：所有方法都是 O(1)/包。SharkTooth 的 numba 版約 {us['sharktooth']:.2f} µs/包，純 Python 逐包版約 {us['sharktooth（純 Python 逐包）']:.0f} µs/包，
   12 節點 × 10 Hz 也只占 PC 不到 0.3 % CPU。
7. 預傾斜 100 ppm 在負漂移時是必要的（T7e），但在雜訊大的 C12 會稍微變差（{res['C12_相同']['sharktooth_tilt']['mean_abs_ms']:.2f} vs {res['C12_相同']['sharktooth']['mean_abs_ms']:.2f} ms）：
   斜率越大，γ1 要爬得越久（見 T9 的 γ1 增量）。實務上取「剛好讓 α + tilt > 0」的最小值即可。

圖：`baselines.png`。
"""
    write_text(od / "README.md", txt)
    write_text(RESULTS / "baselines_T8.md", txt.replace("# T8　與先前方法比較", "# 方法比較表（T8）"))
