"""T6：多節點、Table II / Fig.10 / Fig.11 / Fig.12。

一次模擬（時間戳）→ 三種量測模型評估同一份 SharkTooth 輸出：
  V1 只有時間（訊號無雜訊、無 AFE 差；CC 仍有 0.1 ms 量化與 50 Hz 線性內插誤差）
  V2 + ADC 雜訊（以 A2 穩態 std 0.25 ms 校正）
  V3 + 每節點 AFE 群延遲差（F3）+ 不同 dongle 的固定路徑差（D）
每種都同時算「論文式 RSE（互相關）」與「真值 RSE」。
"""
import csv
import json

import numpy as np

from .. import algorithms
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import ROOT, RESULTS, algo_params, node_series, outdir, pmap, sample_timestamps, write_json, write_text
from ..simulate import simulate


def cfg_for(cfg0, n, L, p):
    c = set_path(set_path(set_path(cfg0, "n_nodes", n), "packet.bytes", L), "per.good", p)
    return c


def sigma2_segments(sim, cfg):
    out = np.zeros(6)
    cnt = np.zeros(6)
    for k in range(len(sim["nodes"])):
        p = node_series(sim, k)
        s2 = (ev.sigma_M(p["t_C"] - p["t_P"], 100) * 1e3) ** 2
        seg = np.clip((p["t_true_last_sample"] // 600).astype(int), 0, 5)
        for j in range(6):
            m = (seg == j) & np.isfinite(s2)
            if m.any():
                out[j] += s2[m].mean()
                cnt[j] += 1
    return out / np.maximum(cnt, 1)


def _calib_job(args):
    cfg, seed = args
    sim = simulate(set_path(cfg, "T", 1200.0), seed)
    return float(np.mean(sigma2_segments(sim, set_path(cfg, "T", 1200.0))[:2]))


def signal_variant(S, meta, cfg, rng, noise_v, afe):
    s = cfg["signal"]
    v = s["amp"] * np.sin(2 * np.pi * s["f"] * (S["t_true"] - afe))
    if noise_v > 0:
        v = v + rng.normal(0, noise_v, v.size)
    return v


def _main_job(args):
    cfg, seed, variants = args
    sim = simulate(cfg, seed)
    prm = algo_params(cfg)
    base_nodes = []
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        out = algorithms.run("sharktooth", p["t_P"], p["t_C"], **prm)
        st = sample_timestamps(sim["nodes"][k], p, out)
        # sample_timestamps 只回傳已收到的樣本；需要它們的真實取樣時間來重算訊號
        S = sim["nodes"][k]["samples"]
        idx = np.searchsorted(S["t_pc_true"], st["sample_true_pc"])
        st["t_true"] = S["t_true"][np.clip(idx, 0, S["t_true"].size - 1)]
        base_nodes.append(st)
    rng = np.random.default_rng(seed + 99991)
    afe_n = rng.normal(0, 1, cfg["n_nodes"])
    res = {}
    for name, v in variants.items():
        nodes = []
        r2 = np.random.default_rng(seed * 7 + 3)
        nv = float(np.sqrt(v["noise_v"] ** 2 + (v.get("noise_len_v", 0.0) * cfg["packet"]["bytes"] / 244.0) ** 2))
        for k, st in enumerate(base_nodes):
            d = afe_n[k] * v["afe_sd"]
            val = signal_variant(dict(t_true=st["t_true"]), None, cfg, r2, nv, d)
            nodes.append(dict(sample_tp=st["sample_tp"], sample_val=val, sample_ts=st["sample_ts"],
                              sample_true_pc=st["sample_true_pc"] + (d if v.get("afe_in_truth") else 0.0)))
        r = ev.paper_rse(nodes, T=cfg["T"], seg_s=600.0)
        rec = dict(cc=r["seg"], true=r["seg_true"])
        if cfg["n_nodes"] == 4 and name in ("V3", "V4", "FIG10"):
            # Fig.10：各對 RSE 的 10 s 平均
            rec["pairs"] = {f"{i+1}-{j+1}": np.nanmean(np.pad(r["rse_cc"][(i, j)], (0, (-len(r["rse_cc"][(i, j)])) % 10),
                                                              constant_values=np.nan).reshape(-1, 10), axis=1).tolist()
                            for i, j in r["pairs"]}
        res[name] = rec
    res["sigma2"] = sigma2_segments(sim, cfg).tolist()
    return res


def paper_table():
    rows = list(csv.DictReader(open(ROOT / "data" / "paper_table2.csv")))
    T = {}
    for r in rows:
        T[(r["metric"], r["config"])] = [float(r[f"seg{k}"]) for k in range(1, 7)]
    return T


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            if not k.startswith("per."):
                cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    from ..digitize import digitize_fig12
    f12 = digitize_fig12()
    s2_target = {}
    for x in f12:
        s2_target.setdefault(x["config"], {})[x["seg"]] = x
    confs = [(n, L) for L in cfg0["bytes_list"] for n in cfg0["nodes_list"]]
    letter = {17: "A", 127: "B", 244: "C"}
    # ---------- 1. 用 Fig.12 的 σ²100 校正每個配置的失敗率 ----------
    grid = cfg0["calib_p_grid"]
    jobs = [(cfg_for(cfg0, n, L, p), s) for (n, L) in confs for p in grid for s in (0, 1)]
    cres = pmap(_calib_job, jobs)
    calib = {}
    i = 0
    for (n, L) in confs:
        s2 = []
        for p in grid:
            s2.append(np.mean(cres[i:i + 2]))
            i += 2
        name = f"{letter[L]}{n}"
        tgt = float(np.mean([s2_target[name][k]["sigma2_100"] for k in range(2, 7)]))
        s2a = np.maximum.accumulate(np.array(s2))
        p_cal = float(np.interp(tgt, s2a, grid)) if tgt > s2a[0] else 0.0
        calib[name] = dict(target_sigma2=tgt, p=p_cal, curve=dict(zip(map(str, grid), s2)))
    write_json(od / "calibration.json", calib)
    # ---------- 2. 校正 ADC 雜訊：A2 第 2–6 段 std = 0.25 ms ----------
    nv_grid = cfg0["noise_v_grid"]
    var_nv = {f"nv{v}": dict(noise_v=v, afe_sd=0.0) for v in nv_grid}
    a2 = cfg_for(cfg0, 2, 17, calib["A2"]["p"])
    r_nv = pmap(_main_job, [(a2, s, var_nv) for s in range(8)])
    std_nv = [float(np.median([np.mean([seg["std"] for seg in r[f"nv{v}"]["cc"][1:]]) for r in r_nv])) for v in nv_grid]
    noise_v = float(np.interp(0.25, std_nv, nv_grid))
    write_json(od / "noise_calibration.json", dict(noise_v_grid=nv_grid, A2_steady_std_ms=std_nv, chosen_noise_v=noise_v))
    variants = {"V1": dict(noise_v=0.0, afe_sd=0.0),
                "V2": dict(noise_v=noise_v, afe_sd=0.0),
                "V3": dict(noise_v=noise_v, afe_sd=cfg0["afe_delay_sd_s"])}
    # ---------- 3. 主實驗 ----------
    jobs = []
    for (n, L) in confs:
        c = cfg_for(cfg0, n, L, calib[f"{letter[L]}{n}"]["p"])
        c = set_path(c, "dongle.base_sd_s", 0.0)
        jobs += [(c, s, variants) for s in cfg0["seeds"]]
    # V3 的 dongle 路徑差需要另一次模擬（影響時間戳）
    res = pmap(_main_job, jobs)
    grid4 = {"V3": variants["V3"]}
    for kv in cfg0["v4_noise_len_v"]:
        for afe in cfg0["v4_afe_sd_s"]:
            grid4[f"G_{kv}_{afe}"] = dict(noise_v=noise_v, afe_sd=afe, noise_len_v=kv)
    jobs3 = []
    for (n, L) in confs:
        c = cfg_for(cfg0, n, L, calib[f"{letter[L]}{n}"]["p"])
        c = set_path(c, "dongle.base_sd_s", cfg0["dongle_path_sd_s"])
        jobs3 += [(c, s, grid4) for s in cfg0["seeds"]]
    res3 = pmap(_main_job, jobs3)
    # 選 V4：第 2–6 段（穩態）與 Table II 的 RMS 差最小的格點
    P = paper_table()
    ns_ = len(cfg0["seeds"])
    score = {}
    for g in grid4:
        e = []
        for ci, (n, L) in enumerate(confs):
            nm = f"{letter[L]}{n}"
            rr = res3[ci * ns_:(ci + 1) * ns_]
            for met in ["absavg", "std"]:
                arr = np.nanmedian(np.array([[sg[met] for sg in r[g]["cc"]] for r in rr], float), axis=0)
                e += list(arr[1:] - np.array(P[(met, nm)][1:]))
        score[g] = float(np.sqrt(np.nanmean(np.array(e) ** 2)))
    g_best = min((g for g in grid4 if g != "V3"), key=lambda g: score[g])
    variants["V4"] = grid4[g_best]
    for r in res3:
        r["V4"] = r[g_best]
    write_json(od / "v4_grid.json", dict(score=score, best=g_best, grid=grid4))
    ns = len(cfg0["seeds"])
    table = {}
    for ci, (n, L) in enumerate(confs):
        name = f"{letter[L]}{n}"
        rr = res[ci * ns:(ci + 1) * ns]
        r3 = res3[ci * ns:(ci + 1) * ns]
        t = dict(sigma2=np.median([r["sigma2"] for r in rr], axis=0).tolist())
        for v in ["V1", "V2"]:
            for kind in ["cc", "true"]:
                for met in ["absavg", "std", "p95"]:
                    arr = np.array([[s[met] for s in r[v][kind]] for r in rr], float)
                    t[f"{v}_{kind}_{met}"] = np.nanmedian(arr, axis=0).tolist()
                    t[f"{v}_{kind}_{met}_iqr"] = (np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)).tolist()
        for v in ["V3", "V4"]:
            for kind in ["cc", "true"]:
                for met in ["absavg", "std", "p95"]:
                    arr = np.array([[s[met] for s in r[v][kind]] for r in r3], float)
                    t[f"{v}_{kind}_{met}"] = np.nanmedian(arr, axis=0).tolist()
                    t[f"{v}_{kind}_{met}_iqr"] = (np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)).tolist()
        if n == 4:
            t["fig10_pairs_seed0"] = r3[0]["V4"].get("pairs")
        table[name] = t
    write_json(od / "summary.json", dict(calibration=calib, noise_v=noise_v, variants=variants, table=table))
    cfg0 = dict(cfg0, _v4=variants["V4"])
    report(od, table, calib, noise_v, cfg0, f12)


def report_only(cfg_path):
    """從 summary.json 重畫圖與 README（不重跑模擬）。"""
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            if not k.startswith("per."):
                cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    S = json.loads((od / "summary.json").read_text())
    from ..digitize import digitize_fig12
    cfg0 = dict(cfg0, _v4=S["variants"]["V4"])
    report(od, S["table"], S["calibration"], S["noise_v"], cfg0, digitize_fig12())


def report(od, table, calib, noise_v, cfg0, f12):
    P = paper_table()
    names = [f"{a}{n}" for a in "ABC" for n in cfg0["nodes_list"]]

    def errs(v, kind):
        e = []
        for nm in names:
            for met in ["absavg", "std"]:
                e += list(np.array(table[nm][f"{v}_{kind}_{met}"]) - np.array(P[(met, nm)]))
        e = np.array(e)
        return float(np.sqrt(np.nanmean(e ** 2))), float(np.nanmean(e))
    # Table II 格式
    def tbl(v, kind, met):
        lines = []
        for nm in names:
            sim_ = table[nm][f"{v}_{kind}_{met}"]
            pap = P[(met, nm)]
            lines.append(f"| {nm} | " + " | ".join(f"{s:.2f} ({p:.2f})" for s, p in zip(sim_, pap)) + " |")
        return "| 配置 | 第 1 段 | 2 | 3 | 4 | 5 | 6 |\n|---|---|---|---|---|---|---|\n" + "\n".join(lines)
    # ---------- 圖：Fig.11 式 ----------
    fig, axs = F.plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    cols = {2: (0.0, 0.447, 0.741), 4: (0.851, 0.325, 0.098), 6: (0.929, 0.694, 0.125), 8: (0.494, 0.184, 0.557),
            10: (0.467, 0.675, 0.188), 12: (0.302, 0.745, 0.933)}
    mk = {"A": "o", "B": "D", "C": "^"}
    for j, met in enumerate(["absavg", "std"]):
        for nm in names:
            n = int(nm[1:])
            axs[j, 0].plot(range(1, 7), P[(met, nm)], marker=mk[nm[0]], color=cols[n], lw=1, ms=5)
            axs[j, 1].plot(range(1, 7), table[nm][f"V4_cc_{met}"], marker=mk[nm[0]], color=cols[n], lw=1, ms=5)
            axs[j, 2].plot(range(1, 7), table[nm][f"V1_true_{met}"], marker=mk[nm[0]], color=cols[n], lw=1, ms=5)
        for a in axs[j]:
            a.set_ylim(0, 1.7 if met == "absavg" else 1.7)
        axs[j, 0].set_ylabel({"absavg": "mean|RSE| (ms)", "std": "std RSE (ms)"}[met])
    axs[0, 0].set_title("論文 Table II / Fig.11")
    axs[0, 1].set_title("模擬 V4（論文式 RSE：互相關）")
    axs[0, 2].set_title("模擬 V1（真值 RSE：只有同步誤差）")
    for a in axs[1]:
        a.set_xlabel("600 s 分段")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cols[n], marker="o", ls="-", label=f"{n} 節點") for n in cols] + \
        [Line2D([], [], color="k", marker=mk[k], ls="", label={"A": "A 17 B", "B": "B 127 B", "C": "C 244 B"}[k]) for k in mk]
    axs[0, 2].legend(handles=h, fontsize=9, ncol=2)
    F.save(fig, od / "fig11_compare.png")
    # ---------- 圖：Fig.12 式 ----------
    fig, axs = F.plt.subplots(2, 3, figsize=(15, 8))
    for seg in range(6):
        ax = axs.ravel()[seg]
        xs_p = [x["sigma2_100"] for x in f12 if x["seg"] == seg + 1]
        ys_p = [x["rse_p95"] for x in f12 if x["seg"] == seg + 1]
        ax.plot(xs_p, ys_p, "k+", ms=8, label="論文")
        xs = [table[nm]["sigma2"][seg] for nm in names]
        ys = [table[nm]["V4_cc_p95"][seg] for nm in names]
        ax.plot(xs, ys, "o", color=F.BLUE, ms=5, label="模擬 V4")
        ax.axvline(225, color="r", ls="--", lw=1)
        ax.set_title(f"第 {seg+1} 段")
        ax.set_xlabel("σ²100 (ms²)")
        ax.set_ylabel("RSE P95 (ms)")
        ax.set_ylim(0, 4)
        ax.legend(fontsize=9)
    F.save(fig, od / "fig12_compare.png")
    # ---------- 圖：Fig.10 式 ----------
    import glob
    from ..digitize import OUT as DIG
    fig, axs = F.plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for fn, c in zip(sorted(glob.glob(str(DIG / "fig10_poly_*.csv"))), F.PAIR_COLORS):
        p = np.loadtxt(fn, delimiter=",", skiprows=1)
        o = np.argsort(p[:, 0])
        axs[0].plot(p[o, 0], p[o, 1], ",", color=c, alpha=0.5)
    pairs = table["C4"].get("fig10_pairs_seed0")
    if not pairs:
        # 用 V4 量測模型重跑 C4 的 seed 0（只為了畫 Fig.10）
        c = cfg_for(cfg0, 4, 244, calib["C4"]["p"])
        c = set_path(c, "dongle.base_sd_s", cfg0["dongle_path_sd_s"])
        pairs = _main_job((c, 0, {"FIG10": cfg0["_v4"]}))["FIG10"]["pairs"]
    for (k, v), c in zip(pairs.items(), F.PAIR_COLORS):
        v = np.array(v, float) * 1e3
        axs[1].plot(np.arange(len(v)) * 10 + 5, v, "-", color=c, lw=0.8, label=f"Apx {k}")
    for ax in axs:
        ax.axhspan(-1, 1, color="y", alpha=0.25)
        for x in range(600, 3601, 600):
            ax.axvline(x, color="b", ls="--", lw=0.7)
        ax.set_xlim(0, 4000)
        ax.set_ylim(-4, 2)
        ax.set_xlabel("Output timestamp (s)")
    axs[0].set_ylabel("RSE (ms)")
    axs[0].set_title("論文 Fig.10（數位化）")
    axs[1].set_title("模擬 C4（V4，10 s 平均）")
    axs[1].legend(fontsize=9, ncol=2)
    F.save(fig, od / "fig10_compare.png")
    rm = {f"{v}_{k}": errs(v, k) for v in ["V1", "V2", "V3", "V4"] for k in ["cc", "true"]}
    cal_tbl = "\n".join(f"| {nm} | {calib[nm]['target_sigma2']:.0f} | {calib[nm]['p']*100:.1f} % | {table[nm]['sigma2'][1]:.0f} |" for nm in names)
    v4 = cfg0["_v4"]

    def steady(v, kind, met, grp):
        return float(np.mean([np.mean(table[nm][f"{v}_{kind}_{met}"][1:]) for nm in names if nm[0] == grp]))

    def steady_p(met, grp):
        return float(np.mean([np.mean(P[(met, nm)][1:]) for nm in names if nm[0] == grp]))

    def seg1(v, kind, met, grp):
        return float(np.mean([table[nm][f"{v}_{kind}_{met}"][0] for nm in names if nm[0] == grp]))
    c12 = table["C12"]
    write_text(od / "README.md", f"""# T6　多節點與 Table II / Fig.10 / Fig.11 / Fig.12

**設定**（`config/T6_tableII.yaml`）：T4 擬合出的 PC / dongle 參數；節點數 2–12 × 封包 17 / 127 / 244 B（18 個配置，配置名同論文），
每 4 個節點一支 dongle；每配置 20 seeds × 3600 s。SharkTooth 照論文字面（穩定期 α = β = 0，該段不計）。
表中數字都是 20 seeds 的中位數（論文也是 20 次重複的中位數）。

### 1. 連線雜訊：用 Fig.12 校正，不用 Table II

Fig.12 是論文唯一「逐配置」量到的雜訊指標（σ²100）。每個配置找每個 event 的失敗率 p，使模擬 σ²100 = Fig.12 第 2–6 段平均。

| 配置 | Fig.12 σ²100 (ms²) | 校正出的 p | 模擬 σ²100 第 2 段 |
|---|---|---|---|
{cal_tbl}

2 節點不論封包長度都在 ≈ 209 ms²（= 純等待的理論值 r(CI−r)）→ **2 節點幾乎沒有重傳**；4 節點起隨封包長度上升
（同一支 dongle 4 × 7.5 ms = 30 ms 剛好塞滿 CI → C5 排程衝突）；> 4 節點再加上多個 piconet 互相干擾（C6）。

### 2. 量測模型（同一份 SharkTooth 輸出，四種評估）

| 模型 | 內容 |
|---|---|
| V1 | 只有時間誤差。訊號無雜訊；論文式 RSE 仍有 0.1 ms 量化與 50 Hz 線性內插誤差 |
| V2 | + ADC 雜訊 {noise_v*1e3:.1f} mV（振幅 500 mV；**用 A2 第 2–6 段 std = 0.25 ms 校正**） |
| V3 | V2 + 每節點 AFE 群延遲差 σ = {cfg0['afe_delay_sd_s']*1e3:.2f} ms（30 Hz 一階高通、元件 ±5 % 在 5 Hz 的相位延遲差）+ 不同 dongle 路徑差 \\|N(0, {cfg0['dongle_path_sd_s']*1e3:.1f} ms)\\| |
| V4（假說） | V2 + 隨封包長度增加的雜訊（244 B 時 +{v4['noise_len_v']*1e3:.0f} mV，∝ 封包長度；例如發射脈衝耦合進高增益 EMG 前端）+ AFE 差 σ = {v4['afe_sd']*1e3:.2f} ms；在第 2–6 段做 4×4 格點搜尋（`v4_grid.json`） |

### 3. 與 Table II 的整體差距（18 配置 × 6 段 × {{mean|RSE|, std}}，ms）

| 模型 | 論文式（互相關）RMS 差 / 平均差 | 真值版 RMS 差 / 平均差 |
|---|---|---|
| V1 只有時間 | {rm['V1_cc'][0]:.3f} / {rm['V1_cc'][1]:+.3f} | {rm['V1_true'][0]:.3f} / {rm['V1_true'][1]:+.3f} |
| V2 + ADC 雜訊 | {rm['V2_cc'][0]:.3f} / {rm['V2_cc'][1]:+.3f} | {rm['V2_true'][0]:.3f} / {rm['V2_true'][1]:+.3f} |
| V3 + AFE / 路徑偏差 | {rm['V3_cc'][0]:.3f} / {rm['V3_cc'][1]:+.3f} | {rm['V3_true'][0]:.3f} / {rm['V3_true'][1]:+.3f} |
| V4 + 隨長度增加的雜訊 | {rm['V4_cc'][0]:.3f} / {rm['V4_cc'][1]:+.3f} | {rm['V4_true'][0]:.3f} / {rm['V4_true'][1]:+.3f} |

### 4. Table II 格式：mean|RSE|（括號內為論文），V4 論文式

{tbl('V4', 'cc', 'absavg')}

### 5. Table II 格式：RSE 標準差（括號內為論文），V4 論文式

{tbl('V4', 'cc', 'std')}

### 6. 只看同步本身：V1 真值 mean|RSE|（括號內為論文）

{tbl('V1', 'true', 'absavg')}

## 結論：論文的 0.2–0.6 ms 從哪來？

1. **只看同步本身（V1 真值：沒有任何量測誤差、沒有節點固定偏差）**：第 2–6 段（最差一對）
   2 節點 A / B / C = {np.mean(table['A2']['V1_true_absavg'][1:]):.2f} / {np.mean(table['B2']['V1_true_absavg'][1:]):.2f} / {np.mean(table['C2']['V1_true_absavg'][1:]):.2f} ms，
   12 節點 = {np.mean(table['A12']['V1_true_absavg'][1:]):.2f} / {np.mean(table['B12']['V1_true_absavg'][1:]):.2f} / {np.mean(table['C12']['V1_true_absavg'][1:]):.2f} ms；
   各組平均 {steady('V1','true','absavg','A'):.2f} / {steady('V1','true','absavg','B'):.2f} / {steady('V1','true','absavg','C'):.2f} ms，約是論文（{steady_p('absavg','A'):.2f} / {steady_p('absavg','B'):.2f} / {steady_p('absavg','C'):.2f} ms）的 {100*np.mean([steady('V1','true','absavg',g)/steady_p('absavg',g) for g in 'ABC']):.0f} %。
   同步誤差的標準差只有 0.02–0.06 ms：SharkTooth 輸出是平滑直線，誤差主要是「偏一個常數」，節點越多、雜訊越大，最差一對的偏移越大。
2. **Table II 的穩態「標準差」幾乎全是量測雜訊**：A2 的 0.25 ms 用 {noise_v*1e3:.0f} mV 的 ADC 雜訊（2.4 % 振幅）就完全解釋；
   std 不隨節點數變、只隨封包長度變（A 0.25、B 0.29、C 0.37–0.41）——**時間誤差做不出這個形狀**（V1–V3 所有配置都停在 0.25 ms），
   需要一個隨封包長度增加的量測誤差（V4：244 B 時 +{v4['noise_len_v']*1e3:.0f} mV）。這是【推測】：BLE 發射脈衝耦合進 ×200–1600 增益的 EMG 前端，或 ISR 時間戳被長封包的 HCI 傳輸延後；模擬無法區分。
3. **mean|RSE| 隨節點數上升，主要來自「最差那一對」的固定偏差**：Eq.27 每段挑 P95 最大的一對，節點越多、挑到的偏移越大。
   需要每個節點有約 {v4['afe_sd']*1e3:.1f} ms 的固定偏移（AFE 元件誤差的量級剛好吻合；Fig.10 收斂後各對 ±0.3 ms 的固定偏移也是它）。
   真正的同步誤差也隨節點數 / 封包長度增加（雜訊多 → δT* 不穩），但只占一部分。
4. **第 1 段大、之後收斂 = SharkTooth 的起始收斂**：V1 真值第 1 段 A / B / C = {seg1('V1','true','absavg','A'):.2f} / {seg1('V1','true','absavg','B'):.2f} / {seg1('V1','true','absavg','C'):.2f} ms，
   std 第 1 段 0.4–1.4 ms（論文 0.33–1.57 ms），雜訊越大越慢收斂。模擬的 A 組第 1 段比論文大（A2 {table['A2']['V4_cc_absavg'][0]:.2f} vs 0.30 ms）——
   【推測】論文的 20 次重複可能在 notification 啟動後才開始計時，或起始雜訊比 Fig.12 平均值低。
5. **C12 的第 1–2 段**模擬偏低（{c12['V4_cc_absavg'][0]:.2f} / {c12['V4_cc_absavg'][1]:.2f} vs 1.27 / 1.16 ms）：論文 C12 第 2 段 IQR 高達 0.79 ms，代表某些重複的雜訊更大或有額外事件（斷線、重連），
   用一個固定的失敗率無法重現。
6. **對「論文說 0.21–0.62 ms」的解讀**：這個數字是「同步誤差 + 量測誤差 + 節點固定偏差」的總和，而且挑最差的一對。
   模擬中 SharkTooth 自己的穩態同步誤差在 2 節點時 < 0.1 ms、12 節點 / 244 B 時約 0.3–0.5 ms（V4 真值第 6 段 C12 = {c12['V4_true_absavg'][5]:.2f} ms，含 dongle 路徑差）。
7. 2 節點的 C2 模擬 std 比論文大（{table['C2']['V4_cc_std'][5]:.2f} vs 0.29）：論文的額外雜訊不只看封包長度，也和同一支 dongle 上的連線數有關（C2 0.30、C4 0.37）。

圖：`fig11_compare.png`（論文 / 模擬 V4 / 模擬 V1 真值）、`fig12_compare.png`（RSE P95 vs σ²100，論文 + 模擬 V4）、`fig10_compare.png`（4 節點各對 RSE，V4）。
""")
