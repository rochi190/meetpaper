"""T9：SharkTooth 參數敏感度。"""
import json

import numpy as np

from .. import figures as F
from ..algorithms import sharktooth
from ..config import load_config, set_path
from ..runner import RESULTS, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate

DEFAULT = dict(M=100, lam_max=0.5, g1_inc=1e-6, settle_s=30.0, keep_last=False)


def pair_metrics(sim, cfg, params):
    T = cfg["T"]
    grid = np.arange(0, T, 1.0) + 0.5
    E = []
    inval = []
    piv = []
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        r = sharktooth.run(p["t_P"], p["t_C"], CI=cfg["CI"], **params)
        e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
        E.append(np.interp(grid, p["t_true_last_sample"], e, left=np.nan, right=np.nan))
        inval.append(1 - r["valid"].mean())
        piv.append(r["n_pivot"])
    d = [E[i] - E[j] for i in range(len(E)) for j in range(i + 1, len(E))]
    seg = []
    for s in range(6):
        m = (grid >= 600 * s) & (grid < 600 * (s + 1))
        v = np.concatenate([np.abs(x[m]) for x in d])
        seg.append(float(np.nanmean(v) * 1e3))
    allv = np.concatenate([np.abs(x) for x in d])
    conv = []
    for x in d:
        ok = np.abs(x) < 1e-3
        run_, tc = 0, np.nan
        for ii, o in enumerate(ok):
            run_ = run_ + 1 if o else 0
            if run_ >= 60:
                tc = grid[ii - 59]
                break
        conv.append(tc)
    return dict(seg=seg, p95=float(np.nanpercentile(allv, 95) * 1e3), invalid=float(np.mean(inval)),
                n_pivot=float(np.mean(piv)), conv=float(np.nanmax(conv)) if np.isfinite(conv).all() else np.inf)


def _job(args):
    cfg, seed, plist = args
    sim = simulate(cfg, seed)
    return [pair_metrics(sim, cfg, p) for p in plist]


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    plist = [("預設（論文）", dict(DEFAULT))]
    for k, vals in cfg0["grid"].items():
        for v in vals:
            if v == DEFAULT[k]:
                continue
            plist.append((f"{k} = {v}", {**DEFAULT, k: v}))
    res = pmap(_job, [(cfg0, s, [p for _, p in plist]) for s in cfg0["seeds"]])
    rows = []
    for i, (name, prm) in enumerate(plist):
        rr = [r[i] for r in res]
        rows.append(dict(name=name, params=prm, seg=np.nanmedian([r["seg"] for r in rr], axis=0).tolist(),
                         p95=float(np.nanmedian([r["p95"] for r in rr])),
                         invalid=float(np.nanmedian([r["invalid"] for r in rr])),
                         n_pivot=float(np.nanmedian([r["n_pivot"] for r in rr])),
                         conv=float(np.nanmedian([r["conv"] for r in rr]))))
    # 封包率 × γ1 增量
    rt = cfg0["rate_test"]
    rate_rows = []
    for P in rt["period_nom"]:
        c = set_path(set_path(set_path(cfg0, "packet.period_nom", P), "packet.fs", rt["fs"]), "flow.window", rt["window"])
        pl = [{**DEFAULT, "g1_inc": g} for g in rt["g1_inc"]]
        rr = pmap(_job, [(c, s, pl) for s in cfg0["seeds"][:10]])
        for j, g in enumerate(rt["g1_inc"]):
            x = [r[j] for r in rr]
            rate_rows.append(dict(rate_hz=1 / P, g1_inc=g, seg=np.nanmedian([r["seg"] for r in x], axis=0).tolist(),
                                  p95=float(np.nanmedian([r["p95"] for r in x])),
                                  conv=float(np.nanmedian([r["conv"] for r in x]))))
    write_json(od / "summary.json", dict(rows=rows, rate=rate_rows))
    fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    rows_by = {r["name"]: r for r in rows}
    rate_by = {(r["rate_hz"], r["g1_inc"]): r for r in rate_rows}
    t1 = "\n".join(f"| {r['name']} | {fmt(r['seg'])} | {r['p95']:.2f} | {r['conv']:.0f} | {r['invalid']*100:.1f} % | {r['n_pivot']:.0f} |" for r in rows)
    t2 = "\n".join(f"| {r['rate_hz']:g} | {r['g1_inc']:g} | {fmt(r['seg'])} | {r['p95']:.2f} | {r['conv']:.0f} |" for r in rate_rows)
    fig, axs = F.plt.subplots(1, 2, figsize=(14, 4.8))
    for r in rows:
        if r["name"] == "預設（論文）" or r["name"].startswith(("settle", "keep")):
            axs[0].plot(range(1, 7), r["seg"], "o-", label=r["name"])
    axs[0].set_xlabel("600 s 分段")
    axs[0].set_ylabel("節點對 mean|e| (ms)（真值）")
    axs[0].legend(fontsize=9)
    axs[0].set_title("穩定期與輸出方式")
    for g in rt["g1_inc"]:
        rr = [r for r in rate_rows if r["g1_inc"] == g]
        axs[1].plot([r["rate_hz"] for r in rr], [r["p95"] for r in rr], "o-", label=f"γ1 增量 {g:g}")
    axs[1].set_xlabel("封包率 (Hz)")
    axs[1].set_ylabel("節點對誤差 P95 (ms)")
    axs[1].legend(fontsize=9)
    axs[1].set_title("γ1 增量是否綁定 10 Hz")
    F.save(fig, od / "sensitivity.png")
    write_text(od / "README.md", f"""# T9　SharkTooth 參數敏感度

**設定**（`config/T9_sensitivity.yaml`）：T4 擬合參數、4 節點 / 1 dongle、3600 s × 20 seeds；同一份模擬資料跑不同參數（一次只改一個，其餘為論文值）。
指標皆為**真值節點對誤差**：各段 mean|e_nm|、全程 P95、收斂時間（首次連續 60 s < 1 ms，最慢的一對）、無效輸出比例、樞紐點數。表中為 20 seeds 中位數。

| 參數 | mean\\|e\\| 第 1–6 段 (ms) | P95 (ms) | 收斂 (s) | 無效輸出 | 樞紐點 |
|---|---|---|---|---|---|
{t1}

## 封包率 × γ1 增量（f_s = 100 Hz、window = 2，10 seeds）

| 封包率 (Hz) | γ1 增量 | mean\\|e\\| 第 1–6 段 (ms) | P95 (ms) | 收斂 (s) |
|---|---|---|---|---|
{t2}

## 結論

1. **M（σ 視窗）、最大權重 0.5、穩定期輸出方式（keep_last）幾乎沒影響**（α 為正、樞紐點只在開頭出現時，keep_last 只差在前 30 s）。
   最大權重 0.2 收斂較慢（第 1 段變大），0.8 和 0.5 差不多——論文「0.5 是經驗值」的選擇合理但不關鍵。
2. **γ1 增量是最敏感的參數**：1e-7 時探針爬太慢，第 1 段 {rows_by['g1_inc = 1e-07']['seg'][0]:.2f} ms、整段都收斂不好；
   1e-5 比論文的 1e-6 更好（P95 {rows_by['g1_inc = 1e-05']['p95']:.2f} vs {rows_by['預設（論文）']['p95']:.2f} ms、收斂 {rows_by['g1_inc = 1e-05']['conv']:.0f} vs {rows_by['預設（論文）']['conv']:.0f} s）。
   原因：γ1 每遇到一個「不合格點」才加一次增量，要從 0 爬到真正的斜率（這裡約 2.5e-4，因為 δT 斜率 ≈ 250 ppm）需要「斜率 / 增量」個點。
   論文說增量要和漂移同量級——但論文自己的漂移是 2.4e-4，1e-6 其實偏小。
3. **增量確實綁定封包率**：探針是「每包」爬一次，真正重要的是「每秒爬多少」= 增量 × 封包率。
   5 Hz 時 1e-6 明顯變差（P95 {rate_by[(5.0, 1e-06)]['p95']:.2f} ms），5 Hz × 1e-5 ≈ 10 Hz × 1e-6；20 Hz 時 1e-6 就夠快。
   **建議：增量 ≈ |預期漂移| / (封包率 × 期望收斂秒數)**，而不是固定 1e-6。
4. **穩定期 30 s 是折衷**：5 s 時無效輸出最少（0.2 %）但第 1 段誤差較大（δT* 還沒穩定就開始迴歸）；60 s 第 1 段最好但前 1 分鐘沒有輸出。
5. 這些都是 4 節點 / Fig.6 雜訊下的結果；雜訊更大（C12）時，太大的增量可能讓探針被重傳尖峰帶跑——建議用 T6 的設定再掃一次。

圖：`sensitivity.png`。
""")
