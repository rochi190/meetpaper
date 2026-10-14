"""T2：P/CI 的公因數（相位鎖定）造成的階梯，與連續延遲多大時會被抹平。"""
import math

import numpy as np

from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import node_series, outdir, pmap, run_node_algos, write_json, write_text
from ..simulate import simulate


def _gcd_ms(P, CI):
    a, b = round(P * 1e5), round(CI * 1e5)
    return math.gcd(a, b) / 100.0


def _job(args):
    cfg, seed = args
    sim = simulate(cfg, seed)
    p = node_series(sim)
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    sig = ev.sigma_M(dt, 100)
    st = run_node_algos(p, cfg, ["sharktooth"])["sharktooth"]
    # 真 α：無延遲時的 δT = clock_PC(t_true) − t_P
    a_true = np.polyfit(tp, p["t_pc_true"] - tp, 1)[0]
    v = st["valid"]
    e = st["err"][v]
    late = tp[v] > tp[0] + cfg["T"] / 2
    return dict(stair=ev.lower_edge_roughness(tp, dt, 5.0), sigma100=float(np.nanmean(sig) * 1e3),
                alpha_bias_ppm=float((st["alpha"][-1] - a_true) * 1e6),
                err_late_absdev_ms=float(np.mean(np.abs(e[late] - np.median(e[late]))) * 1e3),
                err_late_p95_ms=float(np.percentile(np.abs(e[late] - np.median(e[late])), 95) * 1e3),
                bias_ms=float(np.median(e[late]) * 1e3))


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    od = outdir(cfg0["name"])
    rows = []
    for case, adc in cfg0["adc_cases"].items():
        for CI in cfg0["CI_values"]:
            for e1 in cfg0["E1_values"]:
                cfg = set_path(set_path(set_path(cfg0, "CI", CI), "os.median_s", e1), "clock_adc.period_err_ppm", adc)
                res = pmap(_job, [(cfg, s) for s in cfg0["seeds"]])
                P = 0.1 * (1 + adc * 1e-6)
                row = dict(case=case, CI_ms=CI * 1e3, E1_ms=e1 * 1e3, gcd_ms=_gcd_ms(0.1, CI),
                           r_ms=(P % CI) * 1e3,
                           theory_sigma_ms=math.sqrt((P % CI) * 1e3 * (CI * 1e3 - (P % CI) * 1e3)))
                for k in res[0]:
                    vals = [r[k] for r in res]
                    row[k] = float(np.mean(vals))
                    row[k + "_sd"] = float(np.std(vals))
                rows.append(row)
    write_json(od / "summary.json", rows)
    # 圖 1：階梯度
    fig, axs = F.plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, case in zip(axs, cfg0["adc_cases"]):
        for e1 in cfg0["E1_values"]:
            rr = [r for r in rows if r["case"] == case and r["E1_ms"] == e1 * 1e3]
            ax.plot([r["CI_ms"] for r in rr], [r["stair"] for r in rr], "o-", label=f"E1 中位數 {e1*1e3:g} ms")
        ax.set_xlabel("CI (ms)（P = 100 ms 名目）")
        ax.set_title({"locked_245ppm": "封包由節點晶振計時（相對漂移 245 ppm）",
                      "paper_100.93ms": "封包間隔 100.93 ms（T0 量到的）"}[case])
        ax.plot(cfg0["CI_values"] and [c * 1e3 for c in cfg0["CI_values"]],
                [_gcd_ms(0.1, c) for c in cfg0["CI_values"]], "k--", lw=1, label="gcd(P, CI)")
        ax.legend(fontsize=10)
    axs[0].set_ylabel("下緣粗糙度 (ms)")
    F.save(fig, od / "staircase_index.png")
    # 圖 2：例子
    fig, axs = F.plt.subplots(2, 3, figsize=(15, 7.5))
    ex = [("locked_245ppm", 0.030, 0.0005), ("locked_245ppm", 0.030, 0.010), ("locked_245ppm", 0.050, 0.0005),
          ("paper_100.93ms", 0.030, 0.0005), ("paper_100.93ms", 0.045, 0.0005), ("paper_100.93ms", 0.050, 0.0005)]
    for ax, (case, CI, e1) in zip(axs.ravel(), ex):
        cfg = set_path(set_path(set_path(cfg0, "CI", CI), "os.median_s", e1), "clock_adc.period_err_ppm",
                       cfg0["adc_cases"][case])
        cfg = set_path(cfg, "T", 300.0)
        p = node_series(simulate(cfg, 0))
        tp, dt = p["t_P"], p["t_C"] - p["t_P"]
        ax.plot(tp, (dt - dt.min()) * 1e3, ".", ms=1.5, color=F.BLUE)
        ax.set_xlim(0, 300)
        ax.set_ylim(-3, CI * 1e3 + 20)
        ax.set_title(f"{'晶振計時' if case.startswith('locked') else '100.93 ms'}，CI {CI*1e3:g} ms，E1 {e1*1e3:g} ms", fontsize=12)
        ax.set_xlabel("t_P (s)")
        ax.set_ylabel("δT − min (ms)")
    F.save(fig, od / "examples.png")
    tbl = "\n".join(
        f"| {r['case']} | {r['CI_ms']:g} | {r['gcd_ms']:g} | {r['r_ms']:.2f} | {r['E1_ms']:g} | {r['stair']:.1f} | "
        f"{r['sigma100']:.2f}（理論 {r['theory_sigma_ms']:.2f}） | {r['alpha_bias_ppm']:+.3f} ± {r['alpha_bias_ppm_sd']:.3f} | "
        f"{r['err_late_absdev_ms']:.3f} | {r['err_late_p95_ms']:.3f} |" for r in rows)
    write_text(od / "README.md", f"""# T2　相位鎖定（gcd）與「階梯 vs 平滑」

**設定**（`config/T2_gcd.yaml`）：T1 的單節點；P 名目 100 ms，掃 CI = 30、31.25、37.5、45、50 ms 與 E1 中位數 0.5、5、10 ms；
兩種封包計時：(1) 封包由節點晶振計時、相對 dongle 漂移 245 ppm（規格與上一版的假設）；(2) T0 量到的 100.93 ms。每格 5 seeds × 1800 s。
SharkTooth 照論文字面；α 偏差 = 最後的 α̂ − 真 α；誤差 = 後半段真值誤差扣掉中位數（常數偏差另計）。

**下緣粗糙度** = 每 5 s 取 δT 最小值形成下緣，扣掉直線後的 P95 − P5（ms）。平滑斜線約 1 ms；相位鎖定的階梯 ≈ gcd(P, CI)。

| 封包計時 | CI (ms) | gcd(P, CI) | r = P mod CI | E1 (ms) | 下緣粗糙度 (ms) | σ100 (ms) | α 偏差 (ppm) | 後半 mean\\|e\\| (ms) | 後半 P95\\|e\\| (ms) |
|---|---|---|---|---|---|---|---|---|---|
{tbl}

## 結論

1. **階梯從哪來**：封包週期 P 與 CI 有公因數 g 時，等待時間只會落在 g 的倍數相位上；相位只靠兩個時鐘的相對漂移慢慢移動
   （245 ppm 時每 g/245 ppm 秒移動一格：g = 10 ms → 每 41 s 跳一階）。CI = 50 ms 時 g = 50 ms，等待時間幾乎固定 → δT 是每 204 s 跳 50 ms 的階梯。
   這就是上一版模擬「每 50 s 跳 10 ms」的來源。
2. **連續延遲抹不平「下緣」的階梯**：延遲都是正的（lognormal 下界是 0），加大 E1 只會把帶子往上加厚；每 5 s 的最小值仍然落在最低那一層，
   所以下緣粗糙度幾乎不隨 E1 下降（見表與 `staircase_index.png`）。而且 E1 = 5–10 ms 時 σ100 已遠超 Fig.3 的 14.5 ms
   ——**論文 Fig.3 的 σ100 也不允許這麼大的連續延遲**。「靠大抖動抹平」不是論文圖平滑的原因。
3. **論文的真正原因是封包間隔 100.93 ms**（T0）：r = 10.93 ms 讓相位每 3 包偏 2.8 ms、約 3 s 掃過整個 CI，**任何 CI 下都沒有階梯**，
   而且 σ100 只比 √(r(CI−r)) 多一點。
4. **SharkTooth 在階梯下仍大致無偏**：α 偏差在 ±0.5 ppm 內（見表）；但階梯時下緣「點」稀疏（只有每階起點貼著真正的最小延遲），
   δT* 會在階與階之間抬高，後半段誤差的峰對峰較大（CI = 50 ms 鎖定時最明顯）。
""")
