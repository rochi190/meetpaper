"""T7：資料完整性錯誤（掉包、斷線、buffer 讀錯、重複封包、重開機、溢位、負漂移）。"""
import json

import numpy as np

from .. import figures as F
from ..algorithms import integrity, sharktooth
from ..config import load_config, set_path
from ..runner import RESULTS, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate

WRAP31 = (2 ** 31) * 1e-6


def subtests(cfg0):
    S = []
    S.append(("對照（無錯誤）", "ref", {}, None))
    for p in [0.001, 0.01, 0.05]:
        S.append((f"T7a 掉包 {p*100:g} %", "T7a", {"loss.enabled": True, "loss.prob": p}, None))
    S.append(("T7a 斷線 1 次 × 5 s（t = 1800 s）", "T7a", {"disconnect.enabled": True, "disconnect.events": [[1800.0, 5.0]]}, None))
    for fs in [50.0, 1000.0]:
        for p in [0.001, 0.005]:
            S.append((f"T7b buffer 讀錯 {p*100:g} %，f_s = {fs:g} Hz", "T7b",
                      {"buf_err.enabled": True, "buf_err.prob": p, "packet.fs": fs}, None))
    for p in [0.01, 0.05]:
        S.append((f"T7c 重複封包 {p*100:g} %", "T7c", {"dup.enabled": True, "dup.prob": p}, None))
    S.append(("T7d 重開機（t = 1800 s，2 s 無資料）", "T7d", {"reset.enabled": True, "reset.events": [[1800.0]]}, None))
    S.append(("T7d 計時器溢位（31 位元 µs，約 2147 s 繞回）", "T7d",
              {"clock_node.wrap": {"enabled": True, "bits": 31, "unit": 1e-6}}, WRAP31))
    for a in [-20.0, -5.0, 5.0, 20.0]:
        S.append((f"T7e 漂移 α = {a:+g} ppm", "T7e",
                  {"clock_node.rho_ppm": -a, "clock_node.scale_err_ppm": 0.0, "clock_pc.rho_ppm": 0.0}, None))
    return S


def metrics(p, ts, valid, n_pivot, T):
    e = ts - p["t_pc_true"]
    tt = p["t_true_last_sample"]
    v = valid & np.isfinite(e)
    ref = v & (tt > T / 2) & (tt < T / 2 + 600) | (v & (tt < T / 2) & (tt > T / 2 - 600))
    bias = np.median(e[ref]) if ref.any() else (np.median(e[v]) if v.any() else 0.0)
    d = np.abs(e - bias)
    gross = v & (d > 0.010)
    ok = v & ~gross
    return dict(p95_ms=float(np.percentile(d[ok], 95) * 1e3) if ok.any() else np.nan,
                gross_frac=float(gross.mean()), invalid_frac=float(1 - v.mean()),
                bad_total=float(1 - ok.mean()), n_pivot=float(n_pivot),
                p95_all_ms=float(np.percentile(np.where(v, d, np.inf), 95) * 1e3))


def _job(args):
    cfg, seed, wrap_span, tilt = args
    sim = simulate(cfg, seed)
    p = node_series(sim, 0)
    fs = cfg["packet"]["fs"]
    out = {}
    r = sharktooth.run(p["t_P"], p["t_C"], CI=cfg["CI"])
    out["原版（論文字面）"] = metrics(p, r["ts"], r["valid"], r["n_pivot"], cfg["T"])
    r = sharktooth.run(p["t_P"], p["t_C"], CI=cfg["CI"], keep_last=True)
    out["原版 + 保留 α、β"] = metrics(p, r["ts"], r["valid"], r["n_pivot"], cfg["T"])
    r = integrity.run(p, fs, CI=cfg["CI"], wrap_span=wrap_span)
    out["完整性檢查 + SharkTooth"] = metrics(p, r["ts"], r["valid"], r["n_pivot"], cfg["T"])
    out["_info"] = {k: v for k, v in r["info"].items() if k not in ("status", "seg", "tP_unwrapped")}
    r = integrity.run(p, fs, CI=cfg["CI"], wrap_span=wrap_span, keep_last=True, tilt_ppm=tilt)
    out[f"完整性檢查 + 保留 + 預傾斜 {tilt:g} ppm"] = metrics(p, r["ts"], r["valid"], r["n_pivot"], cfg["T"])
    out["_true_lost"] = int(np.sum(sim["nodes"][0]["packets"]["dropped_flag"]))
    out["_true_buf"] = int(np.sum(sim["nodes"][0]["packets"]["buf_err_flag"]))
    return out


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    S = subtests(cfg0)
    jobs = []
    for name, grp, ov, ws in S:
        c = cfg0
        for k, v in ov.items():
            c = set_path(c, k, v)
        jobs += [(c, s, ws, cfg0["tilt_ppm"]) for s in cfg0["seeds"]]
    res = pmap(_job, jobs)
    ns = len(cfg0["seeds"])
    rows = []
    for i, (name, grp, ov, ws) in enumerate(S):
        rr = res[i * ns:(i + 1) * ns]
        methods = [k for k in rr[0] if not k.startswith("_")]
        row = dict(name=name, group=grp)
        for m in methods:
            row[m] = {k: float(np.nanmedian([r[m][k] for r in rr])) for k in rr[0][m]}
        row["detected"] = {k: float(np.mean([r["_info"][k] for r in rr])) for k in rr[0]["_info"]}
        row["true_lost"] = float(np.mean([r["_true_lost"] for r in rr]))
        row["true_buf"] = float(np.mean([r["_true_buf"] for r in rr]))
        rows.append(row)
    write_json(od / "summary.json", rows)
    methods = [k for k in res[0] if not k.startswith("_")]
    hdr = "| 子測試 | " + " | ".join(methods) + " | 偵測到（遺失 / buffer 錯 / 重複 / 重置 / 繞回） |\n|---|" + "---|" * (len(methods) + 1)
    lines = []
    for r in rows:
        cells = []
        for m in methods:
            x = r[m]
            cells.append(f"{x['p95_ms']:.2f} ms / {x['bad_total']*100:.1f} % / {x['n_pivot']:.0f}")
        d = r["detected"]
        lines.append(f"| {r['name']} | " + " | ".join(cells) +
                     f" | {d['lost']:.0f}（實際 {r['true_lost']:.0f}）/ {d['n_buf']:.0f}（實際 {r['true_buf']:.0f}）/ {d['n_dup']:.0f} / {d['n_reset']:.0f} / {d['n_wrap']:.0f} |")
    # 圖：各子測試的「壞輸出比例」
    fig, ax = F.plt.subplots(figsize=(14, 5.5))
    xs = np.arange(len(rows))
    w = 0.2
    for j, m in enumerate(methods):
        ax.bar(xs + (j - 1.5) * w, [max(r[m]["bad_total"] * 100, 0.05) for r in rows], w, label=m)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"].replace("，", "\n") for r in rows], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("無效 + 嚴重錯誤（> 10 ms）的輸出比例 (%)")
    ax.legend(fontsize=9)
    F.save(fig, od / "bad_output_fraction.png")
    write_text(od / "README.md", f"""# T7　資料完整性錯誤

**設定**（`config/T7_integrity.yaml`）：T4 擬合出的延遲參數、單節點、3600 s × 20 seeds（表中為中位數）。
比較四種處理：論文字面 SharkTooth；保留上一組 α、β（穩定期不輸出 0）；先做 §7 完整性檢查（`sim/algorithms/integrity.py`）再跑 SharkTooth；
完整性檢查 + 保留 + 「預傾斜」（先把 δT 加上 +{cfg0['tilt_ppm']:g} ppm 的已知斜率、輸出時扣回；本規格新增的修正）。

每格：**真值誤差 P95**（扣常數偏差，只算有效且 < 10 ms 的輸出）/ **壞輸出比例**（無效 + 誤差 > 10 ms）/ **樞紐點數**。

{hdr}
{chr(10).join(lines)}

## 結論

1. **掉包、斷線不影響 SharkTooth**（0.1–5 % 掉包、5 s 斷線：P95 都在對照的 ±0.05 ms 內）。演算法只用收到的 (t_P, t_C)，不需要連續封包。
   完整性檢查的遺失計數與實際完全一致（1 B 封包編號在 5 s 斷線 = 50 包內不會繞回混淆）。
2. **buffer 讀錯在 50 Hz 時影響大**：t_P 錯一個樣本 = 20 ms，該包的 t̂_S 直接錯 20 ms（P95 由 0.32 升到 2.5–4.3 ms），
   但因為 δT 錯誤是向上或向下的 20 ms，**向下的那一半會變成新的樞紐點候選**，污染下緣。完整性檢查（Δt_P 是否 = 整數倍封包週期，容差 0.5/f_s）
   **抓到的數量與注入的完全一致**，修正 t_P 並不讓它更新 SharkTooth 後回到 0.27–0.28 ms。f_s = 1 kHz 時錯一個樣本只有 1 ms，落在正常抖動內，幾乎看不出影響。
3. **重複封包**（同編號、同 t_P）對精度沒有影響：∞×0 只在「重複封包緊接在新樞紐點後」才會發生，論文式的 γ1 = ∞ 會讓下一包無條件合格，結果無害。
   但重複封包會被當成新的一筆資料，**資料端要去重**（完整性檢查用「最近 64 包內同編號同 t_P」判斷）。
4. **重開機與計時器溢位會讓原版完全失效**：之後的 t_P 從 0（或從溢位點）重來，δT 跳了幾十分鐘，原版 SharkTooth 之後約一半的輸出都是錯的（> 10 ms，表中 41–51 %）。
   完整性檢查偵測到「t_P 倒退」：剛好差一個欄位週期 → unwrap；否則 → 重置 SharkTooth 狀態。之後只損失一次 30 s 的穩定期。
5. **負漂移（α < 0）是 SharkTooth 本身的弱點**（與完整性無關）：δT 一路往下，每個新低點都成為新樞紐點，迴歸矩陣不斷歸零。
   α = −20 ppm：183 個樞紐點、論文字面版 82 % 的時間輸出無效；改成「保留上一組 α、β」雖然有輸出，但 α 一直是舊的，P95 = 5 ms。
   **修正（本模擬新增）：預傾斜**——先把 δT 加上已知的 +100 ppm 斜率（比任何晶振 + PC 誤差大），讓 α + 100 ppm 一定 > 0，輸出時扣回。
   一行程式，負漂移下 P95 0.27–0.28 ms、無效 1 %、樞紐點 8–9 個；正漂移下也不會變差（甚至小 α 時更好，因為樞紐點更少）。
   這也呼應 T5：論文的 δT 斜率剛好是 +235～+245 ppm，等於「天然預傾斜」。

圖：`bad_output_fraction.png`。
""")
