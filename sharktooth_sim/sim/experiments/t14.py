"""T14：實測（3 塊 STM32 藍牙板）和哪一組模擬比較像？

同一套特徵算在實測與三組模擬上：
  paper    = 和論文相符的設定（T4），只改成 3 節點；
  physical = 同上，但時鐘只有物理誤差（T10 / T11 的 exact + qpc）；
  hw       = 照實測條件（原始 tick、QPC、封包間隔 98.9–99.2 ms、20 B、無起始等待、獨立重傳）。
特徵：δT 斜率、封包間隔、對下緣的延遲分布、σ100、起始尖峰、下緣粗糙度、Table II 式 RSE（Eq.23–27）。
"""
import csv
import json

import numpy as np

from .. import algorithms
from .. import evaluate as ev
from .. import figures as F
from ..config import deep_merge, load_config, set_path
from ..runner import ROOT, RESULTS, algo_params, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate
from .t6 import _main_job, paper_table

SCEN_COLOR = {"paper": "#2a78d6", "physical": "#eb6834", "hw": "#1baf7a", "hw_lean": "#eda100"}   # 參考色盤 1–4 格
INK, INK2 = "#0b0b0b", "#52514e"
FEATS = ["slope_ppm", "interval_ms", "P10", "P50", "P90", "P99", "gt_CI", "gt_2CI", "sigma100", "startup_ms", "rough"]


def features(tp, tc):
    """一個節點的 δT 特徵（實測與模擬共用）。"""
    dt = tc - tp
    line = ev.lower_line_lp(tp, dt)
    bs, _ = ev.band_stats(tp, dt, line)
    d = np.diff(tp)
    su = ev.startup_spike(tp, dt, line, n_first=3)
    return dict(slope_ppm=bs["slope_ppm"], interval_ms=float(np.median(d[d < 0.15]) * 1e3),
                P10=bs["P10"], P50=bs["P50"], P90=bs["P90"], P99=bs["P99"],
                gt_CI=bs["frac_gt_CI"] * 100, gt_2CI=bs["frac_gt_2CI"] * 100,
                sigma100=float(np.nanmean(ev.sigma_M(dt, 100)) * 1e3),
                startup_ms=su["max_excess_ms"], rough=ev.lower_edge_roughness(tp, dt))


# --------------------------------------------------------------------------- 實測
def load_hw(folder, lse_hz):
    """Hands_on/st_receiver.py 的 packets.csv → 每個節點的 t_P、t_C、樣本。"""
    rows = list(csv.DictReader(open(folder / "packets.csv", encoding="utf-8")))
    out = {}
    for name in sorted({r["node"] for r in rows}):
        R = [r for r in rows if r["node"] == name]
        tick = np.array([int(r["tP_ticks"]) for r in R], float)
        out[name] = dict(tp=tick / lse_hz, tc=np.array([float(r["t_C"]) for r in R]),
                         fl=np.array([int(r["fl_ticks"]) for r in R], float) / lse_hz,
                         n=np.array([int(r["n"]) for r in R]),
                         s=np.array([[int(r[f"s{k}"]) for k in range(5)] for r in R], float),
                         pkt_id=np.array([int(r["pkt_id"]) for r in R]))
    return out


def hw_rse(nodes, prm, T):
    """實測的 Table II 式 RSE：SharkTooth → 每個樣本的 t̂_S → evaluate.paper_rse（和模擬同一個函式）。"""
    rn = []
    for d in nodes.values():
        r = algorithms.run("sharktooth", d["tp"], d["tc"], **prm)
        n = d["n"][:, None]
        k = np.arange(5)[None, :]
        tps = d["tp"][:, None] - (n - 1 - k) * d["fl"][:, None] / np.maximum(n - 1, 1)
        ts = tps + (r["alpha"][:, None] * tps + r["beta"][:, None])
        ts = np.where(r["valid"][:, None], ts, np.nan)
        rn.append(dict(sample_tp=tps.ravel(), sample_val=d["s"].ravel(), sample_ts=ts.ravel()))
    return ev.paper_rse(rn, T=T, seg_s=600.0)["seg"]


# --------------------------------------------------------------------------- 模擬
def _job(args):
    cfg, seed = args
    sim = simulate(cfg, seed)
    feats = []
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        feats.append(features(p["t_P"], p["t_C"]))
    # 量測模型 V1：只有時間（實測沒有 AFE，訊號直接進 ADC）
    rse = _main_job((cfg, seed, {"V1": dict(noise_v=0.0, afe_sd=0.0)}))["V1"]
    return dict(feats=feats, cc=rse["cc"], true=rse["true"])


def scen_cfg(cfg0, sc):
    """情境可用 base: 繼承另一個情境。"""
    if "base" in sc:
        sc = deep_merge(cfg0["scenarios"][sc["base"]], {k: v for k, v in sc.items() if k != "base"})
    return deep_merge(cfg0, {k: v for k, v in sc.items() if k != "label"})


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
    for k, v in bp.items():
        cfg0 = set_path(cfg0, k, v)
    cfg0 = deep_merge(cfg0, dict(sharktooth=load_config(cfg_path)["sharktooth"]))
    od = outdir(cfg0["name"])
    prm = algo_params(cfg0)

    # 實測
    hw = {}
    for run_id in cfg0["hardware_runs"]:
        nodes = load_hw(ROOT / "data" / run_id, cfg0["hardware_lse_hz"])
        dur = min(d["tp"][-1] for d in nodes.values())
        rec = dict(duration_s=float(dur),
                   feats={n: features(d["tp"], d["tc"]) for n, d in nodes.items()},
                   lost={n: int(np.sum((np.diff(d["pkt_id"]) % 256) - 1)) for n, d in nodes.items()})
        if dur >= 3000:
            rec["rse"] = hw_rse(nodes, prm, cfg0["T"])
        hw[run_id] = rec

    # 模擬
    scen = cfg0["scenarios"]
    jobs = [(scen_cfg(cfg0, scen[s]), seed) for s in scen for seed in cfg0["seeds"]]
    res = pmap(_job, jobs)
    ns = len(cfg0["seeds"])
    sim = {s: res[i * ns:(i + 1) * ns] for i, s in enumerate(scen)}
    write_json(od / "summary.json", dict(hardware=hw, sim=sim, labels={s: scen[s]["label"] for s in scen}))
    report_only(cfg_path)


# --------------------------------------------------------------------------- 匯總、圖、README
def summarize(S):
    hw = S["hardware"]
    long_run = max(hw, key=lambda r: hw[r]["duration_s"])
    hwf = hw[long_run]["feats"]
    out = dict(long_run=long_run, hw_feats={f: [hwf[n][f] for n in sorted(hwf)] for f in FEATS},
               hw_rse=hw[long_run].get("rse"), sim={})
    for s, rr in S["sim"].items():
        F_ = {f: np.array([nd[f] for r in rr for nd in r["feats"]]) for f in FEATS}
        seg = {m: np.array([[x[m] for x in r["cc"]] for r in rr]) for m in ("absavg", "std", "p95")}
        tru = {m: np.array([[x[m] for x in r["true"]] for r in rr]) for m in ("absavg", "std")}
        out["sim"][s] = dict(
            feat_med={f: float(np.nanmedian(v)) for f, v in F_.items()},
            feat_lo={f: float(np.nanpercentile(v, 5)) for f, v in F_.items()},
            feat_hi={f: float(np.nanpercentile(v, 95)) for f, v in F_.items()},
            neg_frac=float(np.mean(F_["slope_ppm"] < 0)),
            slope_min=float(F_["slope_ppm"].min()), slope_max=float(F_["slope_ppm"].max()),
            seg_med={m: np.nanmedian(v, axis=0).tolist() for m, v in seg.items()},
            seg_lo={m: np.nanpercentile(v, 5, axis=0).tolist() for m, v in seg.items()},
            seg_hi={m: np.nanpercentile(v, 95, axis=0).tolist() for m, v in seg.items()},
            true_med={m: np.nanmedian(v, axis=0).tolist() for m, v in tru.items()})
    # 每個特徵：實測（3 節點的中位數）落在哪一組模擬的 5–95 % 範圍內
    hm = {f: float(np.median(v)) for f, v in out["hw_feats"].items()}
    out["hw_med"] = hm
    # 斜率用模擬的全部範圍：三塊板子共用同一台 PC，不是三個獨立抽樣
    out["inside"] = {s: {f: bool((d["slope_min"] if f == "slope_ppm" else d["feat_lo"][f]) <= hm[f]
                                 <= (d["slope_max"] if f == "slope_ppm" else d["feat_hi"][f])) for f in FEATS}
                     for s, d in out["sim"].items()}
    return out


def figures(S, sm, od):
    import matplotlib.pyplot as plt
    labels = S["labels"]
    # 1. 延遲分布：對下緣的殘差直方圖（實測 vs 三組模擬的第 0 個種子）
    from ..simulate import simulate as _sim
    cfg0 = _cfg_cache["cfg0"]
    fig, axes = plt.subplots(1, 1 + len(labels), figsize=(3.9 * (1 + len(labels)), 3.8), sharey=True)
    bins = np.arange(0, 100.1, 2.5)
    nodes = load_hw(ROOT / "data" / sm["long_run"], cfg0["hardware_lse_hz"])
    r_hw = []
    for d in nodes.values():
        _, r = ev.band_stats(d["tp"], d["tc"] - d["tp"])
        r_hw.append(r)
    panels = [("實測（3 塊板子，60 分鐘）", np.concatenate(r_hw), INK)]
    for s in labels:
        c = scen_cfg(cfg0, cfg0["scenarios"][s])
        sim = _sim(c, 0)
        rr = []
        for k in range(c["n_nodes"]):
            p = node_series(sim, k)
            _, r = ev.band_stats(p["t_P"], p["t_C"] - p["t_P"])
            rr.append(r)
        panels.append((f"模擬：{labels[s]}", np.concatenate(rr), SCEN_COLOR[s]))
    for ax, (t, r, col) in zip(axes, panels):
        h, _ = np.histogram(r, bins=bins)
        ax.bar(bins[:-1] + 1.25, h / h.sum() * 100, width=2.3, color=col)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 20)
        ax.set_xlim(0, 100)
        ax.axvline(30, color=INK2, lw=0.8, ls="--")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("延遲（對下緣，ms）")
        ax.text(0.97, 0.95, f"> 1 CI：{(r > 32.5).mean() * 100:.1f} %\n> 2 CI：{(r > 62.5).mean() * 100:.2f} %",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK2)
    axes[0].set_ylabel("比例（%，對數）")
    fig.suptitle("延遲分布：實測的重傳比論文少很多（虛線 = 1 個 CI）", fontsize=13)
    fig.tight_layout()
    fig.savefig(od / "delay_hist.png")
    plt.close(fig)

    # 2. Table II 式 RSE：各段 mean|RSE| 與 std
    if sm["hw_rse"]:
        P = paper_table()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(1, 7)
        for ax, m, nm in zip(axes, ("absavg", "std"), ("mean|RSE|", "RSE 標準差")):
            for cfgname, ls in (("A2", "-"), ("A4", "--")):
                ax.plot(x, P[(m, cfgname)], color=INK2, ls=ls, lw=1.5, marker="o", ms=4,
                        label=f"論文 {cfgname}（17 B）")
            for s in labels:
                d = sm["sim"][s]
                ax.fill_between(x, d["seg_lo"][m], d["seg_hi"][m], color=SCEN_COLOR[s], alpha=0.12, lw=0)
                ax.plot(x, d["seg_med"][m], color=SCEN_COLOR[s], lw=2, marker="o", ms=4, label=f"模擬：{labels[s]}")
            ax.plot(x, [r[m] for r in sm["hw_rse"]], color=INK, lw=2.5, marker="s", ms=7, label="實測")
            ax.set_xticks(x)
            ax.set_xlabel("段（每段 10 分鐘）")
            ax.set_ylabel(f"{nm}（ms）")
            ax.set_ylim(0, None)
        axes[0].legend(fontsize=9, loc="upper right")
        fig.suptitle("Table II 式 RSE（3 節點、最差的一對；模擬帶狀 = 20 次的 5–95 %）", fontsize=13)
        fig.tight_layout()
        fig.savefig(od / "rse_segments.png")
        plt.close(fig)


_cfg_cache = {}


def report_only(cfg_path):
    cfg0 = load_config(cfg_path)
    bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
    for k, v in bp.items():
        cfg0 = set_path(cfg0, k, v)
    cfg0 = deep_merge(cfg0, dict(sharktooth=load_config(cfg_path)["sharktooth"]))
    _cfg_cache["cfg0"] = cfg0
    od = outdir(cfg0["name"])
    S = json.loads((od / "summary.json").read_text(encoding="utf-8"))
    sm = summarize(S)
    write_json(od / "summary_table.json", sm)
    figures(S, sm, od)
    write_readme(S, sm, od)


def _f(v, nd=1):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_readme(S, sm, od):
    labels = S["labels"]
    names = {"slope_ppm": ("δT 斜率（ppm）", 0), "interval_ms": ("封包間隔（ms）", 2),
             "P10": ("延遲 P10（ms）", 1), "P50": ("延遲 P50（ms）", 1), "P90": ("延遲 P90（ms）", 1),
             "P99": ("延遲 P99（ms）", 1), "gt_CI": ("> 1 CI（%）", 2), "gt_2CI": ("> 2 CI（%）", 3),
             "sigma100": ("σ100（ms）", 2), "startup_ms": ("起始尖峰（前 3 包最大，ms）", 0),
             "rough": ("下緣粗糙度（ms）", 2)}
    hwv = sm["hw_feats"]
    head = "| 特徵 | 實測（A / B / C） | " + " | ".join(f"模擬：{labels[s]}" for s in labels) + " |"
    sep = "|---|---|" + "---|" * len(labels)
    rows = []
    for f in FEATS:
        nm, nd = names[f]
        cells = []
        for s in labels:
            d = sm["sim"][s]
            mark = "✔ " if sm["inside"][s][f] else ""
            cells.append(f"{mark}{_f(d['feat_med'][f], nd)}（{_f(d['feat_lo'][f], nd)} ～ {_f(d['feat_hi'][f], nd)}）")
        rows.append(f"| {nm} | {' / '.join(_f(v, nd) for v in hwv[f])} | " + " | ".join(cells) + " |")
    n_in = {s: sum(sm["inside"][s].values()) for s in labels}
    rse_rows = ""
    if sm["hw_rse"]:
        P = paper_table()

        def line(nm, m, vals):
            return f"| {nm} | {m} | " + " | ".join(_f(v, 2) for v in vals) + " |"
        rl = []
        for m, mn in (("absavg", "mean|RSE|"), ("std", "std")):
            rl.append(line("**實測**（20 B、3 節點、1 次）", mn, [r[m] for r in sm["hw_rse"]]))
            for s in labels:
                rl.append(line(f"模擬：{labels[s]}", mn, sm["sim"][s]["seg_med"][m]))
            rl.append(line("論文 A2（17 B、2 節點）", mn, P[(m, "A2")]))
            rl.append(line("論文 A4（17 B、4 節點）", mn, P[(m, "A4")]))
        rse_rows = "\n".join(rl)
    hl = sm["sim"]["hw_lean"]
    ph = sm["sim"]["physical"]
    rse = sm["hw_rse"]
    rse_in = {s: all(sm["sim"][s]["seg_lo"][m][k] <= rse[k][m] <= sm["sim"][s]["seg_hi"][m][k]
                     for m in ("absavg", "std") for k in range(1, 6)) for s in labels}
    txt = f"""# T14　實測（3 塊 STM32 藍牙板）和哪一組模擬比較像？

**實測**：`Hands_on/`（3 塊 B-L475E-IOT01A、PC 內建藍牙 + bleak、Windows QueryPerformanceCounter、20 B、CI 30 ms）。
原始資料複製到 `data/hardware/`：{", ".join(f"`{r.split('/')[-1]}`（{S['hardware'][r]['duration_s'] / 60:.0f} 分鐘）" for r in S['hardware'])}。
兩次都沒有掉包、沒有斷線。下表用 60 分鐘那次。

**模擬**（`config/T14_hardware.yaml`，各 20 次 × 3600 s × 3 節點；SharkTooth 都用「保留 α、β + 預傾斜 100 ppm」，和 `st_analyze.py` 的預設一樣）：

| 模擬 | 設定 |
|---|---|
| {labels['paper']} | T4 原樣：A8 −248 ppm、PC 理想、封包間隔 100.78 ms、重傳 7.9 % + 突發、起始等待 102 ms、244 B |
| {labels['physical']} | 同上，但時鐘只有物理誤差：A8 = 0、PC ±50 ppm（= T10 的 physical、T11 的 exact + qpc） |
| {labels['hw']} | 只填實測的已知條件：A8 = 0（傳原始 tick）、PC ±50 ppm、封包間隔 98.85–99.18 ms、20 B、無起始等待；獨立重傳 1.2 %（用實測反推）。PC 端延遲仍是 T4 擬合的（dongle、序列化、OS） |
| {labels['hw_lean']} | 再拿掉實測沒有的層：nRF dongle 與 Blatann 序列化、GIL；OS 延遲 0.1 ms、韌體打包抖動（B2）關掉；重傳 1.4 %。**這幾項是看實測的延遲分布調的** |

## 1. δT 與延遲的特徵

模擬欄是 60 個節點的中位數（5–95 %）。✔ = 實測（3 節點的中位數）落在這組模擬的 5–95 % 內；
斜率用模擬的全部範圍（三塊板子共用同一台 PC，不是三個獨立抽樣）。

{head}
{sep}
{chr(10).join(rows)}

落在範圍內的特徵數（共 {len(FEATS)} 項）：{"、".join(f"{labels[s]} {n_in[s]}" for s in labels)}。
物理時鐘的模擬中，δT 斜率範圍 {ph['slope_min']:.0f}～{ph['slope_max']:.0f} ppm，{ph['neg_frac'] * 100:.0f} % 的節點是負的；和論文相符的設定做不出負斜率。

![延遲分布](delay_hist.png)

## 2. Table II 式 RSE（Eq.23–27，3 節點中最差的一對，ms）

實測與模擬用同一個函式（`evaluate.paper_rse`，互相關解析度 0.1 ms）。模擬的量測模型是 V1（只有時間誤差，沒有 ADC 雜訊與 AFE 差），因為實測的訊號直接進 ADC、沒有 AFE。
模擬是 20 次的中位數；實測只有 1 次。

| | | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
{rse_rows}

![RSE](rse_segments.png)

## 結論

1. **現有的模擬沒有一組和實測一樣**：和論文相符（T4）、物理時鐘（T10）都是 0–1 / {len(FEATS)} 項。
   只填已知條件（{labels['hw']}）是 {n_in['hw']} 項；再拿掉實測沒有的 dongle 序列化層、縮小 PC 端與韌體抖動（{labels['hw_lean']}）是 {n_in['hw_lean']} 項。
   不需要新的變因，只是現有變因換成實測的值。
2. **時鐘像 T10 / T11 的「物理時鐘」，不像論文**：
   - 實測 δT 斜率 −34～−47 ppm（三塊都往下），論文 +235～+245 ppm。實測傳原始 tick、在 PC 上用 32768 精確換算，所以沒有 A8 換算誤差。
     這支持 T10 / T11：論文的 +240 ppm 不是晶振和 PC 時鐘本身做得出來的。
   - 三塊之間只差 13 ppm，共同部分約 −42 ppm。一般 32.768 kHz 晶振只有 ±20 ppm，所以共同部分至少一半來自 PC 時鐘（QPC）。
   - 斜率往下時論文字面版的 SharkTooth 會失效（T7、T10 的預測）；實測要加預傾斜 100 ppm 才正常輸出，和預測一致。
3. **延遲像論文 Fig.3（最安靜的那張），不像 Fig.6**：
   - 帶寬 = 1 個 CI（30 ms），σ100 約 14.7 ms，和 Fig.3（14.2–14.8 ms）一樣。
   - 重傳約 1.4 %（論文 Fig.6 的設定是 7.9 % 再加突發），而且是獨立的：> 2 CI 只有 0.03 %，獨立重傳 1.4 % 的模擬就是這個值，不需要 T3 / T4 的突發干擾。
   - **沒有起始尖峰和 −40 ms/包 的斜坡**（論文 Fig.6 有，T4 用「第一包等 102 ms + 韌體流量控制」解釋）。所以 C11 起始等待是論文的韌體或連線流程特有的，不是 BLE 一定會有。
   - 等待之外的延遲很小：P10 3.5 ms，純等待（0–30 ms 均勻）是 3.0 ms，只多約 0.5 ms；下緣粗糙度 1.3 ms（論文 Fig.6 約 2.1 ms）。
     T4 的 PC 端延遲（dongle、序列化、OS 合計約 1–2 ms）和 B2 韌體打包抖動（0.2–2 ms，依論文的 FreeRTOS 假設）都比實測大。
     實測用 PC 內建藍牙，沒有 nRF dongle 和 Blatann 的序列化層。
   - 封包間隔 98.9–99.2 ms（比 100 ms 短），論文 100.6–100.9 ms（比 100 ms 長）。兩者都是取樣計時器不是晶振造成的（T0），只是方向不同。
4. **同步誤差**：
   - 實測第 2–6 段 mean|RSE| {min(r['absavg'] for r in rse[1:]):.2f}–{max(r['absavg'] for r in rse[1:]):.2f} ms、std {min(r['std'] for r in rse[1:]):.2f}–{max(r['std'] for r in rse[1:]):.2f} ms，
     落在 20 次 5–95 % 內的模擬：{"、".join(labels[s] for s in rse_in if rse_in[s]) or "無"}{"；" + "、".join(labels[s] for s in rse_in if not rse_in[s]) + " 有幾段實測低於下界" if not all(rse_in.values()) else ""}。
     各組的範圍重疊很多，1 次錄音分不出是哪一組。
   - 比論文 A2 / A4（0.21–0.31 ms、std 約 0.25 ms）小。T6 的拆解：論文穩態的 std 幾乎全是量測雜訊（AFE、EMG 前端），實測訊號直接進 ADC，沒有這部分。
   - 實測第 1 段 {rse[0]['absavg']:.2f} ms，比四組模擬的中位數（{min(sm['sim'][s]['seg_med']['absavg'][0] for s in labels):.2f}–{max(sm['sim'][s]['seg_med']['absavg'][0] for s in labels):.2f} ms）都低。
     T6 / T13 也看到模擬的第 1 段比論文高。實測也比模擬低，表示模擬的起始收斂可能比實際慢，而不是論文的實驗特別好。
   - {labels['hw_lean']}的第 2–6 段中位數（約 {np.median(hl['seg_med']['absavg'][1:]):.2f} ms）反而比{labels['hw']}（約 {np.median(sm['sim']['hw']['seg_med']['absavg'][1:]):.2f} ms）大，而且是節點之間的固定偏差（真值版也一樣），20 次中最差的到 {max(hl['seg_hi']['absavg'][1:]):.2f} ms。
     原因【推測】：延遲很乾淨時，下緣只由少數幾個相位決定（99 ms 的封包 10 包約等於 33 個 CI，相位繞得很慢），各節點的下緣高度會差一點。實測這次沒有出現，要多錄幾次才知道。

## 限制

- 實測只有 1 次 60 分鐘的錄音；論文每個配置是 20 次的中位數。要比 IQR 或判斷第 4 點最後一項，需要多錄幾次。
- 實測的 RSE 沒有真值，只能用論文的方法（5 Hz 正弦）量；模擬另有真值版（`summary.json` 的 `true`）。
- 重傳率是用實測反推的，所以 > 1 CI、σ100 對上是預期中的。{labels['hw_lean']}的 PC 端與韌體設定也是看實測的分布調的。
  **沒有調、直接對上的**：封包間隔、> 2 CI（突發與否）、起始尖峰、斜率的正負與大小範圍。
"""
    write_text(od / "README.md", txt)
