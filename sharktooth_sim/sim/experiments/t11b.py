"""T11b：論文三個斜率（Fig.3 / 4 / 6 = 245 / 235 / 244 ppm）在 T11 各候選組合下的似然。

T11 的五項檢查只問「斜率有沒有落在論文範圍」；這裡問「哪個組合最能解釋這三個數字」。
論文沒寫三張圖來自哪個節點，所以算兩種情境：
  A：三個不同節點、同一台 PC（PC 誤差三張圖共用）；
  B：同一個節點（三個斜率只差量測誤差）。
δT 斜率 ≈ X + ρ_C，X = 節點部分（t_P 時鐘 + 換算常數），交叉項 < 0.1 ppm。
"""
import numpy as np

from ..config import draw, load_config
from ..runner import outdir, write_json, write_text
from .t11 import CONV_LABEL, PC_LABEL, PULL_LABEL, SRC_LABEL, draw_clocks

EXTRA_PULL_LABEL = {"ext10": "外接 10 pF（≈7.2 pF）", "ext12": "外接 12 pF（≈8.2 pF）"}


def smooth_pdf(samples, bw, n_bins=4000):
    """樣本的經驗分布 ⊗ N(0, bw²)，回傳 (格點, 密度)。"""
    edges = np.linspace(samples.min() - 6 * bw, samples.max() + 6 * bw, n_bins + 1)
    h, _ = np.histogram(samples, edges, density=True)
    dx = edges[1] - edges[0]
    t = np.arange(-int(np.ceil(6 * bw / dx)), int(np.ceil(6 * bw / dx)) + 1) * dx
    k = np.exp(-0.5 * (t / bw) ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), np.convolve(h, k / k.sum(), mode="same")


def log_density(grid, dens, x):
    return np.log(np.interp(x, grid, dens, left=0.0, right=0.0) + 1e-300)


def likelihood(cfg, c, rng):
    obs = np.asarray(cfg["obs_ppm"], float)
    sd = cfg["meas_sd_ppm"]
    rho_p, _, conv = draw_clocks(cfg, c, rng, cfg["n_node"])
    X = (1.0 / ((1 + rho_p * 1e-6) * (1 + conv * 1e-6)) - 1) * 1e6
    pc = np.asarray(draw(cfg["levels"]["pc"][c["pc"]]["rho"], rng, cfg["n_pc"]), float)
    g, d = smooth_pdf(X, sd)
    lp = log_density(g, d, obs[None, :] - pc[:, None]).sum(1)          # 情境 A：每個 PC 值下三個節點的聯合機率
    logA = lp.max() + np.log(np.mean(np.exp(lp - lp.max())))
    Y = X + rng.choice(pc, X.size)
    g, d = smooth_pdf(Y, sd / np.sqrt(len(obs)))
    logB = log_density(g, d, np.array([obs.mean()]))[0]                 # 情境 B：同一節點，看三點平均
    return dict(slope_med=float(np.median(Y)), slope_p5=float(np.percentile(Y, 5)),
                slope_p95=float(np.percentile(Y, 95)), logA=float(logA), logB=float(logB))


def run(cfg_path):
    cfg = load_config(cfg_path)
    cfg["levels"]["pull"].update(cfg["extra_pull"])
    od = outdir(cfg["name"])
    rng = np.random.default_rng(0)
    rows = []
    for src, pull, conv, pc in cfg["candidates"]:
        c = dict(src=src, pull=pull, conv=conv, pc=pc)
        rows.append(dict(c, name=f"{src}|{pull}|{conv}|{pc}", **likelihood(cfg, c, rng)))
    for r in rows:
        r["relA"] = float(np.exp(r["logA"] - rows[0]["logA"]))
        r["relB"] = float(np.exp(r["logB"] - rows[0]["logB"]))
    write_json(od / "summary.json", dict(obs_ppm=cfg["obs_ppm"], meas_sd_ppm=cfg["meas_sd_ppm"], rows=rows))
    write_readme(od, cfg, rows)


def write_readme(od, cfg, rows):
    pl = {**PULL_LABEL, **EXTRA_PULL_LABEL}
    by = {r["name"]: r for r in rows}

    def rng_of(key, sc, fmt):
        # 同一個家族在兩種 PC 時鐘下的範圍
        v = sorted(by[f"{key}|{pc}"][f"rel{sc}"] for pc in ("ntp", "qpc"))
        return f"{v[0]:{fmt}}" if f"{v[0]:{fmt}}" == f"{v[1]:{fmt}}" else f"{v[0]:{fmt}}–{v[1]:{fmt}}"

    b15 = by["xt|ext15|exact|ntp"]
    pcb = by["xt|matched|exact|pc_bad"]
    tab = "\n".join(
        f"| {SRC_LABEL[r['src']]} | {pl[r['pull']]} | {CONV_LABEL[r['conv']]} | {PC_LABEL[r['pc']]} | "
        f"{r['slope_med']:+.0f}（{r['slope_p5']:+.0f} ～ {r['slope_p95']:+.0f}） | {r['relA']:.2g} | {r['relB']:.2g} |"
        for r in rows)
    obs = " / ".join(f"{v:.0f}" for v in cfg["obs_ppm"])
    text = f"""# T11b　哪個組合最能解釋論文的三個斜率

**設定**（`config/T11b_slope_likelihood.yaml`）：時鐘模型與 T11 相同。
論文三張圖的 δT 下緣斜率是 {obs} ppm（T0，量測誤差約 ±{cfg['meas_sd_ppm']:.0f} ppm）。
每個組合抽 {cfg['n_node']} 個節點，算「這個組合產生這三個數字的機率」。

T11 的五項檢查只問斜率有沒有落在論文範圍裡，這裡改成比較每個組合能多好地解釋這三個數字。

**相對似然** = 這個組合的機率 ÷ 第一列（30.51 µs、NTP）的機率。
- 0.5 代表比第一列差一倍，差一兩倍不算數。
- 小於 0.1 才算資料明顯不支持。
- 這只看資料，沒有考慮「設計者會不會這樣做」（先驗）。

論文沒寫三張圖是哪個節點，所以分兩種情境：
- **A**：三個不同節點、同一台 PC（PC 誤差三張圖共用）。
- **B**：同一個節點（三個斜率只差量測誤差）。

| t_P 時鐘源 | 負載電容 | 換算 | PC 時鐘 | 斜率 ppm 中位（5–95 %） | 相對似然 A | 相對似然 B |
|---|---|---|---|---|---|---|
{tab}

## 結論

1. **論文的資料分不出 (a) 和 (b)。**
   - (a)：韌體換算用 30.51 µs，負載電容相符。
   - (b)：每腳外接約 12 pF，讓 4 pF 晶振變慢。
   - 和同一種 PC 時鐘的 (a) 比，(b) 每腳 12 pF 的相對似然是 {by['xt|ext12|exact|ntp']['relA']:.2f}（情境 A）／ {by['xt|ext12|exact|ntp']['relB']:.2f}（情境 B），只差不到兩倍。
2. **照抄參考設計的 15 pF 比較不像**：相對似然 {rng_of('xt|ext15|exact', 'A', '.2f')}（A）／ {rng_of('xt|ext15|exact', 'B', '.2f')}（B），範圍是兩種 PC 時鐘。它的斜率中位數 {b15['slope_med']:+.0f} ppm，偏高。
3. **其他候選更不像**：
   - HFADJ 或 30.5 µs 加上不接電容：{rng_of('hfadj|nocap|exact', 'A', '.3f')}（A）／ {rng_of('hfadj|nocap|exact', 'B', '.2f')}（B）。
   - 只靠 PC 時鐘偏差：{pcb['relA']:.2f}（A）／ {pcb['relB']:.2f}（B）。
4. **情境 B 的差距都比較小。** 同一個節點的三個斜率只提供一個數字，沒辦法看出板與板之間的差異。

**限制**：結果取決於板與板差異的假設（晶振誤差 ±20 ppm、C0 1.5–2.0 pF、C1 ±10 %、雜散 0.2 pF），這些都沿用 T11。
資料能做的只有排除，剩下的 (a)、(b) 要靠作者的韌體、電路圖或開發板實測決定。
"""
    write_text(od / "README.md", text)
