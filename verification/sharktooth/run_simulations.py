"""檢查教材中依賴模擬的結論（第 5 部分 5.5、第 6 部分 6.2-1、6.2-2），並和 Gemini 版實作對照。

執行：python3 run_simulations.py [--seeds 3]
輸出為純文字表格；數值單位 ms，除非另註。
"""
import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ble_sim import Channel, DirectOLS, run_sync, simulate_node, worst_pair_by_segment
from sharktooth_claude import SharkTooth, SharkToothBiased
from sharktooth_gemini import SharkToothSync

ALPHAS_PPM = [15, 5, -3, -12]          # 教材 6.2-1 的四個節點
RHO_D_SLOW = 8e-6                     # 使節點對 dongle 的相對漂移 = 23, 13, 5, 4 ppm（教材「4–23 ppm」）
RHO_D_FAST = 158e-6                   # 相對漂移約 143–173 ppm（教材「約 150 ppm」）

LIGHT = Channel(p_loss=0.02, p_spike=0.01)
HEAVY = Channel(p_loss=0.10, p_spike=0.05)


def make_nodes(seed, P, rho_d, channels):
    """rho_d=None 表示 dongle 與節點同頻（無相對漂移，等待時間的相位固定）。"""
    rng = np.random.default_rng(seed)
    nodes = []
    for a_ppm, ch in zip(ALPHAS_PPM, channels):
        alpha = a_ppm * 1e-6
        rho_n = -alpha / (1 + alpha)  # 讓 δT 的真實斜率恰為 α
        theta = rng.uniform(0, 10)
        anchor = rng.uniform(0, 0.030)
        tP, tC, t = simulate_node(rng, rho_n, theta, rho_n if rho_d is None else rho_d,
                                  anchor, ch, P=P)
        nodes.append((alpha, tP, tC, t))
    return nodes


def evaluate(nodes, factory, with_settle=False):
    errs, ts, pivots, alphas, settle = [], [], [], [], []
    for alpha, tP, tC, t in nodes:
        sync = factory()
        core = getattr(sync, "core", sync)
        tS_hat = np.empty_like(tP)
        n_settle = 0
        for i in range(len(tP)):
            r = sync.update(tP[i], tC[i])
            tS_hat[i] = r[0] if isinstance(r, tuple) else r
            if hasattr(core, "settle") and not tP[i] > core.tP0 + core.settle:
                n_settle += 1
        errs.append(tS_hat - t)
        ts.append(t)
        pivots.append(getattr(core, "n_pivots", None))
        alphas.append(getattr(core, "alpha", np.nan) - getattr(sync, "B", 0.0))
        settle.append(100.0 * n_settle / len(tP))
    out = (worst_pair_by_segment(errs, ts) * 1e3, pivots, alphas)
    return out + (settle,) if with_settle else out


def fmt(seg):
    return ", ".join(f"{x:.2f}" if x < 100 else f"{x:.0f}" for x in seg)


# ------------------------------------------------------------------ 情境
def sanity_sigma(seed):
    """δ²T 的 100 點滑動標準差，在無掉包、無尖峰時應約 √200 ≈ 14.1 ms（加一點抖動）。"""
    nodes = make_nodes(seed, 0.100, RHO_D_SLOW, [Channel(p_loss=0, p_spike=0)] * 4)
    out = []
    for _, tP, tC, _ in nodes:
        d2 = np.diff(tC - tP)
        out.append(np.mean([np.std(d2[i:i + 100]) for i in range(0, len(d2) - 100, 50)]) * 1e3)
    return out


def scen_negative_drift(seed, rho_d=RHO_D_SLOW):
    nodes = make_nodes(seed, 0.100, rho_d, [LIGHT] * 4)
    lit = evaluate(nodes, lambda: SharkTooth(0.030), True)
    hold = evaluate(nodes, lambda: SharkTooth(0.030, hold_during_settling=True), True)
    bias = evaluate(nodes, lambda: SharkToothBiased(0.030), True)
    return lit, hold, bias


def scen_gcd(seed):
    res = {}
    for name, P, rho_d in [("P=100ms 慢漂移", 0.100, RHO_D_SLOW),
                           ("P=98.75ms 慢漂移", 0.09875, RHO_D_SLOW),
                           ("P=100ms 快漂移", 0.100, RHO_D_FAST)]:
        nodes = make_nodes(seed, P, rho_d, [LIGHT] * 4)
        res[name] = evaluate(nodes, lambda: SharkToothBiased(0.030))[0]
    return res


def scen_ablation(seed, rho_d):
    res = {}
    for name, chans in [("相同通道、輕度", [LIGHT] * 4),
                        ("相同通道、重度", [HEAVY] * 4),
                        ("一節點掉包 20%", [Channel(p_loss=0.20, p_spike=0.01)] + [LIGHT] * 3)]:
        nodes = make_nodes(seed, 0.100, rho_d, chans)
        res[name] = (evaluate(nodes, DirectOLS)[0],
                     evaluate(nodes, lambda: SharkToothBiased(0.030))[0])
    return res


def scen_gemini(seed):
    """Gemini 版實作（γ1 重設為 1e-8、δT1 = δT^i）vs Claude 版（論文 Eq.11、12：γ1 ← ∞、γ1 = γ0、δT1 = δT*）。
    兩者都先加 B = 100 ppm 讓斜率為正，只比較 Algorithm 2 的差別。"""
    B = 100e-6
    nodes = make_nodes(seed, 0.100, RHO_D_FAST, [LIGHT] * 4)
    shifted = [(a, tP, tC + B * tP, t + 0 * tP) for a, tP, tC, t in nodes]
    out = {}
    for name, factory in [("Claude（論文版）", lambda: SharkTooth(0.030, hold_during_settling=True)),
                          ("Gemini 版", lambda: SharkToothSync(30.0))]:
        errs, ts, alpha_err, pivots = [], [], [], []
        for (alpha, tP, tC, t) in shifted:
            sync = factory()
            tS_hat = run_sync(sync, tP, tC) - B * tP
            errs.append(tS_hat - t); ts.append(t)
            alpha_err.append((sync.alpha - B - alpha) * 1e6)
            pivots.append(sync.n_pivots)
        out[name] = (worst_pair_by_segment(errs, ts) * 1e3, alpha_err, pivots)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    with ProcessPoolExecutor() as ex:
        sig = list(ex.map(sanity_sigma, seeds))
        neg = list(ex.map(scen_negative_drift, seeds))
        neg_iso = list(ex.map(scen_negative_drift, seeds, [None] * len(seeds)))
        gcd = list(ex.map(scen_gcd, seeds))
        abl_fast = list(ex.map(scen_ablation, seeds, [RHO_D_FAST] * len(seeds)))
        abl_slow = list(ex.map(scen_ablation, seeds, [RHO_D_SLOW] * len(seeds)))
        gem = list(ex.map(scen_gemini, seeds))

    print("=== 0. 模擬器自我檢查：無掉包時 σ_100(δ²T)（ms），理論 14.14 + 抖動")
    for s, v in zip(seeds, sig):
        print(f"  seed {s}: " + ", ".join(f"{x:.2f}" for x in v))

    for title, data in [("節點對 dongle 相對漂移 4–23 ppm（教材設定）", neg),
                        ("無相對漂移（隔離「α 正負號」這個因素）", neg_iso)]:
        print(f"\n=== 1. 負漂移（教材 6.2-1）：節點 α = +15, +5, −3, −12 ppm，輕度通道，{title}")
        for s, (lit, hold, bias) in zip(seeds, data):
            print(f"  seed {s}")
            for name, (seg, piv, al, st) in [("字面 Eq.18（穩定期 α=β=0）", lit),
                                             ("穩定期保留上一組 α,β", hold),
                                             ("加斜率 B=100 ppm", bias)]:
                al_s = ", ".join(f"{a*1e6:+.1f}" for a in al)
                st_s = ", ".join(f"{x:.0f}" for x in st)
                print(f"    {name:<22} 樞紐點 {piv}  穩定期占比% [{st_s}]  最終 α̂(ppm) [{al_s}]")
                print(f"      最差一對平均|RSE| 第1–6段: {fmt(seg)}")

    print("\n=== 2. 封包週期與 CI 的 gcd（教材 6.2-2），SharkTooth + B，最差一對平均|RSE| 第2–6段")
    for name in gcd[0]:
        print(f"  {name}")
        for s, g in zip(seeds, gcd):
            print(f"    seed {s}: {fmt(g[name][1:])}")

    for label, abl in [("快漂移（約 150 ppm，教材 5.5 的設定）", abl_fast), ("慢漂移（4–23 ppm）", abl_slow)]:
        print(f"\n=== 3. 消融：直接 OLS vs SharkTooth（教材 5.5），{label}，第2–6段")
        for name in abl[0]:
            print(f"  {name}")
            for s, a in zip(seeds, abl):
                print(f"    seed {s}: 直接 OLS {fmt(a[name][0][1:])} | SharkTooth {fmt(a[name][1][1:])}")
    th = 30 * (0.2 / 0.8 - 0.02 / 0.98)
    print(f"  理論：一節點掉包 20% 時直接 OLS 偏差 ≈ {th:.2f} ms")

    print("\n=== 4. Gemini 版 vs Claude 版 Algorithm 2（同資料，快漂移、輕度通道、加 B）")
    for s, g in zip(seeds, gem):
        print(f"  seed {s}")
        for name, (seg, aerr, piv) in g.items():
            print(f"    {name:<12} 第2–6段: {fmt(seg[1:])}  α̂ 誤差(ppm) "
                  + ", ".join(f"{x:+.2f}" for x in aerr) + f"  樞紐點 {piv}")


if __name__ == "__main__":
    main()
