"""確定性檢查：claude/SharkTooth 導讀教材.pdf 中「不依賴通道模型」的數值例子與推導。

每個測試的 docstring 標出教材頁碼。執行：python3 -m pytest -q test_deterministic.py
"""
import math
from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from sharktooth_claude import SharkTooth

MS = 1e-3


# ---------------------------------------------------------------- 第 1 部分
def test_ppm_drift_per_hour():
    """p.3–4：±20 ppm 一小時 72 ms；兩節點一快一慢差 144 ms。"""
    assert 3600 * 20e-6 == pytest.approx(0.072)
    assert 2 * 3600 * 20e-6 == pytest.approx(0.144)


def test_two_way_offset_example():
    """p.4：雙向交換 offset = 45 ms、delay = 5 ms。"""
    t1, t2, t3, t4 = 100, 150, 152, 112
    assert ((t2 - t1) + (t3 - t4)) / 2 == 45
    assert ((t4 - t1) - (t3 - t2)) / 2 == 5


def test_throughput_budget():
    """p.5：12 節點 1 支 dongle ≈ 1.355 kB/s；EMG 2 kB/s；IMU ≈ 2.5 kB/s；4 節點 ≈ 4.066 kB/s。"""
    assert 12 * 7.5 == 90
    assert 244 / 0.18 / 1000 == pytest.approx(1.3556, abs=1e-3)
    assert 1000 * 2 == 2000
    assert 208 * 6 * 2 == pytest.approx(2500, rel=0.01)
    assert 244 / 0.06 / 1000 == pytest.approx(4.0667, abs=1e-3)
    assert 251 - 4 - 3 == 244


def test_ols_three_points():
    """p.7：(0,1)(1,3)(2,5) → y = 2x + 1。"""
    a, b = np.polyfit([0, 1, 2], [1, 3, 5], 1)
    assert a == pytest.approx(2) and b == pytest.approx(1)


def test_percentile_example():
    """p.8：平均 0.65、中位數 0.35、numpy 預設 P95 ≈ 2.06。"""
    x = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.9, 3.0])
    assert x.mean() == pytest.approx(0.65)
    assert np.median(x) == pytest.approx(0.35)
    assert np.percentile(x, 95) == pytest.approx(2.055, abs=1e-3)


# ---------------------------------------------------------------- 第 3 部分（一）
def test_eq1_numeric_example():
    """p.17：ρ_P = −20 ppm、θ_P = 5 s、θ_C = 100 s → α ≈ 20.0004 ppm、β ≈ 94.9999 s、t_S ≈ 3700.0720 s。"""
    rho_P, theta_P, theta_C = -20e-6, 5.0, 100.0
    alpha = (0 - rho_P) / (1 + rho_P)
    beta = theta_C - (1 + alpha) * theta_P
    assert alpha * 1e6 == pytest.approx(20.0004, abs=1e-4)
    assert beta == pytest.approx(94.9999, abs=1e-4)
    tS = (1 + alpha) * 3605 + beta
    t = (3605 - theta_P) / (1 + rho_P)
    assert tS == pytest.approx(3700.0720, abs=1e-4)
    assert tS == pytest.approx(t + theta_C, abs=1e-9)


def test_eq3_eq4_numeric_example():
    """p.18–19：δT ≈ 95.0860 s。"""
    alpha = 20.0004e-6
    beta = 100 - (1 + alpha) * 5
    dT = alpha * 3605 + beta + 0.014
    assert dT == pytest.approx(95.0860, abs=1e-4)


def _wait_sequence(P, CI, n, w0=0.0):
    """只考慮「等下一個連線事件」：w_i = (w_0 − i·P) mod CI（教材 p.20 步驟 1）。"""
    i = np.arange(n)
    return np.mod(w0 - i * P, CI)


@pytest.mark.parametrize("P,expected_sigma", [(90, 0.00), (100, 14.14), (105, 15.00),
                                              (110, 14.14), (115, 11.18)])
def test_var_d2T_equals_r_times_CI_minus_r(P, expected_sigma):
    """p.20 表格：無抖動、30000 封包，σ(δ²T) = √(r(CI − r))。"""
    CI = 30.0
    w = _wait_sequence(float(P), CI, 30000, w0=7.3)
    sigma = np.std(np.diff(w))
    r = P % CI
    assert sigma == pytest.approx(math.sqrt(r * (CI - r)), abs=0.01)
    assert sigma == pytest.approx(expected_sigma, abs=0.01)


def test_sigma_upper_bound_is_CI_over_2():
    """p.20 步驟 6：σ(δ²T) ≤ CI/2，r = CI/2 時取等號；對所有整數 r 都成立。"""
    CI = 30.0
    for P in np.arange(30.0, 150.0, 0.25):
        w = _wait_sequence(P, CI, 20000, w0=3.1)
        assert np.std(np.diff(w)) <= CI / 2 + 1e-6


def test_iid_uniform_wait_sigma():
    """p.20：若 w 是獨立 U[0, CI)，σ(Δw) = CI/√6 ≈ 12.2 ms。"""
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 30, 2_000_000)
    assert np.std(np.diff(w)) == pytest.approx(30 / math.sqrt(6), rel=0.01)
    assert 30 / math.sqrt(6) == pytest.approx(12.25, abs=0.01)


def test_sawtooth_cycle_r10():
    """p.21：r = 10 ms 時等待 20 → 10 → 0 循環，δ²T = −10, −10, +20。"""
    w = _wait_sequence(100.0, 30.0, 7, w0=20.0)
    assert np.allclose(w, [20, 10, 0, 20, 10, 0, 20])
    assert np.allclose(np.diff(w), [-10, -10, 20, -10, -10, 20])


def test_eq7_example():
    """p.21：(−10, −10, +20) → μ = 0、σ = 14.14 ms。"""
    x = np.array([-10.0, -10.0, 20.0])
    assert x.mean() == 0
    assert np.std(x) == pytest.approx(14.142, abs=1e-3)


# ---------------------------------------------------------------- 第 3 部分（二）追蹤表
TOY_EXTRA_MS = [5, 25, 15, 5, 80, 15, 5, 25]
# 教材 p.25 的表：(λ0, λ, γ0 ppm, γ1 ppm, δT* ms)；λ 為 None 表示不合格點
TOY_TABLE = [
    (1.000, None, 0.0, math.inf, 305.000),
    (1.000, 0.500, 10100.00, 10100.00, 315.100),
    (1.000, 0.500, 7650.00, 7650.00, 320.300),
    (1.000, 0.500, 3925.00, 3925.00, 316.775),
    (0.432, None, 3925.00, 3926.00, 320.700),
    (0.328, 0.500, 3062.50, 3062.50, 320.312),
    (0.358, 0.500, 1631.25, 1631.25, 314.787),
    (0.380, None, 1631.25, 1632.25, 316.419),
]


def _run_toy():
    st = SharkTooth(CI=0.030)
    rows = []
    for i, e in enumerate(TOY_EXTRA_MS):
        tP = float(i)
        dT_true = 0.300 + 200e-6 * tP + e * MS
        prev_dTs, prev_g1, prev_dT0 = st.prev_dTs, st.g1, st.dT0
        _, dT, dTs = st.update(tP, tP + dT_true)
        sigma = float(np.std(st.d2)) if len(st.d2) > 1 else 0.0
        lam0 = 1.0 if sigma == 0 else min(0.030 / (2 * sigma), 1.0)
        pivot = dT < prev_dT0
        qualified = (not pivot) and st.g1 != prev_g1 + 1e-6
        lam = None
        if qualified:
            lam = 0.5 * lam0 if dT >= prev_dTs else 0.5
        rows.append((dT, lam0, lam, st.g0, st.g1, dTs, pivot))
    return rows


def test_toy_trace_table_matches_guide():
    """p.25：8 列追蹤表（δT、λ0、λ、γ0、γ1、δT*）逐格比對。"""
    rows = _run_toy()
    for i, ((dT, lam0, lam, g0, g1, dTs, _), (e_lam0, e_lam, e_g0, e_g1, e_dTs)) in enumerate(
            zip(rows, TOY_TABLE)):
        assert dT / MS == pytest.approx(300 + 0.2 * i + TOY_EXTRA_MS[i], abs=1e-9), i
        assert lam0 == pytest.approx(e_lam0, abs=5e-4), i
        if e_lam is None:
            assert lam is None, i
        else:
            assert lam == pytest.approx(e_lam), i
        assert g0 * 1e6 == pytest.approx(e_g0, abs=5e-3), i
        if math.isinf(e_g1):
            assert math.isinf(g1), i
        else:
            assert g1 * 1e6 == pytest.approx(e_g1, abs=5e-3), i
        assert dTs / MS == pytest.approx(e_dTs, abs=5e-4), i


def test_eq8_eq9_examples():
    """p.23：δT* = 305 + 3.925 × 3 = 316.775 ms；δT_thr = 316.775 + 3.926 × 2 = 324.627 ms。"""
    assert 305 + 3.925 * 3 == pytest.approx(316.775)
    assert 316.775 + 3.926 * 2 == pytest.approx(324.627)


def test_eq10_examples():
    """p.24：σ = 14.14 ms → λ0 = 1；σ = 40 ms → λ0 = 0.375。"""
    assert min(30 / (2 * 14.14), 1) == 1
    assert 30 / (2 * 14.14) == pytest.approx(1.06, abs=0.01)
    assert min(30 / 80, 1) == 0.375


def test_eq12_chord_slope_bias_examples():
    """p.25：ε_i − ε_0 = 10 ms 時，Δt = 10 s 偏 1000 ppm、Δt = 1000 s 偏 10 ppm。"""
    assert 10e-3 / 10 == pytest.approx(1000e-6)
    assert 10e-3 / 1000 == pytest.approx(10e-6)


def test_eq13_example():
    """p.26：λ0 = 0.375 → 線上方 λ = 0.1875。"""
    assert 0.5 * 0.375 == 0.1875


def test_eq14_probe_rises_as_parabola():
    """p.27：合格點後若一直不合格，門檻高出 δT* ≈ (10⁻⁶/P)·Δt²；P = 0.1 s 時 1 ms@10 s、10 ms@31.6 s、30 ms@54.8 s。"""
    P = 0.1
    st = SharkTooth(CI=0.030)
    # 先建立樞紐點與一個合格點，之後全部送很高的點（永遠被擋）
    st.update(0.0, 0.300)
    st.update(P, P + 0.300 + 2e-5)
    t1 = st.tP1
    gaps = {}
    for k in range(1, 600):
        tP = t1 + k * P
        st.update(tP, tP + 10.0)
        t_next = tP + P  # 下一個封包到達時的門檻與輸出線差距
        thr_next = st.dT1 + st.g1 * (t_next - st.tP1)
        dTs_next = st.dT0 + st.g0 * (t_next - st.tP0)
        gaps[round(t_next - t1, 6)] = thr_next - dTs_next
    for dt, expect_ms in [(10.0, 1.0), (31.6, 10.0), (54.8, 30.0)]:
        key = min(gaps, key=lambda d: abs(d - dt))
        assert gaps[key] / MS == pytest.approx(expect_ms, rel=0.02), (dt, gaps[key])
        assert 1e-5 * dt ** 2 / MS == pytest.approx(expect_ms, rel=0.01)


def test_reboot_probe_rise_time():
    """p.45：探針要上升 1000 s 需約 10⁴ s。"""
    assert math.sqrt(1000 / 1e-5) == pytest.approx(1e4)


# ---------------------------------------------------------------- 第 3 部分（三）
def test_rls_three_point_example():
    """p.30：A = [[3,−3],[−3,5]]、b = [13; 9]、det = 6、[α; β] = [2; 1]。"""
    A, b = np.zeros((2, 2)), np.zeros(2)
    for t, y in [(0, 1), (1, 3), (2, 5)]:
        A += np.array([[1, -t], [-t, t * t]])
        b += np.array([t * y, y])
    assert np.allclose(A, [[3, -3], [-3, 5]]) and np.allclose(b, [13, 9])
    det = np.linalg.det(A)
    assert det == pytest.approx(6)
    assert np.allclose(A @ b / det, [2, 1])


def test_rls_equals_polyfit_500_points():
    """p.30：遞迴式與 np.polyfit 一致（教材的具體數值依亂數種子，無法重現，只驗等價）。"""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 3600, 500))
    y = 0.3 + 2e-4 * t + rng.normal(0, 1e-3, 500)
    A, b = np.zeros((2, 2)), np.zeros(2)
    for ti, yi in zip(t, y):
        A += np.array([[1, -ti], [-ti, ti * ti]])
        b += np.array([ti * yi, yi])
    alpha, beta = A @ b / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    a_ref, b_ref = np.polyfit(t, y, 1)
    assert alpha == pytest.approx(a_ref, abs=1e-12)
    assert beta == pytest.approx(b_ref, abs=1e-9)


def test_two_points_determinant():
    """p.30：兩點相距 0.1 s 時 det = (t1 − t2)² = 0.01。"""
    t1, t2 = 100.0, 100.1
    det = 2 * (t1 ** 2 + t2 ** 2) - (t1 + t2) ** 2
    assert det == pytest.approx(0.01, rel=1e-6)


def _rls_float32(t, y):
    A = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    for ti, yi in zip(t.astype(np.float32), y.astype(np.float32)):
        A += np.array([[1, -ti], [-ti, ti * ti]], dtype=np.float32)
        b += np.array([ti * yi, yi], dtype=np.float32)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return (A @ b / det)[0]


def test_float32_breaks_with_large_tP():
    """p.30：36000 點，t_P 從 0 起 float32 仍正確；t_P ≈ 259200 s（開機 3 天）時 float32 算錯，float64 正確。

    教材寫 float32 得 α = −2.15×10⁻⁴（正負號都錯）。這裡用無雜訊直線重現得 +2.38×10⁻⁴（誤差約 19%），
    正負號沒翻；「float32 不可靠、先減參考值即可」的結論成立，但具體數值無法重現。
    """
    t0 = np.arange(36000) * 0.1
    y0 = 0.3 + 2e-4 * t0
    assert _rls_float32(t0, y0) == pytest.approx(2e-4, rel=1e-3)
    t3 = t0 + 259200.0
    y3 = 0.3 + 2e-4 * t3
    a32 = _rls_float32(t3, y3)
    a64 = np.polyfit(t3, y3, 1)[0]
    assert a64 == pytest.approx(2e-4, rel=1e-6)
    assert abs(a32 - 2e-4) / 2e-4 > 0.10  # float32 誤差 > 10%
    assert a32 > 0  # 教材說正負號會翻，這個實作下沒有翻
    # 先減參考值再累加就恢復正確
    assert _rls_float32(t3 - 259200.0, y3 - y3[0]) == pytest.approx(2e-4, rel=1e-3)


def test_eq19_eq20_example():
    """p.31：α = 200 ppm、β = 0.300 s、t_P = 3600 s → δT̂ = 1.02 s、t̂_S = 3601.02 s。"""
    dT_hat = 200e-6 * 3600 + 0.300
    assert dT_hat == pytest.approx(1.02)
    assert 3600 + dT_hat == pytest.approx(3601.02)


def test_eq18_literal_output_is_tP_during_settling():
    """p.31：照 Eq.(18) 字面，穩定期內 α = β = 0 → t̂_S = t_P（秒級誤差）。"""
    st = SharkTooth(CI=0.030)
    for i in range(200):
        tP = i * 0.1
        tS_hat, _, _ = st.update(tP, tP + 1.234 + 1e-4 * tP)
        assert tS_hat == tP


# ---------------------------------------------------------------- 第 3 部分（四）
def test_eq23_eq24_examples():
    """p.32–33：1.0203 − 0.8751 s = 145.2 ms；145.19→145.21 線性平均 145.20。"""
    assert (1.0203 - 0.8751) * 1000 == pytest.approx(145.2)
    assert np.linspace(145.19, 145.21, 50).mean() == pytest.approx(145.20)


def test_eq25_eq26_wrapping_examples():
    """p.34：−108.2 折成 +91.8 → τ_nm = −54.4 ms；RSE 折回例 = −0.4 ms。"""
    wrap = lambda x, T=200.0: (x + T / 2) % T - T / 2
    assert wrap(-108.2) == pytest.approx(91.8)  # 超出 ±100 ms 就被折回
    assert 37.4 - 91.8 == pytest.approx(-54.4)
    assert wrap(2345.2 - 145.6) == pytest.approx(-0.4)


def test_range_constants_d2():
    """p.35：常態樣本全距期望值 d2：N=2 → 1.128、N=4 → 2.059、N=12 → 3.258；12 vs 2 約 2.9 倍。"""
    rng = np.random.default_rng(2)
    for n, d2 in [(2, 1.128), (4, 2.059), (12, 3.258)]:
        x = rng.normal(size=(400_000, n))
        assert np.ptp(x, axis=1).mean() == pytest.approx(d2, abs=0.01)
    assert 3.258 / 1.128 == pytest.approx(2.89, abs=0.01)
    assert 0.41 / 0.21 == pytest.approx(1.95, abs=0.01)
    assert 0.62 / 0.25 == pytest.approx(2.48, abs=0.01)


# ---------------------------------------------------------------- 第 4 部分
def test_fig2_example():
    """p.36：δT_A = 0.75, 0.70；δT_B = 1.25, 1.20；差 0.5。"""
    assert [1.75 - 1, 2.70 - 2] == pytest.approx([0.75, 0.70])
    assert [2.25 - 1, 3.20 - 2] == pytest.approx([1.25, 1.20])


def _folded_mean(mu, sigma):
    return (sigma * math.sqrt(2 / math.pi) * math.exp(-mu ** 2 / (2 * sigma ** 2))
            + mu * (1 - 2 * norm.cdf(-mu / sigma)))


@pytest.mark.parametrize("cfg,mean_abs,std,no_bias,mu_guide", [
    ("A2", 0.21, 0.25, 0.20, 0.08),
    ("B2", 0.22, 0.26, 0.21, 0.09),
    ("C2", 0.25, 0.29, 0.23, 0.12),
    ("A12", 0.41, 0.25, 0.20, 0.40),
    ("B12", 0.50, 0.30, 0.24, 0.49),
    ("C12", 0.62, 0.41, 0.33, 0.59),
])
def test_folded_normal_bias_table(cfg, mean_abs, std, no_bias, mu_guide):
    """p.38–39：由 Table II 第 6 段 (平均|RSE|, std) 反推偏差 μ。"""
    assert 0.798 * std == pytest.approx(no_bias, abs=0.006)
    mu = brentq(lambda m: _folded_mean(m, std) - mean_abs, 0, 5)
    assert mu == pytest.approx(mu_guide, abs=0.006), (cfg, mu)


def test_table1_and_throughput():
    """p.38、41：5 Hz、1 Vpp；18 × 20 × 1 h = 360 h；C12 ≈ 29.3 kB/s、A2 = 0.34 kB/s；17 B 酬載拆解。"""
    assert 10 * math.pi / (2 * math.pi) == 5
    assert 2 * 0.5 == 1.0
    assert 18 * 20 == 360
    assert 12 * 244 / 0.1 / 1000 == pytest.approx(29.28)
    assert 2 * 17 / 0.1 / 1000 == pytest.approx(0.34)
    assert int(50 * 0.1) * 2 + 4 + 2 + 1 == 17


def test_fig12_theory_200ms2():
    """p.39：r(CI − r) = 10 × 20 = 200 ms²；(CI/2)² = 225 ms²。"""
    assert 10 * 20 == 200 and 15 ** 2 == 225


# ---------------------------------------------------------------- 第 5、6 部分
def test_retransmission_bias_theory():
    """p.43：30 × (0.2/0.8 − 0.02/0.98) ≈ 6.9 ms。"""
    assert 30 * (0.2 / 0.8 - 0.02 / 0.98) == pytest.approx(6.89, abs=0.01)


def test_geometric_retransmissions_mean():
    """p.43 的前提：每次嘗試失敗率 p，平均重傳次數 = p/(1−p)。"""
    rng = np.random.default_rng(3)
    for p in (0.02, 0.2):
        k = rng.geometric(1 - p, 2_000_000) - 1
        assert k.mean() == pytest.approx(p / (1 - p), rel=0.02)


def test_crystal_temperature_shift():
    """p.44：−0.03 ppm/°C² × (35 − 25)² = −3 ppm。"""
    assert -0.03 * (35 - 25) ** 2 == pytest.approx(-3)


def test_ble_ci_units_and_gcd():
    """p.45：98.75 ms = 79 × 1.25、30 ms = 24 × 1.25、gcd = 1；100 與 30 的 gcd = 10 ms。"""
    assert 98.75 / 1.25 == 79 and 30 / 1.25 == 24
    assert math.gcd(79, 24) == 1
    assert math.gcd(100, 30) == 10


def test_min_wait_lattice_is_gcd():
    """p.45：w 只落在間距 g = gcd(P, CI) 的相位上，最小等待 = w0 mod g。"""
    for P, CI, w0 in [(100.0, 30.0, 13.7), (98.75, 30.0, 13.7)]:
        w = _wait_sequence(P, CI, 5000, w0=w0)
        g = math.gcd(int(P * 4), int(CI * 4)) / 4
        assert w.min() == pytest.approx(w0 % g, abs=1e-6)
        assert len(np.unique(np.round(w, 6))) == pytest.approx(CI / g)


def test_tP_wraparound():
    """p.46：2³² μs ≈ 71.6 分鐘；2³² / 32768 Hz ≈ 36.4 小時。"""
    assert 2 ** 32 * 1e-6 / 60 == pytest.approx(71.6, abs=0.05)
    assert 2 ** 32 / 32768 / 3600 == pytest.approx(36.4, abs=0.05)


def test_worst_pair_code_snippet_runs():
    """p.35：worst_pair_metrics 片段可執行並選到 P95 最大的那對。"""
    nodes = [1, 2, 3]
    seg = slice(0, 600)
    rng = np.random.default_rng(4)
    scale = {(1, 2): 0.8, (1, 3): 1.1, (2, 3): 0.6}
    RSE = {p: rng.normal(0, s, 600) for p, s in scale.items()}

    def worst_pair_metrics(RSE, seg):
        best = max(combinations(nodes, 2), key=lambda p: np.percentile(np.abs(RSE[p][seg]), 95))
        r = RSE[best][seg]
        return best, np.mean(np.abs(r)), np.std(r), np.percentile(np.abs(r), 95)

    best, *_ = worst_pair_metrics(RSE, seg)
    assert best == (1, 3)
