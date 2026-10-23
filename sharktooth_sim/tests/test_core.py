"""基本正確性測試：python -m pytest tests -q"""
import numpy as np

from sim.algorithms import baselines, sharktooth
from sim.config import load_config
from sim import evaluate as ev
from sim.simulate import received, simulate


def _data(seed=3, T=400.0, **kw):
    cfg = load_config(None, T=T, n_nodes=1, per={"good": 0.09, "bad": 0.6, "p_g2b": 0.01}, **kw)
    p = received(simulate(cfg, seed)["nodes"][0]["packets"])
    return p["t_P"], p["t_C"]


def test_sharktooth_batch_matches_reference():
    tP, tC = _data()
    for kw in [dict(), dict(keep_last=True), dict(forget=0.999, keep_last=True), dict(M=20, g1_inc=1e-5)]:
        ref = sharktooth.SharkTooth(0.03, **kw)
        r = [ref.step(p, c) for p, c in zip(tP, tC)]
        b = sharktooth.run(tP, tC, CI=0.03, **kw)
        np.testing.assert_allclose(b["star"], [x["star"] for x in r], rtol=0, atol=1e-12)
        np.testing.assert_allclose(b["dhat"], [x["dhat"] for x in r], rtol=0, atol=1e-9)
        assert (b["valid"] == np.array([x["valid"] for x in r])).all()


def test_sharktooth_negative_drift_resets_pivots():
    tP, tC = _data(clock_node={"rho_ppm": 30.0}, clock_pc={"rho_ppm": 0.0})   # α ≈ −30 ppm
    out = sharktooth.run(tP, tC)
    assert out["n_pivot"] > 20


def test_lower_hull_matches_lp():
    tP, tC = _data(T=200.0)
    out = baselines.lower_hull(tP, tC)
    a, b = ev.lower_line_lp(tP - tP[0], tC - tP)
    np.testing.assert_allclose(out["dhat"][-1], a * (tP[-1] - tP[0]) + b, atol=2e-6)


def test_cc_phase_equals_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(5):
        tau = rng.uniform(-0.09, 0.09)
        t = 100 + np.arange(50) / 50.0
        x = 0.5 * np.sin(2 * np.pi * 5 * (t - tau)) + rng.normal(0, 0.01, 50)
        fast = ev.cc_delay_epochs(t, x)[0]
        slow = ev.cc_delay_bruteforce(t, x)
        assert abs(fast - slow) <= 1e-4 + 1e-9, (fast, slow, tau)


def test_integrity_detects_buffer_errors_and_reset():
    from sim.algorithms import integrity
    cfg = load_config(None, T=600.0, n_nodes=1, buf_err={"enabled": True, "prob": 0.01, "k": 1},
                      reset={"enabled": True, "events": [[300.0]], "reboot_s": 2.0})
    sim = simulate(cfg, 5)
    p = received(sim["nodes"][0]["packets"])
    c = integrity.classify(p["pkt_id"], p["t_P"], p["first_last_dt"], p["n_samples"], 50.0)
    assert c["n_reset"] == 1
    flagged = c["status"] == 1
    # 每個被注入的 buffer 錯誤都被抓到（重置點本身除外）
    truth = p["buf_err_flag"].astype(bool) & (c["status"] != 3)
    assert (flagged == truth).all()
