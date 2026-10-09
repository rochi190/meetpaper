"""教材第 6 部分 6.2-4：節點重開機（t_P 從 0 重來）後，SharkTooth 不會自己恢復。"""
import numpy as np

from sharktooth_claude import SharkTooth


def test_reboot_gets_stuck():
    rng = np.random.default_rng(0)
    st = SharkTooth(0.030, hold_during_settling=True)
    P = 0.1

    def feed(t0, n):
        errs = []
        for i in range(n):
            t, tP = t0 + i * P, i * P            # t_P 每次都從 0 起算
            tS_hat, _, _ = st.update(tP, t + 0.002 + rng.uniform(0, 0.03))
            errs.append(tS_hat - t)
        return np.array(errs)

    before = feed(1000.0, 6000)                  # 正常 600 s：δT ≈ 1000 s
    assert abs(before[-1]) < 0.01
    pivots = st.n_pivots
    after = feed(1600.0, 30000)                  # 重開機：δT 整個跳 +600 s，接著跑 3000 s
    assert st.n_pivots == pivots                 # 跳高不是新的最小值，不會觸發樞紐點
    assert np.all(np.abs(after) > 500)           # 3000 s 內誤差一直是約 −600 s
