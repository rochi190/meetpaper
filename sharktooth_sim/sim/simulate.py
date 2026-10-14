"""
把各模組串成完整流程（規格 §4.2）：

取樣（ADC 計時器）→ B1 打 t_P → B3 buffer → B4 打包 → B2 進 TX 佇列（C10）
→ C1 等 event（C5 排程）→ C3 重傳 / C4 流量控制 → D1/D2 dongle 序列化 → D3 USB → E1/E2/E3 OS → E4 打 t_C

輸出（每個節點，§4.3）：
  node_id, pkt_id, n_samples, t_P, t_C, t_true_last_sample, n_retx, dropped_flag, buf_err_flag
另外附模擬專用欄位：t_pc_true（樣本真正取樣時刻的 PC 時鐘讀數，算真值誤差用）、dup_flag
"""
from __future__ import annotations

import numpy as np

from . import central, host, link, node
from .clocks import Clock
from .config import draw


def simulate(cfg, seed):
    rng = np.random.default_rng(seed)
    T = cfg["T"]
    cp = cfg["clock_pc"]
    pc = Clock(T, rho_ppm=float(draw(cp["rho_ppm"], rng)), theta=rng.uniform(1000, 2000), tick_hz=None,
               ntp_steps=cp.get("ntp_steps", []), ntp_slews=cp.get("ntp_slews", []), rng=rng)
    sched, dongles = central.schedule(cfg, rng)
    nodes = []
    for n in range(cfg["n_nodes"]):
        nd = node.make_node(cfg, n, rng)
        nd["signal"] = node.signal_values(nd, cfg, rng)
        lk = link.simulate_link(cfg, nd, sched[n], len(dongles), rng)
        nd.update(lk)
        nd["dongle"] = sched[n]["dongle"]
        nd["slot"] = sched[n]["slot"]
        nodes.append(nd)
    # 所有送達 dongle 的封包 → host
    recv_node, recv_idx, recv_air, recv_dg = [], [], [], []
    for nd in nodes:
        ok = np.flatnonzero(np.isfinite(nd["t_air"]))
        recv_node.append(np.full(ok.size, nd["node_id"]))
        recv_idx.append(ok)
        recv_air.append(nd["t_air"][ok])
        recv_dg.append(np.full(ok.size, nd["dongle"]))
    recv = dict(node=np.concatenate(recv_node), idx=np.concatenate(recv_idx),
                t_air=np.concatenate(recv_air), dongle=np.concatenate(recv_dg))
    tC, t_cb, stages, pauses = host.host_pipeline(cfg, recv, dongles, pc, rng)
    out_nodes = []
    for nd in nodes:
        m = recv["node"] == nd["node_id"]
        n_pkt = nd["tP"].size
        t_C = np.full(n_pkt, np.nan)
        t_cbk = np.full(n_pkt, np.nan)
        t_C[recv["idx"][m]] = tC[m]
        t_cbk[recv["idx"][m]] = t_cb[m]
        rec = dict(
            node_id=np.full(n_pkt, nd["node_id"]),
            pkt_id=nd["pkt_id"] % 256,
            pkt_seq=nd["pkt_id"],
            n_samples=nd["n_samples"],
            t_P=nd["tP"],
            t_C=t_C,
            t_true_last_sample=nd["t_true_last"],
            n_retx=np.where(np.isfinite(nd["t_air"]), nd["attempts"] - 1, -1),
            dropped_flag=(nd["drop"] > 0).astype(np.int8),
            drop_reason=nd["drop"],
            buf_err_flag=nd["buf_err"].astype(np.int8),
            t_pc_true=pc.exact(nd["t_true_last"]),
            t_air=nd["t_air"], t_ready=nd["t_ready"], t_callback=t_cbk,
            dup_flag=np.zeros(n_pkt, np.int8),
        )
        # 時間戳中「首末樣本時間差」欄位（2 B，論文 §IV-A）：用節點時鐘
        rec["first_last_dt"] = nd["tp_samp"][nd["last_idx"]] - nd["tp_samp"][nd["first_idx"]]
        if cfg["dup"]["enabled"] and cfg["dup"]["prob"] > 0:
            rec = _add_duplicates(rec, cfg, rng)
        out_nodes.append(dict(
            packets=rec,
            samples=dict(t_true=nd["t_samp"], tp=nd["tp_samp"], val=nd["signal"],
                         t_pc_true=pc.exact(nd["t_samp"]), epoch=nd["epoch"]),
            meta=dict(node_id=nd["node_id"], rho_ppm=nd["rho_ppm"], start_s=nd["start_s"],
                      adc_err_ppm=nd["adc_err_ppm"], dongle=nd["dongle"], slot=nd["slot"],
                      anchor=sched[nd["node_id"]]["anchor"], ci=sched[nd["node_id"]]["ci"],
                      bad_frac=nd["bad_frac"], afe_delay_s=nd.get("afe_delay_s", 0.0),
                      first_idx=nd["first_idx"], last_idx=nd["last_idx"], reset_t=nd["reset_t"]),
        ))
    return dict(nodes=out_nodes, dongles=dongles, pc_rho_ppm=pc.rho0 * 1e6, pauses=pauses, stages=stages,
                recv=recv, pc_clock=pc)


def _add_duplicates(rec, cfg, rng):
    """T7c：已收到的封包以 prob 機率被重複收到一次（同 t_P、同編號，t_C 晚一個 CI）。"""
    ok = np.isfinite(rec["t_C"])
    dup = ok & (rng.random(ok.size) < cfg["dup"]["prob"])
    idx = np.flatnonzero(dup)
    if idx.size == 0:
        return rec
    order = np.argsort(np.r_[np.arange(ok.size), idx + 0.5], kind="stable")
    new = {}
    for k, v in rec.items():
        extra = v[idx].copy()
        if k == "t_C":
            extra = extra + cfg["CI"]
        if k == "dup_flag":
            extra = np.ones(idx.size, np.int8)
        new[k] = np.r_[v, extra][order]
    return new


def received(rec):
    """演算法只看得到已收到的封包（依到達順序）。"""
    m = np.isfinite(rec["t_C"])
    o = np.argsort(rec["t_C"][m], kind="stable")
    return {k: v[m][o] for k, v in rec.items()}
