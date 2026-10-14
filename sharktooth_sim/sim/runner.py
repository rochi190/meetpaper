"""實驗共用工具：跑模擬 + 演算法、算誤差、平行化、寫結果。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import algorithms
from . import evaluate as ev
from .simulate import received, simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def outdir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path, obj):
    Path(path).write_text(json.dumps(_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def pmap(fn, items, n_jobs=None):
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_jobs) as pool:
        return pool.map(fn, items, chunksize=1)


def node_series(sim, k=0):
    """第 k 個節點已收到的封包（依到達順序）。"""
    return received(sim["nodes"][k]["packets"])


def algo_params(cfg):
    p = dict(cfg["sharktooth"])
    p["CI"] = cfg["CI"]
    return p


def run_node_algos(p, cfg, algos):
    """對一個節點跑多個演算法；回傳 name → dict(ts, valid, err(真值誤差, s), ...)。"""
    out = {}
    prm = algo_params(cfg)
    for a in algos:
        r = algorithms.run(a, p["t_P"], p["t_C"], **prm)
        r["err"] = r["ts"] - p["t_pc_true"]
        out[a] = r
    return out


def sample_timestamps(sim_node, p_recv, algo_out, scheme="packet"):
    """把每包的 (α, β) 套到包內每個樣本：樣本 t_P = t_P,last − (N−1−j)·Δ/(N−1)，Δ 為首末樣本時間差欄位。
    回傳所有「已收到」樣本的 (sample_tp, val, sample_ts, sample_true_pc)。"""
    meta = sim_node["meta"]
    S = sim_node["samples"]
    seq = p_recv["pkt_seq"]
    first = meta["first_idx"][seq]
    last = meta["last_idx"][seq]
    nS = last - first + 1
    tot = int(nS.sum())
    rep = np.repeat(np.arange(seq.size), nS)
    j = np.arange(tot) - np.repeat(np.cumsum(nS) - nS, nS)
    idx = np.repeat(first, nS) + j
    dtf = p_recv["first_last_dt"][rep]
    tp_s = p_recv["t_P"][rep] - (nS[rep] - 1 - j) * np.where(nS[rep] > 1, dtf / np.maximum(nS[rep] - 1, 1), 0)
    dhat = algo_out["dhat"][rep]
    if "alpha" in algo_out:
        dhat = algo_out["alpha"][rep] * tp_s + algo_out["beta"][rep]
    ts = tp_s + dhat
    ts = np.where(algo_out["valid"][rep], ts, np.nan)
    # 去掉重複封包（同一樣本出現兩次）
    _, uniq = np.unique(idx, return_index=True)
    return dict(sample_tp=tp_s[uniq], sample_val=S["val"][idx][uniq], sample_ts=ts[uniq],
                sample_true_pc=S["t_pc_true"][idx][uniq])


def pair_true_errors(sim, algo_name, cfg, epoch_s=1.0):
    """真值版節點對誤差：每個節點的 e_n(i) 先內插到共同 1 s epoch（真實時間），再兩兩相減。"""
    T = cfg["T"]
    grid = np.arange(0, T, epoch_s) + epoch_s / 2
    E = []
    for k in range(len(sim["nodes"])):
        p = node_series(sim, k)
        r = algorithms.run(algo_name, p["t_P"], p["t_C"], **algo_params(cfg))
        e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
        tt = p["t_true_last_sample"]
        E.append(np.interp(grid, tt, e, left=np.nan, right=np.nan))
    return grid, E
