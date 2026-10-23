"""
設定：預設值 + YAML 覆蓋（deep merge）。

每個延遲 / 誤差來源（規格 §2 的 ID）都有 `enabled` 開關，方便做消融。
數值參數可以寫成常數，或寫成分布（每個節點各抽一次）：
    {uniform: [lo, hi]}、{normal: [mu, sd]}、{choice: [a, b, ...]}
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "name": "default",
    "T": 3600.0,                 # 擷取長度 (s)
    "seeds": [0],                # 每個 seed 一次完整模擬
    "n_nodes": 1,
    "nodes_per_dongle": 4,       # 論文：每支 dongle 4 個節點
    # ---------------- 封包 / 取樣 -----------------
    "packet": {
        "bytes": 244,            # BLE payload（17 / 127 / 244）
        "period_nom": 0.1,       # 名目封包週期 (s)
        "fs": 50.0,              # 名目取樣率 (Hz)
        "mode": "count",         # B4：count = 湊滿 N 筆就打包；timer = 節點計時器每 period_nom 打包
    },
    # ---------------- A 類：時鐘 -----------------
    "clock_node": {
        "rho_ppm": {"uniform": [-20.0, 20.0]},    # A1 晶振出廠誤差
        "start_s": {"uniform": [0.0, 0.3]},       # 節點收到開始命令、計時器歸零的真實時間
        "tick_hz": 32768.0,                       # A7 量化（None/0 = 不量化）
        "scale_err_ppm": 0.0,                     # A8 tick→時間換算誤差
        "temp": {"enabled": False, "T_start": 23.0, "T_end": 33.0, "tau_s": 300.0,
                 "T0": 25.0, "k_ppm_per_C2": -0.034},   # A2 + A3；另可加 dT_C（取代 T_end）、k1_ppm_per_C、
                                                         # 環境起伏 amb_amp_C / amb_period_s / amb_phase（見 clocks.py）
        "rw": {"enabled": False, "ppm_per_sqrt_hour": 0.05},  # A5 隨機漫步
        "wrap": {"enabled": False, "bits": 32, "unit": 1e-6},  # A10 溢位
    },
    # ADC / 取樣計時器的時鐘（決定實際封包週期）。T0 發現論文資料的封包間隔是 100.64–100.93 ms（以 t_P 計），
    # 即取樣計時器比 32.768 kHz 晶振慢 0.6–0.9 %（【推測】來自 HFRC 衍生的 ADC 計時器）
    "clock_adc": {"period_err_ppm": {"uniform": [6400.0, 9300.0]},
                  "temp_coeff_ppm_per_C": None},    # B6：RC 計時器的溫度係數（要開節點溫度才有作用）
    "clock_pc": {
        "rho_ppm": {"uniform": [-50.0, 50.0]},    # A6
        "tc_resolution": 1e-6,                    # E4
        "temp": {"enabled": False},               # A6 溫度：欄位同 clock_node.temp，通常只用 k1_ppm_per_C
        "ntp_steps": [],                          # [[t, step_s], ...]
        "ntp_slews": [],                          # [[t0, t1, ppm], ...]
    },
    "clock_dongle": {"rho_ppm": {"uniform": [-20.0, 20.0]}},
    # ---------------- B 類：節點韌體 -----------------
    "isr_jitter": {"enabled": True, "max_s": 50e-6},             # B1
    "fw_delay": {"enabled": True, "base_s": 0.2e-3, "max_s": 2e-3},  # B2
    "buf_err": {"enabled": False, "prob": 0.0, "k": 1},          # B3
    "reset": {"enabled": False, "events": [], "reboot_s": 2.0},  # B5：[[t_true, ...]]
    "dup": {"enabled": False, "prob": 0.0},                       # T7c：重複送出同一封包
    # ---------------- C 類：BLE 連線 -----------------
    "CI": 0.030,
    "flow": {"window": 1, "ack_delay_events": 1, "max_pkts_per_event": 1},  # C4 + 韌體流量控制
    "per": {                                     # C3：Gilbert–Elliott（PER 指 ref_bytes 長度的封包）
        "enabled": True, "good": 0.0, "bad": 0.0, "p_g2b": 0.0, "mean_bad_events": 3.0,
        "ref_bytes": 244, "length_scaling": True,
    },
    "piconet_per_add": 0.0,                      # C6：每多一個並行 piconet，good 狀態 PER 增加
    "hop": {"enabled": False, "n_bad": 9, "per_bad_ch": 0.3},   # C7
    "sched": {"enabled": True, "skip_prob": 0.0, "skip_prob_per_extra_node": 0.0},  # C5
    "tx_queue": 8,                               # C10
    "loss": {"enabled": False, "prob": 0.0},     # 其他原因的隨機遺失（T7a 用來設定掉包率）
    "disconnect": {"enabled": False, "events": [], "reconnect_s": 1.0},  # C9：[[t, dur], ...]
    "startup": {"enabled": True, "delay_s": 0.12},  # C11：第一包等待連線 / notification 就緒
    # ---------------- D 類：dongle / USB -----------------
    "dongle": {"enabled": True, "base_s": 0.3e-3, "jitter_s": 0.2e-3},    # D1
    "serial": {"enabled": True, "base_s": 0.3e-3, "per_byte_s": 4e-6},     # D2：每支 dongle 單一伺服器佇列
    "usb": {"enabled": True, "frame_s": 1e-3},                              # D3
    # ---------------- E 類：PC -----------------
    "os": {"enabled": True, "median_s": 0.4e-3, "sigma": 0.6, "per_node_scale": 0.0},  # E1 + E3
    "gil": {"enabled": False, "service_s": 0.15e-3},                        # E3：所有 callback 共用一個直譯器
    "pauses": {"enabled": False, "rate_hz": 0.0, "min_s": 5e-3, "max_s": 80e-3},  # E2：共同模式停頓
    "indep_spikes": {"enabled": False, "prob": 0.0, "min_s": 5e-3, "max_s": 80e-3},
    "background": {"enabled": False, "period_s": 10.0, "pause_s": 0.02},   # E5
    # ---------------- F 類：參考訊號 / 量測 -----------------
    "signal": {"f": 5.0, "amp": 0.5, "noise_v": 0.0, "afe_delay_sd_s": 0.0},
    # ---------------- 演算法 -----------------
    "algorithms": ["sharktooth"],
    "sharktooth": {"M": 100, "lam_max": 0.5, "g1_inc": 1e-6, "settle_s": 30.0, "keep_last": False,
                   "forget": 1.0},
}


REPLACE_KEYS = {"sweep", "adc_cases", "refine"}   # 這些鍵整個取代，不合併


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _is_dist(v) and k not in REPLACE_KEYS:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_dist(v):
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("uniform", "normal", "choice")


def draw(v, rng, size=None):
    """常數或分布 → 抽樣。"""
    if _is_dist(v):
        kind, a = next(iter(v.items()))
        if kind == "uniform":
            return rng.uniform(a[0], a[1], size)
        if kind == "normal":
            return rng.normal(a[0], a[1], size)
        if kind == "choice":
            return rng.choice(a, size)
    if size is None:
        return v
    return np.full(size, v, dtype=float)


def _coerce(o):
    """PyYAML 把 `4e-6`（沒有小數點）讀成字串；把看起來像數字的字串轉回 float。"""
    if isinstance(o, dict):
        return {k: _coerce(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_coerce(v) for v in o]
    if isinstance(o, str):
        try:
            return float(o)
        except ValueError:
            return o
    return o


def load_config(path=None, **overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        y = _coerce(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
        base = y.pop("base", None)
        if base:
            cfg = load_config(Path(path).parent / base)
        cfg = deep_merge(cfg, y)
    return deep_merge(cfg, overrides)


def set_path(cfg, dotted, value):
    """set_path(cfg, 'per.good', 0.1)"""
    c = copy.deepcopy(cfg)
    d = c
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return c
