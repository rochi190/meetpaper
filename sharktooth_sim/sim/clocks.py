"""
A 類：時鐘模型。

clock(t) = θ + ∫_0^t (1 + ρ(τ)) dτ，ρ(t) = ρ_0 + ρ_temp(T(t)) + ρ_rw(t)
  - A1 ρ_0：晶振出廠誤差
  - A2 ρ_temp = k (T − T0)²（音叉晶振；ABS04W：k = −0.02～−0.04 ppm/°C²、T0 = 20～30 °C）
         + k1 (T − T_start)（線性項：PC 的 AT-cut 晶振在室溫附近約 ±0.5 ppm/°C，Murdoch 2006）
  - A3 暖機：T(t) = T_end − (T_end − T_start) e^(−t/τ) + 環境溫度起伏 A sin(2πt/P + φ)
  - A5 隨機漫步
  - A6 PC 時鐘：常數 ρ + 溫度（同 A2/A3 的形式）+ NTP 跳時（step）/ 調速（slew）
  - A7 量化（32.768 kHz tick）
  - A8 tick→時間換算誤差：讀數 × (1 + e)
  - A10 溢位：讀數以 unit 為單位存成 bits 位元，繞回
真實時間 t 是模擬內的上帝時鐘。
"""
from __future__ import annotations

import numpy as np

from .config import draw


def draw_temp(temp, rng):
    """溫度設定的每個數值欄位各抽一次（分布 → 數字）；沒開就回傳 None，不消耗亂數。"""
    if not (temp and temp.get("enabled")):
        return None
    return {k: (float(draw(v, rng)) if k != "enabled" else v) for k, v in temp.items()}


def temperature_curve(temp, g):
    """A3 暖機 + 環境溫度起伏。T_end 沒給時用 T_start + dT_C。"""
    T_end = temp.get("T_end", temp["T_start"] + temp.get("dT_C", 0.0))
    Tc = T_end - (T_end - temp["T_start"]) * np.exp(-np.maximum(g, 0) / temp["tau_s"])
    if temp.get("amb_amp_C"):
        Tc = Tc + temp["amb_amp_C"] * np.sin(2 * np.pi * g / temp["amb_period_s"] + temp.get("amb_phase", 0.0))
    return Tc


class Clock:
    def __init__(self, T, rho_ppm=0.0, theta=0.0, tick_hz=None, scale_err_ppm=0.0,
                 temp=None, rw=None, wrap=None, ntp_steps=(), ntp_slews=(), rng=None, dt_grid=1.0,
                 t_min=-5.0):
        self.rho0 = rho_ppm * 1e-6
        self.theta = theta
        self.tick_hz = tick_hz or None
        self.scale = 1.0 + scale_err_ppm * 1e-6
        self.wrap = wrap if (wrap and wrap.get("enabled")) else None
        self.ntp_steps = sorted([tuple(s) for s in ntp_steps])
        self.const = not ((temp and temp.get("enabled")) or (rw and rw.get("enabled")) or ntp_slews)
        if not self.const:
            g = np.arange(t_min, T + 10.0 + dt_grid, dt_grid)
            rho = np.full(g.size, self.rho0)
            if temp and temp.get("enabled"):
                Tc = temperature_curve(temp, g)
                rho += temp.get("k_ppm_per_C2", 0.0) * 1e-6 * (Tc - temp.get("T0", 25.0)) ** 2
                rho += temp.get("k1_ppm_per_C", 0.0) * 1e-6 * (Tc - temp["T_start"])
                self.temperature = (g, Tc)
            if rw and rw.get("enabled"):
                sd = rw["ppm_per_sqrt_hour"] * 1e-6 * np.sqrt(dt_grid / 3600.0)
                rho += np.cumsum(rng.normal(0, sd, g.size))
            for t0, t1, ppm in ntp_slews:
                rho += np.where((g >= t0) & (g < t1), ppm * 1e-6, 0.0)
            # 梯形積分
            phase = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(g))])
            phase -= np.interp(0.0, g, phase)
            self.g, self.rho, self.phase = g, rho, phase

    # 連續（未量化）讀數
    def exact(self, t):
        t = np.asarray(t, float)
        if self.const:
            c = self.theta + t * (1.0 + self.rho0)
        else:
            c = self.theta + t + np.interp(t, self.g, self.phase)
        for ts, step in self.ntp_steps:
            c = c + np.where(t >= ts, step, 0.0)
        return c

    def rho_at(self, t):
        if self.const:
            return np.full(np.shape(t), self.rho0)
        return np.interp(t, self.g, self.rho)

    # 韌體 / 軟體實際讀到的值
    def read(self, t, resolution=None):
        c = self.exact(t) * self.scale
        if self.tick_hz:
            c = np.floor(c * self.tick_hz) / self.tick_hz
        if resolution:
            c = np.floor(c / resolution) * resolution
        if self.wrap:
            unit = self.wrap["unit"]
            span = (2 ** self.wrap["bits"]) * unit
            c = np.mod(c, span)
        return c

    # 讀數 → 真實時間（未量化讀數的反函數）
    def inverse(self, c):
        c = np.asarray(c, float)
        if self.const and not self.ntp_steps:
            return (c - self.theta) / (1.0 + self.rho0)
        g = self.g if not self.const else np.arange(-5.0, 4e4, 1.0)
        return np.interp(c, self.exact(g), g)


def node_clock(cfg_clock, T, rho_ppm, start_s, rng):
    """節點時鐘：在真實時間 start_s 歸零（收到開始命令）。"""
    return Clock(T, rho_ppm=rho_ppm, theta=-start_s * (1 + rho_ppm * 1e-6),
                 tick_hz=cfg_clock.get("tick_hz"), scale_err_ppm=cfg_clock.get("scale_err_ppm", 0.0),
                 temp=cfg_clock.get("temp"), rw=cfg_clock.get("rw"), wrap=cfg_clock.get("wrap"), rng=rng)
