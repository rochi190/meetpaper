"""
SharkTooth 實測：PC 接收端（v2）
同時連多個 NODE_x，訂閱 notify，每收到一包就立刻打 t_C，存檔供之後復現。

用法：
    pip install bleak
    python st_receiver.py --names NODE_A --minutes 5                  # 一塊板子（Fig. 6）
    python st_receiver.py --names NODE_A NODE_B NODE_C --minutes 30   # 三塊板子（Fig. 9）
    python st_receiver.py ... --note "函數產生器 5 Hz 1 Vpp，筆電插電"  # 附註寫進 meta.json

每次執行會建立一個資料夾 runs/<日期_時間>/，裡面有：
    packets.csv   每一包一列：node, t_C, 解開的欄位, 原始 20 bytes（hex）
    events.csv    連線、notify、斷線、每 5 秒的封包率與「事件迴圈延遲」
    meta.json     這次實驗的設定：參數、板子位址、作業系統、Python / bleak 版本、開始時間

只要留著這個資料夾，就可以用 st_analyze.py 重跑分析（raw_hex 欄位讓之後改格式也能重新解析）。

封包格式（20 bytes，little-endian，和韌體一致）：
    byte 0      pkt_id        封包編號（mod 256）
    byte 1-4    tP_ticks      這包最後一個樣本的時間戳（LSE tick，1 tick = 1/32768 s）
    byte 5-6    fl_ticks      第一個到最後一個樣本的時間差（LSE tick）
    byte 7      n             樣本數（5）
    byte 8-17   s0..s4        5 個 ADC 值（12 bit）
    byte 18     node_byte     節點代號（'A' / 'B' / 'C'）
    byte 19     qdrop         節點端落後太多而跳過的封包數（mod 256）

設計重點：收到封包的 callback 只做兩件事——打 t_C、把資料放進記憶體。
寫硬碟交給另一個執行緒，每秒寫一次，所以存檔不會拖慢 t_C。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import struct
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

from bleak import BleakClient, BleakScanner

VERSION = "2026-11-01b"                   # 每次改版就改這裡，執行時會印出來
CHAR_UUID = "0000abce-8e22-4541-9d4c-21edae82ed19"
PKT = struct.Struct("<BIHB5HBB")          # 1 + 4 + 2 + 1 + 10 + 1 + 1 = 20 bytes
PKT_FIELDS = ["node", "t_C", "pkt_id", "tP_ticks", "fl_ticks", "n",
              "s0", "s1", "s2", "s3", "s4", "node_byte", "qdrop", "raw_hex"]
EVT_FIELDS = ["t_C", "event", "node", "detail"]


def parse(data: bytes):
    if len(data) != PKT.size:
        return None
    return PKT.unpack(data)


class Recorder:
    """收資料（在事件迴圈裡，越快越好）和寫檔（在背景執行緒裡）分開。"""

    def __init__(self, folder: Path, t0: float):
        self.t0 = t0
        self.pkt_buf: list = []
        self.evt_buf: list = []
        self.lock = threading.Lock()
        self.f_pkt = open(folder / "packets.csv", "w", newline="", encoding="utf-8")
        self.f_evt = open(folder / "events.csv", "w", newline="", encoding="utf-8")
        self.w_pkt = csv.writer(self.f_pkt)
        self.w_evt = csv.writer(self.f_evt)
        self.w_pkt.writerow(PKT_FIELDS)
        self.w_evt.writerow(EVT_FIELDS)
        self.count: dict = {}
        self.bad: dict = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    # ---- 在事件迴圈裡呼叫：不碰硬碟 ----
    def on_packet(self, name: str, data: bytes, t_c: float):
        p = parse(data)
        if p is None:
            self.bad[name] = self.bad.get(name, 0) + 1
            self.event("bad_length", name, f"{len(data)} bytes: {data.hex()}")
            return
        with self.lock:
            self.pkt_buf.append((name, f"{t_c:.6f}", *p, data.hex()))
        self.count[name] = self.count.get(name, 0) + 1

    def event(self, kind: str, node: str = "", detail: str = "", at: float | None = None):
        """at：事件發生時的 perf_counter() 讀數；沒給就用現在。"""
        t = (time.perf_counter() if at is None else at) - self.t0
        with self.lock:
            self.evt_buf.append((f"{t:.6f}", kind, node, detail))

    # ---- 背景執行緒：每秒把記憶體裡的資料寫到硬碟 ----
    def _flush(self):
        with self.lock:
            pk, self.pkt_buf = self.pkt_buf, []
            ev, self.evt_buf = self.evt_buf, []
        if pk:
            self.w_pkt.writerows(pk)
            self.f_pkt.flush()
        if ev:
            self.w_evt.writerows(ev)
            self.f_evt.flush()

    def _writer(self):
        while not self._stop.wait(1.0):
            self._flush()

    def close(self):
        self._stop.set()
        self._thread.join()
        self._flush()
        self.f_pkt.close()
        self.f_evt.close()


def lag_probe(loop, stats: dict, stop: threading.Event, period: float = 0.01):
    """量事件迴圈被卡住多久，方法和藍牙封包進來的路徑一樣：
    從另一個執行緒把一個小工作丟進事件迴圈，看它多久之後才被執行。
    （藍牙封包也是由 Windows 的藍牙執行緒丟進事件迴圈，所以這個延遲 = t_C 會被延後多少。）
    舊版用 asyncio.sleep(10 ms) 量，會被 Windows 約 15.6 ms 的計時器精度騙到，一直誤報。"""
    def done(t_post):
        lag = time.perf_counter() - t_post
        stats["max"] = max(stats["max"], lag)
        stats["n_over_5ms"] += lag > 0.005
    while not stop.is_set():
        loop.call_soon_threadsafe(done, time.perf_counter())
        time.sleep(period)


async def main(args):
    t_prog = time.perf_counter()                              # 程式開始（只用來記錄搜尋、連線花多久）
    t0 = t_prog                                               # t_C 的原點：所有板子連好、送出 notify 前會重設
    start_wall = datetime.now()
    folder = Path(args.outdir) / start_wall.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    rec = Recorder(folder, t0)

    meta = dict(
        start_local=start_wall.isoformat(timespec="seconds"),
        names=args.names, minutes=args.minutes, note=args.note,
        receiver_version=VERSION, char_uuid=CHAR_UUID, packet_format="<BIHB5HBB (20 bytes)", lse_hz=32768,
        t_C="time.perf_counter() - t0（秒），t0 = 所有板子連好之後、送出第一個 notify 之前（和論文一樣）",
        perf_counter=dict(vars(time.get_clock_info("perf_counter"))),
        python=sys.version.split()[0], bleak=metadata.version("bleak"),
        platform=platform.platform(), machine=platform.node(),
        devices={},
    )

    def make_cb(name):
        def cb(_char, data: bytearray):
            t_c = time.perf_counter() - rec.t0                # 第一件事就是打 t_C
            rec.on_packet(name, bytes(data), t_c)
        return cb

    def make_disc(name):
        def on_disconnect(_client):
            rec.event("disconnected", name)
            if not closing:                                   # 程式結束時自己斷線，不用警告
                print(f"\n!! {name} 斷線（之後這個節點的資料會中斷）")
        return on_disconnect

    clients = []
    pre_events = []                                           # t0 定下來之前發生的事，之後換算成負的 t_C
    closing = False
    lag = {"max": 0.0, "n_over_5ms": 0}
    stop_probe = threading.Event()
    threading.Thread(target=lag_probe, args=(asyncio.get_running_loop(), lag, stop_probe),
                     daemon=True).start()
    try:
        # 1. 先全部連上，再一起打開 notify。打開 notify = 韌體的「開始命令」（t_P 歸零）
        for name in args.names:
            print(f"搜尋 {name} …")
            try:
                dev = await BleakScanner.find_device_by_name(name, timeout=15.0)
            except Exception as e:                            # 藍牙沒開、沒有藍牙介面卡
                raise SystemExit(f"電腦的藍牙無法使用：{e}\n"
                                 "→ Windows：設定 → 藍牙與裝置 → 把「藍牙」打開，並確認飛航模式是關的")
            if dev is None:
                raise SystemExit(f"找不到 {name}：確認板子有在廣播、手機 App 沒有連著它")
            c = BleakClient(dev, disconnected_callback=make_disc(name))
            await c.connect()
            pre_events.append((time.perf_counter(), "connected", name, dev.address))
            meta["devices"][name] = dev.address
            print(f"  已連線 {name}（{dev.address}）")
            clients.append((name, c))
        # 2. 全部連好了，才把 t_C 歸零。這樣 δT 的起點差（β）只剩「notify 傳到板子」的時間，
        #    和論文一樣約零點幾秒；搜尋、連線花的時間不算進去。
        rec.t0 = time.perf_counter()
        rec.event("t_C_zero", "", f"程式開始後 {rec.t0 - t_prog:.3f} s（搜尋 + 連線花的時間）")
        for at, kind, node, detail in pre_events:
            rec.event(kind, node, detail, at=at)                # 會是負的 t_C
        print(f"全部連好（花了 {rec.t0 - t_prog:.1f} s），t_C 從現在開始算")
        for name, c in clients:
            rec.event("notify_request", name)                 # 這一刻之後，板子收到就把 t_P 歸零
            await c.start_notify(CHAR_UUID, make_cb(name))
            rec.event("notify_on", name)
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"開始記錄 → {folder}（Ctrl+C 提早結束）  [st_receiver {VERSION}]")

        # 2. 每 5 秒印一次封包率（正常約 10 包/秒）和事件迴圈延遲
        end = time.perf_counter() + args.minutes * 60
        last = {n: 0 for n in args.names}
        while time.perf_counter() < end:
            await asyncio.sleep(5.0)
            el = rec.now() / 60
            rates = "  ".join(f"{n}: {(rec.count.get(n, 0) - last[n]) / 5:4.1f}/s" for n in args.names)
            warn = "  ← 電腦卡住過" if lag["max"] > 0.005 else ""
            print(f"[{el:5.1f} min] {rates}   迴圈延遲最大 {lag['max'] * 1e3:5.1f} ms{warn}")
            rec.event("stats", "", f"{rates} | loop_lag_max_ms={lag['max'] * 1e3:.2f} "
                                   f"n_over_5ms={lag['n_over_5ms']}")
            last = {n: rec.count.get(n, 0) for n in args.names}
            lag["max"], lag["n_over_5ms"] = 0.0, 0
    finally:
        closing = True
        stop_probe.set()
        for name, c in clients:
            try:
                await c.stop_notify(CHAR_UUID)
                await c.disconnect()
            except Exception:
                pass
        rec.event("end")
        rec.close()
        meta["end_elapsed_s"] = round(rec.now(), 3)
        meta["packets"] = rec.count
        meta["bad_length"] = rec.bad
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        bad = "沒有" if not rec.bad else rec.bad
        print(f"共收到：{rec.count}  長度不是 20 bytes 的封包：{bad}")
        if rec.count:
            print(f"資料在 {folder}，分析：python st_analyze.py {folder} --ci <實際 CI 秒數>")
        else:
            print(f"沒有收到任何封包，{folder} 可以刪掉")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["NODE_A"])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--note", default="", help="這次實驗的附註，會寫進 meta.json")
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        pass
