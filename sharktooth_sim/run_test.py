"""
執行測試：
    python run_test.py config/T4_fig6.yaml
    python run_test.py config/T0_digitize.yaml config/T1_fig3.yaml ...
    python run_test.py all
每個 YAML 的 `test:` 欄位決定要跑哪個實驗（sim/experiments/tN.py 的 run(cfg_path)）。
"""
import importlib
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def _order(p):
    """依測試編號排序（T0, T1, …, T10, T11, T11b），T_summary 最後；字串排序會讓 T10 跑在 T4 之前。"""
    m = re.match(r"T(\d+)", p.name)
    return (int(m.group(1)) if m else 10 ** 6, p.name)


def main(argv):
    paths = argv[1:]
    if not paths or paths == ["all"]:
        paths = [str(p) for p in sorted((ROOT / "config").glob("T*.yaml"), key=_order)]
    for p in paths:
        spec = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        mod = importlib.import_module(f"sim.experiments.{spec['test'].lower()}")
        t0 = time.time()
        print(f"=== {spec['test']}  ({p})", flush=True)
        mod.run(Path(p))
        print(f"=== {spec['test']} done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv)
