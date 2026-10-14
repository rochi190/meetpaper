"""
執行測試：
    python run_test.py config/T4_fig6.yaml
    python run_test.py config/T0_digitize.yaml config/T1_fig3.yaml ...
    python run_test.py all
每個 YAML 的 `test:` 欄位決定要跑哪個實驗（sim/experiments/tN.py 的 run(cfg_path)）。
"""
import importlib
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main(argv):
    paths = argv[1:]
    if not paths or paths == ["all"]:
        paths = sorted(str(p) for p in (ROOT / "config").glob("T*.yaml"))
    for p in paths:
        spec = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        mod = importlib.import_module(f"sim.experiments.{spec['test'].lower()}")
        t0 = time.time()
        print(f"=== {spec['test']}  ({p})", flush=True)
        mod.run(Path(p))
        print(f"=== {spec['test']} done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv)
