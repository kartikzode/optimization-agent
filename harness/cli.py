"""python -m harness.cli --problem moe --submission problems/moe/submission.py --mode test"""
import argparse
import json
import logging
from pathlib import Path

from .evaluate import MODES, Evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--problem", required=True)
ap.add_argument("--submission", required=True)
ap.add_argument("--mode", choices=MODES, default="test")
ap.add_argument("-v", "--verbose", action="store_true", help="stream Modal's container logs and print the raw evaluator output")
ap.add_argument("--json", action="store_true", help="print the full result dict")
a = ap.parse_args()
# only our own loggers get DEBUG: root-level DEBUG makes Modal's gRPC client log request headers, which carry the API token
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
_log = logging.getLogger("optagent")
_log.addHandler(_h)
_log.setLevel(logging.DEBUG if a.verbose else logging.INFO)
_log.propagate = False

with Evaluator(verbose=a.verbose) as ev:
    res = ev.evaluate(a.problem, Path(a.submission).read_text(), a.mode)

if a.verbose:
    print(res.get("log_text", ""))
res.pop("log_text", None)
if a.json:
    print(json.dumps(res, indent=2))
else:
    print(f"{res['problem']}/{res['mode']}: {res['status']}  ({res['wall_s']:.0f}s)")
    for t in res["tests"]:
        print(f"  test  {t['status']:8} {t['spec']}")
    for b in res["benchmarks"]:
        ms = f"{b['mean_ns'] / 1e6:.3f} ms" if b["mean_ns"] else b["status"]
        print(f"  bench {ms:>10} {b['spec']}")
    if res["geomean_ns"]:
        print(f"  geomean {res['geomean_ns'] / 1e6:.3f} ms")
    for p in res["profiles"]:
        print(f"--- profile {p['spec']}\n{p['report']}")
    if res["error"]:
        print(f"\nERROR:\n{res['error']}")
raise SystemExit(0 if res["status"] == "pass" else 1)
