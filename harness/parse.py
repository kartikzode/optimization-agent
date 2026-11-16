"""Turn eval.py's `key: value` lines (+ exit code / stderr) into a structured result dict."""
import base64
import math
import re

STDERR_TAIL = 3000   # chars of stderr/stdout kept; the end of a traceback is the useful part
ERROR_TAIL = 1500    # chars of any single error message kept


def _tail(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "..." + s[-n:]


_REMOTE = re.compile(r'RemoteTraceback:\s*\n"""\n(.*?)\n"""', re.S)
_FRAME = re.compile(r'^  File "([^"]+)", line \d+')


def condense_traceback(stderr: str) -> str:
    """
    eval.py runs the kernel in a pool worker, so a kernel exception arrives wrapped in a
    multiprocessing RemoteTraceback followed by eval.py's own re-raise. Keep only the worker's
    traceback (the part that mentions submission.py), minus multiprocessing frames, with the
    temp dir stripped. Falls back to the plain tail when the shape is different.
    """
    m = _REMOTE.search(stderr or "")
    if not m:
        return _tail(stderr, STDERR_TAIL)
    kept, skip = [], False
    for line in m.group(1).splitlines():
        f = _FRAME.match(line)
        if f:
            skip = "multiprocessing/" in f.group(1)
        if not skip:
            kept.append(re.sub(r"/tmp/tmp\w+/", "", line))
    return _tail("\n".join(kept), STDERR_TAIL)


def _kv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition(": ")
        if sep:
            out[k.strip()] = v.strip()
    return out


def _indexed(kv: dict[str, str], prefix: str, count: int) -> list[dict[str, str]]:
    rows = [{} for _ in range(count)]
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(\w+)$")
    for k, v in kv.items():
        m = pat.match(k)
        if m and int(m[1]) < count:
            rows[int(m[1])][m[2]] = v
    return rows


def parse_results(text: str, returncode: int, stdout: str = "", stderr: str = "",
                  timed_out: bool = False) -> dict:
    """
    status:
      pass     eval.py finished and every case passed
      fail     eval.py finished and reported a wrong result / error for some case
      crash    eval.py died before finishing (exception in the kernel, CUDA error, OOM, ...)
      timeout  the run exceeded its wall-clock limit
    """
    kv = _kv(text)
    n_tests = int(kv.get("test-count", 0))
    n_bench = int(kv.get("benchmark-count", 0))

    tests = []
    for r in _indexed(kv, "test", n_tests):
        tests.append({"spec": r.get("spec"), "status": r.get("status", "not_run"),
                      "error": _tail(r["error"], ERROR_TAIL) if "error" in r else None})

    benchmarks, profiles = [], []
    for r in _indexed(kv, "benchmark", n_bench):
        if "report" in r:
            report = base64.b64decode(r["report"], altchars=b"+*").decode("utf-8", "replace")
            profiles.append({"spec": r.get("spec"), "report": report})
            continue
        ok = "mean" in r
        benchmarks.append({
            "spec": r.get("spec"),
            "status": "pass" if ok else r.get("status", "not_run"),
            "runs": int(float(r["runs"])) if ok else None,
            "mean_ns": float(r["mean"]) if ok else None,
            "best_ns": float(r["best"]) if ok else None,
            "std_ns": float(r["std"]) if ok else None,
            "error": _tail(r["error"], ERROR_TAIL) if "error" in r else None,
        })

    means = [b["mean_ns"] for b in benchmarks if b["mean_ns"]]
    geomean = math.exp(sum(map(math.log, means)) / len(means)) if means and len(means) == len(benchmarks) else None

    check = kv.get("check")
    if timed_out:
        status = "timeout"
    elif check == "pass" and returncode == 0:
        status = "pass"
    elif check == "fail":
        status = "fail"
    else:
        status = "crash"

    error = None
    if status == "timeout":
        error = "evaluation exceeded its wall-clock limit (infinite loop, deadlock, or far too slow)"
    elif status == "fail":
        failed = [x for x in tests + benchmarks if x["status"] not in ("pass", "not_run") and x["error"]]
        error = failed[0]["error"] if failed else "check failed"
    elif status == "crash":
        error = condense_traceback(stderr) or f"eval.py exited with code {returncode} and no output"

    return {
        "status": status,
        "returncode": returncode,
        "tests": tests,
        "benchmarks": benchmarks,
        "geomean_ns": geomean,
        "profiles": profiles,
        "error": error,
        "stdout_tail": _tail(stdout, STDERR_TAIL),
        "stderr_tail": _tail(stderr, STDERR_TAIL),
    }
