"""
Python API the optimizer's tools call:

    with Evaluator() as ev:
        res = ev.evaluate("moe", code, mode="test")                      # one stage, submission.py only
        out = ev.run_candidate("moe", files, profile=True)               # test -> benchmark -> ncu, one GPU container
        out = ev.compile("moe", files)                                   # build on a CPU container, no GPU

Hold one Evaluator open for a whole submit: it keeps a single Modal app session alive.
"""
import contextlib
import logging
import time
from pathlib import Path

import yaml

from .parse import parse_results

log = logging.getLogger("optagent.harness")

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
MODES = ("test", "benchmark", "leaderboard", "profile")
REQUIRED = ("reference.py", "task.py", "task.yml")
NCU_TIMEOUT_S = 1500
LIGHT_METRICS = "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread"
LIGHT_LAUNCHES = 3000


def list_problems() -> list[str]:
    return sorted(p.name for p in PROBLEMS.iterdir() if p.is_dir() and all((p / f).exists() for f in REQUIRED))


def load_task(problem: str) -> dict:
    return yaml.safe_load((PROBLEMS / problem / "task.yml").read_text())


def tests_txt(task: dict, mode: str, only: list[int] | None = None) -> str:
    """task.yml cases -> the `k: v; k: v` lines eval.py reads (tests for `test`, benchmarks otherwise)."""
    cases = task["tests" if mode == "test" else "benchmarks"]
    if only is not None:
        cases = [cases[i] for i in only]
    return "\n".join("; ".join(f"{k}: {v}" for k, v in c.items()) for c in cases)


def timeout_for(task: dict, mode: str) -> int:
    key = {"test": "test_timeout", "benchmark": "benchmark_timeout"}.get(mode, "ranked_timeout")
    return int(task.get(key, 600))


def stage(problem: str, files: dict[str, str] | str) -> dict[str, str]:
    """Everything eval.py needs, as name -> source. The candidate controls only its own files
    (submission.py plus any .cu/.h it ships); eval.py, utils.py, reference.py, task.py always come from problems/."""
    if isinstance(files, str):
        files = {"submission.py": files}
    p = PROBLEMS / problem
    sources = {"eval.py": PROBLEMS / "eval.py", "utils.py": PROBLEMS / "utils.py",
               "reference.py": p / "reference.py", "task.py": p / "task.py"}
    staged = {name: path.read_text() for name, path in sources.items()}
    for name, text in files.items():
        if name in staged or name == "tests.txt" or name.startswith(("/", "..")) or ".." in Path(name).parts:
            raise ValueError(f"candidate file '{name}' is not allowed")
        staged[name] = text
    return staged


def format_raw(raw: dict, header: str) -> str:
    """Everything the evaluator process produced, for the per-candidate log file."""
    return (f"# {header}\n# returncode={raw['returncode']} timed_out={raw['timed_out']} wall={raw['wall_s']:.1f}s\n\n"
            f"## results (eval.py's key: value output)\n{_redact(raw['results'])}\n\n"
            f"## stdout\n{raw['stdout']}\n\n## stderr\n{raw['stderr']}\n")


def _redact(results: str) -> str:
    return "\n".join(l.split(": ")[0] + f": <{len(l)} chars>" if ".report: " in l else l for l in results.splitlines())


def _app_stopped(e: Exception) -> bool:
    msg = str(e)
    return "is stopped" in msg or "APP_STATE_STOPPED" in msg


class Evaluator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose   # also stream the container's logs (image build, eval output) to this terminal

    def _start(self):
        import modal
        from . import modal_runner
        self._runner = modal_runner
        self._stack = contextlib.ExitStack()
        if self.verbose:
            self._stack.enter_context(modal.enable_output())
        self._stack.enter_context(modal_runner.app.run())
        log.debug("modal app started (gpu=%s)", modal_runner.GPU)

    def _restart(self):
        try:
            self._stack.close()
        except Exception:
            log.debug("closing the stopped modal app failed", exc_info=True)
        self._start()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def _call(self, fn_name: str, *args):
        """Call a Modal function; if the app was stopped under us, restart it and retry once."""
        try:
            return getattr(self._runner, fn_name).remote(*args)
        except Exception as e:
            if not _app_stopped(e):
                raise
            log.warning("the modal app was stopped under us (%s); restarting it and retrying once", e)
            self._restart()
            return getattr(self._runner, fn_name).remote(*args)

    def _remote(self, *args):
        return self._call("run_eval", *args)

    # ---- single stage -------------------------------------------------------------------
    def evaluate(self, problem: str, code: str | dict, mode: str = "test", secret_seed: str | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        task = load_task(problem)
        timeout = timeout_for(task, mode)
        files = stage(problem, code)
        log.debug("modal %s/%s: sending %d files (%d bytes), timeout %ds", problem, mode, len(files),
                  sum(map(len, files.values())), timeout)
        t0 = time.time()
        raw = self._remote(files, mode, tests_txt(task, mode), timeout, secret_seed)
        res = self._parse(raw)
        res.update(problem=problem, mode=mode, wall_s=raw["wall_s"], round_trip_s=time.time() - t0)
        res["log_text"] = format_raw(raw, f"{problem} {mode}")
        self._log_result(problem, mode, res, raw)
        return res

    @staticmethod
    def _parse(raw: dict) -> dict:
        return parse_results(raw["results"], raw["returncode"], raw["stdout"], raw["stderr"], raw["timed_out"])

    @staticmethod
    def _log_result(problem, mode, res, raw):
        head = " ".join((res["error"] or "").split())[:200]
        msg = "%s/%s -> %s (exit %s, eval %.0fs)%s"
        args = (problem, mode, res["status"], raw["returncode"], raw["wall_s"], f": {head}" if head else "")
        (log.info if res["status"] == "pass" else log.warning)(msg, *args)

    # ---- the full pipeline --------------------------------------------------------------
    def pick_shape(self, task: dict, shape: str | int = "last") -> int:
        n = len(task["benchmarks"])
        idx = n - 1 if shape in ("last", None) else 0 if shape == "first" else int(shape)
        if not 0 <= idx < n:
            raise ValueError(f"shape index {idx} out of range (problem has {n} benchmark shapes)")
        return idx

    def run_candidate(self, problem: str, files: dict[str, str], *, profile: bool = True, shape: str | int = "last",
                      ncu_set: str = "detailed", launch_count: int = 40, kernel_regex: str | None = None,
                      secret_seed: str | None = None, profile_only: bool = False) -> dict:
        """
        test -> benchmark -> (only if both pass) Nsight Compute on one benchmark shape, in one GPU container.
        -> {"test": parsed, "benchmark": parsed, "profile": {...} | None, "logs": {name: text}, "gpu_info": str}
        Latency numbers come from the benchmark stage, which never runs under a profiler.
        profile_only skips test/benchmark (re-profiling a candidate that already passed them).
        """
        task = load_task(problem)
        staged = stage(problem, files)
        idx = self.pick_shape(task, shape)
        one = tests_txt(task, "benchmark", [idx])
        plan = [
            {"name": "test", "mode": "test", "tests": tests_txt(task, "test"), "timeout_s": timeout_for(task, "test")},
            {"name": "benchmark", "mode": "benchmark", "tests": tests_txt(task, "benchmark"),
             "timeout_s": timeout_for(task, "benchmark"), "requires": ["test"]},
        ]
        need = [] if profile_only else ["test", "benchmark"]
        if profile_only:
            plan = []
        if profile:
            plan += [
                # pass 1 (cheap): time, DRAM bytes and registers of EVERY launch in the call -> exact totals
                {"name": "ncu_light", "mode": "profile", "tests": one, "timeout_s": NCU_TIMEOUT_S, "requires": need,
                 "ncu": {"metrics": LIGHT_METRICS, "launch_count": LIGHT_LAUNCHES, "prefix": "ncu_light",
                         "keep_rep": False, "export": ["raw"], "kernel_regex": kernel_regex}},
                # pass 2 (detailed): sections for the first `launch_count` launches -> why each kernel is slow
                {"name": "ncu", "mode": "profile", "tests": one, "timeout_s": NCU_TIMEOUT_S, "requires": need,
                 "ncu": {"set": ncu_set, "launch_count": launch_count, "kernel_regex": kernel_regex}},
                {"name": "torch_profile", "mode": "profile", "tests": one, "timeout_s": timeout_for(task, "benchmark"),
                 "requires": need, "only_if_missing": "ncu"},
            ]
        t0 = time.time()
        out = self._call("run_pipeline", staged, plan, secret_seed)
        stages = out["stages"]

        result: dict = {"gpu_info": out.get("gpu_info", ""), "logs": {}, "round_trip_s": time.time() - t0,
                        "shape_index": idx, "shape_spec": one, "test": None, "benchmark": None, "profile": None}
        for name in ("test", "benchmark"):
            raw = stages.get(name)
            if raw and not raw.get("skipped"):
                res = self._parse(raw)
                res.update(problem=problem, mode=name, wall_s=raw["wall_s"])
                result[name] = res
                result["logs"][f"{name}.log"] = format_raw(raw, f"{problem} {name}")
                self._log_result(problem, name, res, raw)
        if profile:
            result["profile"] = self._profile_result(stages, one)
            for key, fname in (("ncu_light", "profile_light.log"), ("ncu", "profile.log"), ("torch_profile", "torch_profile.log")):
                raw = stages.get(key)
                if raw and not raw.get("skipped"):
                    result["logs"][fname] = format_raw(raw, f"{problem} {key}")
        return result

    @staticmethod
    def _profile_result(stages: dict, spec: str) -> dict:
        ncu, tp = stages.get("ncu", {"skipped": True}), stages.get("torch_profile", {"skipped": True})
        light = stages.get("ncu_light", {"skipped": True})
        base = {"spec": spec, "artifacts": {}, "torch_table": "", "status": "skipped", "note": ""}
        if ncu.get("skipped") and tp.get("skipped") and light.get("skipped"):
            base["note"] = "profiling skipped: the test or benchmark stage did not pass"
            return base
        arts = {**(light.get("artifacts") or {}), **(ncu.get("artifacts") or {})}
        if ncu.get("artifacts"):
            base.update(status="ok", artifacts=arts)
            return base
        if light.get("artifacts"):      # the detailed pass failed but the totals are complete
            base.update(status="light_only", artifacts=arts,
                        note="detailed Nsight Compute pass produced no report; totals come from the light pass")
            return base
        text = (ncu.get("stderr") or "") + "\n" + (ncu.get("stdout") or "")
        errs = [l.strip() for l in text.splitlines() if "ERROR" in l or "Error" in l]
        why = (" | ".join(errs[:4]) if errs else text.strip()[-600:]) or "no report written"
        why = why[:700]
        base.update(status="ncu_unavailable", note=f"Nsight Compute produced no report (exit {ncu.get('returncode')}): {why}")
        if tp and not tp.get("skipped"):
            parsed = parse_results(tp["results"], tp["returncode"], tp["stdout"], tp["stderr"], tp["timed_out"])
            base["torch_table"] = "\n\n".join(p["report"] for p in parsed["profiles"])
        return base

    # ---- compile only -------------------------------------------------------------------
    def compile(self, problem: str, files: dict[str, str]) -> dict:
        staged = stage(problem, files)
        raw = self._call("run_compile", staged)
        raw["ok"] = raw["returncode"] == 0 and not raw["timed_out"]
        return raw
