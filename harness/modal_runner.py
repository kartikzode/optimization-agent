"""
Remote side: run problems/eval.py against a staged candidate on a Modal GPU.

Keep this module's imports light (modal + stdlib only): Modal imports it again inside the
container, where yaml/dotenv don't exist. Callers should use harness.evaluate, not this.

Functions
  run_pipeline  one GPU container, several stages in order: test -> benchmark -> ncu (profile).
                Latency is measured with NO profiler attached; Nsight Compute only runs afterwards, and only
                when the earlier stages passed. Sharing the container also shares the nvcc build cache, so a
                CUDA extension is compiled once per candidate, not once per stage.
  run_compile   CPU container: build the candidate (import submission.py) and return nvcc's output.
  run_eval      one stage on its own (used by harness.cli and the tests).
"""
import os

import modal

try:  # local only; harmless if absent (e.g. inside the container)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):  # blank values would shadow ~/.modal.toml
    if not os.environ.get(_k):
        os.environ.pop(_k, None)

GPU = os.environ.get("MODAL_GPU") or "H100"
_ARCH = {"H100": "9.0a", "H200": "9.0a", "B200": "10.0a", "A100": "8.0", "L4": "8.9"}   # TORCH_CUDA_ARCH_LIST
TORCH_ARCH = _ARCH.get(GPU.upper(), "9.0a")

# CUDA *devel* image: nvcc, ninja and gcc for torch extensions (and Triton's launcher stubs), plus Nsight Compute.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # silence the base image's banner
    .apt_install("cuda-nsight-compute-13-0")
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("numpy", "ninja")
    # build for the real target arch (sm_90a gets wgmma/TMA); without this torch picks plain compute_90
    .env({"TORCH_CUDA_ARCH_LIST": TORCH_ARCH, "CUDA_HOME": "/usr/local/cuda"})
)
app = modal.App("optimization-agent", image=image)


def _redact_reports(results: str) -> str:
    """Profile reports are large base64 blobs; keep their size, not their content, in the log."""
    return "\n".join(l.split(": ")[0] + f": <{len(l)} chars>" if ".report: " in l else l for l in results.splitlines())


def _find_ncu() -> str | None:
    import glob
    import shutil
    found = shutil.which("ncu")
    if found:
        return found
    for pat in ("/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda*/bin/ncu", "/usr/local/cuda*/nsight-compute*/ncu"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _ncu_export(ncu: str, rep: str, page: str) -> str:
    import subprocess
    r = subprocess.run([ncu, "--import", rep, "--page", page, "--csv", "--print-units", "base"],
                       capture_output=True, text=True, timeout=300)
    return r.stdout if r.returncode == 0 else f"(ncu --page {page} failed, exit {r.returncode})\n{r.stderr[-2000:]}"


def _eval_process(files: dict[str, str], mode: str, tests: str, timeout_s: int, secret_seed: str | None = None,
                  ncu: dict | None = None) -> dict:
    """Run eval.py once in a scratch dir. With `ncu` the evaluator runs under Nsight Compute (profile mode only)."""
    import signal, subprocess, sys, tempfile, threading, time
    from pathlib import Path

    def tee(stream, sink, buf):
        # echo every line to this container's own stdout/stderr (what the Modal dashboard shows)
        # and keep a copy to return to the caller
        for line in iter(stream.readline, ""):
            buf.append(line)
            sink.write(line)
            sink.flush()

    t0 = time.time()
    label = f"{mode}{'+ncu' if ncu else ''}"
    print(f"[eval] start mode={label} timeout={timeout_s}s files={sorted(files)}", flush=True)
    out: list[str] = []
    err: list[str] = []
    res: list[str] = []
    artifacts: dict[str, object] = {}
    timed_out = False
    ncu_bin = _find_ncu() if ncu else None
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            p = Path(d, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        Path(d, "tests.txt").write_text(tests)

        cmd = [sys.executable, "eval.py", mode, "tests.txt"]
        if ncu:
            if not ncu_bin:
                return {"returncode": 127, "results": "", "stdout": "", "stderr": "ncu is not installed in this image",
                        "timed_out": False, "wall_s": 0.0, "artifacts": {}}
            # --clock-control none: a Modal container may not lock GPU clocks ("Failed to lock GPU clock frequencies"),
            # so profile at whatever clocks the GPU runs. Durations are then slightly noisy; ratios stay meaningful.
            pre = [ncu_bin, "--target-processes", "all", "--nvtx", "--nvtx-include", "custom_kernel/",
                   "--clock-control", "none"]
            if ncu.get("metrics"):      # a light pass: just these counters, over (nearly) every launch
                pre += ["--metrics", ncu["metrics"]]
            else:                       # a detailed pass: a section set, over the first launch_count launches
                pre += ["--set", ncu.get("set", "detailed")]
            pre += ["--launch-count", str(ncu.get("launch_count", 40)), "-f", "-o", str(Path(d, ncu.get("prefix", "ncu")))]
            if ncu.get("kernel_regex"):
                pre += ["--kernel-name", f"regex:{ncu['kernel_regex']}"]
            cmd = pre + cmd

        r, w = os.pipe()
        env = {**os.environ, "EVAL_FD": str(w)}
        if secret_seed:
            env["EVAL_SEED"] = secret_seed
        if ncu:
            env["EVAL_NCU"] = "1"

        proc = subprocess.Popen(cmd, cwd=d, env=env, pass_fds=[w],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                start_new_session=True)   # own process group, so a timeout can kill the pool worker too
        os.close(w)   # the child has its own copy; closing ours lets the reader see EOF when it exits

        def read_results():
            with os.fdopen(r, "r") as f:
                res.append(f.read())

        threads = [threading.Thread(target=tee, args=(proc.stdout, sys.stdout, out)),
                   threading.Thread(target=tee, args=(proc.stderr, sys.stderr, err)),
                   threading.Thread(target=read_results)]
        for t in threads:
            t.start()
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"[eval] TIMEOUT after {timeout_s}s: killing the evaluator's process group", flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
        for t in threads:
            t.join(timeout=15)

        prefix = (ncu or {}).get("prefix", "ncu")
        rep = Path(d, f"{prefix}.ncu-rep")
        if ncu and rep.exists():
            if ncu.get("keep_rep", True):
                artifacts[f"{prefix}.ncu-rep"] = rep.read_bytes()
            for page in ncu.get("export", ["raw", "details"]):
                artifacts[f"{prefix}_{page}.csv"] = _ncu_export(ncu_bin, str(rep), page)
            if "details" in ncu.get("export", ["raw", "details"]):
                # the human-readable report, with NVIDIA's own OPT/WRN analysis messages; readable in a terminal
                import subprocess as _sp
                r = _sp.run([ncu_bin, "--import", str(rep), "--page", "details"], capture_output=True, text=True, timeout=300)
                artifacts[f"{prefix}_details.txt"] = r.stdout if r.returncode == 0 else f"(ncu text export failed: {r.stderr[-500:]})"

    results = "".join(res)
    wall = time.time() - t0
    print(f"[eval] done mode={label} exit={rc} timed_out={timed_out} wall={wall:.1f}s\n"
          f"[eval] results:\n{_redact_reports(results)}", flush=True)
    return {"returncode": rc, "results": results, "stdout": "".join(out), "stderr": "".join(err),
            "timed_out": timed_out, "wall_s": wall, "artifacts": artifacts}


def _passed(raw: dict) -> bool:
    return raw["returncode"] == 0 and not raw["timed_out"] and "check: pass" in raw["results"]


@app.function(gpu=GPU, timeout=60 * 30, retries=0)
def run_eval(files: dict[str, str], mode: str, tests: str, timeout_s: int,
             secret_seed: str | None = None, ncu: dict | None = None) -> dict:
    """One stage. files: name -> source (eval.py, utils.py, reference.py, task.py, submission.py, ...)."""
    return _eval_process(files, mode, tests, timeout_s, secret_seed, ncu)


@app.function(gpu=GPU, timeout=60 * 60, retries=0)
def run_pipeline(files: dict[str, str], plan: list[dict], secret_seed: str | None = None) -> dict:
    """
    plan: [{name, mode, tests, timeout_s, requires: [stage names that must have passed],
            ncu: {...} | None, only_if_missing: stage name | None}]
    A stage whose requirements did not pass is skipped. `only_if_missing` runs a fallback stage only when the
    named stage produced no artifacts (e.g. the torch profiler table when Nsight Compute is unavailable).
    """
    import subprocess
    gpu_info = ""
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total",
             "--format=csv"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        pass
    stages: dict[str, dict] = {}
    for st in plan:
        name = st["name"]
        if any(not stages.get(r, {}).get("passed") for r in st.get("requires", [])):
            stages[name] = {"skipped": True, "passed": False}
            continue
        fb = st.get("only_if_missing")
        if fb and stages.get(fb, {}).get("artifacts"):
            stages[name] = {"skipped": True, "passed": False}
            continue
        raw = _eval_process(files, st["mode"], st["tests"], st["timeout_s"], secret_seed, st.get("ncu"))
        raw["passed"] = _passed(raw)
        stages[name] = raw
    return {"stages": stages, "gpu_info": gpu_info}


@app.function(cpu=8, timeout=60 * 20, retries=0)
def run_compile(files: dict[str, str], timeout_s: int = 900) -> dict:
    """Build the candidate on a CPU container (no GPU needed to compile): import submission.py and report nvcc's output."""
    import subprocess, sys, tempfile, time
    from pathlib import Path
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            p = Path(d, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        env = {**os.environ, "MAX_JOBS": "8", "PYTHONUNBUFFERED": "1"}
        try:
            r = subprocess.run([sys.executable, "-c", "import submission"], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout_s)
            rc, out, err, timed_out = r.returncode, r.stdout, r.stderr, False
        except subprocess.TimeoutExpired as e:
            rc, out, err, timed_out = -1, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
                (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""), True
    print(f"[compile] exit={rc} timed_out={timed_out} wall={time.time() - t0:.1f}s", flush=True)
    return {"returncode": rc, "stdout": out, "stderr": err, "timed_out": timed_out, "wall_s": time.time() - t0}
