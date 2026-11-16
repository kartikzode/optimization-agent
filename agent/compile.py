"""
Compile checks, so a worker finds syntax / template / resource problems before a GPU run is spent.

  local   nvcc on this machine, for .cu files that do not need torch headers. Free and takes seconds; reports
          registers, smem, spills per kernel (ptxas -v). Cross-compiles for the target arch, no local GPU needed.
  modal   build the whole candidate (import submission.py) on a CPU container with torch + nvcc. Required when
          the CUDA source includes torch/ATen headers or is passed to load_inline / cpp_extension.load.

`auto` picks local when every .cu file is torch-free and nvcc exists, else modal.
"""
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .guard import _strip_comments

L = logging.getLogger("optagent.compile")

TORCH_MARKERS = re.compile(r'#\s*include\s*[<"](torch|ATen|c10|pybind11)/|\btorch::|\bat::|\bc10::|\bTORCH_[A-Z_]+|\bPYBIND11_')
NVCC_FLAGS = ["-std=c++17", "-O3", "-lineinfo", "-Xptxas", "-v"]


def needs_torch(src: str) -> bool:
    """True when the source (ignoring comments) includes or uses torch/ATen/pybind11."""
    return bool(TORCH_MARKERS.search(_strip_comments(src)))


def parse_ptxas(text: str) -> list[dict]:
    """ptxas -v output -> [{kernel, arch, regs, smem, stack, spill_stores, spill_loads, cmem}]"""
    out: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if m:
            cur = {"kernel": m.group(1), "arch": m.group(2), "regs": None, "smem": 0, "stack": 0,
                   "spill_stores": 0, "spill_loads": 0}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            cur["stack"], cur["spill_stores"], cur["spill_loads"] = map(int, m.groups())
        m = re.search(r"Used (\d+) registers", line)
        if m:
            cur["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            if s:
                cur["smem"] = int(s.group(1))
    return out


def condense(log: str, max_lines: int = 60) -> str:
    """The lines that matter in a compiler log: errors with their source/caret context, else the tail."""
    lines = log.splitlines()
    keep: list[str] = []
    for i, l in enumerate(lines):
        if re.search(r"\berror\b|Error:|undefined reference|ninja: build stopped|FAILED:", l):
            keep += lines[i:i + 3]
    seen, uniq = set(), []
    for l in keep:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    chosen = uniq or lines[-max_lines:]
    return "\n".join(chosen[:max_lines])


def format_resources(res: list[dict]) -> str:
    if not res:
        return ""
    rows = ["kernel resources (ptxas):"]
    for r in res:
        warn = "  <-- SPILLS" if r["spill_stores"] or r["spill_loads"] else ""
        rows.append(f"  {r['kernel'][:70]}: {r['regs']} regs, {r['smem']} B smem, "
                    f"{r['spill_stores']}/{r['spill_loads']} B spill st/ld{warn}")
    return "\n".join(rows)


def gencode(arch: str) -> str:
    """`sm_90a` -> `-gencode=arch=compute_90a,code=sm_90a`. The `-arch=sm_90a` shorthand also embeds a plain compute_90 PTX
    pass in nvcc 13, and ptxas rejects wgmma/TMA-only instructions there, which blocked every Hopper-specific kernel."""
    m = re.fullmatch(r"sm_(\d+a?)", arch)
    return f"-gencode=arch=compute_{m.group(1)},code=sm_{m.group(1)}" if m else f"-arch={arch}"


def local_nvcc(files: dict[str, str], arch: str, timeout: int = 240) -> dict:
    """nvcc every torch-free .cu file. -> {ran, ok, skipped: [names], log, resources}"""
    nvcc = shutil.which("nvcc")
    cu = {n: t for n, t in files.items() if n.endswith(".cu")}
    skipped = [n for n, t in cu.items() if needs_torch(t)]
    todo = {n: t for n, t in cu.items() if n not in skipped}
    if not nvcc:
        return {"ran": False, "ok": True, "skipped": list(cu), "log": "nvcc not found on this machine", "resources": []}
    if not todo:
        return {"ran": False, "ok": True, "skipped": skipped, "log": "", "resources": []}
    logs, ok = [], True
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            p = Path(d, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        for rel in todo:
            cmd = [nvcc, gencode(arch), *NVCC_FLAGS, f"-I{d}", "-c", str(Path(d, rel)), "-o", str(Path(d, rel + ".o"))]
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=d)
                logs.append(f"$ {' '.join(cmd[:-3])} {rel}  (exit {r.returncode}, {time.time() - t0:.1f}s)\n"
                            f"{r.stdout}{r.stderr}")
                ok &= r.returncode == 0
            except subprocess.TimeoutExpired:
                logs.append(f"$ nvcc {rel}: TIMEOUT after {timeout}s")
                ok = False
    log = "\n".join(logs)
    return {"ran": True, "ok": ok, "skipped": skipped, "log": log, "resources": parse_ptxas(log)}


def modal_compile(evaluator, problem: str, files: dict[str, str]) -> dict:
    raw = evaluator.compile(problem, files)
    log = f"{raw['stdout']}\n{raw['stderr']}".strip()
    return {"ran": True, "ok": raw["ok"], "skipped": [], "log": log, "resources": parse_ptxas(log),
            "wall_s": raw["wall_s"], "timed_out": raw["timed_out"]}


def compile_candidate(files: dict[str, str], *, problem: str, arch: str, where: str = "auto", evaluator_factory=None) -> dict:
    """-> {ok, where, log, summary, resources}. `summary` is the short text a worker should read first."""
    cu = {n: t for n, t in files.items() if n.endswith(".cu")}
    torchy = any(needs_torch(t) for t in cu.values())
    if where == "local" or (where == "auto" and cu and not torchy and shutil.which("nvcc")):
        return _finish(local_nvcc(files, arch), "local")
    if evaluator_factory is None:
        raise RuntimeError("a Modal compile was needed but no evaluator is available")
    with evaluator_factory() as ev:
        return _finish(modal_compile(ev, problem, files), "modal")


def _finish(r: dict, where: str) -> dict:
    res = format_resources(r["resources"])
    if r["ok"]:
        summary = f"compiled OK ({where})" + (f"\n{res}" if res else "")
        if where == "local":
            summary += "\n(kernel code only; the full build with bindings is checked by `compile --where modal` and by submit)"
        if r.get("skipped"):
            summary += f"\nnot checked locally (need torch headers): {', '.join(r['skipped'])}"
    else:
        summary = f"COMPILE FAILED ({where})\n{condense(r['log'])}"
    return {"ok": r["ok"], "where": where, "log": r["log"], "summary": summary, "resources": r["resources"]}
