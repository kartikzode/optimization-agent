"""
Everything that can reject a candidate before it costs a Modal GPU run:
  1. every .py file compiles
  2. guard: contract + reward-hacking patterns, in Python and in the native sources
  3. local nvcc on torch-free .cu files (fast, free, shows registers/smem/spills)

A candidate that includes torch headers cannot be judged here; `optk compile --where modal` (a CPU container
with torch + nvcc) and the GPU run itself do that. Godbolt (a third-party compile service that receives your
source) is only used with use_godbolt=True, on CUDA strings embedded in submission.py.
"""
import ast
import logging

from . import compile as cc
from . import godbolt, guard

log = logging.getLogger("optagent.precheck")

CUDA_PRELUDE = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n"


def _module_strings(tree: ast.Module) -> dict[str, str]:
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            out[n.targets[0].id] = n.value.value
    return out


def extract_cuda_sources(code: str) -> list[str]:
    """String literals (or module-level string constants) given as cuda_sources to load_inline."""
    tree = ast.parse(code)
    consts = _module_strings(tree)

    def strings(node) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name) and node.id in consts:
            return [consts[node.id]]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [s for e in node.elts for s in strings(e)]
        return []

    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "load_inline":
            kw = next((k.value for k in n.keywords if k.arg == "cuda_sources"), None)
            if kw is None and len(n.args) > 2:
                kw = n.args[2]
            if kw is not None:
                found += strings(kw)
    return found


def precheck(files: dict[str, str] | str, *, use_godbolt: bool = False, arch: str = "sm_90a",
             use_local_nvcc: bool = True) -> dict:
    """-> {"ok": bool, "stage": "python"|"guard"|"nvcc"|"godbolt"|"passed", "errors": [str], "notes": [str], "resources": [...]}"""
    if isinstance(files, str):
        files = {"submission.py": files}
    notes: list[str] = []
    for name, src in files.items():
        if name.endswith(".py"):
            try:
                compile(src, name, "exec")
            except SyntaxError as e:
                return {"ok": False, "stage": "python", "errors": [f"{name}: SyntaxError line {e.lineno}: {e.msg}"],
                        "notes": notes, "resources": []}

    violations = guard.check_files(files)
    if violations:
        return {"ok": False, "stage": "guard", "errors": [str(v) for v in violations], "notes": notes, "resources": []}

    resources: list[dict] = []
    cu = [n for n in files if n.endswith(".cu")]
    if use_local_nvcc and cu:
        r = cc.local_nvcc(files, arch)
        if r["ran"]:
            resources = r["resources"]
            if not r["ok"]:
                return {"ok": False, "stage": "nvcc", "errors": [f"nvcc (local, {cc.gencode(arch)}) failed:\n{cc.condense(r['log'])}"],
                        "notes": notes, "resources": resources}
            notes.append(f"nvcc (local, {arch}) compiled: {', '.join(n for n in cu if n not in r['skipped'])}")
        if r["skipped"]:
            notes.append(f"{', '.join(r['skipped'])} need torch headers: compiled on the GPU run instead")
        if not r["ran"] and not r["skipped"]:
            notes.append(r["log"])
    elif not cu:
        notes.append("no .cu files (Triton/PyTorch or CUDA embedded in submission.py); compile happens on the GPU run")

    if use_godbolt:
        for i, src in enumerate(extract_cuda_sources(files["submission.py"])):
            if cc.needs_torch(src):
                notes.append(f"embedded cuda source {i} uses torch headers; Godbolt cannot compile it, skipped")
                continue
            r = godbolt.compile_cuda(CUDA_PRELUDE + src, arch)
            if r["status"] == "error":
                return {"ok": False, "stage": "godbolt", "errors": [
                    f"nvcc ({r['compiler']}, -arch={arch}) failed on embedded cuda source {i}:\n{r['errors']}"],
                    "notes": notes, "resources": resources}
            notes.append(f"embedded cuda source {i}: nvcc {r['status']} ({r['compiler']})")
    return {"ok": True, "stage": "passed", "errors": [], "notes": notes, "resources": resources}
