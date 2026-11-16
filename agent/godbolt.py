"""
Compile CUDA C++ on Compiler Explorer (godbolt.org) via its public REST API.

Godbolt has nvcc but no torch or Triton, so it can only judge standalone CUDA source.
Note: whatever is compiled here is sent to a third-party service.
"""
import json
import re
import urllib.request

API = "https://godbolt.org/api"
_compiler_id: str | None = None


def _http(path: str, payload: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Accept": "application/json", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def resolve_compiler() -> str:
    """Newest plain 'NVCC x.y.z' on godbolt."""
    global _compiler_id
    if _compiler_id is None:
        best = (0, 0, 0), "nvcc130u2"
        for c in _http("/compilers/cuda?fields=id,name"):
            m = re.fullmatch(r"NVCC (\d+)\.(\d+)\.(\d+)", c["name"])
            if m and tuple(map(int, m.groups())) > best[0]:
                best = tuple(map(int, m.groups())), c["id"]
        _compiler_id = best[1]
    return _compiler_id


def compile_cuda(source: str, arch: str = "sm_90a", timeout: float = 90) -> dict:
    """-> {"status": "ok" | "error" | "unavailable", "errors": str, "compiler": id}"""
    try:
        cid = resolve_compiler()
        r = _http(f"/compiler/{cid}/compile", {
            "source": source, "lang": "cuda",
            "options": {"userArguments": f"-arch={arch} -std=c++17 -c",
                        "filters": {"binary": False, "execute": False, "intel": False, "demangle": False,
                                    "labels": False, "directives": True, "commentOnly": True}}}, timeout)
    except Exception as e:  # network down, rate limited, API change: never block the pipeline on it
        return {"status": "unavailable", "errors": f"{type(e).__name__}: {e}", "compiler": _compiler_id}
    if r.get("code") == 0:
        return {"status": "ok", "errors": "", "compiler": cid}
    text = "\n".join(l.get("text", "") for l in r.get("stderr", []))
    return {"status": "error", "errors": text.strip()[:3000], "compiler": cid}
