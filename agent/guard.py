"""
Static check that rejects submissions using known reward-hacking or sandbox-escape patterns.

A tripwire for an LLM that is optimizing a score, not a security boundary: the evaluator also runs
the submission in a separate process, regenerates inputs and hides the secret seed.
"""
import ast
from dataclasses import dataclass

BANNED_MODULES = {
    "reference", "eval", "utils",                       # the checker and the timing harness
    "subprocess", "socket", "requests", "urllib", "http", "ctypes",
    "threading", "multiprocessing", "concurrent",       # work that escapes the timed region
    "gc", "inspect", "importlib", "builtins", "pickle", "shutil",
}
BANNED_CALLS = {"eval", "exec", "__import__", "open", "globals", "breakpoint", "input"}
BANNED_ATTRS = {
    "os.environ", "os.getenv", "os.putenv", "os.system", "os.popen", "os.kill", "os.fork",
    "sys.modules", "sys._getframe", "sys.settrace", "sys.setprofile",
    "torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.set_stream",  # side streams dodge the timing events
    "torch.cuda.Event",
}
# assigning into these rewrites the harness's view of the world
PROTECTED = ("torch.cuda", "torch.Tensor", "torch.testing", "torch.allclose", "torch.isclose",
             "torch.equal", "torch.nn", "time", "sys", "os", "builtins")


@dataclass(frozen=True)
class Violation:
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.detail}"


def _protected(name: str | None) -> bool:
    return bool(name) and any(name == p or name.startswith(p + ".") for p in PROTECTED)


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.aliases: dict[str, str] = {}
        self.violations: list[Violation] = []

    def flag(self, node, rule, detail):
        self.violations.append(Violation(getattr(node, "lineno", 0), rule, detail))

    def resolve(self, node) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        return ".".join([self.aliases.get(node.id, node.id), *reversed(parts)])

    def _module(self, node, name):
        if name.split(".")[0] in BANNED_MODULES:
            self.flag(node, "import", f"import of '{name}' is not allowed")

    def visit_Import(self, node):
        for a in node.names:
            self.aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            self._module(node, a.name)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        self._module(node, mod)
        for a in node.names:
            full = f"{mod}.{a.name}"
            self.aliases[a.asname or a.name] = full
            if full in BANNED_ATTRS:
                self.flag(node, "attr", f"'{full}' is not allowed")

    def visit_Attribute(self, node):
        name = self.resolve(node)
        if name in BANNED_ATTRS:
            self.flag(node, "attr", f"'{name}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in BANNED_CALLS and f.id not in self.aliases:
            self.flag(node, "call", f"'{f.id}()' is not allowed")
        if isinstance(f, ast.Name) and f.id in ("setattr", "delattr") and node.args \
                and _protected(self.resolve(node.args[0])):
            self.flag(node, "monkeypatch", f"{f.id}() on a protected module")
        self.generic_visit(node)

    def _assign(self, node, targets):
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute) and _protected(self.resolve(sub)):
                    self.flag(node, "monkeypatch", f"assignment to '{self.resolve(sub)}' is not allowed")
                    break

    def visit_Assign(self, node):
        self._assign(node, node.targets)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._assign(node, [node.target])
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._assign(node, [node.target])
        self.generic_visit(node)


def _defines_custom_kernel(tree: ast.Module) -> bool:
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "custom_kernel":
            return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "custom_kernel" for t in n.targets):
            return True
        if isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == "custom_kernel" for a in n.names):
            return True
    return False


def check(code: str) -> list[Violation]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Violation(e.lineno or 0, "syntax", e.msg)]
    v = _Visitor()
    v.visit(tree)
    out = v.violations
    if not _defines_custom_kernel(tree):
        out.append(Violation(1, "contract", "module must define custom_kernel(data)"))
    return sorted(out, key=lambda x: x.line)


# ---- native sources (.cu / .cuh / .cpp / .h) ---------------------------------------------------
# Same idea as the Python guard: a tripwire against making the kernel look fast without doing the work
# (side streams dodge the timing events; threads/processes/files/env escape the timed region or read the checker).
import re  # noqa: E402

NATIVE_BANNED = [
    (r"\bcudaStreamCreate\w*|\bcudaStreamNonBlocking\b|getStreamFromPool|\bcudaStreamCreateWithPriority\b",
     "creating CUDA streams (work on a side stream escapes the timing events)"),
    (r"\bcudaLaunchHostFunc\b|\bcudaStreamAddCallback\b", "host callbacks"),
    (r"\bstd::thread\b|\bstd::async\b|\bpthread_\w+|\bfork\s*\(|\bvfork\s*\(", "threads / forking"),
    (r"\bsystem\s*\(|\bpopen\s*\(|\bexecv\w*\s*\(|\bdlopen\s*\(|\bdlsym\s*\(", "spawning processes / loading libraries"),
    (r"\bgetenv\s*\(|\bsecure_getenv\s*\(|\bsetenv\s*\(|\bputenv\s*\(", "reading or writing the environment"),
    (r"#\s*include\s*<\s*(fstream|filesystem)\s*>|\bfopen\s*\(|\bstd::ifstream\b|\bstd::ofstream\b", "file access"),
    (r"\bcudaDeviceReset\b", "resetting the device"),
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def check_native(name: str, src: str) -> list[Violation]:
    out = []
    clean = _strip_comments(src)
    for pat, why in NATIVE_BANNED:
        for m in re.finditer(pat, clean):
            out.append(Violation(clean.count("\n", 0, m.start()) + 1, "native", f"{name}: {why}: '{m.group(0)}'"))
    return out


def check_files(files: dict[str, str]) -> list[Violation]:
    """Guard a whole candidate: submission.py must define custom_kernel; other .py helpers get the same
    Python rules; native sources get the pattern rules. Violations name their file."""
    out: list[Violation] = []
    for name, src in files.items():
        if name == "submission.py":
            vs = check(src)
        elif name.endswith(".py"):
            vs = [v for v in check(src) if v.rule != "contract"]
        elif name.endswith((".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp")):
            vs = check_native(name, src)
        else:
            continue
        out += [Violation(v.line, v.rule, v.detail if name == "submission.py" or v.rule == "native" else f"{name}: {v.detail}")
                for v in vs]
    return out
