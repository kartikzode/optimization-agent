"""
Problem registry: what lives in problems/<name>/, its cost model, and scaffolding for new ones.

A problem directory needs reference.py, task.py, task.yml and submission.py (the baseline).
Optional: cost.py (roofline model).
What is KNOWN about a problem (notes.md, hints.md) lives in knowledge/problems/<name>/, not here: agents may edit the
knowledge directory but never problems/ (the evaluator, references and baselines).

    cost.py:   PEAK = "fp16_tc"                       # which compute peak the roofline uses
               def cost(spec: dict) -> (flops, min_bytes)   # algorithmic work + minimum HBM traffic
"""
import importlib.util
import re
import shutil
from functools import lru_cache
from pathlib import Path

import yaml

from . import config

REQUIRED = ("reference.py", "task.py", "task.yml", "submission.py")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def problem_dir(problem: str) -> Path:
    return config.PROBLEMS / problem


def list_problems() -> list[str]:
    root = config.PROBLEMS
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and all((p / f).exists() for f in REQUIRED))


def require(problem: str) -> Path:
    d = problem_dir(problem)
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise SystemExit(f"problem '{problem}' is not usable: missing {', '.join(missing)} in {d}\n"
                         f"known problems: {', '.join(list_problems()) or '(none)'}")
    return d


def load_task(problem: str) -> dict:
    return yaml.safe_load((problem_dir(problem) / "task.yml").read_text())


def parse_spec(spec: str) -> dict:
    """'a: 1; b: 2' (an eval.py test line) -> {'a': 1, 'b': 2}"""
    out = {}
    for part in spec.split(";"):
        k, _, v = part.partition(":")
        v = v.strip()
        out[k.strip()] = int(v) if re.fullmatch(r"[+-]?\d+", v) else v
    return out


@lru_cache(maxsize=None)
def _cost_module(problem: str):
    path = problem_dir(problem) / "cost.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_cost_{problem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def has_cost(problem: str) -> bool:
    mod = _cost_module(problem)
    return bool(mod and hasattr(mod, "cost"))


def cost(problem: str, spec: str | dict) -> tuple[float, float] | None:
    """(flops, minimum bytes) for one shape, or None when the problem has no model or the spec does not fit it."""
    mod = _cost_module(problem)
    if not mod or not hasattr(mod, "cost"):
        return None
    try:
        flops, nbytes = mod.cost(parse_spec(spec) if isinstance(spec, str) else spec)
        return float(flops), float(nbytes)
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return None


def peak_kind(problem: str) -> str:
    mod = _cost_module(problem)
    return getattr(mod, "PEAK", "fp16_tc") if mod else "fp16_tc"


def knowledge_dir(problem: str) -> Path:
    return config.KNOWLEDGE / "problems" / problem


def notes(problem: str) -> str:
    p = knowledge_dir(problem) / "notes.md"
    return p.read_text() if p.exists() else ""


def scaffold(name: str) -> Path:
    """Copy templates/problem into problems/<name>. Everything in it is a stub to fill in."""
    if not NAME_RE.match(name):
        raise SystemExit("problem names are lowercase letters, digits and underscores, starting with a letter")
    dest, know = problem_dir(name), knowledge_dir(name)
    if dest.exists() or know.exists():
        raise SystemExit(f"{dest if dest.exists() else know} already exists")
    shutil.copytree(config.TEMPLATES / "problem", dest)
    shutil.copytree(config.TEMPLATES / "knowledge_problem", know)      # notes.md, hints.md
    for d in (dest, know):
        for f in d.rglob("*"):
            if f.is_file():
                f.write_text(f.read_text().replace("__NAME__", name))
    return dest
