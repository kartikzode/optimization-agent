"""Paths and GPU settings. There is no LLM configuration here: Claude Code is the optimizer."""
import os
from pathlib import Path

try:  # .env is optional (Modal tokens / MODAL_GPU); the container side has no dotenv
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT / ".env")

PROBLEMS = ROOT / "problems"
WORKSPACE = ROOT / "workspace"
RUNS = WORKSPACE / "runs"
TEMPLATES = ROOT / "templates"
KNOWLEDGE = ROOT / "knowledge"

# nvcc -arch value per GPU
ARCH = {"H100": "sm_90a", "H200": "sm_90a", "B200": "sm_100a", "A100": "sm_80", "L4": "sm_89"}
# the same, spelled the way TORCH_CUDA_ARCH_LIST wants it
TORCH_ARCH = {"sm_90a": "9.0a", "sm_100a": "10.0a", "sm_80": "8.0", "sm_89": "8.9"}


def modal_status() -> str:
    """'env' (token vars), '~/.modal.toml', or 'missing'."""
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return "env"
    if (Path.home() / ".modal.toml").exists():
        return "~/.modal.toml"
    return "missing"


def gpu_name() -> str:
    return os.getenv("MODAL_GPU") or "H100"


def gpu_arch() -> str:
    return ARCH.get(gpu_name().upper(), "sm_90a")


def run_dir(problem: str, run: str | None = None) -> Path:
    """One run directory per problem by default (workspace/runs/<problem>); pass `run` for a fresh start."""
    return RUNS / (run or problem)
