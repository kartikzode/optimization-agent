"""
Continuous-run control: should the orchestrator start another round?

    ./optk continue --start [--max-hours 8] [--max-gpu-hours 4] [--patience 6]   begin a continuous run (resets the budgets)
    ./optk continue                          ask before every round: exit 0 = go on, exit 1 = stop (and why)
    ./optk continue --stop                   create workspace/STOP: the kill switch (also: `touch workspace/STOP`)
    ./optk continue --clear                  remove it

A round continues only while: no STOP file, wall-clock budget left, GPU budget left. Plateau is *not* a reason to stop: it
switches the advice from "exploit" (refine the current best) to "explore" (try something structurally different).
An improvement counts only if it beats the previous best by more than SIG (1%), since the benchmark noise is about that large.
"""
import json
import time
from pathlib import Path

from . import config
from .store import Store

SIG = 0.01      # a new best must be at least 1% faster than the old one to count as progress


def stop_file() -> Path:
    return config.WORKSPACE / "STOP"


def state_file(store: Store) -> Path:
    return store.dir / "CONTINUE.json"


def start(store: Store, max_hours: float, max_gpu_hours: float, patience: int) -> dict:
    st = {"started_at": time.time(), "max_hours": max_hours, "max_gpu_hours": max_gpu_hours, "patience": patience}
    state_file(store).write_text(json.dumps(st, indent=2))
    return st


def plateau(store: Store) -> dict:
    """Progress that clears the noise floor: the last significant new best and how many candidates came after it."""
    best = None
    last_sig = None
    for c in store.all():
        if c.kind == "seed" or not c.ok:
            continue
        if best is None or c.geomean_ns < best * (1 - SIG):
            last_sig = c.id
        if best is None or c.geomean_ns < best:
            best = c.geomean_ns
    tried = [c for c in store.all() if c.kind != "seed" and c.status not in ("running", "infra") and (last_sig is None or c.id > last_sig)]
    return {"last_significant": last_sig, "since": len(tried), "best_ns": best}


def check(store: Store, now: float | None = None) -> tuple[bool, str]:
    """-> (go_on, one-line explanation incl. the mode advice)."""
    now = now or time.time()
    if stop_file().exists():
        return False, f"STOP file present ({stop_file()}): remove with `./optk continue --clear` to resume"
    f = state_file(store)
    if not f.exists():
        return False, "no continuous run is active: start one with `./optk continue --start` (it sets the budgets)"
    st = json.loads(f.read_text())
    hours = (now - st["started_at"]) / 3600
    gpu_h = sum(c.wall_s for c in store.all() if c.created_at >= st["started_at"]) / 3600
    p = plateau(store)
    best = f"best #{store.best().id} {store.best().geomean_ns / 1e6:.3f} ms" if store.best() else "no correct candidate yet"
    head = f"elapsed {hours:.1f}/{st['max_hours']:g} h · GPU {gpu_h:.2f}/{st['max_gpu_hours']:g} h · {best}"
    if hours >= st["max_hours"]:
        return False, f"STOP: wall-clock budget used ({head})"
    if gpu_h >= st["max_gpu_hours"]:
        return False, f"STOP: GPU budget used ({head})"
    since = p["since"]
    if since < st["patience"]:
        mode = f"EXPLOIT: {since} candidate(s) since the last significant (>{SIG:.0%}) improvement (#{p['last_significant']}); refine the current best"
    elif since < 3 * st["patience"]:
        mode = (f"EXPLORE: {since} candidates without a significant improvement (patience {st['patience']}); stop refining, "
                "try structurally different ideas (see the skill's plateau playbook)")
    else:
        mode = (f"EXPLORE-HARD: {since} candidates without a significant improvement; the current design is probably at its ceiling. "
                "Re-diagnose from the profile, audit the best kernel, then try a different algorithm")
    return True, f"CONTINUE ({head}). {mode}"
