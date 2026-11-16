"""
One global, append-only event log: workspace/events.jsonl. Everything that can stall writes here, automatically:

  * every `./optk` command (op_start / op_heartbeat / op_end, with the pipeline phase it is in),
  * every tool call of every Claude Code agent (via hooks: agent/hooklog.py),
  * permission prompts, notifications, sub-agent start/stop.

An operation that has an `op_start` and no `op_end` is *in flight*. `./optk where` pairs them up, ages them, and says what is
stuck: a Modal call that never returned, a tool call waiting on a permission prompt, an agent that has gone quiet.

Stdlib only and cheap to import: the hook script imports this on every tool call.
"""
import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "workspace" / "events.jsonl"
MAX_BYTES = 20 * 1024 * 1024
FIELD_MAX = 400

_phase: dict[str, str] = {}      # op id -> what it is doing right now
_current: list[str] = []         # ids of the optk spans running in this process (innermost last)
_lock = threading.Lock()


def _clip(v):
    if isinstance(v, str) and len(v) > FIELD_MAX:
        return v[:FIELD_MAX] + "…"
    return v


def emit(kind: str, **fields) -> None:
    """Append one JSON line. Never raises: logging must not break the thing it observes."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": round(time.time(), 3), "t": time.strftime("%H:%M:%S"), "pid": os.getpid(), "kind": kind}
        rec.update({k: _clip(v) for k, v in fields.items() if v not in (None, "")})
        line = (json.dumps(rec, default=str) + "\n").encode()
        with _lock:
            if LOG.exists() and LOG.stat().st_size > MAX_BYTES:
                LOG.replace(LOG.with_name("events.1.jsonl"))
            fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)      # a single write of one short line: atomic enough across processes
            finally:
                os.close(fd)
    except Exception:
        pass


def set_phase(op: str | None, text: str) -> None:
    """Say what the running operation is doing now; heartbeats repeat it, and the transition is logged."""
    op = op or (_current[-1] if _current else None)
    if op:
        _phase[op] = text
        emit("phase", op=op, phase=text)


@contextlib.contextmanager
def span(name: str, *, heartbeat: float = 60.0, **fields):
    """Log op_start / op_end around a block, plus a heartbeat every `heartbeat` seconds while it runs.
    Yields the op id (pass it to set_phase). Exceptions are recorded and re-raised."""
    op = uuid.uuid4().hex[:8]
    t0 = time.time()
    emit("op_start", op=op, name=name, src="optk", **fields)
    stop = threading.Event()

    def beat():
        while not stop.wait(heartbeat):
            emit("op_heartbeat", op=op, name=name, elapsed=round(time.time() - t0), phase=_phase.get(op, ""))

    _current.append(op)
    th = None
    if heartbeat and heartbeat > 0:
        th = threading.Thread(target=beat, daemon=True)
        th.start()
    status, err = "ok", None
    try:
        yield op
    except SystemExit as e:
        status, err = ("ok", None) if e.code in (0, None) else ("exit", str(e.code))
        raise
    except BaseException as e:      # noqa: BLE001  record KeyboardInterrupt too, then re-raise
        status, err = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        stop.set()
        if op in _current:
            _current.remove(op)
        emit("op_end", op=op, name=name, status=status, err=err, dur=round(time.time() - t0, 1), phase=_phase.pop(op, ""))


# ---- reading / analysis ---------------------------------------------------------------------------
def read(path: Path | None = None, since: float | None = None) -> list[dict]:
    p = path or LOG
    out: list[dict] = []
    if not p.exists():
        return out
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since is None or r.get("ts", 0) >= since:
                out.append(r)
    return out


def _alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def actor_of(r: dict) -> str:
    a = r.get("agent") or r.get("actor") or ""
    return a or ("main" if r.get("src") == "hook" else "")


def analyze(events: list[dict], now: float | None = None, stall_s: float = 600.0) -> dict:
    """-> {"open": [...], "waiting": [...], "agents": {...}, "last_ts": float}.
    open: operations with a start and no end, oldest first, each with age, last heartbeat, phase and a verdict:
      'dead'    an optk process that no longer exists (killed / crashed without writing an end)
      'stalled' older than its threshold with no heartbeat for 3 minutes (or a hook op older than stall_s)
      'running' otherwise
    waiting: permission prompts / notifications that nothing has followed since (you may need to answer them)."""
    now = now or time.time()
    started: dict[str, dict] = {}
    beats: dict[str, dict] = {}
    for r in events:
        k, op = r.get("kind"), r.get("op")
        if k == "op_start" and op:
            started[op] = r
        elif k == "op_end" and op:
            started.pop(op, None)
            started.pop("perm-" + op, None)
        elif k in ("op_heartbeat", "phase") and op:
            beats.setdefault(op, {}).update(ts=r["ts"], phase=r.get("phase", beats.get(op, {}).get("phase", "")))
    open_ops = []
    for op, r in started.items():
        age = now - r["ts"]
        b = beats.get(op, {})
        quiet = now - b.get("ts", r["ts"])
        verdict = "running"
        if r.get("src") == "optk" and not _alive(r.get("pid")):
            verdict = "dead"
        elif r.get("src") == "optk" and age > 120 and quiet > 180:
            verdict = "stalled"        # an optk process heartbeats every 60 s: silence means it is hung
        elif r.get("src") == "hook" and age > stall_s:
            verdict = "stalled"
        open_ops.append({**r, "age": age, "phase": b.get("phase", ""), "quiet": quiet, "verdict": verdict})
    open_ops.sort(key=lambda x: -x["age"])

    last_activity = max((r["ts"] for r in events if r.get("kind") != "notification"), default=0)
    waiting = [r for r in events if r.get("kind") in ("notification",) and r["ts"] >= last_activity - 1]
    waiting += [o for o in open_ops if str(o.get("name", "")).startswith("permission prompt")]

    agents: dict[str, dict] = {}
    for r in events:
        a = actor_of(r)
        if a and r.get("kind") in ("op_start", "op_end", "agent_start", "agent_stop", "phase", "op_heartbeat"):
            d = agents.setdefault(a, {"last_ts": 0, "last": "", "stopped": False})
            d["last_ts"] = r["ts"]
            if r["kind"] == "op_start":
                d["last"] = f"{r.get('name', '')} {r.get('detail', '')}".strip()[:110]
            d["stopped"] = r["kind"] == "agent_stop"
    return {"open": open_ops, "waiting": waiting, "agents": agents, "last_ts": last_activity}


def fmt_age(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 120 else f"{s // 60}m{s % 60:02d}s" if s < 7200 else f"{s // 3600}h{(s % 3600) // 60:02d}m"


def format_event(r: dict) -> str:
    k = r.get("kind", "?")
    who = actor_of(r)
    who = f"[{who}] " if who else ""
    if k == "op_start":
        return f"{r['t']} {who}START {r.get('name', '')} {r.get('detail', '') or r.get('argv', '')}".rstrip()
    if k == "op_end":
        extra = f" ({r['err']})" if r.get("err") else ""
        return f"{r['t']} {who}END   {r.get('name', '')} {r.get('status', '')} {r.get('dur', '')}s{extra}"
    if k == "op_heartbeat":
        return f"{r['t']} {who}...   {r.get('name', '')} still running {r.get('elapsed', '')}s  {r.get('phase', '')}"
    if k == "phase":
        return f"{r['t']} {who}PHASE {r.get('phase', '')}"
    return f"{r['t']} {who}{k.upper()} " + " ".join(f"{a}={b}" for a, b in r.items() if a not in ("ts", "t", "pid", "kind", "agent", "actor"))
