"""
Claude Code hook -> events.jsonl. Wired up in .claude/settings.json for every tool call of every agent
(orchestrator, kernel-workers, modal-runner), permission prompts, notifications and sub-agent start/stop.

    python -m agent.hooklog <pre|post|postfail|perm|permdenied|notify|substart|substop|stop>   (hook JSON on stdin)

Always exits 0 and prints nothing: a logging hook must never block or confuse the agent.
"""
import json
import sys

from . import events


def _summ(tool: str, inp: dict) -> str:
    """One line saying what the tool call is about."""
    inp = inp or {}
    if tool == "Bash":
        return " ".join(str(inp.get("command", "")).split())      # one line, even for heredocs
    if tool in ("Read", "Write", "Edit", "NotebookEdit"):
        return str(inp.get("file_path", ""))
    if tool in ("Grep", "Glob"):
        return str(inp.get("pattern", ""))
    if tool in ("Agent", "Task"):
        return f"{inp.get('subagent_type', '')}: {inp.get('description', '')} | {str(inp.get('prompt', ''))[:120]}"
    if tool == "SendMessage":
        return f"to={inp.get('to', '')}: {str(inp.get('message', ''))[:150]}"
    try:
        return json.dumps(inp, default=str)[:200]
    except Exception:
        return ""


def handle(kind: str, d: dict) -> None:
    tool = d.get("tool_name", "")
    tid = d.get("tool_use_id") or ""
    agent_type, agent_id = d.get("agent_type") or "", d.get("agent_id") or ""
    agent = (agent_type + (f":{agent_id[:6]}" if agent_id else "")) if (agent_type or agent_id) else "main"
    base = {"src": "hook", "agent": agent, "session": (d.get("session_id") or "")[:8]}

    if kind in ("perm", "permdenied", "notify", "substart", "substop", "stop"):
        base["keys"] = ",".join(sorted(d))      # low-volume events keep the payload's field names, to debug agent labelling
    if kind in ("pre", "post", "postfail", "perm", "permdenied") and not tool:
        return                  # malformed payload: record nothing rather than a phantom operation
    if kind == "pre":
        events.emit("op_start", op=tid or f"{base['session']}-{tool}-{hash(json.dumps(d.get('tool_input'), default=str, sort_keys=True)) & 0xffff:x}",
                    name=f"tool:{tool}", detail=_summ(tool, d.get("tool_input")), **base)
    elif kind in ("post", "postfail"):
        resp = d.get("tool_response")
        err = d.get("error") or (str(resp)[:200] if kind == "postfail" else None)
        if tid:
            events.emit("op_end", op=tid, name=f"tool:{tool}", status="ok" if kind == "post" else "error", err=err, **base)
            events.emit("op_end", op="perm-" + tid, name="permission prompt", status="done", **base)   # closes a pending prompt
    elif kind == "perm":
        events.emit("op_start", op="perm-" + (tid or tool), name=f"permission prompt: {tool}", detail=_summ(tool, d.get("tool_input")), **base)
    elif kind == "permdenied":
        events.emit("permission_denied", tool=tool, detail=_summ(tool, d.get("tool_input")), reason=d.get("reason"), **base)
        if tid:
            events.emit("op_end", op="perm-" + tid, name="permission prompt", status="denied", **base)
            events.emit("op_end", op=tid, name=f"tool:{tool}", status="denied", **base)
    elif kind == "notify":
        events.emit("notification", message=d.get("message"), type=d.get("notification_type"), **base)
    elif kind == "substart":
        events.emit("agent_start", agent_type=agent_type, agent_id=agent_id, **{**base, "agent": f"{agent_type}:{agent_id[:6]}" if agent_id else agent_type})
    elif kind == "substop":
        events.emit("agent_stop", agent_type=agent_type, agent_id=agent_id, **{**base, "agent": f"{agent_type}:{agent_id[:6]}" if agent_id else agent_type})
    elif kind == "stop":
        events.emit("turn_stop", note="main session finished its turn (waiting for you, or done)", **base)
    else:
        events.emit("hook_unknown", hook=kind, **base)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    try:
        handle(argv[0] if argv else "unknown", data if isinstance(data, dict) else {})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
