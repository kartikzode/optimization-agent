import json
import os
import time

import pytest

from agent import cli, events, hooklog


@pytest.fixture(autouse=True)
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "LOG", tmp_path / "events.jsonl")
    return tmp_path / "events.jsonl"


def kinds():
    return [(r["kind"], r.get("name") or r.get("phase")) for r in events.read()]


def test_span_logs_start_phase_and_end_with_duration():
    with events.span("optk submit", heartbeat=0, argv="submit x", actor="w1") as op:
        events.set_phase(None, "#3 WAITING on Modal")
    ev = events.read()
    assert [r["kind"] for r in ev] == ["op_start", "phase", "op_end"]
    assert ev[0]["actor"] == "w1" and ev[1]["phase"] == "#3 WAITING on Modal" and ev[2]["status"] == "ok"
    assert ev[2]["phase"] == "#3 WAITING on Modal" and ev[0]["op"] == ev[2]["op"] == op


def test_span_records_exceptions_and_nonzero_exits_and_reraises():
    with pytest.raises(ValueError):
        with events.span("optk compile", heartbeat=0):
            raise ValueError("boom")
    with pytest.raises(SystemExit):
        with events.span("optk submit", heartbeat=0):
            raise SystemExit("no baseline yet")
    ends = [r for r in events.read() if r["kind"] == "op_end"]
    assert ends[0]["status"] == "error" and "ValueError: boom" in ends[0]["err"]
    assert ends[1]["status"] == "exit" and "no baseline" in ends[1]["err"]


def test_heartbeat_repeats_the_phase_while_an_operation_hangs():
    with events.span("optk submit", heartbeat=0.05):
        events.set_phase(None, "WAITING on Modal")
        time.sleep(0.3)
    beats = [r for r in events.read() if r["kind"] == "op_heartbeat"]
    assert len(beats) >= 3 and beats[0]["phase"] == "WAITING on Modal"


def start(op, name, ts, **kw):
    return {"ts": ts, "t": "00:00:00", "pid": kw.pop("pid", os.getpid()), "kind": "op_start", "op": op, "name": name, **kw}


def test_open_operations_are_paired_aged_and_judged():
    now = 10_000.0
    ev = [start("a", "optk submit", now - 1500, src="optk", actor="w1"),                    # started, heartbeats stopped
          {"ts": now - 600, "t": "", "kind": "op_heartbeat", "op": "a", "phase": "#4 WAITING on Modal"},
          start("b", "optk status", now - 30, src="optk"),
          {"ts": now - 29, "t": "", "kind": "op_end", "op": "b", "status": "ok"},          # finished: not open
          start("c", "tool:Bash", now - 700, src="hook", agent="kernel-worker:b7e2aa", detail="sleep 9999"),
          start("d", "optk submit", now - 500, src="optk", pid=2_999_999)]                  # no such process
    info = events.analyze(ev, now, stall_s=600)
    by = {o["op"]: o for o in info["open"]}
    assert set(by) == {"a", "c", "d"}
    assert by["a"]["verdict"] == "stalled" and by["a"]["phase"] == "#4 WAITING on Modal"
    assert by["c"]["verdict"] == "stalled"                    # a tool call open for 700 s > 600 s
    assert by["d"]["verdict"] == "dead"                       # its process is gone without an end
    assert [o["op"] for o in info["open"]] == ["a", "c", "d"]  # oldest first


def test_a_live_heartbeating_operation_is_running_not_stalled():
    now = 10_000.0
    ev = [start("a", "optk submit", now - 1500, src="optk"),
          {"ts": now - 20, "t": "", "kind": "op_heartbeat", "op": "a", "phase": "waiting"}]
    assert events.analyze(ev, now)["open"][0]["verdict"] == "running"


def test_pending_permission_prompt_is_reported_until_the_tool_finishes():
    hooklog.handle("pre", {"tool_name": "Bash", "tool_use_id": "T9", "tool_input": {"command": "./optk submit d"}, "agent_type": "modal-runner", "agent_id": "abc123456"})
    hooklog.handle("perm", {"tool_name": "Bash", "tool_use_id": "T9", "tool_input": {"command": "./optk submit d"}, "agent_type": "modal-runner", "agent_id": "abc123456"})
    hooklog.handle("notify", {"message": "Claude needs your permission to use Bash", "agent_type": "modal-runner", "agent_id": "abc123456"})
    info = events.analyze(events.read())
    assert any(str(w.get("name", "")).startswith("permission prompt") for w in info["waiting"])
    assert any(w.get("kind") == "notification" for w in info["waiting"])
    hooklog.handle("post", {"tool_name": "Bash", "tool_use_id": "T9", "agent_type": "modal-runner", "agent_id": "abc123456"})
    info = events.analyze(events.read())
    assert info["open"] == []                                 # approved and finished: nothing pending


def test_hook_maps_tool_calls_with_agent_identity_and_one_line_commands():
    hooklog.handle("pre", {"session_id": "sess123456", "tool_name": "Bash", "tool_use_id": "T1",
                           "tool_input": {"command": "python - <<'EOF'\nprint(1)\nEOF"}, "agent_type": "kernel-worker", "agent_id": "b7e2aa10ff"})
    hooklog.handle("pre", {"tool_name": "Agent", "tool_use_id": "T2",
                           "tool_input": {"subagent_type": "kernel-worker", "description": "try smem tiling", "prompt": "do it"}})
    hooklog.handle("pre", {})                                  # malformed: ignored
    a, b = events.read()
    assert a["agent"] == "kernel-worker:b7e2aa" and "\n" not in a["detail"] and a["session"] == "sess1234"
    assert b["agent"] == "main" and "kernel-worker: try smem tiling" in b["detail"]


def test_failed_and_denied_tools_close_their_operation_with_the_reason():
    hooklog.handle("pre", {"tool_name": "Write", "tool_use_id": "T3", "tool_input": {"file_path": "/x.py"}})
    hooklog.handle("postfail", {"tool_name": "Write", "tool_use_id": "T3", "error": "permission denied"})
    hooklog.handle("pre", {"tool_name": "Bash", "tool_use_id": "T4", "tool_input": {"command": "rm -rf /"}})
    hooklog.handle("permdenied", {"tool_name": "Bash", "tool_use_id": "T4", "reason": "blocked"})
    ends = {r["op"]: r for r in events.read() if r["kind"] == "op_end"}
    assert ends["T3"]["status"] == "error" and "permission denied" in ends["T3"]["err"] and ends["T4"]["status"] == "denied"
    assert events.analyze(events.read())["open"] == []


def test_hook_entry_point_never_fails_even_on_garbage(monkeypatch, capsys):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("not json at all"))
    assert hooklog.main(["pre"]) == 0 and capsys.readouterr().out == ""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"tool_name": "Bash", "tool_use_id": "z", "tool_input": {"command": "ls"}})))
    assert hooklog.main(["nonsense-event"]) == 0


def test_every_optk_command_is_logged_and_where_and_events_are_not(capsys):
    cli.main(["problems"])
    with pytest.raises(SystemExit):
        cli.main(["status", "--problem", "no_such_problem"])
    names = [(r["kind"], r.get("name")) for r in events.read()]
    assert ("op_start", "optk problems") in names and ("op_end", "optk problems") in names
    assert ("op_start", "optk status") in names
    assert any(r["kind"] == "op_end" and r.get("status") == "exit" for r in events.read())
    before = len(events.read())
    cli.main(["where"])
    cli.main(["events"])
    assert len(events.read()) == before                          # the diagnosis commands leave no trace


def test_where_reports_stalled_dead_and_waiting_with_a_verdict(capsys):
    now = time.time()
    events.emit("op_start", op="s1", name="optk submit", src="optk", actor="w1", argv="submit d")
    events.emit("op_start", op="p1", name="permission prompt: Bash", src="hook", agent="kernel-worker:b7e2aa", detail="./optk compile d")
    # age the log by rewriting timestamps
    rows = events.read()
    rows[0]["ts"] = now - 3000
    rows[1]["ts"] = now - 400
    events.LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cli.main(["where"])
    out = capsys.readouterr().out
    assert "WAITING ON YOU" in out and "permission prompt: Bash" in out and "kernel-worker:b7e2aa" in out
    assert "VERDICT: a permission prompt" in out and "stalls the agent until you answer" in out
    assert "!! STALLED" in out or "!! DEAD" in out              # the 50-minute-old submit is flagged


def test_where_with_no_events_explains_how_logging_starts(capsys):
    cli.main(["where"])
    assert "no events" in capsys.readouterr().out
