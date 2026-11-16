import logging
import time

from agent import log as logsetup
from harness import modal_runner
from harness.evaluate import format_raw

# a stand-in for problems/eval.py: reports through EVAL_FD like the real one, and prints on both streams
FAKE_EVAL = '''
import os, sys, time
mode = sys.argv[1]
print("hello from stdout", flush=True)
print("ERR: something went wrong", file=sys.stderr, flush=True)
if mode == "hang":
    time.sleep(120)
with os.fdopen(int(os.environ["EVAL_FD"]), "w") as f:
    f.write("check: pass\\n")
    f.write("benchmark.0.report: " + "A" * 500 + "\\n")
sys.exit(3 if mode == "bad-exit" else 0)
'''


def run_remote_body(mode, timeout_s=30):
    return modal_runner.run_eval.local({"eval.py": FAKE_EVAL}, mode, "x: 1", timeout_s)


def test_container_streams_eval_output_to_its_own_logs(capfd):
    out = run_remote_body("test")
    seen = capfd.readouterr()
    # these are the lines the Modal dashboard shows
    assert "hello from stdout" in seen.out and "ERR: something went wrong" in seen.err
    assert "[eval] start mode=test" in seen.out and "[eval] done mode=test exit=0" in seen.out
    assert "check: pass" in seen.out and "A" * 100 not in seen.out        # profile blobs are summarised
    # and the caller still gets everything
    assert out["returncode"] == 0 and "check: pass" in out["results"] and "A" * 500 in out["results"]
    assert "hello from stdout" in out["stdout"] and "ERR: something went wrong" in out["stderr"]


def test_nonzero_exit_is_reported_and_logged(capfd):
    out = run_remote_body("bad-exit")
    assert out["returncode"] == 3 and "exit=3" in capfd.readouterr().out


def test_timeout_kills_the_evaluator_and_says_so(capfd):
    t0 = time.time()
    out = run_remote_body("hang", timeout_s=2)
    assert out["timed_out"] and time.time() - t0 < 20
    assert "TIMEOUT after 2s" in capfd.readouterr().out
    assert "hello from stdout" in out["stdout"]          # output produced before the hang is kept


def test_candidate_helper_files_are_staged_next_to_submission():
    prog = ('import os, sys\n'
            'src = open("kernels/k.cu").read()\n'
            'with os.fdopen(int(os.environ["EVAL_FD"]), "w") as f:\n'
            '    f.write("check: pass\\nlen: %d\\n" % len(src))\n')
    out = modal_runner.run_eval.local({"eval.py": prog, "kernels/k.cu": "// kernel"}, "test", "x: 1", 30)
    assert out["returncode"] == 0 and "len: 9" in out["results"]


def test_missing_ncu_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(modal_runner, "_find_ncu", lambda: None)
    out = modal_runner._eval_process({"eval.py": "pass"}, "profile", "x: 1", 5, ncu={"set": "full"})
    assert out["returncode"] == 127 and "ncu is not installed" in out["stderr"]


def test_pipeline_skips_later_stages_when_an_earlier_one_fails():
    passing = ('import os, sys\nwith os.fdopen(int(os.environ["EVAL_FD"]), "w") as f:\n    f.write("check: %s\\n" % sys.argv[1])\n'
               'sys.exit(0 if sys.argv[1] == "pass" else 112)\n')
    # eval.py is invoked as `eval.py <mode>`; use the mode itself as the verdict
    plan = [{"name": "test", "mode": "fail", "tests": "x: 1", "timeout_s": 20},
            {"name": "benchmark", "mode": "pass", "tests": "x: 1", "timeout_s": 20, "requires": ["test"]}]
    out = modal_runner.run_pipeline.local({"eval.py": passing}, plan)
    assert out["stages"]["test"]["passed"] is False and out["stages"]["benchmark"] == {"skipped": True, "passed": False}


def test_console_is_concise_and_the_file_has_everything(tmp_path, capsys):
    root = logsetup.setup(tmp_path, verbose=False)
    lg = logging.getLogger("optagent.test")
    lg.debug("debug detail")
    lg.warning("careful")
    try:
        raise ValueError("boom")
    except ValueError:
        lg.exception("it failed")
    out = capsys.readouterr().out
    assert "debug detail" not in out and "[warn] careful" in out and "[error] it failed" in out
    assert "Traceback" not in out                           # no tracebacks on the terminal
    text = (tmp_path / "agent.log").read_text()
    assert "debug detail" in text and "Traceback" in text and "ValueError: boom" in text
    root.handlers.clear()


def test_format_raw_has_every_stream():
    raw = {"returncode": 1, "timed_out": False, "wall_s": 2.5, "results": "check: fail", "stdout": "o", "stderr": "e"}
    t = format_raw(raw, "moe test")
    assert "returncode=1" in t and "check: fail" in t and "## stdout\no" in t and "## stderr\ne" in t
