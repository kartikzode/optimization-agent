import base64
from harness.parse import parse_results

PASS = """test-count: 2
test.0.spec: dhidden: 512; seed: 1
test.0.status: pass
test.1.spec: dhidden: 1024; seed: 2
test.1.status: pass
check: pass
"""
FAIL = """test-count: 2
test.0.spec: a: 1
test.0.status: pass
test.1.spec: a: 2
test.1.status: fail
test.1.error: mismatch at [0]: 1.0 vs 2.0
check: fail
"""
BENCH = """benchmark-count: 2
benchmark.0.spec: a: 1
benchmark.0.runs: 10
benchmark.0.mean: 2000000.0
benchmark.0.std: 1.0
benchmark.0.err: 0.1
benchmark.0.best: 1900000.0
benchmark.0.worst: 2100000.0
benchmark.1.spec: a: 2
benchmark.1.runs: 10
benchmark.1.mean: 8000000.0
benchmark.1.std: 1.0
benchmark.1.err: 0.1
benchmark.1.best: 7900000.0
benchmark.1.worst: 8100000.0
check: pass
"""

def test_pass():
    r = parse_results(PASS, 0)
    assert r["status"] == "pass" and [t["status"] for t in r["tests"]] == ["pass", "pass"] and r["error"] is None

def test_fail_reports_first_error():
    r = parse_results(FAIL, 112)
    assert r["status"] == "fail" and "mismatch" in r["error"]

def test_crash_uses_stderr_and_marks_unrun_cases():
    partial = "test-count: 2\ntest.0.spec: a: 1\n"
    r = parse_results(partial, 1, stderr="Traceback...\nValueError: boom")
    assert r["status"] == "crash" and r["tests"][0]["status"] == "not_run" and "boom" in r["error"]

def test_no_output_crash():
    assert parse_results("", 111)["status"] == "crash"

def test_timeout():
    assert parse_results("", -1, timed_out=True)["status"] == "timeout"

def test_geomean():
    r = parse_results(BENCH, 0)
    assert abs(r["geomean_ns"] - 4_000_000) < 1

def test_profile_custom_base64_alphabet():
    text = "x ~~ *+*?" * 20
    enc = base64.b64encode(text.encode(), b"+*").decode()
    r = parse_results(f"benchmark-count: 1\nbenchmark.0.spec: a: 1\nbenchmark.0.report: {enc}\ncheck: pass\n", 0)
    assert r["profiles"][0]["report"] == text and r["benchmarks"] == []


WORKER_CRASH = '''multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/tmp/tmpab12cd/eval.py", line 161, in _run_single_test
    submission_output = custom_kernel(_clone_data(data))
  File "/tmp/tmpab12cd/submission.py", line 42, in custom_kernel
    return moe(x)
NameError: name 'Optional' is not defined
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/tmp/tmpab12cd/eval.py", line 493, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.12/multiprocessing/pool.py", line 774, in get
    raise self._value
NameError: name 'Optional' is not defined
'''

def test_crash_error_is_condensed_to_the_workers_traceback():
    r = parse_results("test-count: 1\n", 1, stderr=WORKER_CRASH)
    assert r["status"] == "crash"
    assert 'File "submission.py", line 42' in r["error"] and "NameError" in r["error"]
    assert "multiprocessing" not in r["error"] and "tmpab12cd" not in r["error"]
    assert "direct cause" not in r["error"]
