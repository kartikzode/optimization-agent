import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from agent import pipeline
from agent.store import Store
from dashboard.build import render_html
from dashboard.data import build_payload, list_runs
from dashboard.serve import make_handler

from conftest import FakeEvaluator, passing_result, profile_ok

KW = dict(gpu="H100", arch="sm_90a", log=lambda *_: None)


def submit(store, files, result, **kw):
    return pipeline.submit(store, "moe", files, evaluator_factory=lambda: FakeEvaluator(result), **KW, **kw)


def make_run(tmp_path, evil=""):
    s = Store(tmp_path / "r1")
    s.set_meta("problem", "moe")
    s.set_meta("gpu", "H100")
    seed = submit(s, {"submission.py": "def custom_kernel(d):\n    return d\n"},
                  passing_result(ms=(2000.0, 8000.0), profile=profile_ok(6_000_000_000, 30, 10)), kind="seed", hypothesis="baseline")
    files = {"submission.py": "def custom_kernel(d):\n    return d  # v2\n", "kernel.cu": "__global__ void k(){}\n"}
    submit(s, {"submission.py": files["submission.py"] + evil}, {"gpu_info": "", "logs": {"test.log": "boom"}, "round_trip_s": 1, "shape_index": 1,
           "shape_spec": "", "test": {"status": "crash", "error": "boom", "benchmarks": [], "tests": [], "geomean_ns": None},
           "benchmark": None, "profile": None}, parent_id=seed.id, worker="w1", strategy="s1", hypothesis=evil)
    submit(s, files, passing_result(ms=(1000.0, 4000.0), profile=profile_ok(2_000_000_000, 70, 60)),
           parent_id=seed.id, worker="w1", strategy="s1", hypothesis="tile")
    submit(s, files, passing_result(ms=(1500.0, 6000.0), profile=profile_ok(4_000_000_000, 50, 30)),
           parent_id=seed.id, worker="w2", strategy="s2", hypothesis="other")
    s.close()
    return tmp_path


def test_payload_marks_new_bests_speedups_roofline_points_and_groups(tmp_path):
    make_run(tmp_path)
    assert [r["id"] for r in list_runs(tmp_path)] == ["r1"]
    d = build_payload(tmp_path, "r1")
    c = d["candidates"]
    assert [x["new_best"] for x in c] == [True, False, True, False]
    assert c[2]["speedup"] == 2.0 and c[1]["speedup"] is None and c[2]["vs_parent"] == 2.0
    assert len(d["shapes"]) == 2 and c[2]["benchmarks"][0]["pct_roof"] > 0
    # measured intensity rises as DRAM traffic falls: the roofline x coordinate moves right
    assert c[0]["roof_point"]["ai"] < c[2]["roof_point"]["ai"] and c[1]["roof_point"] is None
    assert abs(c[2]["profile"]["totals"]["sm_pct"] - 57.6) < 1e-9          # time-weighted: (70*4 + 8*1) / 5
    assert c[2]["profile"]["kernels"][0]["short"] == "gemm_kernel"
    # a "win" beats its parent by >= 2%: s2 is slower than the best but still faster than its parent (the baseline)
    assert {x["name"]: (x["attempts"], x["wins"]) for x in d["strategies"]} == {"s1": (2, 1), "s2": (1, 1)}
    assert {x["name"]: (x["attempts"], x["wins"]) for x in d["workers"]} == {"w1": (2, 1), "w2": (1, 1)}
    assert "kernel.cu" in c[2]["diff"] and c[2]["diff"].startswith("--- #1/")
    assert "# Iteration #3" in c[2]["brief"]
    rels = {link["rel"] for link in c[2]["links"]}
    assert {"candidates/0003/brief.md", "candidates/0003/logs/test.log", "candidates/0003/profile/ncu.ncu-rep"} <= rels
    json.dumps(d)                                                   # must be serialisable as is


def test_payload_without_a_cost_model_still_builds(tmp_path):
    s = Store(tmp_path / "r2")
    s.set_meta("problem", "no_such_problem")
    c = s.reserve(kind="seed")
    s.finish(c.id, status="pass", geomean_ns=1e9, result={"benchmark": {"benchmarks": [
        {"spec": "x: 1", "mean_ns": 1e9, "std_ns": 0, "runs": 3, "status": "pass"}]}})
    s.close()
    d = build_payload(tmp_path, "r2")
    assert d["shapes"] == [] and d["peaks"] is None and d["candidates"][0]["benchmarks"][0].get("tflops") is None


def test_running_candidates_show_up_so_the_live_dashboard_sees_in_flight_work(tmp_path):
    s = Store(tmp_path / "r3")
    s.set_meta("problem", "moe")
    s.reserve(kind="improve", worker="w9", hypothesis="in flight")
    s.close()
    d = build_payload(tmp_path, "r3")
    assert d["candidates"][0]["status"] == "running" and d["workers"] == []      # not counted as an attempt yet


def test_embedded_data_cannot_break_out_of_the_script_tag(tmp_path):
    evil = "</script><script>window.__pwned=1</script>"
    make_run(tmp_path, evil=evil)
    html = render_html(build_payload(tmp_path, "r1"))
    assert evil not in html and "\\u003c/script>" in html
    assert render_html(None).count("window.__DASH__ = null") == 1     # live mode placeholder


def test_old_llm_loop_runs_still_render(tmp_path):
    import sqlite3
    d = tmp_path / "old"
    d.mkdir()
    db = sqlite3.connect(d / "store.db")
    db.executescript("""CREATE TABLE candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id INTEGER, parent2_id INTEGER,
        kind TEXT NOT NULL, strategy TEXT NOT NULL DEFAULT '', hypothesis TEXT NOT NULL DEFAULT '', code TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, geomean_ns REAL, result TEXT NOT NULL DEFAULT '{}', profile TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '', repair_depth INTEGER NOT NULL DEFAULT 0, reward REAL NOT NULL DEFAULT 0,
        llm_in INTEGER NOT NULL DEFAULT 0, llm_out INTEGER NOT NULL DEFAULT 0, wall_s REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO candidates (kind,code,status,geomean_ns,created_at) VALUES ('seed','legacy','pass',1e9,1.0);
        INSERT INTO candidates (parent_id,kind,code,status,geomean_ns,created_at) VALUES (1,'improve','legacy2','pass',5e8,2.0);""")
    db.commit()
    db.close()
    p = build_payload(tmp_path, "old")
    assert [c["speedup"] for c in p["candidates"]] == [1.0, 2.0] and p["candidates"][1]["diff"]


def serve(runs):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runs))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_server_serves_logs_but_not_outside_the_run_or_odd_file_types(tmp_path):
    make_run(tmp_path)
    (tmp_path / "secret.txt").write_text("outside the run")
    srv, base = serve(tmp_path)
    try:
        code, body = get(f"{base}/raw/r1/candidates/0003/logs/test.log")
        assert code == 200 and b"bench ok" in body or code == 200
        assert get(f"{base}/raw/r1/candidates/0003/brief.md")[0] == 200
        assert get(f"{base}/raw/r1/candidates/0003/profile/ncu.ncu-rep")[0] == 200
        assert get(f"{base}/raw/r1/..%2Fsecret.txt")[0] == 404          # traversal
        assert get(f"{base}/raw/r1/store.db")[0] == 404                 # not an allowed type
        assert get(f"{base}/raw/nope/candidates/0001/brief.md")[0] == 404
        code, body = get(f"{base}/api/data?run=r1")
        assert code == 200 and json.loads(body)["run_id"] == "r1"
    finally:
        srv.shutdown()
