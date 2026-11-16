import sqlite3
import threading

import pytest

from agent.store import Store, collect_files


def test_reserve_finish_roundtrip_and_files_on_disk(tmp_path):
    s = Store(tmp_path / "run")
    c = s.reserve(kind="seed", worker="w1", strategy="baseline", hypothesis="h")
    assert c.status == "running" and c.id == 1
    s.save_kernel(c.id, {"submission.py": "code", "k/kernel.cu": "// cu"})
    c = s.finish(c.id, status="pass", geomean_ns=5e6, result={"benchmark": {"benchmarks": []}}, wall_s=3.0)
    assert c.ok and (tmp_path / "run" / "candidates" / "0001" / "kernel" / "k" / "kernel.cu").read_text() == "// cu"
    assert s.files(1) == {"submission.py": "code", "k/kernel.cu": "// cu"}
    assert '"status": "pass"' in (tmp_path / "run" / "candidates" / "0001" / "meta.json").read_text()
    assert s.seed().id == 1 and s.seed_ns() == 5e6


def test_best_kernel_is_copied_to_best_dir(tmp_path):
    s = Store(tmp_path / "run")
    for ns, code in ((9e6, "slow"), (4e6, "fast"), (6e6, "mid")):
        c = s.reserve(kind="improve")
        s.save_kernel(c.id, {"submission.py": code})
        s.finish(c.id, status="pass", geomean_ns=ns)
    assert s.best().id == 2 and (tmp_path / "run" / "best" / "submission.py").read_text() == "fast"


def test_parallel_reservations_never_collide(tmp_path):
    s = Store(tmp_path / "run")
    ids, lock = [], threading.Lock()

    def go():
        c = s.reserve(kind="improve")
        with lock:
            ids.append(c.id)

    ts = [threading.Thread(target=go) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(ids) == list(range(1, 13))


def test_two_processes_style_connections_share_one_database(tmp_path):
    a, b = Store(tmp_path / "run"), Store(tmp_path / "run")
    x, y = a.reserve(kind="improve"), b.reserve(kind="improve")
    assert {x.id, y.id} == {1, 2} and a.count() == b.count() == 2


def test_old_llm_loop_stores_still_open(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    db = sqlite3.connect(d / "store.db")
    db.executescript("""CREATE TABLE candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id INTEGER, parent2_id INTEGER,
        kind TEXT NOT NULL, strategy TEXT NOT NULL DEFAULT '', hypothesis TEXT NOT NULL DEFAULT '', code TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, geomean_ns REAL, result TEXT NOT NULL DEFAULT '{}', profile TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '', repair_depth INTEGER NOT NULL DEFAULT 0, reward REAL NOT NULL DEFAULT 0,
        llm_in INTEGER NOT NULL DEFAULT 0, llm_out INTEGER NOT NULL DEFAULT 0, wall_s REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO candidates (kind,code,status,geomean_ns,created_at) VALUES ('seed','legacy code','pass',1e9,1.0);""")
    db.commit()
    db.close()
    s = Store(d)
    c = s.get(1)
    assert c.ok and c.profile_summary == {} and s.files(1) == {"submission.py": "legacy code"}


def test_ancestors_walk_up_the_lineage(tmp_path):
    s = Store(tmp_path / "run")
    a = s.reserve(kind="seed")
    b = s.reserve(kind="improve", parent_id=a.id)
    c = s.reserve(kind="improve", parent_id=b.id)
    assert [x.id for x in s.ancestors(c.id)] == [b.id, a.id]


def test_collect_files_rules(tmp_path):
    d = tmp_path / "k"
    (d / "sub").mkdir(parents=True)
    (d / "submission.py").write_text("x")
    (d / "kernel.cu").write_text("// k")
    (d / "sub" / "h.cuh").write_text("// h")
    (d / "candidate.json").write_text("{}")          # worker bookkeeping: not shipped
    (d / ".compile.log").write_text("log")           # hidden: not shipped
    (d / "data.bin").write_bytes(b"\x00")            # not a text source: not shipped
    assert sorted(collect_files(d)) == ["kernel.cu", "sub/h.cuh", "submission.py"]
    (d / "eval.py").write_text("evil")
    with pytest.raises(ValueError, match="eval.py"):
        collect_files(d)
    (d / "eval.py").unlink()
    (d / "submission.py").unlink()
    with pytest.raises(ValueError, match="submission.py"):
        collect_files(d)
