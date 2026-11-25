"""The ./optk workflow end to end with a fake Modal: scaffold -> compile -> submit -> status/show/compare/best/journal."""
import json
import shutil

import pytest

from agent import cli, config

from conftest import FakeEvaluator, failing_result, passing_result, profile_ok

pytestmark = pytest.mark.usefixtures("runs_dir")


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def use_fake(monkeypatch, result):
    monkeypatch.setattr(cli, "evaluator_factory", lambda verbose: (lambda: FakeEvaluator(result)))


def test_the_whole_loop(capsys, monkeypatch, runs_dir):
    use_fake(monkeypatch, passing_result(ms=(2000.0, 8000.0), profile=profile_ok(dram_each=6_000_000_000, sm=30, tensor=10)))
    code, out = run(capsys, "submit", "--seed", "--problem", "moe")
    assert code == 0 and "#1 PASS" in out

    code, out = run(capsys, "new-kernel", "--problem", "moe", "--name", "w1", "--from-best", "--strategy", "smem tiling",
                    "--hypothesis", "tile through shared memory")
    kdir = runs_dir / "moe" / "scratch" / "w1"
    assert code == 0 and (kdir / "submission.py").exists()
    meta = json.loads((kdir / "candidate.json").read_text())
    assert meta["parent"] == 1 and meta["strategy"] == "smem tiling" and meta["worker"] == "w1"
    (kdir / "kernel.cu").write_text("__global__ void k() {}\n")

    use_fake(monkeypatch, passing_result(ms=(1000.0, 4000.0), profile=profile_ok(dram_each=2_000_000_000, sm=70, tensor=60)))
    code, out = run(capsys, "submit", str(kdir), "--problem", "moe")
    assert code == 0 and "#2 PASS" in out and "2.00x vs parent #1" in out
    assert json.loads((kdir / "candidate.json").read_text())["submitted"] == [2]

    code, out = run(capsys, "status", "--problem", "moe")
    assert "baseline   #1" in out and "best       #2" in out and "smem tiling" in out and "2.00x vs baseline" in out
    assert "% of its roof" in out                                         # roofline position of the best kernel
    code, out = run(capsys, "show", "2", "--problem", "moe")
    assert "# Iteration #2" in out and "tile through shared memory" in out
    code, out = run(capsys, "compare", "1", "2", "--problem", "moe")
    assert "#1" in out and "#2" in out and "improved: DRAM traffic" in out
    code, out = run(capsys, "best", "--problem", "moe")
    assert "best #2" in out and str(runs_dir / "moe" / "best") in out
    code, out = run(capsys, "journal", "--problem", "moe")
    assert "#1" in out and "tile through shared memory" in out
    code, out = run(capsys, "problems")
    assert "moe:" in out and "best #2" in out


def test_submit_demands_a_hypothesis_and_a_baseline(capsys, monkeypatch, runs_dir):
    use_fake(monkeypatch, passing_result())
    run(capsys, "new-kernel", "--problem", "moe", "--name", "w1")
    kdir = runs_dir / "moe" / "scratch" / "w1"
    with pytest.raises(SystemExit, match="hypothesis"):
        cli.main(["submit", str(kdir), "--problem", "moe"])
    with pytest.raises(SystemExit, match="no baseline"):
        cli.main(["submit", str(kdir), "--problem", "moe", "--hypothesis", "h"])


def test_failed_submit_exits_1_and_prints_the_error(capsys, monkeypatch, runs_dir):
    use_fake(monkeypatch, passing_result())
    run(capsys, "submit", "--seed", "--problem", "moe")
    run(capsys, "new-kernel", "--problem", "moe", "--name", "w2", "--hypothesis", "h")
    use_fake(monkeypatch, failing_result())
    code, out = run(capsys, "submit", str(runs_dir / "moe" / "scratch" / "w2"), "--problem", "moe")
    assert code == 1 and "#2 FAIL" in out and "mismatch at [0]" in out


def test_second_seed_needs_force(capsys, monkeypatch):
    use_fake(monkeypatch, passing_result())
    run(capsys, "submit", "--seed", "--problem", "moe")
    with pytest.raises(SystemExit, match="already has a baseline"):
        cli.main(["submit", "--seed", "--problem", "moe"])


def test_compile_runs_the_guard_first(capsys, runs_dir):
    run(capsys, "new-kernel", "--problem", "moe", "--name", "bad")
    kdir = runs_dir / "moe" / "scratch" / "bad"
    (kdir / "kernel.cu").write_text("void f() { cudaStream_t s; cudaStreamCreate(&s); }\n")
    code, out = run(capsys, "compile", str(kdir), "--problem", "moe")
    assert code == 1 and "REJECTED by the guard" in out and "cudaStreamCreate" in out


@pytest.mark.skipif(not shutil.which("nvcc"), reason="no local nvcc")
def test_the_cuda_template_compiles_locally_and_reports_resources(capsys, runs_dir):
    run(capsys, "new-kernel", "--problem", "moe", "--name", "t")
    code, out = run(capsys, "compile", str(runs_dir / "moe" / "scratch" / "t"), "--problem", "moe", "--where", "local")
    assert code == 0 and "compiled OK (local)" in out and "regs" in out


def test_new_kernel_refuses_to_overwrite_and_from_best_needs_a_best(capsys):
    run(capsys, "new-kernel", "--problem", "moe", "--name", "dup")
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(["new-kernel", "--problem", "moe", "--name", "dup"])
    with pytest.raises(SystemExit, match="no correct candidate"):
        cli.main(["new-kernel", "--problem", "moe", "--name", "x", "--from-best"])


def test_problem_must_be_named_when_ambiguous_and_must_exist():
    with pytest.raises(SystemExit, match="not usable|known problems"):
        cli.pick_problem("no_such_problem")


def test_continue_budgets_kill_switch_and_plateau_modes(capsys, monkeypatch, runs_dir, tmp_path):
    import time
    from agent import budget
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    use_fake(monkeypatch, passing_result(ms=(2000.0, 8000.0)))
    run(capsys, "submit", "--seed", "--problem", "moe")
    code, out = run(capsys, "continue", "--problem", "moe")
    assert code == 1 and "no continuous run is active" in out
    code, out = run(capsys, "continue", "--start", "--max-hours", "1", "--max-gpu-hours", "1", "--patience", "2", "--problem", "moe")
    assert code == 0 and "CONTINUE" in out and "EXPLOIT" in out
    # candidates that do not clear the 1% noise floor push the run into explore mode
    store = cli.open_store("moe", None)
    for ms in (1000.0, 998.0, 999.0, 1001.0):
        c = store.reserve(kind="improve", parent_id=1)
        store.finish(c.id, status="pass", geomean_ns=ms * 1e6, wall_s=10.0)
    assert budget.plateau(store) == {"last_significant": 2, "since": 3, "best_ns": 998.0e6}
    code, out = run(capsys, "continue", "--problem", "moe")
    assert code == 0 and "EXPLORE" in out
    # the kill switch stops it, clearing resumes it
    run(capsys, "continue", "--stop")
    code, out = run(capsys, "continue", "--problem", "moe")
    assert code == 1 and "STOP file present" in out
    run(capsys, "continue", "--clear")
    assert run(capsys, "continue", "--problem", "moe")[0] == 0
    # budgets
    ok, why = budget.check(store, now=time.time() + 2 * 3600)
    assert not ok and "wall-clock budget used" in why
    st = json.loads(budget.state_file(store).read_text())
    st["max_gpu_hours"] = 0.001
    budget.state_file(store).write_text(json.dumps(st))
    assert "GPU budget used" in budget.check(store)[1]
