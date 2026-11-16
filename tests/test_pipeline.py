import json

from agent import pipeline
from agent.store import Store

from conftest import FakeEvaluator, failing_result, passing_result, profile_ok

KW = dict(gpu="H100", arch="sm_90a", log=lambda *_: None)
PY = {"submission.py": "def custom_kernel(d):\n    return d\n"}


def submit(store, files, ev, **kw):
    return pipeline.submit(store, "moe", files, evaluator_factory=lambda: ev, **KW, **kw)


def test_seed_then_improvement_is_recorded_with_logs_profile_brief_and_journal(tmp_path):
    s = Store(tmp_path / "run")
    seed = submit(s, PY, FakeEvaluator(passing_result(ms=(2000.0, 8000.0), profile=profile_ok(dram_each=6_000_000_000, sm=30, tensor=10))),
                  kind="seed", hypothesis="baseline")
    assert seed.ok and s.seed_ns() == seed.geomean_ns
    files = {"submission.py": PY["submission.py"], "kernel.cu": "__global__ void k(){}"}
    better = submit(s, files, FakeEvaluator(passing_result(ms=(1000.0, 4000.0), profile=profile_ok(dram_each=2_000_000_000, sm=70, tensor=60))),
                    parent_id=seed.id, worker="w1", strategy="smem tiling", hypothesis="tile the GEMM in shared memory")
    d = tmp_path / "run" / "candidates" / f"{better.id:04d}"
    assert (d / "kernel" / "kernel.cu").exists() and (d / "meta.json").exists()
    assert sorted(p.name for p in (d / "logs").iterdir()) == ["benchmark.log", "gpu_info.txt", "precheck.json", "profile.log", "test.log"]
    assert sorted(p.name for p in (d / "profile").iterdir()) == ["ncu.ncu-rep", "ncu_details.csv", "ncu_details.txt", "ncu_light_raw.csv", "ncu_raw.csv", "summary.json"]
    assert (d / "profile" / "ncu.ncu-rep").read_bytes() == b"\x00rep"

    brief = (d / "brief.md").read_text()
    assert "tile the GEMM in shared memory" in brief and "new best" in brief and "2.00× vs parent" in brief
    assert "What changed vs #1" in brief and "DRAM traffic" in brief and "measured arithmetic intensity" in brief
    assert "Nsight Compute's own analysis of `gemm_kernel`" in brief and "limited by the number of required registers" in brief
    assert "(logs/test.log)" in brief and "(profile/ncu.ncu-rep)" in brief            # links to the real files
    journal = (tmp_path / "run" / "JOURNAL.md").read_text()
    assert "#1" in journal and "#2" in journal and "tile the GEMM" in journal
    assert (tmp_path / "run" / "best" / "kernel.cu").exists()                          # best kernel mirrored


def test_profile_summary_has_exact_totals_and_measured_intensity(tmp_path):
    s = Store(tmp_path / "run")
    c = submit(s, PY, FakeEvaluator(passing_result(profile=profile_ok(dram_each=4_000_000_000))), kind="seed", hypothesis="b")
    t = c.profile_summary["totals"]
    assert c.profile_summary["ok"] and t["dram_bytes"] == 5_000_000_000 and t["kernel_launches"] == 2
    assert t["ai_measured"] == t["flops"] / 5e9 and not t["truncated"]
    assert c.profile_summary["shape_label"] == "seqlen 8192"
    assert json.loads((tmp_path / "run" / "candidates" / "0001" / "profile" / "summary.json").read_text())["ok"]


def test_wrong_output_is_recorded_with_the_error_and_is_not_profiled(tmp_path):
    s = Store(tmp_path / "run")
    c = submit(s, PY, FakeEvaluator(failing_result()), hypothesis="h")
    assert c.status == "fail" and "mismatch" in c.error and not c.profile_summary
    brief = (tmp_path / "run" / "candidates" / "0001" / "brief.md").read_text()
    assert "No latency: fail" in brief and "mismatch at [0]" in brief and "Not profiled" in brief


def test_guard_rejection_never_reaches_modal(tmp_path):
    s = Store(tmp_path / "run")
    ev = FakeEvaluator(passing_result())
    c = submit(s, {"submission.py": "import subprocess\ndef custom_kernel(d): pass\n"}, ev, hypothesis="h")
    assert c.status == "rejected" and "subprocess" in c.error and ev.calls == []


def test_unavailable_ncu_keeps_the_latency_and_explains(tmp_path):
    s = Store(tmp_path / "run")
    prof = {"spec": "s", "status": "ncu_unavailable", "note": "Nsight Compute produced no report (exit 9): clock lock failed",
            "torch_table": "aten::mm  ...", "artifacts": {}}
    c = submit(s, PY, FakeEvaluator(passing_result(profile=prof)), kind="seed", hypothesis="b")
    assert c.ok and c.profile_summary["ok"] is False
    brief = (tmp_path / "run" / "candidates" / "0001" / "brief.md").read_text()
    assert "clock lock failed" in brief and "aten::mm" in brief
    assert "profile unavailable" in pipeline.terminal_report(s, c)


def test_modal_trouble_is_an_infra_status_not_a_failed_kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_: None)
    s = Store(tmp_path / "run")
    ev = FakeEvaluator(exc=ConnectionError("modal unreachable"))
    c = submit(s, PY, ev, hypothesis="h")
    assert c.status == "infra" and "modal unreachable" in c.error and len(ev.calls) == 3      # retried
    assert s.best() is None


def test_no_profile_flag_skips_ncu_in_the_request(tmp_path):
    s = Store(tmp_path / "run")
    ev = FakeEvaluator(passing_result(profile=None))
    ev.result["profile"] = None
    c = submit(s, PY, ev, kind="seed", hypothesis="b", profile=False)
    assert c.ok and ev.calls[0][2]["profile"] is False and not c.profile_summary


def test_terminal_report_names_status_latency_comparisons_and_paths(tmp_path):
    s = Store(tmp_path / "run")
    seed = submit(s, PY, FakeEvaluator(passing_result(ms=(2000.0, 8000.0))), kind="seed", hypothesis="b")
    c = submit(s, PY, FakeEvaluator(passing_result(ms=(1000.0, 4000.0))), parent_id=seed.id, hypothesis="h", worker="w3")
    text = pipeline.terminal_report(s, c)
    assert "#2 PASS" in text and "2.00x vs parent #1" in text and "NEW BEST" in text
    assert "brief:" in text and str(tmp_path / "run" / "candidates" / "0002" / "brief.md") in text
    assert "vs #1:" in text
