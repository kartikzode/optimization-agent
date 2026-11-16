import pytest
import yaml

from agent import config, problems, roofline

SPEC = ("dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nsharedexperts: 1; nexpertspertoken: 4; "
        "bs: 1; seqlen: {n}; seed: 9371")


def test_moe_flops_and_intensity_match_hand_calculation():
    flops, nbytes = problems.cost("moe", SPEC.format(n=2048))
    routed = 2048 * 4 * 6 * 7168 * 2048
    shared = 2048 * 6 * 7168 * 2048
    router = 2 * 2048 * 7168 * 32
    assert flops == routed + shared + router
    weights = 7168 * 32 + 32 * 3 * 7168 * 2048 + 3 * 7168 * 2048
    assert nbytes == (weights + 2 * 2048 * 7168) * 2
    assert 290 < flops / nbytes < 320          # right around the H100 ridge point (989 / 3.35 = 295)


def test_point_caps_the_roof_and_survives_bad_specs():
    pk = roofline.peaks_for("H100")
    p = roofline.point("moe", SPEC.format(n=2048), 1e9, pk)
    assert p["roof_tflops"] <= 989.0 and abs(p["pct_roof"] - 100 * p["tflops"] / p["roof_tflops"]) < 1e-9
    assert roofline.point("moe", "bs: 1; seed: 2", 1e9, pk) is None                  # keys do not match the model
    assert roofline.point("no-such-problem", SPEC.format(n=1), 1e9, pk) is None
    assert "tflops" not in roofline.point("moe", SPEC.format(n=2048), None, pk)      # shape-only point, no timing


def test_peaks_depend_on_gpu_and_precision():
    assert roofline.peaks_for("H100", "fp16_tc")["tflops"] == 989.0
    assert roofline.peaks_for("H100", "fp32")["tflops"] < 100
    assert roofline.peaks_for("A100", "fp8_tc") is None and roofline.peaks_for("nope") is None


def test_shape_labels_use_only_the_varying_keys():
    assert roofline.shape_labels([SPEC.format(n=2048), SPEC.format(n=8192)]) == ["seqlen 2048", "seqlen 8192"]
    assert roofline.shape_labels(["a: 1; seed: 5"]) == ["shape 1"]


def test_problem_registry_and_peak_kind():
    assert "moe" in problems.list_problems() and problems.has_cost("moe") and problems.peak_kind("moe") == "fp16_tc"
    assert problems.notes("moe").startswith("# moe")                # read from knowledge/problems/moe/notes.md
    assert not (problems.problem_dir("moe") / "notes.md").exists() and not (problems.problem_dir("moe") / "hints.md").exists()


def test_scaffold_creates_a_complete_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROBLEMS", tmp_path / "problems")
    monkeypatch.setattr(config, "KNOWLEDGE", tmp_path / "knowledge")
    problems._cost_module.cache_clear()
    dest = problems.scaffold("vecscale")
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["cost.py", "reference.py", "submission.py", "task.py", "task.yml"]      # nothing the agents write is in problems/
    know = config.KNOWLEDGE / "problems" / "vecscale"
    assert sorted(p.name for p in know.iterdir()) == ["hints.md", "notes.md"]
    assert "__NAME__" not in "".join(p.read_text() for p in [*dest.iterdir(), *know.iterdir()])
    assert "vecscale" in problems.list_problems()
    task = yaml.safe_load((dest / "task.yml").read_text())
    assert task["tests"] and task["benchmarks"]
    assert problems.cost("vecscale", "size: 1000; seed: 1") == (1000.0, 8000.0)
    with pytest.raises(SystemExit, match="already exists"):
        problems.scaffold("vecscale")
    with pytest.raises(SystemExit, match="lowercase"):
        problems.scaffold("Bad-Name")


def test_scaffold_task_keys_are_parseable_by_the_evaluator_regex():
    import re
    task = yaml.safe_load((config.TEMPLATES / "problem" / "task.yml").read_text())
    pat = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"          # eval.py get_test_cases
    for case in task["tests"] + task["benchmarks"]:
        for k, v in case.items():
            assert re.fullmatch(pat, f"{k}: {v}"), (k, v)


def test_require_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROBLEMS", tmp_path)
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "reference.py").write_text("")
    with pytest.raises(SystemExit, match="missing task.py"):
        problems.require("half")
