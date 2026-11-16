from types import SimpleNamespace as NS

import pytest

from harness.evaluate import Evaluator, stage


def evaluator_with(remote, restarts):
    ev = Evaluator()
    ev._runner = NS(run_eval=NS(remote=remote))
    ev._restart = lambda: restarts.append(1)
    return ev


def test_stopped_modal_app_is_restarted_and_the_call_retried():
    calls, restarts = [], []

    def remote(*a):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("ConflictError: function fu-123 is stopped")
        return "ok"

    assert evaluator_with(remote, restarts)._remote("x") == "ok" and restarts == [1] and len(calls) == 2


def test_other_errors_are_not_swallowed_by_the_restart_logic():
    restarts = []

    def remote(*a):
        raise ValueError("something else")

    with pytest.raises(ValueError):
        evaluator_with(remote, restarts)._remote("x")
    assert restarts == []


def test_stage_ships_helper_files_but_never_lets_a_candidate_replace_the_evaluator():
    files = stage("moe", {"submission.py": "x", "kernel.cu": "// k", "sub/dir.h": "// h"})
    assert files["kernel.cu"] == "// k" and files["sub/dir.h"] == "// h" and files["eval.py"] != "x"
    for bad in ("eval.py", "reference.py", "../evil.py", "/abs.py", "tests.txt"):
        with pytest.raises(ValueError):
            stage("moe", {"submission.py": "x", bad: "boom"})
