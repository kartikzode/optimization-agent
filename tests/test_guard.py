from agent.guard import check

OK = '''
import torch
import triton
import triton.language as tl
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True

def custom_kernel(data):
    x, w, cfg = data
    return x @ w["a"]
'''

def rules(code):
    return sorted({v.rule for v in check(code)})

def test_clean_submission_passes():
    assert check(OK) == []

def test_missing_entrypoint():
    assert rules("import torch\ndef foo(data): pass") == ["contract"]

def test_syntax_error():
    assert rules("def custom_kernel(:") == ["syntax"]

def test_banned_imports():
    for mod in ("reference", "utils", "subprocess", "threading", "inspect", "importlib"):
        assert "import" in rules(f"import {mod}\ndef custom_kernel(d): pass"), mod
    assert "import" in rules("from reference import ref_kernel\ndef custom_kernel(d): pass")

def test_reads_env_even_through_alias():
    assert "attr" in rules("import os as o\ndef custom_kernel(d):\n    return o.environ['EVAL_SEED']")
    assert "attr" in rules("from os import environ\ndef custom_kernel(d): pass")

def test_side_stream_and_events():
    assert "attr" in rules("import torch\ndef custom_kernel(d):\n    s = torch.cuda.Stream()")
    assert "attr" in rules("from torch import cuda\ndef custom_kernel(d):\n    s = cuda.Stream()")
    assert "attr" in rules("import torch\ndef custom_kernel(d):\n    e = torch.cuda.Event()")

def test_monkeypatching():
    assert "monkeypatch" in rules("import torch\ntorch.cuda.synchronize = lambda: None\ndef custom_kernel(d): pass")
    assert "monkeypatch" in rules("import time\ntime.perf_counter_ns = lambda: 0\ndef custom_kernel(d): pass")
    assert "monkeypatch" in rules("import torch\nsetattr(torch.cuda, 'x', 1)\ndef custom_kernel(d): pass")

def test_builtin_calls():
    assert "call" in rules("def custom_kernel(d):\n    return eval('1')")
    assert "call" in rules("def custom_kernel(d):\n    open('/etc/passwd')")

def test_tf32_flags_are_allowed():
    assert check("import torch\ntorch.backends.cudnn.benchmark = True\ndef custom_kernel(d): pass") == []
