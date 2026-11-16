import shutil

import pytest

from agent import precheck as pc

TRITON = "import triton\ndef custom_kernel(d):\n    return d\n"
CUDA_OK = '''
import torch
from torch.utils.cpp_extension import load_inline
SRC = """__global__ void k(float* x, int n) { int i = threadIdx.x; if (i < n) x[i] *= 2.f; }"""
mod = load_inline(name="m", cpp_sources="", cuda_sources=SRC, functions=[])
def custom_kernel(d):
    return d
'''
CUDA_TORCH = '''
from torch.utils.cpp_extension import load_inline
mod = load_inline("m", ["x"], ["#include <torch/extension.h>\\n__global__ void k() {}"], functions=[])
def custom_kernel(d):
    return d
'''
PY = "def custom_kernel(d):\n    return d\n"
GOOD_CU = "__global__ void k(float* x) { x[threadIdx.x] *= 2.f; }\n"
needs_nvcc = pytest.mark.skipif(not shutil.which("nvcc"), reason="no local nvcc")


def test_extracts_named_constant_and_positional_sources():
    assert len(pc.extract_cuda_sources(CUDA_OK)) == 1 and "__global__" in pc.extract_cuda_sources(CUDA_OK)[0]
    assert len(pc.extract_cuda_sources(CUDA_TORCH)) == 1


def test_triton_is_skipped_not_rejected(monkeypatch):
    monkeypatch.setattr(pc.godbolt, "compile_cuda", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    r = pc.precheck(TRITON)
    assert r["ok"] and r["stage"] == "passed" and "Triton" in r["notes"][0]


def test_syntax_error_stops_at_python_stage():
    r = pc.precheck("def custom_kernel(:")
    assert not r["ok"] and r["stage"] == "python"


def test_syntax_error_in_a_helper_file_names_the_file():
    r = pc.precheck({"submission.py": PY, "helper.py": "def f(:"})
    assert not r["ok"] and r["stage"] == "python" and "helper.py" in r["errors"][0]


def test_guard_violation():
    r = pc.precheck("import subprocess\ndef custom_kernel(d): pass")
    assert not r["ok"] and r["stage"] == "guard"


def test_native_guard_rejects_side_streams_in_cu_files():
    r = pc.precheck({"submission.py": PY, "k.cu": "void f(){ cudaStream_t s; cudaStreamCreate(&s); }"}, use_local_nvcc=False)
    assert not r["ok"] and r["stage"] == "guard" and "k.cu" in r["errors"][0]


def test_godbolt_is_off_unless_asked(monkeypatch):
    monkeypatch.setattr(pc.godbolt, "compile_cuda", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert pc.precheck(CUDA_OK)["ok"]


def test_godbolt_error_rejects_when_enabled(monkeypatch):
    monkeypatch.setattr(pc.godbolt, "compile_cuda",
                        lambda src, arch: {"status": "error", "errors": "identifier x is undefined", "compiler": "nvcc1"})
    r = pc.precheck(CUDA_OK, use_godbolt=True)
    assert not r["ok"] and r["stage"] == "godbolt" and "undefined" in r["errors"][0]


def test_godbolt_outage_does_not_block(monkeypatch):
    monkeypatch.setattr(pc.godbolt, "compile_cuda",
                        lambda src, arch: {"status": "unavailable", "errors": "down", "compiler": None})
    assert pc.precheck(CUDA_OK, use_godbolt=True)["ok"]


def test_torch_header_source_is_skipped(monkeypatch):
    monkeypatch.setattr(pc.godbolt, "compile_cuda", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    r = pc.precheck(CUDA_TORCH, use_godbolt=True)
    assert r["ok"] and any("torch headers" in n for n in r["notes"])


@needs_nvcc
def test_local_nvcc_accepts_good_cu_and_reports_resources():
    r = pc.precheck({"submission.py": PY, "k.cu": GOOD_CU})
    assert r["ok"] and r["resources"] and r["resources"][0]["regs"] > 0


@needs_nvcc
def test_local_nvcc_rejects_broken_cu_with_the_compiler_message():
    r = pc.precheck({"submission.py": PY, "k.cu": "__global__ void k(float* x) { x[threadIdx.x] = nope; }"})
    assert not r["ok"] and r["stage"] == "nvcc" and "nope" in r["errors"][0]


def test_cu_with_torch_includes_is_not_compiled_locally():
    r = pc.precheck({"submission.py": PY, "k.cu": '#include <torch/extension.h>\n__global__ void k() {}'})
    assert r["ok"] and any("need torch headers" in n or "torch headers" in n for n in r["notes"])
