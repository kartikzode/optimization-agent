"""Shared fixtures: a fake Modal evaluator that returns canned pipeline results, and real-shaped ncu CSVs."""
import pytest

from agent import config

SPEC = ("dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nsharedexperts: 1; nexpertspertoken: 4; "
        "bs: 1; seqlen: {n}; seed: 9371")

HEADER = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC",'
          '"gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum",'
          '"sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
          '"launch__registers_per_thread","sm__warps_active.avg.pct_of_peak_sustained_active",'
          '"sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active"')
UNITS = '"","","","","","","","","","","","nsecond","byte","byte","%","%","register/thread","%","%"'


def ncu_csv(dram_each: int, sm: float, tensor: float, extra_kernel: bool = True) -> str:
    def row(i, name, ns, dram, sm_, mem, regs, occ, ten):
        return (f'"{i}","1","python","host","{name}","1","7","(256, 1, 1)","(132, 1, 1)","0","9.0",'
                f'"{ns:,}","{dram:,}","0","{sm_}","{mem}","{regs}","{occ}","{ten}"')
    rows = [row(0, "gemm_kernel(float *)", 4_000_000, dram_each, sm, 40.0, 168, 22.0, tensor)]
    if extra_kernel:
        rows.append(row(1, "silu_mul(float *)", 1_000_000, dram_each // 4, 8.0, 70.0, 32, 60.0, 0.0))
    return "\n".join(["==PROF== Connected", HEADER, UNITS, *rows])


def light_csv(dram_each: int, extra_kernel: bool = True) -> str:
    head = '"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","launch__registers_per_thread"'
    rows = [head, '"","","nsecond","byte","byte","register/thread"', f'"0","gemm_kernel(float *)","4,000,000","{dram_each:,}","0","168"']
    if extra_kernel:
        rows.append(f'"1","silu_mul(float *)","1,000,000","{dram_each // 4:,}","0","32"')
    return "\n".join(rows)


def parsed(status="pass", ms=(1000.0, 4000.0), error=None):
    benches = [{"spec": SPEC.format(n=n), "status": "pass", "runs": 10, "mean_ns": m * 1e6, "best_ns": m * 1e6,
                "std_ns": m * 1e4, "error": None} for n, m in zip((2048, 8192), ms)]
    geo = (ms[0] * ms[1]) ** 0.5 * 1e6
    return {"status": status, "returncode": 0, "tests": [], "benchmarks": benches, "geomean_ns": geo, "profiles": [],
            "error": error, "stdout_tail": "", "stderr_tail": ""}


NCU_TEXT = """[447] python3@127.0.0.1
  gemm_kernel (2, 66, 1)x(384, 1, 1), Context 1, Stream 7, Device 0, CC 9.0

    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       307.90
    ----------------------- ----------- ------------

    INF   Utilizing more than 80% of the device.

    Section: Occupancy
    OPT   Est. Local Speedup: 81.25% The theoretical occupancy (18.8%) is limited by the number of required registers,
          and the required amount of shared memory.

    Section: Memory Workload Analysis
    WRN   Uncoalesced global accesses cost 40% of the sectors.

  gemm_kernel (2, 66, 1)x(384, 1, 1), Context 1, Stream 7, Device 0, CC 9.0

    Section: Occupancy
    OPT   Est. Local Speedup: 81.25% same message again.

  silu_mul (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 9.0

    Section: Occupancy
    OPT   silu advice.
"""


def profile_ok(dram_each=3_000_000_000, sm=50.0, tensor=40.0):
    return {"spec": SPEC.format(n=8192), "status": "ok", "note": "", "torch_table": "",
            "artifacts": {"ncu.ncu-rep": b"\x00rep", "ncu_raw.csv": ncu_csv(dram_each, sm, tensor),
                          "ncu_details.csv": "details", "ncu_details.txt": NCU_TEXT, "ncu_light_raw.csv": light_csv(dram_each)}}


class FakeEvaluator:
    """Stands in for harness.evaluate.Evaluator: no Modal, canned results, records what it was asked."""

    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run_candidate(self, problem, files, **kw):
        self.calls.append((problem, sorted(files), kw))
        if self.exc:
            raise self.exc
        return self.result

    def compile(self, problem, files):
        return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "wall_s": 1.0, "ok": True}


def passing_result(ms=(1000.0, 4000.0), profile=None):
    return {"gpu_info": "NVIDIA H100", "logs": {"test.log": "# test ok", "benchmark.log": "# bench ok",
                                                "profile.log": "# ncu ok"},
            "round_trip_s": 1.0, "shape_index": 1, "shape_spec": SPEC.format(n=8192),
            "test": parsed(), "benchmark": parsed(ms=ms), "profile": profile or profile_ok()}


def failing_result(stage="test", status="fail", error="mismatch at [0]: 1.0 vs 2.0"):
    bad = parsed(status=status, error=error)
    return {"gpu_info": "", "logs": {f"{stage}.log": f"# {stage}\nstderr: {error}"}, "round_trip_s": 1.0, "shape_index": 1,
            "shape_spec": SPEC.format(n=8192), "test": bad if stage == "test" else parsed(), "benchmark": None, "profile": None}


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setattr(config, "RUNS", d)
    return d
