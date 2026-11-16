"""
Roofline math shared by the iteration briefs and the dashboard.

Hardware peaks are vendor datasheet numbers (dense, no sparsity). The FLOP and minimum-byte model of a problem
lives in problems/<name>/cost.py. A kernel's position on the roofline is

    y: achieved TFLOP/s  = algorithmic FLOPs / benchmark latency          (benchmark runs without a profiler)
    x: arithmetic intensity, two flavours
       algorithmic = FLOPs / minimum bytes     a property of the shape: the best any kernel could do
       measured    = FLOPs / DRAM bytes ncu saw  what this kernel actually did; rises as reuse improves (fusion,
                                                 tiling, caching in smem/L2), and approaches the algorithmic value
"""
from . import problems

PEAKS = {
    "H100": {"name": "H100 SXM", "tbps": 3.35, "tflops": {"fp16_tc": 989.0, "tf32_tc": 494.0, "fp8_tc": 1979.0, "fp32": 67.0}},
    "H200": {"name": "H200", "tbps": 4.8, "tflops": {"fp16_tc": 989.0, "tf32_tc": 494.0, "fp8_tc": 1979.0, "fp32": 67.0}},
    "B200": {"name": "B200", "tbps": 8.0, "tflops": {"fp16_tc": 2250.0, "tf32_tc": 1125.0, "fp8_tc": 4500.0, "fp32": 80.0}},
    "A100": {"name": "A100 80GB", "tbps": 2.039, "tflops": {"fp16_tc": 312.0, "tf32_tc": 156.0, "fp32": 19.5}},
    "L4": {"name": "L4", "tbps": 0.3, "tflops": {"fp16_tc": 121.0, "tf32_tc": 60.0, "fp8_tc": 242.0, "fp32": 30.3}},
}
KIND_LABEL = {"fp16_tc": "fp16/bf16 tensor cores", "tf32_tc": "tf32 tensor cores", "fp8_tc": "fp8 tensor cores",
              "fp32": "fp32 CUDA cores"}


def peaks_for(gpu: str, kind: str = "fp16_tc") -> dict | None:
    p = PEAKS.get(gpu.upper())
    if not p or kind not in p["tflops"]:
        return None
    return {"tflops": p["tflops"][kind], "tbps": p["tbps"], "kind": kind,
            "label": f"{p['name']}, {KIND_LABEL.get(kind, kind)} (dense)"}


def shape_labels(specs: list[str]) -> list[str]:
    """Short names from the keys that differ between shapes (ignoring seed)."""
    parsed = [problems.parse_spec(s) for s in specs]
    varying = [k for k in parsed[0] if k != "seed" and len({str(p.get(k)) for p in parsed}) > 1] if parsed else []
    return [" · ".join(f"{k} {p[k]}" for k in varying) or f"shape {i + 1}" for i, p in enumerate(parsed)]


def point(problem: str, spec: str, mean_ns: float | None, peaks: dict | None) -> dict | None:
    """Achieved performance of one shape against its roof, or None when the problem has no cost model."""
    c = problems.cost(problem, spec)
    if c is None or not peaks:
        return None
    flops, nbytes = c
    ai = flops / nbytes
    roof = min(peaks["tflops"], peaks["tbps"] * ai)
    out = {"flops": flops, "bytes": nbytes, "intensity": ai, "roof_tflops": roof}
    if mean_ns:
        tflops = flops / (mean_ns * 1e-9) / 1e12
        out.update(tflops=tflops, pct_roof=100 * tflops / roof)
    return out
