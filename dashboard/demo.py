"""
Create a SYNTHETIC run so the dashboard can be previewed before a real run has results.
Nothing here was measured: timings, profiles and kernels are made up, and the dashboard shows a banner saying so.

    python -m dashboard.demo        # -> workspace/runs/demo-synthetic
"""
import random
import shutil

from agent import brief, config, problems
from agent.store import Store

SPECS = ["dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nsharedexperts: 1; nexpertspertoken: 4; bs: 1; seqlen: 2048; seed: 9371",
         "dhidden: 7168; dexpert: 2048; nroutedexperts: 32; nsharedexperts: 1; nexpertspertoken: 4; bs: 1; seqlen: 8192; seed: 1212"]
STRATS = ["hoist setup out of the call", "sort tokens by expert", "grouped GEMM kernel", "fuse silu*up epilogue",
          "cp.async pipeline", "WGMMA + TMA", "L2-aware tile order", "persistent kernel"]
HYPS = ["Build nothing per call: use the weights in `data` directly and keep one preallocated output.",
        "Sort (token, expert) pairs on the GPU once; every expert reads a contiguous slice.",
        "One grouped-GEMM launch walks all experts' token blocks instead of a Python loop.",
        "Fuse SiLU(gate) * up into the GEMM epilogue so the intermediate never goes to HBM.",
        "Multi-stage cp.async pipeline to hide global-load latency.",
        "Replace SIMT FMA with WGMMA fed by TMA loads and mbarriers.",
        "Rasterise tiles in groups so concurrent CTAs share A/B tiles in L2.",
        "Persistent CTAs with a tile scheduler to remove wave quantisation."]
WORKERS = ["w1", "w2", "w3"]


def profile(spec_idx, ai, sm, mem, tensor, occ, launches, dram, gpu_ns, flops, min_bytes, bound):
    kernels = [{"name": "grouped_gemm_kernel", "short": "grouped_gemm_kernel", "launches": max(1, launches // 6),
                "duration_ns": gpu_ns * 0.8, "share": 0.8, "bound": bound, "sm_pct": sm, "mem_pct": mem, "dram_pct": mem * 0.7,
                "occupancy_pct": occ, "tensor_pct": tensor, "regs": 168, "smem": 213_000, "dram_bytes": dram * 0.9},
               {"name": "route_topk_kernel", "short": "route_topk_kernel", "launches": 1, "duration_ns": gpu_ns * 0.2,
                "share": 0.2, "bound": "latency-bound", "sm_pct": 12, "mem_pct": 20, "dram_pct": 15, "occupancy_pct": 60,
                "tensor_pct": 0, "regs": 40, "smem": 0, "dram_bytes": dram * 0.1}]
    return {"ok": True, "spec": SPECS[spec_idx], "shape_label": "seqlen 8192", "shape_index": spec_idx, "status": "ok",
            "kernels": kernels, "hidden_kernels": 0,
            "totals": {"kernel_launches": launches, "distinct_kernels": 2, "gpu_time_ns": gpu_ns, "dram_bytes": dram,
                       "max_regs": 168, "detailed_launches": launches, "truncated": False, "sm_pct": sm, "mem_pct": mem,
                       "dram_pct": mem * 0.7, "occupancy_pct": occ, "tensor_pct": tensor, "bound": bound, "flops": flops,
                       "min_bytes": min_bytes, "ai_measured": flops / dram, "ai_algorithmic": flops / min_bytes,
                       "traffic_vs_min": dram / min_bytes}}


def main():
    rd = config.RUNS / "demo-synthetic"
    shutil.rmtree(rd, ignore_errors=True)
    s, rng = Store(rd), random.Random(7)
    for k, v in (("problem", "moe"), ("gpu", "H100"), ("synthetic", True)):
        s.set_meta(k, v)
    costs = [problems.cost("moe", sp) for sp in SPECS]
    flops, min_bytes = costs[1]

    def add(kind, status, base_ms=None, parent=None, i=0, prof=None, err="", worker="", strat=""):
        c = s.reserve(kind=kind, parent_id=parent, worker=worker, strategy=strat, hypothesis=HYPS[i % len(HYPS)] if kind != "seed" else "baseline")
        s.save_kernel(c.id, {"submission.py": f"# synthetic v{c.id}\ndef custom_kernel(d):\n    return d\n"})
        result, geo = {}, None
        if base_ms:
            ns = [base_ms * 1e6, base_ms * 3.9e6]
            result = {"benchmark": {"benchmarks": [{"spec": sp, "mean_ns": n, "std_ns": n * .01, "runs": 20, "status": "pass"}
                                                   for sp, n in zip(SPECS, ns)]}, "precheck": {"notes": ["synthetic"]}}
            geo = (ns[0] * ns[1]) ** 0.5
        c = s.finish(c.id, status=status, geomean_ns=geo, result=result, error=err, wall_s=150, profile_summary=prof)
        s.write_brief(c.id, brief.build_brief(s, c, problem="moe", gpu="H100"))
        s.append_journal(brief.journal_entry(s, c))
        return c

    seed = add("seed", "pass", 6500.0, prof=profile(1, 0, 71, 62, 70, 23, 346, 12.2e9, 9.7e6, flops, min_bytes, "balanced"))
    best, bms = seed, 6500.0
    plan = [("pass", .004), ("crash", 0), ("pass", .6), ("rejected", 0), ("pass", .8), ("fail", 0), ("pass", .7), ("pass", .85),
            ("timeout", 0), ("pass", .8), ("crash", 0), ("pass", .9), ("pass", .95), ("fail", 0), ("pass", .93)]
    for n, (st, f) in enumerate(plan, start=1):
        strat, worker = STRATS[n % len(STRATS)], WORKERS[n % 3]
        if st == "pass":
            ms = bms * f * rng.uniform(.9, 1.05) if n > 1 else 24.0
            progress = min(1.0, n / len(plan))
            prof = profile(1, 0, 30 + 55 * progress + rng.uniform(-4, 4), 62 - 18 * progress + rng.uniform(-4, 4),
                           10 + 80 * progress + rng.uniform(-5, 5), 23 - 8 * progress, max(3, int(346 * (1 - progress)) + 3),
                           max(min_bytes * 1.05, 12.2e9 * (1 - .82 * progress) + rng.uniform(-3e8, 3e8)),
                           max(2e6, ms * 1e6 * .85 * (n > 1)) if n > 1 else 9.7e6, flops, min_bytes,
                           "compute-bound" if progress > .5 else "memory-bound")
            c = add("improve", "pass", ms, parent=best.id, i=n, prof=prof, worker=worker, strat=strat)
            if c.geomean_ns < best.geomean_ns:
                best, bms = c, ms
        else:
            err = {"crash": "Traceback (most recent call last):\n  File \"submission.py\", line 42, in custom_kernel\nRuntimeError: CUDA error: an illegal memory access was encountered",
                   "fail": "mismatch found! custom implementation doesn't match reference: Number of mismatched elements: 3649642",
                   "timeout": "evaluation exceeded its wall-clock limit",
                   "rejected": "nvcc (local, -arch=sm_90a) failed:\nkernel.cu(31): error: identifier \"tile_m\" is undefined"}[st]
            add("improve", st, None, parent=best.id, i=n, err=err, worker=worker, strat=strat)
    print(f"wrote {rd}  (synthetic)")


if __name__ == "__main__":
    main()
