from agent import brief


def summ(time_ns=5e6, dram=4e9, launches=10, sm=40.0, mem=60.0, tensor=20.0, kernels=None, ai=250.0, regs=128):
    ks = kernels or [{"name": "gemm", "short": "gemm", "share": 0.8, "bound": "compute-bound", "launches": 2}]
    return {"ok": True, "kernels": ks, "totals": {"gpu_time_ns": time_ns, "dram_bytes": dram, "kernel_launches": launches,
            "ai_measured": ai, "sm_pct": sm, "mem_pct": mem, "tensor_pct": tensor, "occupancy_pct": 20.0, "max_regs": regs}}


def test_improvements_and_regressions_are_labelled_by_direction():
    before = summ(time_ns=10e6, dram=8e9, launches=40, sm=30, tensor=10, ai=120)
    after = summ(time_ns=5e6, dram=4e9, launches=6, sm=60, tensor=55, ai=240)
    ch = brief.compare_profiles(before, after)
    text = "\n".join(ch)
    assert "improved: kernel time under ncu (relative) 10.000 ms → 5.000 ms (-50%)" in text
    assert "improved: DRAM traffic 8.00 GB → 4.00 GB (-50%)" in text
    assert "improved: kernel launches 40 → 6" in text
    assert "improved: measured arithmetic intensity 120 FLOP/B → 240 FLOP/B" in text
    assert "improved: SM throughput 30% → 60%" in text and "improved: tensor-pipe active 10% → 55%" in text
    worse = brief.compare_profiles(after, before)
    assert any(c.startswith("regressed: DRAM traffic") for c in worse)


def test_changes_below_the_noise_threshold_are_not_reported():
    a = summ()
    b = summ(time_ns=5.1e6, dram=4.05e9, sm=41.0)
    assert brief.compare_profiles(a, b) == []


def test_new_and_removed_kernels_and_a_new_dominant_kernel_are_reported():
    old = summ(kernels=[{"name": "slow", "short": "slow", "share": 0.9, "bound": "latency-bound", "launches": 30}])
    new = summ(kernels=[{"name": "fused", "short": "fused", "share": 0.95, "bound": "compute-bound", "launches": 1}])
    text = "\n".join(brief.compare_profiles(old, new))
    assert "new kernel: fused" in text and "gone: slow (was 90% of time)" in text and "dominant kernel: slow (90%) → fused (95%)" in text


def test_unavailable_profiles_compare_to_nothing():
    assert brief.compare_profiles({"ok": False}, summ()) == [] and brief.compare_profiles({}, {}) == []


def test_formatters():
    assert brief.fmt_ns(6.6e9) == "6.60 s" and brief.fmt_ns(1.5e6) == "1.500 ms" and brief.fmt_ns(2500) == "2.5 µs" and brief.fmt_ns(None) == "–"
    assert brief.fmt_bytes(2.5e9) == "2.50 GB" and brief.fmt_bytes(1500) == "1.50 KB" and brief.fmt_bytes(12) == "12 B"
