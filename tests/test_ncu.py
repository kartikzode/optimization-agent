from harness import ncu

HEADER = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC",'
          '"gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum",'
          '"sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
          '"dram__throughput.avg.pct_of_peak_sustained_elapsed","launch__registers_per_thread",'
          '"sm__warps_active.avg.pct_of_peak_sustained_active","sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active"')
UNITS = '"","","","","","","","","","","","nsecond","byte","byte","%","%","%","register/thread","%","%"'


def row(i, name, ns, rd, wr, sm, mem, dram, regs, occ, tensor):
    return (f'"{i}","1","python","host","{name}","1","7","(256, 1, 1)","(132, 1, 1)","0","9.0",'
            f'"{ns:,}","{rd:,}","{wr:,}","{sm}","{mem}","{dram}","{regs}","{occ}","{tensor}"')


CSV = "\n".join([
    "==PROF== Connected to process 1 (/usr/bin/python3)",
    HEADER, UNITS,
    row(0, "gemm_kernel(float *, float *)", 1_000_000, 2_000_000_000, 500_000_000, 70.5, 40.2, 38.0, 168, 24.1, 58.0),
    row(1, "gemm_kernel(float *, float *)", 1_000_000, 2_000_000_000, 500_000_000, 71.5, 41.8, 39.0, 168, 23.9, 60.0),
    row(2, "silu_mul(float *)", 500_000, 100_000_000, 100_000_000, 10.0, 80.0, 75.0, 24, 70.0, ""),
])


def test_aggregates_launches_by_kernel_and_weights_by_time():
    s = ncu.summarize(CSV)
    assert s["ok"] and [k["short"] for k in s["kernels"]] == ["gemm_kernel", "silu_mul"]
    g = s["kernels"][0]
    assert g["launches"] == 2 and g["duration_ns"] == 2_000_000 and g["regs"] == 168
    assert abs(g["sm_pct"] - 71.0) < 1e-9 and abs(g["share"] - 2_000_000 / 2_500_000) < 1e-9
    assert g["bound"] == "compute-bound" and s["kernels"][1]["bound"] == "memory-bound"
    t = s["totals"]
    assert t["kernel_launches"] == 3 and t["distinct_kernels"] == 2 and t["gpu_time_ns"] == 2_500_000
    assert t["dram_bytes"] == 2 * 2_500_000_000 + 200_000_000 and t["max_regs"] == 168


def test_measured_intensity_is_algorithmic_flops_over_dram_bytes():
    s = ncu.summarize(CSV, flops=1e12, min_bytes=1e9)
    t = s["totals"]
    assert abs(t["ai_measured"] - 1e12 / 5.2e9) < 1e-6
    assert abs(t["ai_algorithmic"] - 1000.0) < 1e-9 and t["traffic_vs_min"] > 1     # more traffic than the minimum


def test_missing_metrics_are_none_not_errors():
    only_time = '"ID","Kernel Name","gpu__time_duration.sum"\n"","","nsecond"\n"0","k","2,000"'
    s = ncu.summarize(only_time)
    assert s["ok"] and s["kernels"][0]["duration_ns"] == 2000 and s["kernels"][0]["sm_pct"] is None
    assert s["totals"]["dram_bytes"] is None and s["kernels"][0]["bound"] == "unknown"


def test_units_are_converted_when_ncu_does_not_use_base_units():
    csv_text = '"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum"\n"","","usecond","Gbyte"\n"0","k","5","2"'
    k = ncu.parse_kernels(csv_text)[0]
    assert k["duration_ns"] == 5000 and k["dram_read"] == 2e9


def test_no_rows_means_unavailable_with_a_reason():
    s = ncu.summarize("==ERROR== nothing\n")
    assert s["ok"] is False and "no kernel rows" in s["reason"]


def test_long_template_kernel_names_are_shortened():
    n = "void cutlass::Kernel<cutlass::gemm::kernel::GemmUniversal<very::long::template::args, more, and, more>>(Params)"
    assert len(ncu.short_name(n)) <= 60 and "(" not in ncu.short_name(n)


def test_classify_latency_bound_when_nothing_is_busy():
    assert ncu.classify({"sm_pct": 12, "mem_pct": 20, "dram_pct": 15}) == "latency-bound"
    assert ncu.classify({}) == "unknown"


LIGHT_HEADER = ('"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","launch__registers_per_thread"')
LIGHT_UNITS = '"","","nsecond","byte","byte","register/thread"'


def light_csv(n_gemm=5):
    rows = [f'"{i}","gemm_kernel(float *, float *)","1,000,000","2,000,000,000","500,000,000","168"' for i in range(n_gemm)]
    rows.append(f'"{n_gemm}","silu_mul(float *)","500,000","100,000,000","100,000,000","24"')
    return "\n".join([LIGHT_HEADER, LIGHT_UNITS, *rows])


def test_light_pass_gives_exact_totals_and_the_detailed_pass_gives_the_diagnosis():
    s = ncu.summarize(CSV, raw_light=light_csv(5), flops=1e12, min_bytes=1e9, launch_limit=3)
    t = s["totals"]
    assert t["kernel_launches"] == 6 and t["detailed_launches"] == 3 and not t["truncated"]
    assert t["dram_bytes"] == 5 * 2_500_000_000 + 200_000_000            # all five gemm launches, not just the sampled two
    assert abs(t["ai_measured"] - 1e12 / t["dram_bytes"]) < 1e-9
    g = s["kernels"][0]
    assert g["launches"] == 5 and g["detailed_launches"] == 2 and g["bound"] == "compute-bound" and g["sm_pct"] > 70


def test_a_detailed_pass_that_hit_its_launch_limit_withholds_the_intensity():
    s = ncu.summarize(CSV, flops=1e12, min_bytes=1e9, launch_limit=3)       # 3 launches seen, limit 3: probably cut off
    t = s["totals"]
    assert t["truncated"] and t["dram_bytes"] is None and t["ai_measured"] is None and t["ai_algorithmic"] == 1000.0
    assert not ncu.summarize(CSV, flops=1e12, min_bytes=1e9, launch_limit=40)["totals"]["truncated"]
