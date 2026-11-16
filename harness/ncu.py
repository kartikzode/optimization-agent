"""
Nsight Compute output -> a compact, comparable summary.

Input is `ncu --import X.ncu-rep --page raw --csv --print-units base`: one row per profiled kernel launch, one column
per metric. We keep the metrics that answer "what is this kernel bound by and did the last change help", aggregate
launches of the same kernel, and total the whole call.

Caveat for readers: ncu serialises kernels and locks clocks, so durations here are *relative*. Use them to compare
iterations and to see where time goes; the benchmark stage owns the real latency.
"""
import csv
import io
import re

# summary field -> candidate metric names (first one present wins; names differ a little between ncu versions)
METRICS = {
    "duration_ns": ["gpu__time_duration.sum"],
    "grid": ["launch__grid_size"],
    "block": ["launch__block_size"],
    "regs": ["launch__registers_per_thread"],
    "smem_static": ["launch__shared_mem_per_block_static"],
    "smem_dynamic": ["launch__shared_mem_per_block_dynamic"],
    "sm_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "mem_pct": ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
    "dram_pct": ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l1_pct": ["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l2_pct": ["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l1_hit": ["l1tex__t_sector_hit_rate.pct"],
    "l2_hit": ["lts__t_sector_hit_rate.pct"],
    "occupancy_pct": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "occupancy_max_pct": ["sm__maximum_warps_per_active_cycle_pct"],
    "issue_pct": ["smsp__issue_active.avg.pct_of_peak_sustained_active"],
    "tensor_pct": ["sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
                   "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
                   "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"],
    "dram_read": ["dram__bytes_read.sum"],
    "dram_write": ["dram__bytes_write.sum"],
}
# time-weighted when averaged over a kernel's launches / over the whole call
AVG_FIELDS = ["sm_pct", "mem_pct", "dram_pct", "l1_pct", "l2_pct", "l1_hit", "l2_hit", "occupancy_pct",
              "occupancy_max_pct", "issue_pct", "tensor_pct"]

_TIME = {"nsecond": 1.0, "usecond": 1e3, "msecond": 1e6, "second": 1e9, "ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
_BYTES = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if not s or s.upper() == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _rows(text: str) -> tuple[list[str], list[str], list[list[str]]]:
    """-> (header, units row, data rows). ncu prints ==PROF== noise before the table; skip to the header."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if start is None:
        return [], [], []
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header, rest = rows[0], rows[1:]
    units: list[str] = []
    if rest and not (rest[0][0] or "").strip().isdigit():   # the units row has an empty ID cell
        units, rest = rest[0], rest[1:]
    return header, units, [r for r in rest if r and (r[0] or "").strip().isdigit()]


def _scale(value: float | None, unit: str, kind: str) -> float | None:
    if value is None:
        return None
    if kind == "time":
        return value * _TIME.get(unit, 1.0)
    if kind == "bytes":
        return value * _BYTES.get(unit, 1.0)
    return value


def short_name(name: str, n: int = 60) -> str:
    name = re.sub(r"\(.*", "", name)            # drop the parameter list
    name = re.sub(r"<.*", "<..>", name) if len(name) > n else name
    return name if len(name) <= n else name[: n - 1] + "…"


def parse_kernels(raw_csv: str) -> list[dict]:
    """One dict per profiled launch with the fields in METRICS (None when a metric was not collected)."""
    header, units, rows = _rows(raw_csv)
    if not header:
        return []
    col = {name: i for i, name in enumerate(header)}
    name_i = col.get("Kernel Name")
    out = []
    for r in rows:
        def get(field: str) -> float | None:
            for m in METRICS[field]:
                i = col.get(m)
                if i is not None and i < len(r):
                    v = _num(r[i])
                    if v is not None:
                        unit = units[i] if i < len(units) else ""
                        kind = "time" if field == "duration_ns" else "bytes" if field in ("dram_read", "dram_write") else ""
                        return _scale(v, unit, kind)
            return None
        k = {f: get(f) for f in METRICS}
        k["name"] = r[name_i] if name_i is not None and name_i < len(r) else "?"
        out.append(k)
    return out


def _wavg(items: list[dict], field: str) -> float | None:
    pairs = [(k[field], k["duration_ns"] or 0.0) for k in items if k.get(field) is not None]
    if not pairs:
        return None
    w = sum(d for _, d in pairs)
    return sum(v * d for v, d in pairs) / w if w else sum(v for v, _ in pairs) / len(pairs)


def classify(k: dict) -> str:
    """A coarse first diagnosis from the speed-of-light percentages (ncu's own rule of thumb)."""
    sm, mem = k.get("sm_pct"), max(k.get("mem_pct") or 0, k.get("dram_pct") or 0)
    if sm is None and not mem:
        return "unknown"
    sm = sm or 0
    if sm < 40 and mem < 40:
        return "latency-bound"          # neither pipe nor memory is busy: stalls, low occupancy, tiny grid, launch gaps
    if mem >= 60 and mem > sm:
        return "memory-bound"
    if sm >= 60 and sm >= mem:
        return "compute-bound"
    return "balanced"


def summarize(raw_full: str = "", *, raw_light: str = "", flops: float | None = None, min_bytes: float | None = None,
              top: int = 10, launch_limit: int | None = None) -> dict:
    """
    {"kernels": [...top kernels aggregated by name...], "totals": {...}, ...}

    Two inputs, because ncu has a cost/coverage trade-off:
      raw_light  a cheap pass (time, DRAM bytes, registers) over EVERY launch in the call: exact totals and shares
      raw_full   a detailed pass (throughputs, occupancy, tensor pipe) over the first `launch_limit` launches: why
    Totals (launches, time, DRAM bytes) come from the light pass when there is one. Without it, a detailed pass that hit
    its launch limit has seen only part of the call, so DRAM bytes and the measured intensity are withheld rather than
    reported wrong (`truncated`).
    flops/min_bytes are the problem's algorithmic cost of the profiled shape; with them we report the *measured*
    arithmetic intensity (algorithmic FLOPs / DRAM bytes the kernels actually moved).
    """
    full = parse_kernels(raw_full) if raw_full else []
    light = parse_kernels(raw_light) if raw_light else []
    launches = light or full
    if not launches:
        return {"ok": False, "reason": "no kernel rows in the ncu CSV (counters unavailable, or nothing in the "
                                       "custom_kernel NVTX range)", "kernels": [], "totals": {}}
    truncated = not light and launch_limit is not None and len(full) >= launch_limit

    detail: dict[str, list[dict]] = {}
    for k in full:
        detail.setdefault(k["name"], []).append(k)
    by_name: dict[str, list[dict]] = {}
    for k in launches:
        by_name.setdefault(k["name"], []).append(k)
    total_ns = sum(k["duration_ns"] or 0 for k in launches)
    kernels = []
    for name, ks in by_name.items():
        t = sum(k["duration_ns"] or 0 for k in ks)
        ds = detail.get(name, [])
        agg = {"name": name, "short": short_name(name), "launches": len(ks), "duration_ns": t,
               "share": t / total_ns if total_ns else 0.0,
               "grid": (ds or ks)[0]["grid"], "block": (ds or ks)[0]["block"],
               "regs": max((k["regs"] or 0) for k in ks) or None,
               "smem": max(((k["smem_static"] or 0) + (k["smem_dynamic"] or 0)) for k in (ds or ks)) or None,
               "dram_bytes": sum((k["dram_read"] or 0) + (k["dram_write"] or 0) for k in ks),
               "detailed_launches": len(ds)}
        for f in AVG_FIELDS:
            agg[f] = _wavg(ds, f) if ds else None
        agg["bound"] = classify(agg) if ds else "not detailed"     # outside the detailed pass's launch window
        kernels.append(agg)
    kernels.sort(key=lambda k: -k["duration_ns"])

    dram = sum(k["dram_bytes"] for k in kernels)
    have_dram = not truncated and any(k["dram_read"] is not None or k["dram_write"] is not None for k in launches)
    totals = {"kernel_launches": len(launches), "distinct_kernels": len(kernels), "gpu_time_ns": total_ns,
              "dram_bytes": dram if have_dram else None, "max_regs": max((k["regs"] or 0) for k in launches) or None,
              "detailed_launches": len(full), "truncated": truncated}
    for f in AVG_FIELDS:
        totals[f] = _wavg(full, f)       # time-weighted over the launches the detailed pass saw
    totals["bound"] = classify(totals)
    if flops is not None:
        totals["flops"] = flops
        totals["min_bytes"] = min_bytes
        totals["ai_measured"] = flops / dram if have_dram and dram > 0 else None
        totals["ai_algorithmic"] = flops / min_bytes if min_bytes else None
        totals["traffic_vs_min"] = dram / min_bytes if have_dram and min_bytes else None   # 1.0 = ideal reuse
    return {"ok": True, "kernels": kernels[:top], "hidden_kernels": max(0, len(kernels) - top), "totals": totals}
