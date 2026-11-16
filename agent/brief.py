"""
The per-iteration brief: what was tried, the latency, what the profiler says changed, and where the logs are.

Written to candidates/<id>/brief.md, and a short version is appended to the run's JOURNAL.md. It is generated
from measurements (no LLM): the *why* comes from the hypothesis the worker recorded, the *what happened* from
the benchmark and Nsight Compute numbers, compared with the parent.
"""
import datetime as dt
from pathlib import Path

from . import problems, roofline
from .store import Candidate, Store

PCT_FIELDS = [("sm_pct", "SM throughput"), ("mem_pct", "memory throughput"), ("dram_pct", "DRAM throughput"),
              ("occupancy_pct", "achieved occupancy"), ("tensor_pct", "tensor-pipe active"),
              ("l2_hit", "L2 hit rate"), ("issue_pct", "issue-slot utilisation")]

HINTS = {
    "latency-bound": "neither the math pipes nor memory are busy: look for warp stalls, too few warps/CTAs "
                     "(grid below the SM count), many tiny launches, dependent loads; add parallelism or "
                     "pipelining (cp.async/TMA multi-stage), fuse launches",
    "memory-bound": "limited by DRAM/L2 traffic: raise data reuse (smem tiling, bigger tiles, L2-friendly rasterisation), "
                    "vectorise and coalesce, fuse to avoid HBM round-trips, store narrower types",
    "compute-bound": "math pipes are the limit: make sure tensor cores (WGMMA/MMA) carry the work, keep them fed "
                     "(async copies, double buffering, warp specialisation), tune tile/cluster shape, trim non-MMA instructions",
    "balanced": "both compute and memory are moderately busy: check the stall reasons and the roofline chart "
                "before choosing between a reuse fix and a pipeline fix",
    "unknown": "no throughput counters: open the .ncu-rep in Nsight Compute",
}


def fmt_ns(ns) -> str:
    if ns is None:
        return "–"
    return f"{ns / 1e9:.2f} s" if ns >= 1e9 else f"{ns / 1e6:.3f} ms" if ns >= 1e6 else f"{ns / 1e3:.1f} µs" if ns >= 1e3 else f"{ns:.0f} ns"


def fmt_bytes(b) -> str:
    if b is None:
        return "–"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def _pct(x) -> str:
    return "–" if x is None else f"{x:.0f}%"


def _change(old, new, lower_is_better: bool, fmt, label, min_rel=0.05) -> str | None:
    if old is None or new is None or old == 0:
        return None
    rel = (new - old) / abs(old)
    if rel == 0 or abs(rel) < min_rel:
        return None
    better = (rel < 0) == lower_is_better
    return f"{'improved' if better else 'regressed'}: {label} {fmt(old)} → {fmt(new)} ({rel:+.0%})"


def compare_profiles(prev: dict, cur: dict) -> list[str]:
    """Bullets about what changed between two profile summaries (harness.ncu.summarize output)."""
    if not (prev and prev.get("ok") and cur and cur.get("ok")):
        return []
    p, c = prev["totals"], cur["totals"]
    out = [x for x in (
        _change(p.get("gpu_time_ns"), c.get("gpu_time_ns"), True, fmt_ns, "kernel time under ncu (relative)"),
        _change(p.get("kernel_launches"), c.get("kernel_launches"), True, lambda v: f"{v:.0f}", "kernel launches", 0.0),
        _change(p.get("dram_bytes"), c.get("dram_bytes"), True, fmt_bytes, "DRAM traffic"),
        _change(p.get("ai_measured"), c.get("ai_measured"), False, lambda v: f"{v:.0f} FLOP/B", "measured arithmetic intensity"),
        _change(p.get("max_regs"), c.get("max_regs"), True, lambda v: f"{v:.0f}", "max registers/thread", 0.15),
    ) if x]
    for f, label in PCT_FIELDS:
        a, b = p.get(f), c.get(f)
        if a is not None and b is not None and abs(b - a) >= 4:
            up_good = f in ("sm_pct", "tensor_pct", "occupancy_pct", "l2_hit", "issue_pct")
            tag = ("improved" if (b > a) == up_good else "regressed") if f != "mem_pct" and f != "dram_pct" else "changed"
            out.append(f"{tag}: {label} {_pct(a)} → {_pct(b)}")
    old, new = {k["name"]: k for k in prev["kernels"]}, {k["name"]: k for k in cur["kernels"]}
    for n, k in new.items():
        if n not in old:
            out.append(f"new kernel: {k['short']} ({k['launches']}×, {k['share']:.0%} of time, {k['bound']})")
    for n, k in old.items():
        if n not in new:
            out.append(f"gone: {k['short']} (was {k['share']:.0%} of time)")
    if prev["kernels"] and cur["kernels"] and prev["kernels"][0]["name"] != cur["kernels"][0]["name"]:
        out.append(f"dominant kernel: {prev['kernels'][0]['short']} ({prev['kernels'][0]['share']:.0%}) → "
                   f"{cur['kernels'][0]['short']} ({cur['kernels'][0]['share']:.0%})")
    return out


def profile_section(summ: dict, comparator: Candidate | None, *, shape_label: str, torch_table: str = "",
                    latency_ns: float | None = None) -> list[str]:
    lines = []
    if not summ or not summ.get("ok"):
        why = (summ or {}).get("reason") or "no profile was collected"
        lines.append(f"_No Nsight Compute profile: {why}_")
        if torch_table:
            lines += ["", "Torch profiler fallback:", "```", torch_table[:3000], "```"]
        return lines
    t = summ["totals"]
    lines.append(f"Nsight Compute on shape **{shape_label}** (relative timing: ncu serialises kernels; "
                 f"the benchmark owns the real latency).")
    lines.append("")
    facts = [f"{t['kernel_launches']} launches ({t['distinct_kernels']} distinct kernels)",
             f"ncu kernel time {fmt_ns(t['gpu_time_ns'])}"]
    if t.get("dram_bytes") is not None:
        facts.append(f"DRAM traffic {fmt_bytes(t['dram_bytes'])}" + (
            f" ({t['traffic_vs_min']:.2f}× the minimum {fmt_bytes(t['min_bytes'])})" if t.get("traffic_vs_min") else ""))
    if t.get("ai_measured"):
        facts.append(f"measured arithmetic intensity {t['ai_measured']:.0f} FLOP/B"
                     + (f" (algorithmic {t['ai_algorithmic']:.0f})" if t.get("ai_algorithmic") else ""))
    lines.append("- " + " · ".join(facts))
    if latency_ns and t.get("gpu_time_ns"):
        busy = t["gpu_time_ns"] / latency_ns
        if busy < 0.5:
            lines.append(f"- ⚠ **host-bound**: the GPU kernels add up to {fmt_ns(t['gpu_time_ns'])} (ncu, serialised) but one call takes "
                         f"{fmt_ns(latency_ns)}: the GPU is busy about {busy:.1%} of the time. Look at launch count, host syncs, "
                         "per-call allocation/initialisation and Python loops before touching a kernel.")
        else:
            lines.append(f"- GPU busy ≈ {min(busy, 1.0):.0%} of the measured call (kernel time {fmt_ns(t['gpu_time_ns'])} of {fmt_ns(latency_ns)}).")
    if t.get("truncated"):
        lines.append(f"- ⚠ the detailed pass hit its launch limit ({t['detailed_launches']} launches) and the call has more: DRAM "
                     "traffic and measured intensity are withheld (they would be wrong). Re-run `./optk profile --cand N --launch-count 200`.")
    elif t.get("detailed_launches") and t["detailed_launches"] < t["kernel_launches"]:
        lines.append(f"- throughput / occupancy metrics are from the first {t['detailed_launches']} of {t['kernel_launches']} "
                     "launches; time, DRAM traffic and intensity cover all of them.")
    lines.append(f"- SM {_pct(t.get('sm_pct'))} · memory {_pct(t.get('mem_pct'))} · DRAM {_pct(t.get('dram_pct'))} · "
                 f"occupancy {_pct(t.get('occupancy_pct'))} · tensor pipe {_pct(t.get('tensor_pct'))} · "
                 f"max regs/thread {int(t['max_regs']) if t.get('max_regs') else '–'}")
    if summ["kernels"]:
        top = summ["kernels"][0]
        lines.append(f"- **Verdict:** dominant kernel `{top['short']}` ({top['share']:.0%} of time) is {top['bound']}. "
                     f"{HINTS.get(top['bound'], '')}")
    adv = summ.get("advice")
    if adv:
        lines += ["", f"### Nsight Compute's own analysis of `{adv['kernel'][:60]}`", ""]
        lines += [f"- **{m['kind']}** ({m['section']}): {m['text']}" for m in adv["messages"]]
    if comparator is not None:
        ch = compare_profiles(comparator.profile_summary, summ)
        lines += ["", f"### What changed vs #{comparator.id}", *([f"- {c}" for c in ch] or ["- no profile metric moved by more than the noise threshold"])]
    lines += ["", "### Top kernels", "", "| kernel | launches | ncu time | share | bound | SM | mem | occ | regs | smem |",
              "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for k in summ["kernels"]:
        lines.append(f"| `{k['short']}` | {k['launches']} | {fmt_ns(k['duration_ns'])} | {k['share']:.0%} | {k['bound']} | "
                     f"{_pct(k.get('sm_pct'))} | {_pct(max(k.get('mem_pct') or 0, k.get('dram_pct') or 0) or None)} | "
                     f"{_pct(k.get('occupancy_pct'))} | {int(k['regs']) if k.get('regs') else '–'} | {fmt_bytes(k['smem']) if k.get('smem') else '–'} |")
    if summ.get("hidden_kernels"):
        lines.append(f"\n_{summ['hidden_kernels']} more small kernels not shown (see profile/ncu_raw.csv)._")
    return lines


def _bench(c: Candidate) -> list[dict]:
    return (c.result.get("benchmark") or {}).get("benchmarks", [])


def gpu_busy(c: Candidate) -> float | None:
    """Fraction of one call the GPU spends in kernels (ncu kernel time / benchmark latency of the profiled shape).
    ncu serialises and flushes caches, so its time is an upper bound on real GPU time: a small ratio is conclusive."""
    ps = c.profile_summary or {}
    t, idx, bm = ps.get("totals") or {}, ps.get("shape_index"), _bench(c)
    if not t.get("gpu_time_ns") or idx is None or idx >= len(bm) or not bm[idx].get("mean_ns"):
        return None
    return t["gpu_time_ns"] / bm[idx]["mean_ns"]


def verdict(c: Candidate) -> str:
    """What limits the call as a whole: host-bound when the GPU is mostly idle, else the kernels' own bound."""
    busy = gpu_busy(c)
    if busy is not None and busy < 0.5:
        return "host-bound"
    return ((c.profile_summary or {}).get("totals") or {}).get("bound", "")


def _links(store: Store, c: Candidate) -> list[str]:
    d = store.cand_dir(c.id)
    items = [("precheck", "logs/precheck.json"), ("test log", "logs/test.log"), ("benchmark log", "logs/benchmark.log"),
             ("profile log", "logs/profile.log"), ("torch profile log", "logs/torch_profile.log"),
             ("Nsight Compute report (open in ncu-ui)", "profile/ncu.ncu-rep"), ("ncu raw metrics (csv)", "profile/ncu_raw.csv"),
             ("ncu details (csv)", "profile/ncu_details.csv"), ("profile summary (json)", "profile/summary.json"),
             ("kernel sources", "kernel/")]
    return [f"- [{label}]({rel})" for label, rel in items if (d / rel).exists()]


def build_brief(store: Store, c: Candidate, *, problem: str, gpu: str, torch_table: str = "") -> str:
    parent = store.get(c.parent_id) if c.parent_id else None
    prior = [x for x in store.all() if x.id < c.id and x.ok]
    best_before = min(prior, key=lambda x: x.geomean_ns) if prior else None
    seed = store.seed()
    peaks = roofline.peaks_for(gpu, problems.peak_kind(problem))
    when = dt.datetime.fromtimestamp(c.created_at).strftime("%Y-%m-%d %H:%M")
    title = c.strategy or c.hypothesis.split(".")[0][:80] or c.kind
    L = [f"# Iteration #{c.id}: {title}", "",
         f"`{c.status}` · {c.kind} · parent {'#' + str(c.parent_id) if c.parent_id else '–'} · "
         f"worker {c.worker or '–'} · {when} · evaluation {c.wall_s:.0f} s", "",
         "## What was tried", "", c.hypothesis or "_no hypothesis recorded_", ""]

    L += ["## Latency", ""]
    if c.ok:
        refs = []
        for label, ref in (("parent", parent), ("best so far", best_before), ("baseline", seed)):
            if ref is not None and ref.ok and ref.id != c.id:
                refs.append(f"{ref.geomean_ns / c.geomean_ns:.2f}× vs {label} #{ref.id} ({fmt_ns(ref.geomean_ns)})")
        star = " ★ **new best**" if best_before is None or c.geomean_ns < best_before.geomean_ns else ""
        L.append(f"**{fmt_ns(c.geomean_ns)}** (geometric mean over the benchmark shapes){star}")
        if refs:
            L.append("- " + "\n- ".join(refs))
        specs = [b["spec"] for b in _bench(c)]
        labels = roofline.shape_labels(specs)
        L += ["", "| shape | latency | vs parent | achieved TFLOP/s | % of roof |", "|---|---:|---:|---:|---:|"]
        pb = _bench(parent) if parent and parent.ok else []
        for i, b in enumerate(_bench(c)):
            pt = roofline.point(problem, b["spec"], b["mean_ns"], peaks) or {}
            vs = f"{pb[i]['mean_ns'] / b['mean_ns']:.2f}×" if i < len(pb) and pb[i].get("mean_ns") and b.get("mean_ns") else "–"
            L.append(f"| {labels[i]} | {fmt_ns(b['mean_ns'])} ± {fmt_ns(b.get('std_ns'))} | {vs} | "
                     f"{pt['tflops']:.1f} | {pt['pct_roof']:.1f}% |" if pt else
                     f"| {labels[i]} | {fmt_ns(b['mean_ns'])} ± {fmt_ns(b.get('std_ns'))} | {vs} | – | – |")
    else:
        L.append(f"**No latency: {c.status}.**")
        if c.error:
            L += ["", "```", c.error[-1800:], "```"]
        res = ((c.result.get("precheck") or {}).get("resources")) or []
        if res:
            L += ["", "ptxas resources before the failure:", *[f"- `{r['kernel'][:60]}`: {r['regs']} regs, {r['smem']} B smem" for r in res]]
    notes = (c.result.get("precheck") or {}).get("notes") or []
    if notes:
        L += ["", "Pre-check notes: " + "; ".join(notes)]

    L += ["", "## Profile", ""]
    comparator = None
    for a in ([parent] if parent else []) + store.ancestors(c.id)[1:] + ([seed] if seed else []):
        if a is not None and a.id != c.id and a.profile_summary.get("ok"):
            comparator = a
            break
    prof_meta = c.profile_summary or {}
    if c.ok or prof_meta:
        shape = prof_meta.get("shape_label") or prof_meta.get("spec") or "–"
        idx = prof_meta.get("shape_index")
        bm = _bench(c)
        lat = bm[idx]["mean_ns"] if idx is not None and idx < len(bm) else None
        L += profile_section(prof_meta, comparator, shape_label=shape, torch_table=torch_table, latency_ns=lat)
    else:
        L.append("_Not profiled: the candidate did not pass correctness and benchmarking._")

    L += ["", "## Logs and artifacts", "", f"Directory: `{store.cand_dir(c.id)}`", "", *_links(store, c), ""]
    return "\n".join(L)


def journal_entry(store: Store, c: Candidate) -> str:
    """A few lines for JOURNAL.md: enough for the orchestrator to rebuild context after a compaction."""
    parent = store.get(c.parent_id) if c.parent_id else None
    head = f"### #{c.id} · {c.status} · {c.strategy or c.kind} · worker {c.worker or '–'}"
    if c.ok:
        sp = f"{parent.geomean_ns / c.geomean_ns:.2f}× vs #{parent.id}, " if parent and parent.ok else ""
        res = f"{fmt_ns(c.geomean_ns)} ({sp}{(store.seed_ns() or c.geomean_ns) / c.geomean_ns:.2f}× vs baseline)"
        t = (c.profile_summary or {}).get("totals") or {}
        if t:
            res += (f"; DRAM {fmt_bytes(t.get('dram_bytes'))}, AI {t['ai_measured']:.0f} FLOP/B, " if t.get("ai_measured") else "; ") \
                   + f"SM {_pct(t.get('sm_pct'))}, mem {_pct(t.get('mem_pct'))}, {verdict(c)}"
    else:
        res = " ".join((c.error or "").split())[:240]
    return f"{head}\n- tried: {' '.join((c.hypothesis or '').split())[:300]}\n- result: {res}\n- brief: {store.cand_dir(c.id) / 'brief.md'}"
