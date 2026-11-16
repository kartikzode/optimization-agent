"""Run store -> JSON payload for the dashboard (also used by the live server)."""
import difflib
import time
from pathlib import Path

from agent import config, problems, roofline
from agent.store import Store

MAX_CODE, MAX_DIFF, MAX_ERR, MAX_BRIEF = 80_000, 30_000, 4_000, 24_000
LINKS = [("Brief", "brief.md"), ("Precheck", "logs/precheck.json"), ("Test log", "logs/test.log"),
         ("Benchmark log", "logs/benchmark.log"), ("Profile log", "logs/profile.log"),
         ("Torch profile", "logs/torch_profile.txt"),
         ("Nsight Compute report (.ncu-rep)", "profile/ncu.ncu-rep"), ("ncu raw metrics (csv)", "profile/ncu_raw.csv"),
         ("ncu details (text)", "profile/ncu_details.txt"), ("ncu details (csv)", "profile/ncu_details.csv"), ("Profile summary", "profile/summary.json"),
         ("Metadata", "meta.json")]


def list_runs(runs_dir: Path) -> list[dict]:
    out = []
    for d in sorted(Path(runs_dir).glob("*/store.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({"id": d.parent.name, "mtime": d.stat().st_mtime})
    return out


def _diff(store: Store, parent_id: int | None, cid: int) -> str:
    if not parent_id:
        return ""
    try:
        a, b = store.files(parent_id), store.files(cid)
    except KeyError:
        return ""
    chunks = []
    for name in sorted(set(a) | set(b)):
        if a.get(name) == b.get(name):
            continue
        chunks.append("\n".join(difflib.unified_diff(a.get(name, "").splitlines(), b.get(name, "").splitlines(),
                                                     f"#{parent_id}/{name}", f"#{cid}/{name}", n=3, lineterm="")))
    return "\n".join(chunks)[:MAX_DIFF]


def _profile_view(ps: dict) -> dict | None:
    if not ps:
        return None
    if not ps.get("ok"):
        return {"ok": False, "reason": ps.get("reason") or "", "shape_label": ps.get("shape_label")}
    keep = ("short", "launches", "duration_ns", "share", "bound", "sm_pct", "mem_pct", "dram_pct", "occupancy_pct",
            "tensor_pct", "regs", "smem", "dram_bytes")
    return {"ok": True, "shape_label": ps.get("shape_label"), "shape_index": ps.get("shape_index"),
            "totals": ps["totals"], "kernels": [{k: kern.get(k) for k in keep} for kern in ps["kernels"][:8]],
            "hidden_kernels": ps.get("hidden_kernels", 0)}


def build_payload(runs_dir: Path, run_id: str) -> dict:
    run_dir = Path(runs_dir) / run_id
    if not (run_dir / "store.db").exists():
        raise FileNotFoundError(run_id)
    store = Store(run_dir)
    try:
        cands = store.all()
        problem = store.get_meta("problem", "moe")
        gpu = store.get_meta("gpu", config.gpu_name())
        seed = next((c for c in cands if c.kind == "seed" and c.ok), None) or next((c for c in cands if c.kind == "seed"), None)
        seed_ns = store.seed_ns() or (seed.geomean_ns if seed else None)
        known = problems.problem_dir(problem).exists()
        peaks = roofline.peaks_for(gpu, problems.peak_kind(problem)) if known else None
        by_id = {c.id: c for c in cands}

        ref = next((c for c in cands if (c.result.get("benchmark") or {}).get("benchmarks")), None)
        specs = [b["spec"] for b in ref.result["benchmark"]["benchmarks"]] if ref else []
        labels = roofline.shape_labels(specs)
        shapes = []
        if peaks and known and problems.has_cost(problem):
            for spec, label in zip(specs, labels):
                p = roofline.point(problem, spec, None, peaks)
                if p is None:
                    shapes = []
                    break
                shapes.append({"label": label, "spec": spec, "flops": p["flops"], "bytes": p["bytes"],
                               "intensity": p["intensity"], "roof_tflops": p["roof_tflops"]})

        best_ns, out = None, []
        for c in cands:
            bench = []
            for i, b in enumerate((c.result.get("benchmark") or {}).get("benchmarks", [])):
                row = {"label": labels[i] if i < len(labels) else f"shape {i + 1}", "mean_ns": b.get("mean_ns"),
                       "std_ns": b.get("std_ns"), "runs": b.get("runs"), "status": b.get("status")}
                pt = roofline.point(problem, b["spec"], b["mean_ns"], peaks) if shapes and b.get("mean_ns") else None
                if pt:
                    row.update(pt)
                bench.append(row)
            new_best = c.ok and (best_ns is None or c.geomean_ns < best_ns)
            if new_best:
                best_ns = c.geomean_ns
            parent = by_id.get(c.parent_id)
            prof = _profile_view(c.profile_summary)
            # roofline position: measured intensity (x) and benchmark throughput of the profiled shape (y)
            rp = None
            if prof and prof["ok"] and prof["shape_index"] is not None and prof["totals"].get("ai_measured"):
                bi = prof["shape_index"]
                if bi < len(bench) and bench[bi].get("tflops"):
                    rp = {"shape": bi, "ai": prof["totals"]["ai_measured"], "tflops": bench[bi]["tflops"],
                          "pct_roof": bench[bi].get("pct_roof")}
            d = store.cand_dir(c.id)
            brief_file = d / "brief.md"
            out.append({
                "id": c.id, "parent_id": c.parent_id, "kind": c.kind, "worker": c.worker,
                "strategy": c.strategy, "hypothesis": c.hypothesis, "status": c.status,
                "geomean_ns": c.geomean_ns, "speedup": (seed_ns / c.geomean_ns) if c.ok and seed_ns else None,
                "vs_parent": (parent.geomean_ns / c.geomean_ns) if c.ok and parent and parent.ok else None,
                "new_best": new_best, "wall_s": c.wall_s, "created_at": c.created_at,
                "error": c.error[-MAX_ERR:], "diff": _diff(store, c.parent_id, c.id),
                "files": {k: v[:MAX_CODE] for k, v in store.files(c.id).items()},
                "benchmarks": bench, "profile": prof, "roof_point": rp,
                "notes": (c.result.get("precheck") or {}).get("notes", []),
                "brief": brief_file.read_text()[:MAX_BRIEF] if brief_file.exists() else "",
                "links": [{"label": lab, "rel": f"candidates/{c.id:04d}/{rel}"} for lab, rel in LINKS if (d / rel).exists()],
            })

        strat: dict[str, dict] = {}
        work: dict[str, dict] = {}
        for c in cands:
            if c.kind == "seed" or c.status in ("running", "infra"):
                continue
            parent = by_id.get(c.parent_id)
            win = bool(c.ok and parent and parent.ok and c.geomean_ns <= parent.geomean_ns * 0.98)
            for table, key in ((strat, c.strategy or "(none)"), (work, c.worker or "(none)")):
                s = table.setdefault(key, {"name": key, "attempts": 0, "correct": 0, "wins": 0})
                s["attempts"] += 1
                s["correct"] += c.ok
                s["wins"] += win

        def order(v):
            return sorted(v.values(), key=lambda s: (-s["wins"] / s["attempts"], -s["attempts"]))

        last = max((c.created_at for c in cands), default=0)
        stamp = sum(len(json_len(c.profile_summary)) + len(c.status) for c in cands)
        return {
            "run_id": run_id, "problem": problem, "gpu": gpu, "synthetic": bool(store.get_meta("synthetic", False)),
            "generated_at": time.time(), "version": f"{len(cands)}:{last}:{stamp}",
            "seed_ns": seed_ns, "peaks": peaks, "shapes": shapes, "candidates": out,
            "strategies": order(strat), "workers": order(work),
            "has_journal": (run_dir / "JOURNAL.md").exists(),
        }
    finally:
        store.close()


def json_len(obj) -> str:
    import json
    return json.dumps(obj, default=str)
