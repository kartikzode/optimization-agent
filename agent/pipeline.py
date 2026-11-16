"""
submit: one candidate through the whole evaluation and into the record.

    reserve an id -> save the kernel files -> precheck (python, guard, local nvcc)
      -> Modal H100, one container: test -> benchmark (no profiler)  -> Nsight Compute (only if both passed)
      -> logs/, profile/, meta.json, brief.md, JOURNAL.md

Nothing here calls an LLM. The hypothesis and strategy are supplied by whoever wrote the kernel.
"""
import json
import logging
import time

from . import brief, problems, roofline
from .precheck import precheck
from .store import Candidate, Store
from harness import ncu as ncu_mod
from harness import ncu_text

L = logging.getLogger("optagent.pipeline")


class InfraError(RuntimeError):
    """Modal/network trouble. Not the candidate's fault."""


def _run_remote(evaluator_factory, problem, files, *, retries=3, log=print, **kw) -> dict:
    last = None
    for attempt in range(retries):
        try:
            with evaluator_factory() as ev:
                return ev.run_candidate(problem, files, **kw)
        except Exception as e:      # noqa: BLE001  Modal raises many types; all of them mean "not the candidate's fault"
            last = e
            L.exception("modal run failed (attempt %d/%d)", attempt + 1, retries)
            log(f"  [modal] run failed ({type(e).__name__}: {str(e)[:150]}); " + ("retrying" if attempt + 1 < retries else "giving up"))
            time.sleep(5 * (attempt + 1))
    raise InfraError(f"{type(last).__name__}: {last}") from last


def _profile_summary(store: Store, cand_id: int, problem: str, prof: dict, benchmarks: list[dict], launch_count: int = 40) -> dict:
    """Write the profile artifacts and return the summary stored on the candidate."""
    specs = [b["spec"] for b in benchmarks]
    labels = roofline.shape_labels(specs) if specs else []
    spec = prof["spec"]
    idx = specs.index(spec) if spec in specs else None
    base = {"spec": spec, "shape_label": labels[idx] if idx is not None else spec, "shape_index": idx, "status": prof["status"]}
    if prof.get("torch_table"):
        store.write_log(cand_id, "torch_profile.txt", prof["torch_table"])
    if prof["status"] not in ("ok", "light_only"):
        return {**base, "ok": False, "reason": prof["note"], "kernels": [], "totals": {}}
    for name, data in prof["artifacts"].items():
        store.write_log(cand_id, name, data, sub="profile")
    cost = problems.cost(problem, spec)
    summ = ncu_mod.summarize(prof["artifacts"].get("ncu_raw.csv", ""), raw_light=prof["artifacts"].get("ncu_light_raw.csv", ""),
                             flops=cost[0] if cost else None, min_bytes=cost[1] if cost else None, launch_limit=launch_count)
    summ.update(base)
    txt = prof["artifacts"].get("ncu_details.txt", "")
    if txt and summ.get("ok"):      # NVIDIA's own analysis for the most expensive kernel the detailed pass saw
        summ["advice"] = ncu_text.top_advice(txt, [k["name"] for k in summ["kernels"]])
    store.write_log(cand_id, "summary.json", json.dumps(summ, indent=2), sub="profile")
    return summ


def submit(store: Store, problem: str, files: dict[str, str], *, evaluator_factory, gpu: str, arch: str,
           kind: str = "improve", parent_id: int | None = None, worker: str = "", strategy: str = "",
           hypothesis: str = "", profile: bool = True, shape: str | int = "last", ncu_set: str = "detailed",
           launch_count: int = 40, kernel_regex: str | None = None, log=print) -> Candidate:
    t0 = time.time()
    cand = store.reserve(kind=kind, parent_id=parent_id, worker=worker, strategy=strategy, hypothesis=hypothesis)
    store.save_kernel(cand.id, files)
    log(f"#{cand.id}: precheck ...")

    def done(status, **kw) -> Candidate:
        c = store.finish(cand.id, status=status, wall_s=time.time() - t0, **kw)
        torch_table = ""
        tt = store.cand_dir(c.id) / "logs" / "torch_profile.txt"
        if tt.exists():
            torch_table = tt.read_text()
        store.write_brief(c.id, brief.build_brief(store, c, problem=problem, gpu=gpu, torch_table=torch_table))
        store.append_journal(brief.journal_entry(store, c))
        if kind == "seed" and c.ok:
            store.set_meta("seed_geomean_ns", c.geomean_ns)
        L.info("candidate #%d (%s): %s%s in %.0fs", c.id, kind, c.status,
               f", {c.geomean_ns / 1e6:.2f} ms" if c.ok else "", time.time() - t0)
        return c

    pre = precheck(files, arch=arch)
    store.write_log(cand.id, "precheck.json", json.dumps(pre, indent=2))
    if not pre["ok"]:
        L.warning("precheck rejected #%d at stage '%s'", cand.id, pre["stage"])
        return done("rejected", error="\n".join(pre["errors"]), result={"precheck": pre})

    log(f"#{cand.id}: running on Modal ({gpu}): test -> benchmark" + (" -> ncu" if profile else ""))
    try:
        out = _run_remote(evaluator_factory, problem, files, log=log, profile=profile, shape=shape, ncu_set=ncu_set,
                          launch_count=launch_count, kernel_regex=kernel_regex)
    except InfraError as e:
        return done("infra", error=str(e), result={"precheck": pre})

    for name, text in out["logs"].items():
        store.write_log(cand.id, name, text)
    if out.get("gpu_info"):
        store.write_log(cand.id, "gpu_info.txt", out["gpu_info"])
    test, bench = out["test"], out["benchmark"]
    result = {"precheck": pre, "test": test, "benchmark": bench}

    if test is None or test["status"] != "pass":
        return done(test["status"] if test else "crash", error=(test or {}).get("error") or "test stage did not run",
                    result=result)
    if bench is None or bench["status"] != "pass" or not bench["geomean_ns"]:
        return done(bench["status"] if bench else "crash", error=(bench or {}).get("error") or "benchmark stage did not run",
                    result=result)

    summary = {}
    if profile and out.get("profile"):
        log(f"#{cand.id}: correct, {bench['geomean_ns'] / 1e6:.3f} ms; collecting the Nsight Compute profile")
        summary = _profile_summary(store, cand.id, problem, out["profile"], bench["benchmarks"], launch_count)
    return done("pass", geomean_ns=bench["geomean_ns"], result=result, profile_summary=summary)


# ---- what the worker/runner reads back in the terminal ------------------------------------------
def terminal_report(store: Store, c: Candidate, *, max_changes: int = 6) -> str:
    seed, best_before = store.seed(), None
    prior = [x for x in store.all() if x.id < c.id and x.ok]
    if prior:
        best_before = min(prior, key=lambda x: x.geomean_ns)
    parent = store.get(c.parent_id) if c.parent_id else None
    out = []
    if c.ok:
        bits = []
        for label, ref in (("parent", parent), ("best", best_before), ("baseline", seed)):
            if ref is not None and ref.ok and ref.id != c.id:
                bits.append(f"{ref.geomean_ns / c.geomean_ns:.2f}x vs {label} #{ref.id}")
        star = "  NEW BEST" if best_before is None or c.geomean_ns < best_before.geomean_ns else ""
        out.append(f"#{c.id} PASS  {brief.fmt_ns(c.geomean_ns)}  ({', '.join(bits) or 'first result'}){star}")
        specs = [b["spec"] for b in (c.result.get("benchmark") or {}).get("benchmarks", [])]
        labels = roofline.shape_labels(specs)
        out.append("  shapes: " + " | ".join(f"{labels[i]}: {brief.fmt_ns(b['mean_ns'])}"
                                             for i, b in enumerate(c.result["benchmark"]["benchmarks"])))
    else:
        out.append(f"#{c.id} {c.status.upper()}")
        out.append("  " + (c.error or "(no error text)").strip()[-1500:].replace("\n", "\n  "))
    ps = c.profile_summary or {}
    if ps.get("ok"):
        t = ps["totals"]
        top = ps["kernels"][0] if ps["kernels"] else None
        out.append(f"  profile [{ps.get('shape_label')}]: {t['kernel_launches']} launches, SM {brief._pct(t.get('sm_pct'))}, "
                   f"mem {brief._pct(t.get('mem_pct'))}, occupancy {brief._pct(t.get('occupancy_pct'))}"
                   + (f", DRAM {brief.fmt_bytes(t['dram_bytes'])}, AI {t['ai_measured']:.0f} FLOP/B" if t.get("ai_measured") else "")
                   + (f"; top kernel {top['short']} {top['share']:.0%} ({top['bound']})" if top else ""))
        adv = ps.get("advice")
        if adv:
            out.append(f"  ncu says about {adv['kernel'][:40]}: " + " | ".join(
                f"[{m['kind']}] {m['text'][:170]}" for m in adv["messages"][:2]) + "  (full: ./optk ncu %d --advice)" % c.id)
        busy = brief.gpu_busy(c)
        if busy is not None and busy < 0.5:
            out.append(f"  HOST-BOUND: GPU kernels are ~{busy:.1%} of the call; fix launches/syncs/allocation/Python overhead first")
        comparator = None
        for a in ([parent] if parent else []) + store.ancestors(c.id)[1:] + ([seed] if seed else []):
            if a is not None and a.id != c.id and a.profile_summary.get("ok"):
                comparator = a
                break
        if comparator:
            ch = brief.compare_profiles(comparator.profile_summary, ps)
            out.append(f"  vs #{comparator.id}: " + ("; ".join(ch[:max_changes]) if ch else "no profile metric moved"))
    elif c.ok:
        out.append(f"  profile unavailable: {(ps.get('reason') or 'not collected')[:300]}")
    d = store.cand_dir(c.id)
    out.append(f"  brief: {d / 'brief.md'}")
    out.append(f"  logs:  {d / 'logs'}   profile: {d / 'profile'}")
    return "\n".join(out)


def reprofile(store: Store, problem: str, cand: Candidate, *, evaluator_factory, gpu: str, shape: str | int = "last",
              ncu_set: str = "detailed", launch_count: int = 40, kernel_regex: str | None = None, log=print) -> Candidate:
    """Run Nsight Compute again on a finished candidate (another shape, another set, one kernel) and refresh its brief."""
    if not cand.ok:
        raise SystemExit(f"#{cand.id} did not pass ({cand.status}); only correct candidates are profiled")
    files = store.files(cand.id)
    log(f"#{cand.id}: profiling on Modal ({gpu}), shape={shape}, set={ncu_set}" + (f", kernels ~ /{kernel_regex}/" if kernel_regex else ""))
    out = _run_remote(evaluator_factory, problem, files, log=log, profile=True, profile_only=True, shape=shape,
                      ncu_set=ncu_set, launch_count=launch_count, kernel_regex=kernel_regex)
    for name, text in out["logs"].items():
        store.write_log(cand.id, name, text)
    summary = _profile_summary(store, cand.id, problem, out["profile"], cand.result["benchmark"]["benchmarks"], launch_count)
    store.set_profile_summary(cand.id, summary)
    c = store.get(cand.id)
    tt = store.cand_dir(c.id) / "logs" / "torch_profile.txt"
    store.write_brief(c.id, brief.build_brief(store, c, problem=problem, gpu=gpu,
                                              torch_table=tt.read_text() if tt.exists() else ""))
    return c
