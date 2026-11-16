"""
`./optk <command>`: the tools Claude Code (orchestrator, workers, runner) uses. See CLAUDE.md for the workflow.

  doctor                     check Modal credentials, local nvcc/ncu, knowledge base
  problems                   list problems and their run status
  new-problem NAME           scaffold problems/NAME from templates/problem
  new-kernel                 scaffold a scratch kernel dir (CUDA template, the baseline, or a copy of a candidate)
  compile DIR                compile check (local nvcc for torch-free .cu, else a Modal CPU build)
  submit DIR | --seed        precheck -> Modal test+benchmark (no profiler) -> Nsight Compute if correct -> brief
  profile --cand N           re-run Nsight Compute on a finished candidate (other shape / kernel / section set)
  status                     baseline, best, leaderboard, recent iterations
  show N                     the brief of one iteration          compare A B   latency + profile diff
  ncu N [--kernel R] [--section S] [--advice]   the Nsight Compute report as terminal text
  where                      what is running / STUCK / waiting on you (global event log)      events   the raw log
  note "text" [--kind K] [--cand N..]   record your understanding in NOTES.md      notes   print them
  brief N                    regenerate a brief from stored data (no GPU)
  best                       path of the fastest correct kernel  journal       the run's running log
  kb ...                     knowledge base (PDFs and docs): ingest, search, read, add
"""
import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from . import brief, compile as cc, config, events, pipeline, problems, roofline
from . import log as logsetup
from .store import Store, collect_files


# ---- helpers --------------------------------------------------------------------------------------
def pick_problem(name: str | None) -> str:
    known = problems.list_problems()
    if name:
        problems.require(name)
        return name
    if len(known) == 1:
        return known[0]
    raise SystemExit(f"--problem is required (known: {', '.join(known) or 'none'})")


def open_store(problem: str, run: str | None) -> Store:
    d = config.run_dir(problem, run)
    store = Store(d)
    if store.get_meta("problem") is None:
        store.set_meta("problem", problem)
        store.set_meta("gpu", config.gpu_name())
    return store


def evaluator_factory(verbose: bool):
    def make():
        from harness.evaluate import Evaluator
        return Evaluator(verbose=verbose)
    return make


def read_meta(kdir: Path) -> dict:
    f = kdir / "candidate.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_meta(kdir: Path, meta: dict):
    (kdir / "candidate.json").write_text(json.dumps(meta, indent=2) + "\n")


def _ago(ts: float) -> str:
    s = max(0, int(dt.datetime.now().timestamp() - ts))
    return f"{s}s" if s < 90 else f"{s // 60}m" if s < 5400 else f"{s // 3600}h"


# ---- commands -------------------------------------------------------------------------------------
def cmd_doctor(a) -> int:
    ok = True

    def line(label, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else '!!'}] {label}: {detail}")

    print("optk doctor")
    ms = config.modal_status()
    line("modal credentials", ms != "missing", ms if ms != "missing" else "run `modal setup` (or set MODAL_TOKEN_ID/SECRET in .env)")
    try:
        import modal
        line("modal package", True, modal.__version__)
    except ImportError:
        line("modal package", False, "pip install -r requirements.txt")
    line("target GPU", True, f"{config.gpu_name()} ({config.gpu_arch()})")
    nvcc, ncu = shutil.which("nvcc"), shutil.which("ncu")
    line("local nvcc", nvcc, nvcc or "not found: torch-free .cu files are compiled on Modal instead (slower)")
    print(f"  [..] local ncu: {ncu or 'not found (not needed: profiling runs on Modal)'}")
    probs = problems.list_problems()
    line("problems", probs, ", ".join(probs) or "none: ./optk new-problem NAME")
    from . import kb
    docs = kb.list_docs()
    print(f"  [..] knowledge base: {len(docs)} documents (./optk kb list)")
    return 0 if ok else 1


def cmd_problems(a) -> int:
    for p in problems.list_problems():
        task = problems.load_task(p)
        d = config.RUNS / p
        info = "no run yet"
        if (d / "store.db").exists():
            s = Store(d)
            best, seed = s.best(), s.seed()
            info = f"{s.count()} iterations"
            if best and seed:
                info += f", best #{best.id} {brief.fmt_ns(best.geomean_ns)} ({seed.geomean_ns / best.geomean_ns:.2f}x vs baseline)"
            elif seed is None:
                info += ", no baseline yet (./optk submit --seed)"
            s.close()
        extra = "" if problems.has_cost(p) else "  [no cost.py: roofline disabled]"
        print(f"{p}: {len(task['tests'])} tests, {len(task['benchmarks'])} benchmark shapes; {info}{extra}")
    return 0


def cmd_new_problem(a) -> int:
    dest = problems.scaffold(a.name)
    print(f"created {dest}\nfill in: task.yml, task.py, reference.py (generate_input, ref_kernel, tolerance), submission.py (baseline),"
          f"\ncost.py (FLOPs and minimum bytes). Also created knowledge/problems/{a.name}/ (notes.md, hints.md: what is known, first ideas)."
          f"\nThen: ./optk submit --seed --problem {a.name}")
    return 0


def cmd_new_kernel(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    name = a.name or f"k{dt.datetime.now():%m%d-%H%M%S}"
    dest = store.dir / "scratch" / name
    if dest.exists():
        raise SystemExit(f"{dest} already exists; pick another --name")
    parent = a.parent
    if a.from_candidate or a.from_best:
        src = store.best() if a.from_best else store.get(a.from_candidate)
        if src is None:
            raise SystemExit("no correct candidate yet; submit the baseline first (./optk submit --seed)")
        files = store.files(src.id)
        parent = parent or src.id
        origin = f"copy of #{src.id}"
    elif a.template == "baseline":
        files = {"submission.py": (problems.problem_dir(problem) / "submission.py").read_text()}
        origin = "the baseline submission"
    else:
        files = {f.name: f.read_text() for f in sorted((config.TEMPLATES / "cuda").iterdir()) if f.is_file()}
        origin = "the CUDA template"
    for rel, text in files.items():
        f = dest / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    write_meta(dest, {"problem": problem, "parent": parent, "worker": a.worker or name, "strategy": a.strategy or "",
                      "hypothesis": a.hypothesis or "", "submitted": []})
    print(f"{dest}\n  from {origin}; files: {', '.join(files)}\n  next: edit, `./optk compile {dest} --problem {problem}`, "
          f"fill in candidate.json (strategy, hypothesis), hand it over for ./optk submit")
    return 0


def cmd_compile(a) -> int:
    problem = pick_problem(a.problem)
    kdir = Path(a.dir)
    try:
        files = collect_files(kdir)
    except ValueError as e:
        raise SystemExit(f"cannot compile {kdir}: {e}")
    from . import guard
    bad = guard.check_files(files)
    if bad:
        print("REJECTED by the guard (would be rejected at submit):\n  " + "\n  ".join(map(str, bad)))
        return 1
    r = cc.compile_candidate(files, problem=problem, arch=config.gpu_arch(), where=a.where,
                             evaluator_factory=evaluator_factory(a.verbose))
    (kdir / ".compile.log").write_text(r["log"])
    print(r["summary"])
    print(f"full log: {kdir / '.compile.log'}")
    return 0 if r["ok"] else 1


def cmd_submit(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    logsetup.setup(store.dir, a.verbose)
    kdir = Path(a.dir) if a.dir else None
    meta = read_meta(kdir) if kdir else {}
    if a.seed:
        base = problems.problem_dir(problem) / "submission.py"
        if store.seed() and not a.force:
            raise SystemExit(f"run '{store.dir.name}' already has a baseline (#{store.seed().id}); use --force to add another")
        files, kind, parent = {"submission.py": base.read_text()}, "seed", None
        hypothesis, strategy, worker = a.hypothesis or "baseline: problems/%s/submission.py" % problem, "baseline", "seed"
    else:
        if not kdir:
            raise SystemExit("give the kernel directory (./optk new-kernel creates one), or --seed")
        try:
            files = collect_files(kdir)
        except ValueError as e:
            raise SystemExit(f"cannot submit {kdir}: {e}")
        hypothesis = a.hypothesis or meta.get("hypothesis", "")
        strategy = a.strategy or meta.get("strategy", "")
        worker = a.worker or meta.get("worker", "")
        parent = a.parent if a.parent is not None else meta.get("parent")
        if not hypothesis.strip():
            raise SystemExit("every iteration needs a hypothesis (what changed, why it should be faster): "
                             "put it in candidate.json or pass --hypothesis")
        if parent is None and store.best():
            parent = store.best().id
            print(f"(no parent given: using the current best #{parent})")
        kind = "repair" if a.repair else "improve"
        if store.seed() is None:
            raise SystemExit("this run has no baseline yet: ./optk submit --seed --problem %s" % problem)
    gpu = store.get_meta("gpu", config.gpu_name())
    cand = pipeline.submit(store, problem, files, evaluator_factory=evaluator_factory(a.verbose), gpu=gpu,
                           arch=config.gpu_arch(), kind=kind, parent_id=parent, worker=worker, strategy=strategy,
                           hypothesis=hypothesis, profile=not a.no_profile, shape=a.shape, ncu_set=a.ncu_set,
                           launch_count=a.launch_count, kernel_regex=a.kernel_regex, log=print)
    if kdir:
        meta["submitted"] = [*meta.get("submitted", []), cand.id]
        meta.setdefault("problem", problem)
        write_meta(kdir, meta)
    print(pipeline.terminal_report(store, cand))
    return 0 if cand.ok else 1


def cmd_profile(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    logsetup.setup(store.dir, a.verbose)
    cand = store.get(a.cand)
    c = pipeline.reprofile(store, problem, cand, evaluator_factory=evaluator_factory(a.verbose),
                           gpu=store.get_meta("gpu", config.gpu_name()), shape=a.shape, ncu_set=a.ncu_set,
                           launch_count=a.launch_count, kernel_regex=a.kernel_regex, log=print)
    print(pipeline.terminal_report(store, c))
    return 0


def cmd_status(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    cands = store.all()
    seed, best = store.seed(), store.best()
    gpu = store.get_meta("gpu", config.gpu_name())
    print(f"problem {problem} · run {store.dir.name} · {gpu} · {len(cands)} iterations")
    if seed:
        print(f"baseline   #{seed.id}  {brief.fmt_ns(seed.geomean_ns)}")
    else:
        print("baseline   none yet: ./optk submit --seed")
    if best:
        sp = f"  ({seed.geomean_ns / best.geomean_ns:.2f}x vs baseline)" if seed and seed.ok else ""
        t = best.profile_summary.get("totals") or {}
        prof = f"  AI {t['ai_measured']:.0f} FLOP/B, {brief.verdict(best)}" if t.get("ai_measured") else ""
        print(f"best       #{best.id}  {brief.fmt_ns(best.geomean_ns)}{sp}{prof}\n           {store.dir / 'best'}")
        if problems.has_cost(problem):
            bench = (best.result.get("benchmark") or {}).get("benchmarks", [])
            pk = roofline.peaks_for(gpu, problems.peak_kind(problem))
            if bench:
                pt = roofline.point(problem, bench[-1]["spec"], bench[-1]["mean_ns"], pk)
                if pt and "pct_roof" in pt:
                    print(f"           largest shape: {pt['tflops']:.1f} TFLOP/s = {pt['pct_roof']:.1f}% of its roof "
                          f"({pt['roof_tflops']:.0f} TFLOP/s, algorithmic AI {pt['intensity']:.0f} FLOP/B)")
    ok = sorted([c for c in cands if c.ok], key=lambda c: c.geomean_ns)[:5]
    if ok:
        print("leaderboard: " + " | ".join(f"#{c.id} {brief.fmt_ns(c.geomean_ns)}" for c in ok))
    running = [c for c in cands if c.status == "running"]
    if running:
        print("running:     " + ", ".join(f"#{c.id} ({_ago(c.created_at)} ago)" for c in running))
    print(f"\nlast {min(a.n, len(cands))} iterations:")
    for c in cands[-a.n:]:
        lat = brief.fmt_ns(c.geomean_ns) if c.ok else c.status
        sp = f"{seed.geomean_ns / c.geomean_ns:.2f}x" if seed and seed.ok and c.ok else "-"
        par = f"<-#{c.parent_id}" if c.parent_id else "     "
        print(f"  #{c.id:<3} {par:<6} {lat:>10} {sp:>6}  {c.worker or '-':<10} {(c.strategy or c.kind)[:28]:<28} "
              f"{' '.join(c.hypothesis.split())[:70]}")
    recent = store.notes(3)
    if recent:
        print("\nlatest notes (./optk notes for all):\n" + "\n\n".join(recent))
    if (store.dir / "JOURNAL.md").exists():
        print(f"\njournal: {store.dir / 'JOURNAL.md'}")
    return 0


def cmd_show(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    p = store.cand_dir(a.id) / "brief.md"
    if not p.exists():
        raise SystemExit(f"#{a.id} has no brief yet (still running? see ./optk status)")
    print(p.read_text())
    return 0


def cmd_where(a) -> int:
    """What is running, what is stuck, who is waiting on you: read from the global event log."""
    now = time.time()
    evs = events.read(since=now - a.hours * 3600)
    if not evs:
        print(f"no events in the last {a.hours:g} h ({events.LOG}). Hooks log tool calls once a NEW claude session has started in this repo;\n"
              "./optk commands log themselves.")
        return 0
    info = events.analyze(evs, now, stall_s=a.stall_min * 60)
    print(f"{events.LOG}: {len(evs)} events in the last {a.hours:g} h; last activity {events.fmt_age(now - info['last_ts'])} ago")
    ops = [o for o in info["open"] if not str(o.get("name", "")).startswith("permission prompt")]
    perms = [o for o in info["open"] if str(o.get("name", "")).startswith("permission prompt")]
    print("\nIN FLIGHT (started, no end yet):" if ops else "\nIN FLIGHT: nothing")
    for o in ops:
        mark = {"dead": "!! DEAD   ", "stalled": "!! STALLED", "running": "   running"}[o["verdict"]]
        who = events.actor_of(o)
        line = f"  {mark} {events.fmt_age(o['age']):>7}  [{who or '-'}] {o.get('name', '')} {o.get('detail') or o.get('argv', '')}"[:200]
        print(line)
        if o.get("phase"):
            print(f"{'':14}phase: {o['phase']}" + (f"   (last heartbeat {events.fmt_age(o['quiet'])} ago)" if o.get("src") == "optk" else ""))
        if o["verdict"] == "dead":
            print(f"{'':14}the process (pid {o.get('pid')}) is gone without writing an end: it crashed or was killed")
    if perms or info["waiting"]:
        print("\nWAITING ON YOU (a permission prompt or notification with nothing after it):")
        for r in perms:
            print(f"  !! {events.fmt_age(r['age'])} [{events.actor_of(r) or '-'}] {r.get('name')} {r.get('detail', '')}"[:200])
        for r in info["waiting"]:
            if r.get("kind") == "notification":
                print(f"  !! {r['t']} [{events.actor_of(r) or '-'}] {r.get('message', '')}")
    if info["agents"]:
        print("\nAGENTS (last activity):")
        for name, d in sorted(info["agents"].items(), key=lambda kv: -kv[1]["last_ts"]):
            print(f"  {name:<28} {events.fmt_age(now - d['last_ts']):>7} ago  {'(finished) ' if d['stopped'] else ''}{d['last']}"[:200])
    print(f"\nlast {a.n} events:")
    for r in evs[-a.n:]:
        print("  " + events.format_event(r)[:220])
    stalled = [o for o in ops if o["verdict"] != "running"]
    print("\nVERDICT: ", end="")
    if perms:
        r = perms[0]
        print(f"a {r.get('name')} has been waiting {events.fmt_age(r['age'])} for your approval: that stalls the agent until you answer.")
    elif stalled:
        o = stalled[0]
        print(f"{o['verdict'].upper()}: {o.get('name')} {o.get('detail') or o.get('argv', '')} ({events.fmt_age(o['age'])}), phase '{o.get('phase') or 'unknown'}'.")
    elif ops:
        print(f"{len(ops)} operation(s) running normally; oldest {events.fmt_age(ops[0]['age'])}.")
    elif now - info["last_ts"] > 300:
        print(f"nothing in flight and quiet for {events.fmt_age(now - info['last_ts'])}: the orchestrator is idle, finished, or waiting for you at its prompt.")
    else:
        print("nothing in flight right now.")
    return 0


def cmd_events(a) -> int:
    now = time.time()
    evs = events.read(since=now - a.hours * 3600)
    if a.actor:
        evs = [r for r in evs if a.actor in (events.actor_of(r) or "")]
    if a.kind:
        evs = [r for r in evs if a.kind in r.get("kind", "") or a.kind in str(r.get("name", ""))]
    if a.open:
        evs = events.analyze(evs, now)["open"]
    for r in evs[-a.n:]:
        print(events.format_event(r)[:260])
    return 0


def cmd_ncu(a) -> int:
    """Print a candidate's Nsight Compute report as text in the terminal (no GUI needed)."""
    from harness import ncu_text
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    pdir = store.cand_dir(a.id) / "profile"
    txt = pdir / "ncu_details.txt"
    if not txt.exists():
        rep = pdir / "ncu.ncu-rep"
        ncu = shutil.which("ncu")
        if not rep.exists():
            raise SystemExit(f"#{a.id} has no Nsight Compute report (not profiled, or profiling failed: see brief.md)")
        if not ncu:
            raise SystemExit("this report predates the saved text export and no local `ncu` is installed to convert it")
        import subprocess
        r = subprocess.run([ncu, "--import", str(rep), "--page", "details"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"ncu --import failed: {r.stderr[-300:]}")
        txt.write_text(r.stdout)
    print(ncu_text.render(txt.read_text(), kernel=a.kernel, section=a.section, advice_only=a.advice, max_lines=a.max_lines))
    return 0


def cmd_note(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    text = " ".join(a.text).strip()
    if not text:
        raise SystemExit("give the note text: ./optk note \"what you understood and why\" [--cand 3 5] [--kind insight]")
    print(store.add_note(a.kind, text, a.cand))
    print(f"saved to {store.dir / 'NOTES.md'}")
    return 0


def cmd_notes(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    notes = store.notes(a.n)
    print("\n\n".join(notes) if notes else "(no notes yet: ./optk note \"...\")")
    return 0


def cmd_brief(a) -> int:
    """Regenerate brief.md (and the journal-free parts) from the stored measurements; no GPU needed."""
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    c = store.get(a.id)
    txt = store.cand_dir(c.id) / "profile" / "ncu_details.txt"
    if c.profile_summary.get("ok") and not c.profile_summary.get("advice") and txt.exists():   # backfill for older candidates
        from harness import ncu_text
        c.profile_summary["advice"] = ncu_text.top_advice(txt.read_text(), [k["name"] for k in c.profile_summary["kernels"]])
        store.set_profile_summary(c.id, c.profile_summary)
        c = store.get(c.id)
    tt = store.cand_dir(c.id) / "logs" / "torch_profile.txt"
    text = brief.build_brief(store, c, problem=problem, gpu=store.get_meta("gpu", config.gpu_name()),
                             torch_table=tt.read_text() if tt.exists() else "")
    store.write_brief(c.id, text)
    print(f"wrote {store.cand_dir(c.id) / 'brief.md'}")
    return 0


def cmd_compare(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    x, y = store.get(a.a), store.get(a.b)
    print(f"#{x.id} {brief.fmt_ns(x.geomean_ns)}  ->  #{y.id} {brief.fmt_ns(y.geomean_ns)}"
          + (f"   ({x.geomean_ns / y.geomean_ns:.2f}x)" if x.ok and y.ok else ""))
    bx, by = (x.result.get("benchmark") or {}).get("benchmarks", []), (y.result.get("benchmark") or {}).get("benchmarks", [])
    labels = roofline.shape_labels([b["spec"] for b in by or bx])
    for i, lab in enumerate(labels):
        if i < len(bx) and i < len(by):
            print(f"  {lab}: {brief.fmt_ns(bx[i]['mean_ns'])} -> {brief.fmt_ns(by[i]['mean_ns'])}")
    ch = brief.compare_profiles(x.profile_summary, y.profile_summary)
    print("profile changes:\n" + ("\n".join(f"  - {c}" for c in ch) if ch else "  (none, or one side has no profile)"))
    print(f"kernel diff: diff -ru {store.cand_dir(x.id) / 'kernel'} {store.cand_dir(y.id) / 'kernel'}")
    return 0


def cmd_best(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    best = store.best()
    if not best:
        raise SystemExit("no correct candidate yet")
    print(f"best #{best.id} {brief.fmt_ns(best.geomean_ns)}\nkernel: {store.dir / 'best'}\nbrief:  {store.cand_dir(best.id) / 'brief.md'}")
    return 0


def cmd_journal(a) -> int:
    problem = pick_problem(a.problem)
    store = open_store(problem, a.run)
    p = store.dir / "JOURNAL.md"
    if not p.exists():
        print("(journal is empty)")
        return 0
    blocks = p.read_text().strip().split("\n\n### ")
    blocks = [blocks[0]] + ["### " + b for b in blocks[1:]] if len(blocks) > 1 else blocks
    print("\n\n".join(blocks[-a.n:]))
    return 0


def cmd_kb(a) -> int:
    from . import kb
    return kb.main(a.rest)


# ---- parser ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="optk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        p.add_argument("--problem", help="problem name (optional when only one exists)")
        if run:
            p.add_argument("--run", help="run id (default: the problem's own run, workspace/runs/<problem>)")
        p.add_argument("-v", "--verbose", action="store_true", help="stream Modal container logs and debug lines")

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("problems").set_defaults(fn=cmd_problems)

    p = sub.add_parser("new-problem")
    p.add_argument("name")
    p.set_defaults(fn=cmd_new_problem)

    p = sub.add_parser("new-kernel", help="scaffold a scratch kernel directory")
    common(p)
    p.add_argument("--name", help="scratch dir name (default: timestamp)")
    p.add_argument("--template", choices=["cuda", "baseline"], default="cuda")
    p.add_argument("--from-candidate", type=int, metavar="ID", help="start from a copy of this candidate's kernel")
    p.add_argument("--from-best", action="store_true", help="start from a copy of the best candidate")
    p.add_argument("--parent", type=int)
    p.add_argument("--worker")
    p.add_argument("--strategy")
    p.add_argument("--hypothesis")
    p.set_defaults(fn=cmd_new_kernel)

    p = sub.add_parser("compile", help="compile check; exit 1 on failure")
    common(p, run=False)
    p.add_argument("dir")
    p.add_argument("--where", choices=["auto", "local", "modal"], default="auto")
    p.set_defaults(fn=cmd_compile)

    def ncu_opts(p):
        p.add_argument("--shape", default="last", help="benchmark shape to profile: last (default), first, or an index")
        p.add_argument("--ncu-set", default="detailed", help="Nsight Compute section set for the detailed pass (detailed, full = much larger report, basic, roofline)")
        p.add_argument("--launch-count", type=int, default=40, help="max kernel launches to profile")
        p.add_argument("--kernel-regex", help="profile only kernels whose name matches this regex")

    p = sub.add_parser("submit", help="evaluate a kernel directory (or the baseline with --seed) and record the iteration")
    common(p)
    p.add_argument("dir", nargs="?")
    p.add_argument("--seed", action="store_true", help="evaluate problems/<p>/submission.py as the baseline")
    p.add_argument("--force", action="store_true")
    p.add_argument("--parent", type=int)
    p.add_argument("--strategy")
    p.add_argument("--hypothesis")
    p.add_argument("--worker")
    p.add_argument("--repair", action="store_true", help="this is a fix of a failed candidate, not a new idea")
    p.add_argument("--no-profile", action="store_true", help="latency only: skip Nsight Compute")
    ncu_opts(p)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("profile", help="re-run Nsight Compute on a correct candidate")
    common(p)
    p.add_argument("--cand", type=int, required=True)
    ncu_opts(p)
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("status")
    common(p)
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(fn=cmd_status)

    for name, fn, arg in (("show", cmd_show, "id"), ("brief", cmd_brief, "id"), ("best", cmd_best, None),
                          ("journal", cmd_journal, None)):
        p = sub.add_parser(name)
        common(p)
        if arg:
            p.add_argument(arg, type=int)
        if name == "journal":
            p.add_argument("-n", type=int, default=6)
        p.set_defaults(fn=fn)
    p = sub.add_parser("note", help="record what you understood (insight / decision / dead-end) in the run's NOTES.md")
    common(p)
    p.add_argument("text", nargs="*")
    p.add_argument("--kind", default="insight", choices=["insight", "decision", "hypothesis", "dead-end", "question", "plan"])
    p.add_argument("--cand", type=int, nargs="*", help="iterations this note is about")
    p.set_defaults(fn=cmd_note)
    p = sub.add_parser("notes", help="print the run's notes")
    common(p)
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(fn=cmd_notes)

    p = sub.add_parser("where", help="what is running / stuck / waiting on you, from the global event log (workspace/events.jsonl)")
    p.add_argument("-n", type=int, default=12, help="how many recent events to show")
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--stall-min", type=float, default=10, help="a tool call older than this is called stalled")
    p.set_defaults(fn=cmd_where)
    p = sub.add_parser("events", help="print the event log (filters: --actor, --kind, --open)")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--actor")
    p.add_argument("--kind")
    p.add_argument("--open", action="store_true", help="only operations without an end")
    p.set_defaults(fn=cmd_events)

    p = sub.add_parser("ncu", help="print the Nsight Compute report as text (kernel/section filters, --advice for NVIDIA's rules)")
    common(p)
    p.add_argument("id", type=int)
    p.add_argument("--kernel", help="regex on the kernel name")
    p.add_argument("--section", help="regex on the section name, e.g. 'memory|occupancy|warp'")
    p.add_argument("--advice", action="store_true", help="only NVIDIA's analysis messages (OPT / WRN / INF)")
    p.add_argument("--max-lines", type=int, default=250)
    p.set_defaults(fn=cmd_ncu)

    p = sub.add_parser("compare")
    common(p)
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("kb", help="knowledge base: ./optk kb --help")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_kb)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    # only our own loggers get DEBUG: root-level DEBUG makes Modal's gRPC client log request headers (with the token)
    if not logging.getLogger("optagent").handlers:
        logsetup.setup(None, getattr(a, "verbose", False))
    if a.cmd in ("where", "events"):          # the diagnosis commands do not log themselves
        return a.fn(a) or 0
    actor = os.environ.get("OPTK_ACTOR", "")
    if not actor and getattr(a, "dir", None):
        try:
            actor = read_meta(Path(a.dir)).get("worker", "")
        except Exception:
            actor = ""
    cli_args = " ".join(argv if argv is not None else sys.argv[1:])
    with events.span(f"optk {a.cmd}", argv=cli_args[:300], actor=actor, problem=getattr(a, "problem", None)):
        rc = a.fn(a) or 0
        events.set_phase(None, f"finished rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
