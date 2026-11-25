---
name: optimize
description: Run the multi-agent kernel-optimization loop on a problem. The main session is the orchestrator (Sonnet 5.5 for now, Fable 5.2 when available); kernel-worker sub-agents (Sonnet 5.5) write, compile and self-check CUDA candidates; the orchestrator itself schedules the finished ones on Modal with `./optk submit`. Use as /optimize <problem> [rounds|forever] [workers].
---

# /optimize: orchestrator playbook

Arguments: `<problem>` (required), `[rounds]` (default 3), `[workers per round]` (default 3). You are the **orchestrator**: you decide, you do not write kernels or run GPU jobs yourself (unless a step is trivial).
Read `CLAUDE.md` first if you have not this session: rules, commands, how to read the profiler, the tricks playbook.

## 0. Setup (once)
1. `./optk doctor` must be clean. `./optk status --problem <p>`.
2. No baseline yet? `./optk submit --seed --problem <p>` (a few minutes; it is iteration #1 and the speedup reference). Read its brief.
3. Read `knowledge/problems/<p>/notes.md` and `hints.md`, then `problems/<p>/reference.py`, `task.yml`, `cost.py`. Open the dashboard if the user wants it: `.venv/bin/python -m dashboard.serve`.

## 1. Each round
**Diagnose.** `./optk status` (shows the latest notes), `./optk notes`, `./optk journal`, the brief of the best and of the last few iterations. State in two or three sentences: where the time goes (top kernel, bound, measured vs algorithmic arithmetic intensity, % of roof), and the biggest gap to close. Search the knowledge base (`./optk kb search`) when a decision depends on hardware detail.

**Plan** `workers` experiments. Each is ONE hypothesis: parent candidate, the change, why it should help, and the profiler signature you expect (e.g. "DRAM bytes down ~30%, SM% up"). Make them *different* (not three tile sizes of one idea) unless an idea is clearly dominant; vary risk (one safe step, one bold). Never assign the same scratch name twice.

**Dispatch workers in parallel** (one Agent call each, `subagent_type: "kernel-worker"`, run in the background). The prompt must be self-contained:
- problem, run, scratch name (e.g. `r2-w1`), parent candidate id (or "baseline"/"best"), the hypothesis, the strategy tag, anything the diagnosis found that they need;
- "finish with `READY: <dir>`, `QUESTION: ...` or `BLOCKED: ...`".
Keep each worker's agent id: you need it to reply.

**Answer questions.** When a worker returns `QUESTION:`, decide and reply with `SendMessage(to: <agent id>, ...)` so it continues with its context. Prefer a decisive answer plus the reason; if the idea looks infeasible, say "drop it" or give a smaller variant. Do not let a worker wander: after two exchanges, narrow the task.

**Run (you schedule the GPU).** When a worker reports `READY: <dir>`, submit it yourself right away:
`./optk submit <dir> --problem <p>` with the Bash tool's **`run_in_background: true`** (a submit takes ~2 minutes; in the foreground it would stop you answering other workers' questions). Launch one background submit per ready candidate: they run in parallel, each on its own H100, so you choose how many run at once (and what they cost). You are notified as each finishes; read its output (a compact report: status, latency vs parent/best/baseline, profile line, NVIDIA's advice, paths). You may read `./optk show <id>` for the full brief. Workers never submit.

**Failures.** The worker already compiled and self-checked, so a failure here is a *runtime* one: `rejected` (guard / precheck), `fail` (wrong output), `crash`, `timeout`. Resume **the same worker** with `SendMessage(to: <its agent id>, ...)` containing the report's verbatim error and the log path (`workspace/runs/<run>/candidates/<id>/logs/`), and ask for a fix: at most 2 repair rounds per candidate; the repair is submitted with `--repair`. `infra` means Modal trouble, not the kernel: resubmit once. If a failure shows the hypothesis is wrong, drop it and record why.

**Learn.** For every finished candidate compare with its parent (`./optk compare A B`, the brief's "what changed" section). Decide: keep (new parent), drop, or follow up. Be honest about noise: a gain smaller than the reported std is not a gain.
- **Record your understanding** before moving on: `./optk note "..." --kind insight|decision|dead-end --cand <ids>` (why the result happened, what the profile implies, why you chose the next step). Also note the plan at the start of each round (`--kind plan`). After a context compaction, `./optk notes` + `./optk status` are how you recover your own reasoning.
- If something general was learned, **edit `CLAUDE.md`'s playbook now**: *Verified here* (measured win, with the iteration), *Refuted / pitfalls* (with evidence). Update `knowledge/problems/<p>/notes.md` with problem-specific findings.
- One short status message to the user per round: best latency and speedup, % of roofline, what was tried, what is next.

## Continuous mode: `/optimize <problem> forever [workers]`
Instead of a fixed number of rounds, keep going until the budgets run out or the user stops you.
1. At the start run `./optk continue --start --max-hours <H> --max-gpu-hours <G>` (defaults 8 h and 4 GPU-hours; use what the user gave you). **Before every round** run `./optk continue`: exit 0 means start a round, exit 1 means stop (STOP file, or a budget used) and report. The user can stop you from another terminal with `./optk continue --stop`.
2. It also tells you the mode. **EXPLOIT**: the last few candidates still produced >1% improvements, so refine the best (tile shapes, pipeline depth, fusion, glue). **EXPLORE**: no significant improvement for `patience` candidates, so do *not* keep nudging the same design. Re-diagnose from `./optk ncu <best> --advice` and the roofline, search the knowledge base, list what has never been tried (`./optk notes`, `knowledge/problems/<p>/hints.md`) and assign structurally different ideas, one per worker, including at least one bold one. **EXPLORE-HARD**: also audit the best kernel (below) before trying a different algorithm.
3. A plateau is not a reason to stop; only the budgets, the STOP file or the user are. Realistic ceiling: well-tuned H100 GEMMs reach about 70-75% of the dense peak (the H100 worklog's 764 TFLOP/s vs cuBLAS 716 of 989), so past that expect diminishing returns and say so in your round summaries.
4. **Audit the best every ~5 rounds** (a worker, read-only task): does the kernel make shape or value assumptions the benchmark happens to satisfy (divisibility, fixed expert counts, specific seeds)? Does it do all the reference's work? Test it on extra shapes if the problem allows. Optimizing for many rounds against a fixed handful of shapes drifts toward overfitting; report anything suspicious to the user instead of burying it.
5. **Context hygiene for long runs:** keep each round's reasoning short, write the plan and the insight with `./optk note` every round, and every ~5 rounds a "state of the world" note (best, what worked, what is exhausted, what to try next). After a compaction, recover from `./optk status`, `./optk notes -n 10` and `./optk continue`, not from memory. Do not paste whole briefs or logs into your context; use `./optk show`, `ncu --advice` and `compare`.
6. One status line to the user every few rounds (best latency, % of roof, mode, budgets used), not every round.

## 2. Stopping
(Fixed-rounds mode; in continuous mode `./optk continue` decides.)
Stop when the rounds are used, **or** three consecutive rounds produce no improvement over the best, **or** the best kernel is within ~10% of its roof (tensor-pipe near its limit, measured AI near the algorithmic value), **or** the user says stop. Finish with: `./optk status`, the best kernel's path (`./optk best`), a summary of what worked and what did not, and the CLAUDE.md entries you added.

## If something seems stuck
Do not guess and do not re-submit. Run `./optk where`: it lists what is in flight with its age and phase, flags STALLED / DEAD operations and any permission prompt waiting for the user, and names the likely culprit. A worker, or one of your own background submits, that has been silent for several minutes while `where` shows a healthy heartbeat is just waiting on Modal; a `DEAD` op or a pending permission prompt needs action (tell the user). Everything agents do is logged to `workspace/events.jsonl` automatically.

## Orchestration rules
- Workers write, compile and self-check code; you decide and you schedule the GPU. Do not write kernels in your own context: delegate, and read only the workers' reports and the briefs.
- Everything important is on disk (`JOURNAL.md`, briefs, `status`): after a long session or a context compaction, rebuild state from `./optk status` and `./optk journal`, not from memory.
- Never edit `problems/`, `harness/`, `agent/`, `dashboard/` to make a result look better. If a tool is broken, say so and stop.
- Parallel workers must use distinct scratch dirs; a worker must never submit to Modal itself.
