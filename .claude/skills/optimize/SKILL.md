---
name: optimize
description: Run the multi-agent kernel-optimization loop on a problem. The main session is the orchestrator (run it on Fable 5.2); kernel-worker sub-agents (Sonnet 5.5) write and compile CUDA candidates, the modal-runner sub-agent runs them on Modal and reports latency and profiler facts. Use as /optimize <problem> [rounds] [workers].
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

**Run.** When workers report `READY`, gather their directories and dispatch **one** `modal-runner` (`subagent_type: "modal-runner"`) with all of them (it runs the submits in parallel and returns one report). Do not run GPU jobs yourself while a runner is active.

**Failures.** For a `rejected`/`fail`/`crash`/`timeout` candidate, send the runner's verbatim error and log path back to *its* worker (`SendMessage`) for a repair: at most 2 repair rounds per candidate; a repair is submitted with `--repair`. If a failure shows the hypothesis is wrong, drop it and record why.

**Learn.** For every finished candidate compare with its parent (`./optk compare A B`, the brief's "what changed" section). Decide: keep (new parent), drop, or follow up. Be honest about noise: a gain smaller than the reported std is not a gain.
- **Record your understanding** before moving on: `./optk note "..." --kind insight|decision|dead-end --cand <ids>` (why the result happened, what the profile implies, why you chose the next step). Also note the plan at the start of each round (`--kind plan`). After a context compaction, `./optk notes` + `./optk status` are how you recover your own reasoning.
- If something general was learned, **edit `CLAUDE.md`'s playbook now**: *Verified here* (measured win, with the iteration), *Refuted / pitfalls* (with evidence). Update `knowledge/problems/<p>/notes.md` with problem-specific findings.
- One short status message to the user per round: best latency and speedup, % of roofline, what was tried, what is next.

## 2. Stopping
Stop when the rounds are used, **or** three consecutive rounds produce no improvement over the best, **or** the best kernel is within ~10% of its roof (tensor-pipe near its limit, measured AI near the algorithmic value), **or** the user says stop. Finish with: `./optk status`, the best kernel's path (`./optk best`), a summary of what worked and what did not, and the CLAUDE.md entries you added.

## If something seems stuck
Do not guess and do not re-submit. Run `./optk where`: it lists what is in flight with its age and phase, flags STALLED / DEAD operations and any permission prompt waiting for the user, and names the likely culprit. A runner or worker that has been silent for several minutes while `where` shows a healthy heartbeat is just waiting on Modal; a `DEAD` op or a pending permission prompt needs action (tell the user). Everything agents do is logged to `workspace/events.jsonl` automatically.

## Orchestration rules
- Workers write code and reason about correctness; the runner runs; you decide. Do not do their jobs in your own context: delegate, and read only their reports and the briefs.
- Everything important is on disk (`JOURNAL.md`, briefs, `status`): after a long session or a context compaction, rebuild state from `./optk status` and `./optk journal`, not from memory.
- Never edit `problems/`, `harness/`, `agent/`, `dashboard/` to make a result look better. If a tool is broken, say so and stop.
- Parallel workers must use distinct scratch dirs; a worker must never submit to Modal itself.
