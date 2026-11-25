# Design notes

## Idea

Claude Code is the optimizer. The repo supplies what an agent cannot do by itself: a trusted way to run and time a kernel on
a real H100, Nsight Compute data for every correct candidate, and a durable, searchable record. Decisions stay in the model;
measurements stay in code. Nothing in the repo calls an LLM.

## Roles

    orchestrator (the main session: Sonnet 5.5 for now, Fable 5.2 when available)
        diagnoses from briefs + roofline -> assigns ONE hypothesis per worker -> answers worker questions
        -> runs `./optk submit` (background jobs, it decides how many H100s run at once) -> reads the reports -> curates CLAUDE.md
          |  Agent / SendMessage                                  ^
    kernel-worker x N (Sonnet 5.5, parallel)                      |  READY: <dir>  (or QUESTION / BLOCKED)
        new-kernel -> edit .cu -> compile (local nvcc, then Modal CPU build) -> self-review -> READY
        resumed with the verbatim error if the GPU run fails -> fix -> READY again

Sub-agents cannot spawn sub-agents, so the orchestrator is the main session. A worker that needs a decision stops with
`QUESTION:`; the orchestrator resumes it with `SendMessage`. Everything checkable without a GPU (guard, syntax, nvcc, the full
torch build, register/spill stats) is the worker's own tight loop; only the orchestrator spends GPU time. An earlier design had
a third agent (modal-runner) that ran the submits and relayed the report. It added a hop (about 20 s per candidate on top of a
110 s submit) and no judgement, since `submit` already prints a compact, complete report, so it was removed.

## Why a one-container pipeline

`run_pipeline` runs test -> benchmark -> Nsight Compute in one GPU container: the extension is built once, the container
start cost is paid once, and **latency is measured with no profiler attached**. Profiling runs only after correctness and
benchmarking passed, so a broken kernel never costs an ncu run, and ncu never perturbs a timing.

## Profiling design

- Roofline x is the *measured* arithmetic intensity (algorithmic FLOPs / DRAM bytes), y the benchmark throughput, so each point
  mixes two independent measurements. Algorithmic intensity (FLOPs / minimum bytes) is the ceiling that ideal reuse would reach.
- Two ncu passes because coverage and detail trade off: a light pass (time, DRAM bytes, registers) over every launch gives exact
  totals; a detailed pass over the first N launches explains why. A single capped pass under-counts DRAM traffic silently
  (it overstated the baseline's intensity ~8x), so truncation is detected and the number withheld when no light pass exists.
- ncu runs with `--clock-control none` (Modal containers cannot lock clocks). Durations are relative; ratios and bytes are reliable.
- `gpu_busy` = ncu kernel time / benchmark latency flags host-bound candidates (the baseline MoE: GPU ~0.1% busy).

## Records

SQLite is the index, files are the record. Every candidate keeps its exact sources, raw evaluator output, ncu report and CSVs, a
generated brief, and a JOURNAL.md entry; the journal and `./optk status` are enough to rebuild the orchestrator's state after a
context compaction. Ids are allocated atomically so parallel workers never collide.

## Multiple problems

A problem is a directory (reference, task spec, baseline, cost model, notes). `./optk new-problem` scaffolds one; runs, baselines,
journals and dashboard entries are per problem. Spec keys must be letters only (the evaluator parses `[a-zA-Z]+`).

## Safety

The evaluator and checker are never part of a candidate (staged from `problems/`); candidate file names are validated; a static
guard rejects reward-hacking patterns in Python and CUDA sources; the secret seed is removed from the environment before the
submission process starts. The guard is a tripwire, not a boundary: fresh seeds and the reference check do the real work.
