# optimization-agent

Tooling for **Claude Code to optimize CUDA kernels** on a Modal H100. The repo does not call any LLM: it is the toolbox
(compile, run, profile), the record (every log, every Nsight Compute report, a brief per iteration) and the dashboard.
Claude Code is the optimizer; in multi-agent mode a Fable 5.2 orchestrator directs Sonnet 5.5 kernel-writer and runner agents.

**Start with [CLAUDE.md](CLAUDE.md)**: it is what the agents read (commands, rules, how to read the profiler, the tricks playbook).

## Layout

    CLAUDE.md      guide for Claude Code: workflow, rules, commands, profiler reading, living optimization playbook
    optk           the tool entry point: ./optk submit | compile | profile | status | kb ...
    .claude/       agents/kernel-worker.md, agents/modal-runner.md (Sonnet 5.5), skills/optimize (the orchestrator playbook)
    problems/      task definitions and the trusted evaluator (candidates must not edit these)
      eval.py        test / benchmark / profile driver; reports key: value lines on fd $EVAL_FD
      <name>/        reference.py, task.py, task.yml, submission.py (baseline), cost.py (roofline), notes.md, hints.md
    templates/     cuda/ (kernel skeleton), problem/ (scaffold for ./optk new-problem)
    agent/         the tools behind ./optk: store, precheck + guard, compile, pipeline, briefs, roofline, knowledge base
    harness/       Modal side: one GPU container runs test -> benchmark (no profiler) -> Nsight Compute; CPU container compiles
    knowledge/     PDFs and docs the agents search (./optk kb), curated resource list, hardware notes
    workspace/     runs: one directory per problem; candidates, logs, profiles, briefs (gitignored)
    dashboard/     latency by iteration, roofline with measured intensity, AI trend, profiler trends, iteration blocks
    tests/         no GPU or network needed

## Setup

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/modal setup                 # one-time; or put the tokens in .env (see .env.example)
    ./optk doctor
    ./knowledge/fetch_core.sh             # optional: download the core kernel-optimization references into the knowledge base

## Use it

    ./optk submit --seed --problem moe          # baseline = iteration #1 (the speedup reference)
    claude --model claude-fable-5-2             # then:  /optimize moe 3 3     (3 rounds, 3 workers per round)
    .venv/bin/python -m dashboard.serve         # live dashboard at http://127.0.0.1:8765
    ./optk status                               # where things stand, from the terminal
    ./optk where                                # what is running / STUCK / waiting on you (global event log)
    ./optk ncu 3 --advice                       # Nsight Compute report as terminal text, with NVIDIA's own analysis
    .venv/bin/python -m pytest tests -q

Doing it by hand uses the same tools: `./optk new-kernel` -> edit the `.cu` -> `./optk compile DIR` -> `./optk submit DIR`.

## What one `submit` does

    precheck (python compile, guard on .py and .cu, local nvcc with ptxas stats)
      -> one Modal H100 container: correctness tests -> benchmark (NO profiler: this is the latency)
         -> only if both passed: Nsight Compute (a light pass over every launch, a detailed pass over the first 40)
      -> workspace/runs/<problem>/candidates/<id>/{kernel/, logs/, profile/, brief.md, meta.json} + JOURNAL.md

The **brief** says what was tried, the latency and its change vs parent/best/baseline, what the profiler shows
(measured arithmetic intensity, DRAM traffic, SM / memory / tensor-pipe, host-bound or not, what changed vs the parent), and links to the logs.

## Problem contract

A candidate is a directory: `submission.py` defining `custom_kernel(data)` plus its `.cu/.cuh/.cpp` files. `eval.py`
calls `generate_input(**spec)`, runs the kernel, and checks it with `check_implementation` from the problem's `reference.py`.
Add problems with `./optk new-problem NAME`; each gets its own run, baseline, roofline model and dashboard entry.

## Logs: where to look when something fails

    ./optk status / show N       the brief of an iteration
    workspace/runs/<run>/
      agent.log                  everything, with timestamps and tracebacks (console shows only short lines)
      candidates/<id>/logs/      precheck.json, test.log, benchmark.log, profile_light.log, profile.log, torch_profile.log
      candidates/<id>/profile/   ncu.ncu-rep (open in Nsight Compute), ncu_raw.csv, ncu_details.csv, summary.json
    Modal dashboard              container stdout/stderr per call
    -v / --verbose               stream the container logs to your terminal
