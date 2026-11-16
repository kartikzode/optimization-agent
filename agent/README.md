# agent/

The tools behind `./optk`. Nothing here calls an LLM: Claude Code (the orchestrator and its sub-agents) drives these commands.
It never runs kernels itself: it talks to `harness/`, and writes only under `workspace/`.

    cli.py        the ./optk commands (doctor, problems, new-problem, new-kernel, compile, submit, profile, ncu, status, show, brief, compare, best, journal, kb)
    pipeline.py   submit(): reserve id -> precheck -> Modal pipeline -> logs/profile/summary -> brief -> journal; terminal_report(); reprofile()
    store.py      SQLite index + files per candidate (kernel/, logs/, profile/, brief.md, meta.json); parallel-safe id allocation
    precheck.py   python compile -> guard (py + native) -> local nvcc on torch-free .cu (Godbolt only on request)
    guard.py      static tripwire against reward hacking / sandbox escape: AST rules for .py, pattern rules for .cu/.cpp
    compile.py    local nvcc (ptxas -v: registers, smem, spills) or a Modal CPU build of the whole candidate
    brief.py      iteration brief + journal entry; profile comparison ("what improved / regressed vs the parent")
    roofline.py   hardware peaks, shape labels, achieved-vs-roof points (shared by briefs and the dashboard)
    problems.py   problem registry, cost-model loader (problems/<p>/cost.py), scaffolding for new problems
    kb.py         knowledge base: ingest PDFs/docs, BM25 search, read pages, add by file or URL
    events.py     global event log (workspace/events.jsonl): spans, heartbeats, stall analysis;  hooklog.py  Claude Code hooks -> events
    config.py     paths and GPU settings (.env)        log.py   console vs agent.log logging
    godbolt.py    Compiler Explorer client (opt-in; sends source to godbolt.org)

Runs are per problem: `workspace/runs/<problem>` unless `--run` says otherwise. Several workers can submit at once
(each `submit` is its own process with its own Modal app; ids are allocated atomically in SQLite).

Statuses: `running` (in flight), `pass`, `fail` (wrong output), `crash`, `timeout`, `rejected` (guard / compile before the GPU),
`infra` (Modal trouble: not the kernel's fault; excluded from the strategy and worker statistics).
