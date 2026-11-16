# harness/

Runs one candidate against one problem on Modal and returns structured results. No LLM code here.

    evaluate.py      Evaluator: stage files -> Modal -> parse. run_candidate (GPU pipeline), compile (CPU build), evaluate (one stage)
    modal_runner.py  Modal app + image + the remote functions (below)
    parse.py         eval.py's `key: value` output + exit code + stderr -> result dict (status pass | fail | crash | timeout)
    ncu_text.py      Nsight Compute text report: split per kernel, filter, extract NVIDIA's OPT/WRN analysis messages
    ncu.py           Nsight Compute CSV -> per-kernel and total summary (DRAM bytes, measured arithmetic intensity, SM / memory / tensor-pipe, bound)
    cli.py           python -m harness.cli --problem moe --submission <file> --mode test|benchmark|profile   (single stage, no recording)
    cleanup.py       list / stop leftover Modal apps

Remote functions (`modal_runner.py`; the image is CUDA 13 devel + torch 2.9.1 + ninja + Nsight Compute, `TORCH_CUDA_ARCH_LIST` set for the GPU):

    run_pipeline   one GPU container, stages in order, later stages skipped if a required one failed:
                     test -> benchmark                       latency, no profiler attached
                     -> ncu_light (every launch: time, DRAM bytes, regs)   -> ncu (detailed sections, first N launches)
                     -> torch profiler table only if ncu produced nothing
                   The container shares one nvcc build cache across stages, so an extension is compiled once per candidate.
    run_compile    CPU container (8 cores): `import submission` builds the extension; returns nvcc's output
    run_eval       one stage (harness.cli and the tests)

Nsight Compute notes: counters are accessible on Modal, but a container cannot lock GPU clocks, so it runs with `--clock-control none`
(durations are relative; the benchmark stage owns latency). Profiling is limited to the `custom_kernel` NVTX range, after an untimed
warm-up call (the edit in `problems/eval.py`'s ncu path), on one benchmark shape (the last by default).

Result: `status` (pass | fail | crash | timeout), per-case `tests` / `benchmarks` (mean_ns, best_ns, std_ns), `geomean_ns`, `error`
(first failure, or the stderr tail on a crash), `wall_s`.

A candidate controls only its own files (submission.py plus helper sources, text only, no path escapes); eval.py, utils.py,
reference.py, task.py are always staged from problems/. Secrets (`EVAL_SEED`) never reach the submission process.
