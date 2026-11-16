# CLAUDE.md: GPU kernel optimization, run by Claude Code

This repo is the **toolbox and the record** for optimizing CUDA kernels on a Modal H100. There is no LLM client in it:
Claude Code is the optimizer. The repo runs candidates on the GPU, profiles them, saves every log, and shows the results.

    ./optk <command>      the one tool entry point (wraps `.venv/bin/python -m agent`); `./optk --help`

## The team (multi-agent mode, `/optimize`)

| role | model | job |
|---|---|---|
| **orchestrator** | the main session; run it on **Fable 5.2** (`claude --model claude-fable-5-2`) | reads results, diagnoses the bottleneck, decides the next experiments, assigns one idea per worker, answers worker questions, keeps this file's tricks section current |
| **kernel-worker** (several, parallel) | Sonnet 5.5 (`.claude/agents/kernel-worker.md`) | writes one CUDA candidate, reasons about correctness, compiles until clean, asks the orchestrator when stuck, hands over `READY: <dir>` |
| **modal-runner** | Sonnet 5.5 (`.claude/agents/modal-runner.md`) | `./optk submit`s ready candidates, returns latency + profiler facts + log paths |

Sub-agents cannot spawn sub-agents, so the **orchestrator must be the main session**. Back-and-forth works like this: a worker
ends with `QUESTION: ...`; the orchestrator answers with `SendMessage` to that worker (which resumes it with its context). A failed
candidate goes back to its worker the same way, with the runner's verbatim error. Playbook: `.claude/skills/optimize/SKILL.md`.
Working alone (no sub-agents) is fine for small steps: you then do all three roles; follow the same loop.

## Hard rules

1. **Never edit `problems/` (the evaluator, reference, task.yml, baseline), `harness/`, `agent/`, `dashboard/` while optimizing.** `problems/` is read-only for agents (enforced in `.claude/settings.json`); the problem knowledge files live in `knowledge/problems/<p>/` (`notes.md`, `hints.md`), which you should keep current. Candidates live only in `workspace/runs/<run>/scratch/<name>/`. A tool bug is not yours to patch: report it (as the runner did for the nvcc issue) and let the maintainer fix it.
2. **No cheating.** The guard rejects: importing `reference`/`eval`/`utils`, subprocess/threads/files/env access, side CUDA streams, monkeypatching torch/time/sys, and in `.cu`: `cudaStreamCreate*`, host callbacks, threads, `system/popen/dlopen`, `getenv`, file I/O. Caching outputs across calls, hard-coding shapes/seeds, skipping work the reference does: all invalid, and fresh seeds expose them. A speedup that does not survive "is the same work being done?" is not a result.
3. **CUDA is the target.** `.cu` files with a thin `submission.py` wrapper (template below). Triton/torch-only candidates are allowed for a quick experiment but the goal is hand-written CUDA.
4. **Precision follows the reference**: tolerance is in `problems/<p>/reference.py` (`rtol/atol`). fp32 accumulation for fp16/bf16 inputs. Do not loosen anything to "win".
5. **Latency is measured without a profiler.** Never judge speed from an ncu number (clocks are not locked; kernels are serialised). ncu explains *why*; the benchmark says *how fast*.
6. **One idea per candidate.** Bundling changes makes the measurement unreadable. Record the hypothesis before submitting; every iteration needs one.
7. **Do not re-submit the same kernel hoping for a better number.** Runs cost GPU time; noise is reported (`± std`).
8. Secrets: never print `.env` or Modal tokens. Nothing in this repo sends source to a third party unless you pass `use_godbolt` (off by default).

## Commands

    ./optk doctor                              Modal creds, local nvcc, problems, knowledge base
    ./optk problems                            problems and their run status
    ./optk new-problem NAME                    scaffold problems/NAME (reference, task.yml, cost.py) + knowledge/problems/NAME (notes.md, hints.md)
    ./optk submit --seed --problem P           evaluate the baseline submission.py: iteration #1, the speedup reference
    ./optk new-kernel --problem P --name W1 [--from-candidate N | --from-best | --template cuda|baseline]
                                               scratch dir with the files + candidate.json (parent, worker, strategy, hypothesis)
    ./optk compile DIR --problem P [--where local|modal]   exit 1 on failure; prints regs / smem / spills (ptxas -v)
    ./optk submit DIR --problem P [--parent N] [--no-profile] [--shape last|first|IDX]
                                               precheck -> H100 test -> benchmark (no profiler) -> Nsight Compute if correct -> brief
    ./optk profile --cand N [--shape i] [--kernel-regex R] [--launch-count K] [--ncu-set detailed|full]
                                               re-profile a correct candidate (another shape, one kernel, more launches)
    ./optk status [-n 10]                      baseline, best (+ % of roofline), leaderboard, running, recent iterations
    ./optk show N | compare A B | best | journal [-n 6]
    ./optk note "why X happened" [--kind insight|decision|dead-end|plan|question] [--cand 3 5]    ./optk notes [-n 10]
                                               the orchestrator's memory of what it UNDERSTOOD (NOTES.md in the run; shown by `status`)
    ./optk ncu N [--advice] [--kernel REGEX] [--section REGEX]   the Nsight Compute report as TERMINAL TEXT (no GUI); --advice = NVIDIA's own
                                               OPT/WRN analysis messages per kernel ("limited by registers", "uncoalesced accesses", ...)
    ./optk kb search "query" | read SLUG --pages 3-5 | add FILE_OR_URL | list     knowledge base
    .venv/bin/python -m dashboard.serve        live dashboard http://127.0.0.1:8765   (or `dashboard.build --open`)
    .venv/bin/python -m pytest tests -q        no GPU or network needed (a few tests use local nvcc / modal.local)

`--problem` may be omitted when only one problem exists; `--run` picks another run directory (default `workspace/runs/<problem>`).

## One iteration

1. **Diagnose** (orchestrator): `./optk status`, read the last briefs (`./optk show N`) and `JOURNAL.md`. Where is the time? What does the roofline say (memory- or compute-bound, how far from the roof, is measured arithmetic intensity rising)?
2. **Assign** distinct hypotheses, one per worker: parent candidate, the change, the reason, the expected profiler signature.
3. **Write + compile** (worker): `new-kernel`, edit `.cu`, `compile` until clean (read ptxas: spills, regs, smem), reason about correctness, fill `candidate.json`.
4. **Run** (runner): `submit DIR`. Test gate -> benchmark -> ncu only if correct. Report back.
5. **Learn** (orchestrator): **write down what you understood** with `./optk note` (one to three sentences per iteration or round: why it
   worked or not, what the profile showed, the decision and its reason, dead ends). This is what survives context compaction and what the next
   round starts from; `JOURNAL.md` only has the facts. Then read the brief's "Nsight Compute's own analysis" and, for the dominant kernel, `./optk ncu N --advice --kernel <name>`; compare with the parent, read "what changed", decide keep / drop / repair / next. If a result teaches something general, **add it to the playbook below**.

Failure handling: `rejected` = guard/compile (fix and resubmit with `--repair`); `fail` = wrong output (the log shows mismatches; check indexing, bounds, races, accumulation); `crash` = exception or CUDA error (traceback in the log; `compute-sanitizer` ideas in the KB); `timeout` = deadlock or far too slow; `infra` = Modal trouble, not the kernel's fault.

## When something stalls: `./optk where`

Everything that can hang is logged automatically (agents do not have to remember to log) to ONE file, `workspace/events.jsonl`:
- every `./optk` command: start, a **heartbeat every 60 s with the phase it is in** (e.g. `#7 WAITING on Modal H100: test -> benchmark -> ncu`), end with status and duration;
- every tool call of every agent (orchestrator, workers, runner) via Claude Code hooks (`.claude/settings.json` -> `agent/hooklog.py`), plus **permission prompts**,
  notifications, sub-agent start/stop and failed/denied tools.

An operation with a start and no end is *in flight*. `./optk where` pairs them up and says what is stuck: **STALLED** (no heartbeat / tool call far too long),
**DEAD** (the optk process is gone without an end: crashed or killed), **WAITING ON YOU** (a permission prompt nobody answered: the usual reason an unattended agent freezes),
plus per-agent last activity, the last events, and a one-line VERDICT. `./optk events [--actor X] [--kind K] [--open] [-n 60]` prints the raw log.
If a run seems stuck, run `./optk where` first; if the in-flight op is a Modal call, the Modal dashboard (or rerunning with `-v`) shows the container side.
Hooks only log sessions started after `.claude/settings.json` existed (a new `claude` session picks them up).

## Where everything is saved

    workspace/runs/<run>/                       one run per problem (default run = problem name)
      (global, not per run: workspace/events.jsonl, the stall/diagnosis log)
      store.db  JOURNAL.md (facts, automatic)  NOTES.md (understanding, written by you)  best/  scratch/<worker>/  agent.log
      candidates/<id>/
        kernel/            the exact files that were evaluated
        brief.md           what was tried, latency table, profiler summary, what changed vs parent, links to logs
        meta.json          status, latency, lineage, profile summary
        logs/              precheck.json  test.log  benchmark.log  profile_light.log  profile.log  torch_profile.log  gpu_info.txt
        profile/           ncu.ncu-rep (open in ncu-ui)  ncu_details.txt (readable report)  ncu_raw.csv  ncu_details.csv  ncu_light_raw.csv  summary.json

Raw evaluator output is always kept (stdout, stderr, exit code). The Modal dashboard also shows container logs.

## Reading the numbers

**Roofline.** y = achieved TFLOP/s (algorithmic FLOPs from `problems/<p>/cost.py` / benchmark latency). x = arithmetic intensity (AI):
*algorithmic* AI = FLOPs / minimum bytes (best any kernel could do); *measured* AI = FLOPs / DRAM bytes ncu saw. Measured AI below algorithmic
means data is re-read from HBM; **rising measured AI over iterations (toward the dashed "ideal reuse" line) is the sign of better tiling/fusion/caching.**
H100 SXM: 989 TFLOP/s fp16 tensor, 3.35 TB/s -> ridge AI about 295 FLOP/B. Below the ridge a kernel is memory-bound, above it compute-bound.

**ncu verdict -> next move** (the `bound` field; thresholds are rules of thumb):

| signal | meaning | try |
|---|---|---|
| brief says **host-bound** (ncu kernel time << latency) | the GPU is idle most of the call: the time is Python, launches, syncs, allocation, initialisation | fix that first; no kernel change will show up in the latency until it is gone |
| SM% and memory% both < 40% | latency-bound: stalls, too few warps/CTAs, tiny/many launches | more parallelism (grid >= 132 SMs x blocks/SM), fuse launches, async copies + multi-stage pipeline, check occupancy limiters (regs, smem) |
| memory% high, SM% low | memory-bound | raise reuse (smem tiling, bigger tiles), vectorize (16 B loads), coalesce, fuse to skip HBM round-trips, narrower storage types |
| SM% high, tensor-pipe% low | compute-bound but not on tensor cores | use MMA/WGMMA, not FMA |
| tensor-pipe% high (>80%) | near the roof for this shape | remaining: tile-quantization/wave effects, epilogue overlap, L2 rasterization, clusters + TMA multicast |
| measured AI << algorithmic AI | redundant DRAM traffic | block for L2/smem, fuse producer+consumer, persistent CTAs with L2-aware tile order |
| many launches, small kernels | launch/sync-bound | fuse, grouped kernel, device-side scheduling, no host syncs |
| registers high + spills (ptxas `spill`) | register pressure | `__launch_bounds__`, smaller tile per thread, split accumulators, recompute vs store |
| occupancy low but tensor-pipe high | fine on Hopper (WGMMA kernels run at low occupancy by design) | do not chase occupancy blindly |

ncu facts for this setup: runs on the Modal H100 with `--clock-control none` (containers cannot lock clocks), profiles one benchmark shape (the last, largest by default)
inside the `custom_kernel` NVTX range after a warm-up call. Two passes: a *light* pass over **every** launch (exact time/DRAM/launch totals) and a *detailed* pass
(throughputs, occupancy, tensor pipe) over the first `--launch-count` launches. If the call has more launches than the detailed pass covered, the brief says so; if the detailed
pass fails, you still get totals and a torch-profiler table. `--ncu-set full` adds source counters and makes a very large report (>100 MB for 40 kernels): use it for one candidate, not every iteration.

## Kernel contract (CUDA)

- A candidate is a directory: `submission.py` (defines `custom_kernel(data)`; the evaluator imports it) plus `.cu/.cuh/.h/.cpp` files. `./optk new-kernel` copies `templates/cuda/`.
- Pattern: `kernel.cu` = plain CUDA + a C++ launcher over raw pointers (no torch includes => `compile` checks it locally in seconds and prints ptxas stats); `submission.py` holds the torch wrapper as `CPP_SRC` and calls `load_inline(cpp_sources=..., cuda_sources=Path(kernel.cu).read_text(), functions=[...])`.
- **Build at import time**, never inside `custom_kernel`. Flags: `-O3 -std=c++17 -lineinfo -Xptxas -v`; the Modal image sets `TORCH_CUDA_ARCH_LIST=9.0a` (sm_90a: WGMMA/TMA), so do not add `-arch`. Add `--use_fast_math` only if the tolerance allows.
- The H100 baseline numbers: 132 SMs, 227 KB smem/block (opt in with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`), 64K registers/SM, 255 regs/thread, 50 MB L2, 3.35 TB/s HBM3, clusters up to 8 (16 non-portable).
- No host syncs (`.item()`, `.cpu()`, `bincount().cpu()`), no per-call allocation of big buffers if avoidable (allocate outputs once per call, never random-init anything), no per-call Python loops over experts/heads.
- Shapes in `task.yml`: tests include small awkward sizes, benchmarks the real ones. Spec keys are letters only (`eval.py` parses `[a-zA-Z]+`).
- Compile on Modal's CPU container (`compile --where modal`) before the first submit of a new design: it builds exactly what the GPU run will build, including bindings.

## Multiple problems

`problems/<name>/` = `reference.py` (`generate_input`, `ref_kernel`, `check_implementation`), `task.py`, `task.yml` (tests, benchmarks, timeouts), `submission.py` (baseline), plus
`cost.py` (FLOPs and minimum bytes per shape: powers the roofline). What we *know* lives separately in `knowledge/problems/<name>/`: `notes.md` (what we know), `hints.md` (first ideas). `./optk new-problem NAME` scaffolds both.
Each problem gets its own run (`workspace/runs/<name>`), seed, journal and dashboard entry. Read `knowledge/problems/<p>/notes.md` before working on a problem, and update it when you learn something specific.

## Knowledge base

`knowledge/` holds PDFs and saved docs (CUDA guides, tuning guides, papers, worklogs); `./optk kb search "..."` ranks pages, `kb read SLUG --pages N-M` prints them. **Search before guessing** about
a hardware limit, an instruction's semantics, or a pattern (wgmma layouts, mbarrier phases, swizzles). To add material: drop PDFs into `knowledge/pdfs/` or `./optk kb add <file|url>`.
`knowledge/resources.md` is the curated link list; `knowledge/fetch_core.sh` re-downloads the core set.

---

# Optimization playbook (living: keep this section current)

**Protocol.** When a candidate teaches something *general* (a technique that gave a verified speedup, a pitfall that cost an iteration, a profiler signature that predicts a fix),
add one entry under *Verified here* in the same turn: `- [problem #iter] technique: what changed -> measured effect (latency x, profile signature). Why it works / when it fails.`
Link the iteration (`workspace/runs/<run>/candidates/<id>/brief.md`). Promote an item from *Candidates* to *Verified* only after a measured win, and move refuted ones to *Refuted* with the evidence.
Entries are short, concrete and sourced. Do not add speculation to *Verified*.

## Verified here

- [infra] Nsight Compute works on Modal H100 containers but needs `--clock-control none`; `--set full` is >100 MB per 40 kernels, so the default is the `detailed` set plus a light pass for exact totals. A detailed pass capped at N launches silently under-counts DRAM bytes: for the MoE baseline (346 launches in the call, 40 profiled) the measured intensity read 2311 FLOP/B instead of the true 296, an 8x error. Always check `truncated`/`detailed_launches` in the brief.
- [moe #1 baseline] The baseline is **host-bound**: 6.54 s per call at seqlen 8192, but its 346 GPU kernels add up to only ~9.7 ms under ncu (GPU busy ~0.1%). 91% of the CPU time is `aten::uniform_`: it builds `nn.Module`s with random-initialised `nn.Linear` weights on every call (the reference does this too, and the timed call includes it). Its GEMMs are already cuBLAS `nvjet_*` kernels at ~92% SM and ~97% tensor-pipe (213 KB smem, 168 regs/thread), so the first win is *not* a better GEMM: delete the per-call construction and the Python loop over experts. It also moves 12.19 GB through DRAM against a 3.14 GB minimum (3.9x), so fusion/reuse is the second win. Read this off any brief: "host-bound" line, then "DRAM traffic ... x the minimum". (`workspace/runs/moe/candidates/0001/brief.md`)

- [moe #2] Delete per-call setup: no `nn.Module`/`nn.Linear`/`nn.Parameter` construction, weights from `data` used directly with `x @ W`, reference loop otherwise unchanged -> 6.61 s to 7.33 ms (901x; 5.80 ms @2048, 9.27 ms @8192), ncu kernel time 9.26 ms of 9.27 ms, so the call is GPU-bound afterwards. When a brief says host-bound, this is the whole win; do it before anything else. What remains at 8192: cuBLAS GEMMs ~60% (compute-bound, 76-91% SM), gather/index_add/elementwise glue ~35% over 314 launches. (`workspace/runs/moe/candidates/0002/brief.md`)
- [moe #3] One grouped GEMM launch instead of a per-expert loop (device-side counting sort by expert, upper-bound grid with early exit, shared expert as one more group, fused `silu(gate)*up` epilogue, per-(token, slot) output buffer + combine instead of scatter-add; `mma.sync` fp16, fp32 accumulators; no host sync) -> 314 launches to 7, seqlen 2048 from 5.80 to 3.19 ms (1.8x). The small shape (256 tokens/expert) is launch- and glue-bound under cuBLAS, so fusion pays there even with a plain `mma.sync` kernel. Passed all tests on its first GPU run. (`workspace/runs/moe/candidates/0003/brief.md`)

## Candidates (from the literature, not yet measured here)

**Memory system**
- Coalesce global accesses: consecutive threads touch consecutive addresses; check `l1_hit`, sectors/request in ncu. Strided access to a 32x32 smem tile is a 32-way bank conflict: pad the inner dimension (`tile[32][33]`) or swizzle. (NVIDIA transpose blog; Best Practices guide)
- Vectorize: 16-byte `float4`/`uint4` loads/stores, aligned; also for smem. (siboehm matmul kernel 6: ~3% on top of tiling)
- Fuse producers/consumers to avoid HBM round-trips; keep intermediates in registers/smem (the `silu(gate)*up` epilogue, routing-weight multiply and scatter-add for MoE).
- Raise arithmetic intensity by tiling: block tiling (smem) -> thread/register tiling -> warp tiling. Each level cut DRAM/smem traffic per FLOP (siboehm: 0.3 -> ~22 TFLOP/s fp32 on A6000 through 1D/2D blocktiling, vectorization, autotuning, warptiling).
- Prefer fewer threads with more registers per thread (ILP) over maximal occupancy for GEMM-like kernels (Volkov, SC08).
- L2-aware scheduling: order tiles (grouped / Hilbert rasterization) so concurrently running CTAs share A/B tiles; the H100 worklog reached 83% L2 hit rate (cuBLAS 70%) and +4% from this; 128 CTAs on a 16x8 region beat 132. 50 MB L2 on H100.

**Hopper (sm_90a) tensor-core path** (H100 worklog, 32 -> 764 TFLOP/s bf16 4096^3, cuBLAS 716)
- Fp32-style SIMT GEMM reaches ~4% of cuBLAS on H100. The jump is: **WGMMA (tensor cores, warpgroup MMA) + TMA loads + mbarriers**: 32 -> 317 TFLOP/s.
- Bigger output tiles (128x128 -> 128x256 with 2 consumer warpgroups): 498 -> 610; shift registers to consumers with `setmaxnreg` (e.g. 240 consumer / 24 producer): 631.
- Producer/consumer **warp specialization** with a multi-stage smem ring buffer (QSIZE 3-5) hides load latency: 317 -> 498.
- Persistent kernel + tile scheduler + L2-friendly order: 660. Faster barrier API (`cuda::ptx` mbarrier wait/arrive): +10% -> 704. Thread-block **clusters with TMA multicast** (2 CTAs share B tile loads): 734. Async stores, micro-opts, Hilbert curve order: 747 -> 764.
- Hopper tuning facts: max 64 warps/SM, 32 blocks/SM, 228 KB smem/SM, TMA handles 1D-5D tensors, no registers or SM instructions needed for the copy, one thread issues it; clusters share distributed smem (keep accesses coalesced/32B aligned). Dynamic smem above 48 KB needs an opt-in attribute.
- Wave quantization: tiles / SMs not an integer => idle SMs in the last wave. Stream-K or split-K / persistent scheduling fixes it. cuBLAS chooses among hundreds of kernels by shape (split-K for small M/N with large K); one kernel does not fit all shapes.
- DeepGEMM: compact Hopper FP8/BF16 grouped GEMMs for MoE (contiguous and masked layouts), JIT-compiled; read it for how to group variable-sized expert batches into one launch.

**Reductions / scans / softmax** (NVIDIA reduction slides: 15-21x over the naive kernel, step by step)
- Replace divergent interleaved addressing with sequential addressing (no divergence, no bank conflicts); do the first add during the global load (halve idle threads); unroll the last warp / use `__shfl_down_sync` instead of smem+barriers; template the block size for full unrolling; process several elements per thread (algorithmic cascading). Online softmax keeps one pass over memory.

**Launch and host overheads**
- Hoist all one-time setup (module construction, weight transposes, extension builds, buffer allocation) out of the timed call; avoid host syncs; fuse small kernels or capture CUDA graphs only if the guard allows (it currently forbids side streams, so graphs launched on the current stream are the safe form).
- Sort/bucket tokens by expert on the GPU once and give each expert a contiguous slice; compute group offsets without `.cpu()`.

**Profiling discipline**
- Warm up before timing; the evaluator clears L2 between runs, so cold-cache numbers are what you get. Compare like with like (same shape, same seed set). ncu replays kernels and locks nothing here: use it for ratios and bottlenecks, not for absolute time.
- Use `compute-sanitizer` (memcheck/racecheck/synccheck) when a kernel is wrong in non-obvious ways; a race often passes small tests and fails large ones.

## Refuted / pitfalls

- [moe #3] Pitfall: a 2D grid `dim3(rowTiles, colTiles)` with `rt = blockIdx.x` schedules row tiles fastest, so the CTAs running together are different row tiles (here: different experts) of one column tile and share neither A nor B. Each A tile is then re-read from HBM once per column tile: 27.7 GB of DRAM traffic (8.8x the minimum), L2 hit 42%, measured intensity 130 FLOP/B, both GEMMs memory-bound at 52-60% tensor pipe, and seqlen 8192 at 12.11 ms was slower than cuBLAS-in-a-loop (9.27 ms, #2). Signature to look for: "DRAM traffic Nx the minimum" with N close to the number of column tiles. Decide the CTA order on purpose. (`workspace/runs/moe/candidates/0003/brief.md`)
- [moe #4] Fixing that order alone (1D grid, `rt = lin / NCT`, `ct = lin % NCT`: group -> row tile -> column tile) cut DRAM traffic 27.7 -> 16.1 GB (-42%), raised L2 hit 42 -> 60% and measured intensity 130 -> 224 FLOP/B, but latency moved 6.21 -> 6.06 ms (8192: 12.11 ± 0.74 -> 11.64 ± 0.69 ms), which is inside the noise. Lesson: ncu's "memory-bound" label on a kernel with tensor pipe ~60% and DRAM throughput well under 100% does not mean HBM is the limiter; when traffic drops 42% and time does not, the inner loop is (here `mma.sync` + `ldmatrix` at 25% occupancy; cuBLAS's wgmma kernels do the same GEMMs at 76-91% SM). Check DRAM throughput %, not only the label, before spending a round on traffic. (`workspace/runs/moe/candidates/0004/brief.md`)
- [infra, moe #5] FIXED: any `.cu` using `wgmma` was rejected by the submit precheck because `nvcc -arch=sm_90a` (nvcc 13) also compiles a plain `compute_90` pass where ptxas rejects `wgmma.mma_async`. `agent/compile.py:local_nvcc` now passes `-gencode=arch=compute_90a,code=sm_90a`; WGMMA/TMA candidates pass the precheck. Lesson: never use the `-arch=sm_XXa` shorthand for Hopper-only code. (`workspace/runs/moe/candidates/0005/logs/precheck.json` shows the old failure.)
- Add: what was tried, the evidence, and why it did not work.
