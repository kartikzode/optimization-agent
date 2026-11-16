# dashboard/

Results of a run, read from its SQLite store (`workspace/runs/<run>/store.db`) and files.

    python -m dashboard.serve [--port 8765]            live; polls every 8 s, so iterations show up (as "Running") while workers submit
    python -m dashboard.build [--run ID] [--open]      self-contained dashboard.html (data embedded, works offline; log links are relative file links)
    python -m dashboard.demo                           SYNTHETIC run for previewing the UI (clearly bannered)

What it shows
- **KPIs**: best latency (and speedup over the baseline), % of the roofline, measured arithmetic intensity vs algorithmic and ridge, iteration counts.
- **Latency by iteration** (toggle: latency in ms or speedup), log scale, best-so-far line, failures in a strip with distinct shapes
  (wrong output / crash / timeout / rejected / running / infra). Latency comes from the benchmark stage, which never runs under a profiler.
- **Roofline**: achieved TFLOP/s (benchmark) against *measured* arithmetic intensity (algorithmic FLOPs / DRAM bytes from Nsight Compute),
  with the best-so-far trajectory drawn as arrows and the "ideal reuse" intensity as a dashed line: better kernels move right and up.
- **Arithmetic intensity by iteration**: the same number as a trend against the ideal-reuse and ridge-point lines, so progress is unmistakable.
- **Profiler metrics by iteration**: SM throughput, memory throughput, occupancy, tensor-pipe activity.
- **Which ideas paid off**, by strategy or by worker (share of attempts that beat their parent by at least 2%).
- **Iteration blocks**: one card per candidate (latency, vs baseline and vs parent, profile one-liner, hypothesis, worker). Click one for tabs:
  Overview (per-shape table), **Brief** (the generated write-up), **Profile** (totals and top kernels), **Logs** (links to every log and the
  `.ncu-rep`), Diff vs parent, Files, Error.
- A table view for every chart, light/dark theme.

Roofline caveats
- Peaks are vendor datasheet numbers (dense). The FLOP / minimum-byte model of a problem is `problems/<name>/cost.py`; without one the
  roofline card says so and everything else still works. Measured intensity needs an Nsight Compute profile: iterations without one are not on the roofline.
- If the detailed ncu pass hit its launch limit and no light pass exists, DRAM bytes (and so measured intensity) are withheld rather than reported wrong.

Log links in the live dashboard are served from `/raw/<run>/<path>` with an allow-list of run ids and file types, and a check that the
resolved path stays inside the run directory. Text from kernels, logs and errors is only ever inserted with `textContent`, and embedded JSON is escaped.
