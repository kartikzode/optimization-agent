# workspace/

The agents' sandbox and the record of every run. Everything here is generated and gitignored except this file.

    workspace/runs/<run>/                 one run per problem (default run name = problem name)
        store.db                          index: status, latency, lineage, profile summary per candidate
        JOURNAL.md                        one short entry per finished iteration (what was tried, result, link to the brief)
        agent.log                         full timestamped log with tracebacks
        best/                             copy of the fastest correct kernel
        scratch/<worker>/                 where workers build kernels before submitting (candidate.json holds parent, strategy, hypothesis)
        candidates/<id>/
            kernel/                       the exact files that were evaluated
            brief.md                      what was tried, latency, profiler summary, what changed vs parent, links
            meta.json                     status, latency, lineage, profile summary, error
            logs/                         precheck.json  test.log  benchmark.log  profile_light.log  profile.log  torch_profile.log  gpu_info.txt
            profile/                      ncu.ncu-rep  ncu_raw.csv  ncu_details.csv  ncu_light_raw.csv  summary.json
        dashboard.html                    snapshot written by `python -m dashboard.build`

`.ncu-rep` files are large (tens of MB with the default `detailed` set, more with `--ncu-set full`): prune old ones when disk matters.
Runs from the earlier LLM-loop layout (`workspace/runs/2026*`, `smoke1`) still open in the dashboard.
