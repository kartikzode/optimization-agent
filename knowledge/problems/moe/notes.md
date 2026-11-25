# moe: what we know

(Living notes for this problem. Add findings here: what worked, what did not, why. Keep it short.)

- Baseline (`submission.py`) builds `nn.Module`s per call and loops over experts in Python: launch-bound and sync-bound.
- Algorithmic arithmetic intensity is about 300 FLOP/byte at the benchmark shapes, right at the H100 ridge point
  (989 TFLOP/s / 3.35 TB/s): a good kernel can be compute-bound, so tensor cores (WGMMA on Hopper) matter.
- See hints.md (next to this file) for the first ideas to try.
- #2 (no per-call init, torch/cuBLAS loop): 7.33 ms (5.80 @2048, 9.27 @8192). #3 (hand-written grouped mma.sync GEMM, 7 launches): 6.21 ms (3.19 @2048, 12.11 @8192).
- #4 (#3 + column-fastest CTA order): 6.06 ms (3.15 @2048, 11.64 @8192): DRAM traffic -42% but latency within noise of #3. At 8192 the cuBLAS loop (#2, 9.27 ms) still beats the hand-written `mma.sync` kernel; the inner loop is the limiter, not HBM.
- Untested: `workspace/runs/moe/scratch/r2-w2` (#4's order + wgmma m64n64k16 inner loop). Blocked at the submit precheck by a local nvcc arch issue (see CLAUDE.md pitfalls); its smem descriptor encoding was written from memory and has never run.
- Floors: seqlen 8192 is 3.6e12 FLOP (~3.7 ms at the fp16 roof); seqlen 2048 is 0.9e12 FLOP (~0.9 ms) but still has to read all 3.14 GB of weights (~0.94 ms), so the small shape is bound by weight traffic and launch count, not arithmetic.
- Keep the router as the reference's exact torch ops (`F.linear`, `softmax`, `topk(sorted=False)`): expert selection then matches bit for bit on near-ties; it costs ~60 µs.
- Routed tokens per expert: ~T*k/E (256 at seqlen 2048, 1024 at 8192); the shared expert is a group of all T tokens.
- #3's launcher requires d_hidden % 128 == 0 and d_expert % 64 == 0 (true for every shape in task.yml); a general kernel needs partial column tiles.
- Best #21 3.47 ms (2048: 1.88, 8192: 6.41; 1904x baseline, 57% of fp16 roof at 8192). Design: route (torch router + route_kernel), gather_kernel makes A rows contiguous, moe_gemm = 1 TMA producer thread + 2 consumer WGs (wgmma m64n128 x2, 128x256 tile, setmaxnreg 40/232, 4 stages 197 KB), persistent strided raster (RS=8 row tiles), combine_kernel.
- What mattered in order: delete per-call init (#2) > grouped GEMM (#3) > wgmma (#6, 1.36x) > TMA pipeline + 128x256 tile (#18, 1.21x) > raster (#9, 1.05x). Dead ends: branching around wgmma (#12), cp.async producer WG (#14), contiguous persistent chunks (#20), clusters+multicast at 160 KB smem (#19).
- Open: 2048 shape is weight-streaming bound (floor ~0.94 ms vs 1.88); glue kernels ~9% at 8192 (gather 4%, combine 3%).
