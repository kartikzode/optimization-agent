# moe: what we know

(Living notes for this problem. Add findings here: what worked, what did not, why. Keep it short.)

- Baseline (`submission.py`) builds `nn.Module`s per call and loops over experts in Python: launch-bound and sync-bound.
- Algorithmic arithmetic intensity is about 300 FLOP/byte at the benchmark shapes, right at the H100 ridge point
  (989 TFLOP/s / 3.35 TB/s): a good kernel can be compute-bound, so tensor cores (WGMMA on Hopper) matter.
- See hints.md for the first ideas to try.
