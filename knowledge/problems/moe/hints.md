# First ideas for the MoE problem (most promising first; one per candidate, CUDA)

- Delete the per-call setup: do not build `nn.Module`s or random-initialise anything per call (the baseline spends ~91% of its time in `aten::uniform_`); use the tensors in `data` directly (expert and shared-expert weights are stored [in, out], so `x @ W` works as is; check the router weight's shape in `generate_input`).
- Sort the (token, expert) pairs by expert id once, on the GPU, and process each expert's contiguous block of tokens; compute group offsets without a host sync (no `.cpu()`, no `bincount().cpu()`).
- Replace the Python loop over experts with one grouped-GEMM kernel that walks all experts' token blocks in a single launch (see DeepGEMM's grouped layouts in the knowledge base).
- Fuse `SiLU(x @ W_gate) * (x @ W_up)` into one kernel so the [tokens, d_expert] intermediates are not written to HBM twice.
- Fuse the router: softmax + top-k; apply the routing weight in the down-projection epilogue before the scatter-add.
- Handle the shared expert as a dense GEMM chain, fused the same way, and add it into the output in the same epilogue.
- Then the Hopper path: WGMMA + TMA + mbarrier pipeline, warp specialization, persistent tile scheduler with an L2-friendly order (see CLAUDE.md playbook).
