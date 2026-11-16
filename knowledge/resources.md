# Curated resources for GPU kernel optimization

Distilled from https://github.com/kartikzode/gpu-perf-engineering-resources (the kernel / GPU / profiling parts; the
serving and distributed sections are omitted because this repo optimizes single-GPU kernels). `./optk kb list` shows
which of these are already downloaded and searchable (`./knowledge/fetch_core.sh` fetches the core set). To add a PDF
you downloaded: `./optk kb add path/to.pdf`.

## Start here
- CUDA C++ basics: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
- Programming Massively Parallel Processors (textbook, the main reference for tiling, coalescing, reductions, scans)
- Roofline paper: https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf (compute vs bandwidth vs arithmetic intensity)
- GPU Mode lectures (practical companion): https://github.com/gpu-mode/lectures

## GPU fundamentals
- CUDA Programming Guide (normative): https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (coalescing, shared memory, occupancy, sync)
- Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/ (TMA, clusters, async execution)
- NVCC driver: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/ (wgmma, cp.async, mbarrier, memory model)
- CUDA binary utilities (cuobjdump, nvdisasm): https://docs.nvidia.com/cuda/cuda-binary-utilities/

## Kernel optimization: exercises that teach the fundamentals
- Matrix transpose (coalescing, smem tiling, bank conflicts): https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
- Parallel reduction (divergence, sequential addressing, unrolling, warp shuffles): https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- Decoupled look-back scan (single-pass prefix scan): https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
- Online softmax (one-pass normalizer): https://arxiv.org/abs/1805.02867
- Batch-invariant kernels / determinism: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

## Matrix multiplication (the core of most kernels, and of MoE)
- Volkov, benchmarking GPUs for dense linear algebra (registers, not occupancy): https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf
- siboehm, CUDA matmul from naive to cuBLAS-like: https://siboehm.com/articles/22/CUDA-MMM
- Anatomy of high-performance matmul kernels (layouts, tiling, PTX, roofline): https://www.aleksagordic.com/blog/matmul
- Outperforming cuBLAS on H100, a worklog: https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- Colfax, TMA tutorial: https://research.colfax-intl.com/tutorial-hopper-tma/
- CUTLASS 3.x design: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html
- Stream-K (wave quantization): https://arxiv.org/abs/2301.03598
- DeepGEMM (compact FP8 GEMM for Hopper, grouped GEMMs for MoE): https://github.com/deepseek-ai/DeepGEMM
- tcgen05 for dummies (Blackwell GEMM): https://gau-nernst.github.io/tcgen05/

## Tensor cores and low precision
- OCP FP8 spec (E4M3/E5M2); OCP microscaling (MX) formats; NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

## Attention
- FlashAttention 1/2/3: https://arxiv.org/abs/2205.14135, https://arxiv.org/abs/2307.08691, https://arxiv.org/abs/2407.08608
- FlashInfer: https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html

## CUTLASS / CuTe
- CuTe layout algebra: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html
- CuTe GEMM tutorial: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
- CUTLASS pipelines (producer/consumer, async stages): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html

## Profiling, benchmarking, correctness
- Nsight Compute profiling guide (metrics, sections, replay, roofline): https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/
- Compute Sanitizer (memcheck, racecheck, initcheck, synccheck): https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/
- CUTLASS GEMM measurement methodology: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html

## Mixture-of-experts (this repo's first problem)
- DeepSeek-V3 (routed + shared experts): https://arxiv.org/html/2412.19437
- GShard (token dispatch/combine, capacity): https://arxiv.org/abs/2006.16668
- DeepEP, EPLB: https://github.com/deepseek-ai/DeepEP, https://github.com/deepseek-ai/EPLB

## AI-generated kernels (what to learn from, what to avoid)
- KernelBench: https://proceedings.mlr.press/v267/ouyang25a.html
- KernelGuard (reward-hacking defenses: static rules, physics-floor checks): https://icml.cc/virtual/2026/67922
- SOL-ExecBench (measure against a speed-of-light model): https://github.com/nvidia/sol-execbench
