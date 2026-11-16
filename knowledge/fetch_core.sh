#!/usr/bin/env bash
# Seed the knowledge base with the core kernel-optimization resources from the curated list
# (https://github.com/kartikzode/gpu-perf-engineering-resources). Re-runnable: ./knowledge/fetch_core.sh
# Each page is downloaded once, converted to text and indexed; see `./optk kb list`.
cd "$(dirname "$0")/.." || exit 1
add() { echo "=== $2"; ./optk kb add "$1" --name "$2" 2>&1 | tail -2; }
add https://siboehm.com/articles/22/CUDA-MMM siboehm-cuda-matmul
add https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html cuda-c-best-practices-guide
add https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html hopper-tuning-guide
add https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ nvidia-matrix-transpose
add https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf nvidia-parallel-reduction
add https://research.colfax-intl.com/tutorial-hopper-tma/ colfax-hopper-tma
add https://www.aleksagordic.com/blog/matmul aleksagordic-matmul-anatomy
add https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog h100-matmul-worklog
add https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html nsight-compute-profiling-guide
add https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf roofline-paper
add https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf volkov-dense-linear-algebra
add https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md deepgemm-readme
echo "ALL DONE"
