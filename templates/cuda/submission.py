"""
CUDA candidate skeleton: this file is the Python entry point the evaluator imports; the kernel lives in kernel.cu.

How it fits together
  * kernel.cu is plain CUDA with a C++ launcher taking raw pointers. It has no torch includes, so
    `./optk compile` can check it locally with nvcc in seconds (and shows registers / smem / spills).
  * CPP_SRC is the thin torch wrapper (tensor checks, output allocation, stream). load_inline builds both
    and binds `launch`.
  * The extension is built at IMPORT time, so compilation never lands inside the timed region. The evaluator
    warms up before timing, but a build inside custom_kernel would still hit the first call.

Replace the toy `scale` kernel with the real thing; custom_kernel(data) must return what the problem's
reference returns (see problems/<name>/reference.py and task.py).
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).resolve().parent

CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// defined in kernel.cu
void launch_scale(const float* x, float* y, float alpha, int n, cudaStream_t stream);

torch::Tensor scale(torch::Tensor x, double alpha) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 CUDA");
    auto y = torch::empty_like(x);
    launch_scale(x.data_ptr<float>(), y.data_ptr<float>(), (float)alpha, (int)x.numel(),
                 at::cuda::getCurrentCUDAStream());
    return y;
}
"""

# TORCH_CUDA_ARCH_LIST is set in the Modal image (9.0a for H100), so no -arch flag here.
# -lineinfo lets Nsight Compute map metrics back to source lines; -Xptxas -v prints registers/smem/spills.
_ext = load_inline(
    name="kernel_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=(HERE / "kernel.cu").read_text(),
    functions=["scale"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-Xptxas", "-v"],   # add "--use_fast_math" only if the tolerance allows
    verbose=True,
)


def custom_kernel(data):
    raise NotImplementedError(
        "template: implement custom_kernel for this problem (read problems/<name>/reference.py), "
        "e.g. call _ext.scale(x, 2.0) on a CUDA fp32 tensor to see the toolchain work")
