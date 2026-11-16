"""Baseline for __NAME__ (the seed iteration). Optimized candidates are built from templates/cuda instead."""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return data * 2.0          # TODO: same as the reference, or a straightforward torch implementation
