"""Ground truth for __NAME__. The evaluator imports generate_input and check_implementation from here."""
import torch
from task import input_t, output_t
from utils import make_match_reference


def generate_input(size: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(size, device="cuda", dtype=torch.float32, generator=gen)


def ref_kernel(data: input_t) -> output_t:
    return data * 2.0          # TODO: the real computation, written for clarity not speed


# rtol/atol decide what counts as correct: choose them for the precision the kernel is allowed to use
check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
