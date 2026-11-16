from typing import TypedDict, TypeVar
import torch

input_t = TypeVar("input_t")
output_t = TypeVar("output_t")


class TestSpec(TypedDict):
    size: int
    seed: int
