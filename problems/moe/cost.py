"""Roofline model for the MoE problem: algorithmic FLOPs and the minimum HBM traffic of one call."""

PEAK = "fp16_tc"   # fp16/bf16 tensor-core peak
FP16_BYTES = 2


def cost(a: dict) -> tuple[float, float]:
    """DeepSeek-style MoE layer, fp16: router + top-k routed experts + shared expert (3 GEMMs each).

    flops: 2*FLOPs per MAC. bytes: every weight, the input and the output touched exactly once.
    """
    T = a["bs"] * a["seqlen"]
    dh, de, E = a["dhidden"], a["dexpert"], a["nroutedexperts"]
    k, ns = a["nexpertspertoken"], a["nsharedexperts"]
    flops = 2 * T * dh * E                       # router
    flops += T * k * 6 * dh * de                 # routed experts: gate, up, down per selected expert
    flops += T * 6 * dh * de * ns                # shared expert
    weights = dh * E + E * 3 * dh * de + 3 * dh * de * ns
    acts = 2 * T * dh                            # read x, write out
    return float(flops), float((weights + acts) * FP16_BYTES)
