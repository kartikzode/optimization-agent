"""Roofline model for __NAME__: algorithmic FLOPs and minimum HBM bytes for one call. Powers the dashboard roofline."""

PEAK = "fp32"        # fp16_tc | tf32_tc | fp8_tc | fp32 : which compute peak applies to this kernel


def cost(a: dict) -> tuple[float, float]:
    """a: one shape from task.yml (keys as written there). Return (flops, minimum bytes moved)."""
    n = a["size"]
    return float(n), float(2 * n * 4)      # TODO: replace; here: 1 FLOP per element, read + write fp32
