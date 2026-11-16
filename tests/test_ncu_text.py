from harness import ncu_text

from conftest import NCU_TEXT


def test_blocks_and_advice_are_extracted_with_wrapped_lines_joined():
    blocks = ncu_text.split_kernels(NCU_TEXT)
    assert [b["name"] for b in blocks] == ["gemm_kernel", "gemm_kernel", "silu_mul"]
    msgs = ncu_text.advice(blocks[0])
    assert [m["kind"] for m in msgs] == ["INF", "OPT", "WRN"]
    assert msgs[1]["section"] == "Occupancy" and "registers, and the required amount of shared memory." in msgs[1]["text"]


def test_advice_only_view_dedupes_repeated_launches_and_filters():
    out = ncu_text.render(NCU_TEXT, advice_only=True)
    assert out.count("gemm_kernel (2, 66, 1)") == 1 and "[+1 more launches of this kernel]" in out
    assert "silu advice" in out
    only = ncu_text.render(NCU_TEXT, kernel="silu", advice_only=True)
    assert "gemm_kernel" not in only and "silu advice" in only
    assert "Uncoalesced" not in ncu_text.render(NCU_TEXT, section="occupancy", advice_only=True)


def test_full_view_filters_by_section_and_truncates():
    sec = ncu_text.render(NCU_TEXT, kernel="gemm", section="Speed Of Light")
    assert "Duration" in sec and "Occupancy" not in sec
    assert "truncated at 3 lines" in ncu_text.render(NCU_TEXT, max_lines=3)
    assert "no kernel matched" in ncu_text.render(NCU_TEXT, kernel="nothing")


def test_top_advice_picks_the_first_named_kernel_present_with_messages():
    a = ncu_text.top_advice(NCU_TEXT, ["void other<..>", "gemm_kernel", "silu_mul"])
    assert a["kernel"] == "gemm_kernel" and [m["kind"] for m in a["messages"]] == ["OPT", "WRN"]     # INF excluded
    assert ncu_text.top_advice(NCU_TEXT, ["absent"]) is None
