import pytest

from agent import kb


@pytest.fixture
def kbdir(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "KB", tmp_path)
    monkeypatch.setattr(kb, "CACHE", tmp_path / ".cache")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "smem.md").write_text(
        "# Shared memory\n\nShared memory is divided into 32 banks. A 32x32 tile read by column causes a 32-way bank conflict; "
        "pad the tile to 33 columns to avoid bank conflicts.\n\n" + "filler paragraph about something else entirely.\n\n" * 5)
    (tmp_path / "docs" / "tma.md").write_text(
        "# TMA\n\nThe Tensor Memory Accelerator copies tiles asynchronously between global and shared memory using an mbarrier.\n")
    (tmp_path / "README.md").write_text("not indexed")
    return tmp_path


def test_ingest_search_ranks_the_relevant_doc_first(kbdir):
    docs = kb.ingest(log=lambda *_: None)
    assert sorted(d["slug"] for d in docs) == ["smem", "tma"]               # README is skipped
    hits = kb.search("bank conflict padding", n=3)
    assert hits[0]["slug"] == "smem" and "33" in hits[0]["snippet"]
    assert kb.search("mbarrier asynchronous copy")[0]["slug"] == "tma"
    assert kb.search("zebra giraffe") == []
    assert [h["slug"] for h in kb.search("bank conflict", doc="tma")] == []   # --doc restricts the search
    assert (kbdir / "INDEX.md").read_text().count("| smem |") == 1


def test_ingest_is_incremental_and_notices_changes_and_deletions(kbdir):
    kb.ingest(log=lambda *_: None)
    msgs = []
    kb.ingest(log=msgs.append)
    assert msgs == []                                                       # nothing changed: nothing re-extracted
    (kbdir / "docs" / "new.md").write_text("# New\n\nwarp specialization splits producer and consumer warps.")
    kb.ingest(log=msgs.append)
    assert any("new.md" in m for m in msgs)
    assert kb.search("producer consumer warps")[0]["slug"] == "new"
    (kbdir / "docs" / "new.md").unlink()
    kb.ingest(log=lambda *_: None)
    assert kb.search("producer consumer warps") == []


def test_add_copies_a_file_in_and_indexes_it(kbdir, tmp_path):
    src = tmp_path / "outside" / "guide.txt"
    src.parent.mkdir()
    src.write_text("Occupancy is limited by registers, shared memory and block size per SM.")
    dest = kb.add(str(src), name="my guide", log=lambda *_: None)
    assert dest.parent == kbdir / "docs" and kb.search("occupancy registers")[0]["slug"] == "my-guide"
    with pytest.raises(SystemExit, match="no such file"):
        kb.add(str(tmp_path / "missing.pdf"), log=lambda *_: None)
    bad = tmp_path / "x.exe"
    bad.write_text("x")
    with pytest.raises(SystemExit, match="unsupported"):
        kb.add(str(bad), log=lambda *_: None)


def test_pdf_pages_are_extracted_one_per_page(kbdir):
    pypdf = pytest.importorskip("pypdf")
    (kbdir / "pdfs").mkdir()
    w = pypdf.PdfWriter()
    w.add_blank_page(200, 200)
    w.add_blank_page(200, 200)
    with open(kbdir / "pdfs" / "blank.pdf", "wb") as f:
        w.write(f)
    docs = kb.ingest(log=lambda *_: None)
    blank = next(d for d in docs if d["slug"] == "blank")
    assert len(blank["pages"]) == 2


def test_html_is_converted_to_text_without_scripts_or_nav():
    title, text = kb.html_to_text("<html><head><title>T &amp; U</title><script>evil()</script></head>"
                                  "<body><nav>menu</nav><h2>Heading</h2><p>Real <b>content</b> here.</p></body></html>")
    assert title == "T & U" and "## Heading" in text and "Real content here." in text
    assert "evil" not in text and "menu" not in text


def test_read_reports_pages_and_truncates(kbdir, capsys):
    kb.main(["read", "smem", "--pages", "1"])
    out = capsys.readouterr().out
    assert "smem page 1/" in out and "32 banks" in out
    with pytest.raises(SystemExit, match="unknown document"):
        kb.main(["read", "nope"])
