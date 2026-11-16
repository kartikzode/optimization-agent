# knowledge/

The agents' reference shelf. Put PDFs here (the CUDA programming guide, tuning guides, papers, anything you want Claude to consult),
then search them instead of guessing:

    ./optk kb add path/to/cuda-programming-guide.pdf      copy in (or just drop the file in knowledge/pdfs/)
    ./optk kb add https://example.com/guide.html          download a PDF, or convert an HTML page to text
    ./optk kb list                                        what is indexed
    ./optk kb search "wgmma accumulator layout" -n 6      ranked pages with snippets
    ./optk kb read cuda-c-best-practices-guide --pages 31-33

    pdfs/            PDFs (yours)                     docs/        markdown / text / pages saved from the web
    hardware/        short GPU fact sheets (h100.md)  resources.md  curated links (from the gpu-perf-engineering-resources list)
    INDEX.md         generated index                  .cache/      extracted text per page (derived, gitignored)
    fetch_core.sh    downloads the core kernel-optimization references listed in resources.md

Text is extracted once (pypdf, or poppler's `pdftotext` when installed) and cached; changed or removed files are picked up on the next
search. Search is BM25 over pages (a "page" of a markdown/text doc is a ~3000-character chunk). Claude Code can also open a PDF
directly with the Read tool and a page range; the knowledge base is for finding *which* pages.
