"""
Logging for a run: concise lines on the console, everything (debug detail + tracebacks) in
<run dir>/agent.log. Library code uses logging.getLogger("optagent.<module>").
"""
import logging
import sys
from pathlib import Path

ROOT = "optagent"


class _Console(logging.Formatter):
    """Message only, with a [warn]/[error] prefix; tracebacks go to the file, not the terminal."""

    def format(self, record: logging.LogRecord) -> str:
        saved = record.exc_info, record.exc_text
        record.exc_info = record.exc_text = None
        try:
            msg = super().format(record)
        finally:
            record.exc_info, record.exc_text = saved
        if record.levelno >= logging.ERROR:
            return f"  [error] {msg}"
        if record.levelno >= logging.WARNING:
            return f"  [warn] {msg}"
        return msg


def setup(run_dir: Path | None = None, verbose: bool = False) -> logging.Logger:
    root = logging.getLogger(ROOT)
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False
    con = logging.StreamHandler(sys.stdout)
    con.setLevel(logging.DEBUG if verbose else logging.INFO)
    con.setFormatter(_Console("%(message)s"))
    root.addHandler(con)
    if run_dir is not None:
        fh = logging.FileHandler(Path(run_dir) / "agent.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
    return root
