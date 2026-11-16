"""
Write a self-contained dashboard.html for one run (data embedded, works offline, no server).

    python -m dashboard.build                       # newest run -> workspace/runs/<id>/dashboard.html
    python -m dashboard.build --run 20251107-0921 --open
"""
import argparse
import json
import webbrowser
from pathlib import Path

from agent.config import ROOT

from .data import build_payload, list_runs

TEMPLATE = Path(__file__).with_name("template.html")


def render_html(payload: dict | None) -> str:
    """payload=None -> the page fetches /api/* (live server mode)."""
    blob = json.dumps(payload, separators=(",", ":")) if payload is not None else "null"
    blob = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.read_text().replace("/*__DATA__*/null", blob, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run id (default: newest)")
    ap.add_argument("--runs-dir", default=str(ROOT / "workspace" / "runs"))
    ap.add_argument("--out", help="output file (default: <run dir>/dashboard.html)")
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    a = ap.parse_args()

    runs = list_runs(Path(a.runs_dir))
    if not runs:
        raise SystemExit(f"no runs found under {a.runs_dir}")
    run_id = a.run or runs[0]["id"]
    out = Path(a.out) if a.out else Path(a.runs_dir) / run_id / "dashboard.html"
    out.write_text(render_html(build_payload(Path(a.runs_dir), run_id)))
    print(f"wrote {out}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
