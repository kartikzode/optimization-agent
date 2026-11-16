"""
Live dashboard: re-reads the SQLite stores on every request, so it updates while the loop runs.

    python -m dashboard.serve [--port 8765]      # then open http://127.0.0.1:8765
"""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from agent.config import ROOT

from .build import render_html
from .data import build_payload, list_runs

L = logging.getLogger("optagent.dashboard")

# what /raw/ may serve: text shown inline, the Nsight Compute report as a download
_TXT = "text/plain; charset=utf-8"
RAW_TYPES = {".md": _TXT, ".log": _TXT, ".txt": _TXT, ".json": _TXT, ".csv": _TXT, ".py": _TXT, ".cu": _TXT,
             ".cuh": _TXT, ".h": _TXT, ".cpp": _TXT, ".ncu-rep": "application/octet-stream"}


def make_handler(runs_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                self._route()
            except Exception as e:      # a half-written store, a bad payload, ...: say so instead of dropping the connection
                L.exception("error serving %s", self.path)
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

        def _route(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == "/":
                return self._send(200, render_html(None).encode(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                return self._send(200, json.dumps(list_runs(runs_dir)).encode(), "application/json")
            if url.path == "/api/data":
                run = (q.get("run") or [""])[0]
                if run not in {r["id"] for r in list_runs(runs_dir)}:   # allow-list: no path traversal
                    return self._send(404, b'{"error":"unknown run"}', "application/json")
                return self._send(200, json.dumps(build_payload(runs_dir, run)).encode(), "application/json")
            if url.path.startswith("/raw/"):
                return self._raw(url.path[len("/raw/"):])
            if url.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            self._send(404, b"not found", "text/plain")

        def _raw(self, rest: str):
            """Serve a log / profile / kernel file of a run. Allow-listed run ids and extensions; the resolved path
            must stay inside the run directory (no traversal, no symlink escape)."""
            run, _, rel = unquote(rest).partition("/")
            if run not in {r["id"] for r in list_runs(runs_dir)} or not rel:
                return self._send(404, b"not found", "text/plain")
            base = (runs_dir / run).resolve()
            target = (base / rel).resolve()
            if base not in target.parents or not target.is_file() or target.suffix not in RAW_TYPES:
                return self._send(404, b"not found", "text/plain")
            self._send(200, target.read_bytes(), RAW_TYPES[target.suffix])

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-dir", default=str(ROOT / "workspace" / "runs"))
    a = ap.parse_args()
    from agent import log as logsetup
    logsetup.setup(None)    # scoped to our own loggers; never configure the root logger
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Path(a.runs_dir)))   # localhost only
    print(f"dashboard on http://127.0.0.1:{a.port}  (runs: {a.runs_dir})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
