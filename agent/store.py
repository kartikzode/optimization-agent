"""
Candidate store for one run (one problem). SQLite is the index; the files are the record.

    workspace/runs/<run>/
      store.db                     index (status, latency, lineage, profile summary)
      JOURNAL.md                   one short entry per finished iteration, newest last
      best/                        the fastest correct kernel so far (a copy of its kernel/ dir)
      scratch/<name>/              where workers build kernels before they are submitted
      candidates/<id>/
        kernel/                    the exact files that were evaluated (submission.py, *.cu, ...)
        meta.json                  everything in the index row, readable without SQLite
        brief.md                   what was tried, latency, profiler summary, links to the logs
        logs/                      precheck.json, compile.log, test.log, benchmark.log, profile.log
        profile/                   ncu.ncu-rep (open in Nsight Compute), ncu_raw.csv, ncu_details.csv, summary.json

Rows from the older LLM-loop layout (code in the row, candidates/<id>/submission.py) still load.
"""
import json
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    kind TEXT NOT NULL,              -- seed | improve | repair
    worker TEXT NOT NULL DEFAULT '',
    strategy TEXT NOT NULL DEFAULT '',
    hypothesis TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,            -- running | pass | fail | crash | timeout | rejected
    geomean_ns REAL,                 -- only for status=pass
    result TEXT NOT NULL DEFAULT '{}',
    profile_summary TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    wall_s REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
# columns added after the first schema: (name, DDL). Older stores are upgraded in place.
ADDED = [("worker", "TEXT NOT NULL DEFAULT ''"), ("profile_summary", "TEXT NOT NULL DEFAULT '{}'")]

TEXT_EXT = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".txt", ".json", ".md"}
MAX_FILE_BYTES = 400_000
MAX_FILES = 40
RESERVED = {"eval.py", "utils.py", "reference.py", "task.py", "tests.txt"}   # always staged from problems/


@dataclass
class Candidate:
    id: int
    kind: str
    status: str
    parent_id: int | None = None
    worker: str = ""
    strategy: str = ""
    hypothesis: str = ""
    geomean_ns: float | None = None
    result: dict = field(default_factory=dict)
    profile_summary: dict = field(default_factory=dict)
    error: str = ""
    wall_s: float = 0.0
    created_at: float = 0.0
    code: str = ""          # legacy rows only
    profile: str = ""       # legacy rows only (torch profiler table)

    @property
    def ok(self) -> bool:
        return self.status == "pass" and self.geomean_ns is not None


_FIELDS = {f.name for f in fields(Candidate)}


def collect_files(src: Path) -> dict[str, str]:
    """The text files of a kernel directory, as relative-path -> source. Raises ValueError with a reason."""
    src = Path(src)
    if not src.is_dir():
        raise ValueError(f"{src} is not a directory")
    out: dict[str, str] = {}
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if not p.is_file() or any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if p.suffix not in TEXT_EXT:
            continue
        if rel.name in ("candidate.json", "NOTES.md"):   # worker bookkeeping, not part of the kernel
            continue
        if str(rel) in RESERVED:
            raise ValueError(f"{rel} is staged from problems/ and cannot be part of a candidate")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{rel} is larger than {MAX_FILE_BYTES // 1000} KB")
        out[rel.as_posix()] = p.read_text()
    if "submission.py" not in out:
        raise ValueError("the kernel directory must contain submission.py (it defines custom_kernel)")
    if len(out) > MAX_FILES:
        raise ValueError(f"more than {MAX_FILES} files")
    return out


class Store:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        (self.dir / "candidates").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "store.db", check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(candidates)")}
        for name, ddl in ADDED:
            if name not in have:
                self.db.execute(f"ALTER TABLE candidates ADD COLUMN {name} {ddl}")
        self.db.commit()

    def close(self):
        self.db.close()

    # ---- paths --------------------------------------------------------------------------
    def cand_dir(self, cand_id: int) -> Path:
        return self.dir / "candidates" / f"{cand_id:04d}"

    # ---- writes -------------------------------------------------------------------------
    def reserve(self, *, kind: str, parent_id=None, worker="", strategy="", hypothesis="") -> Candidate:
        """Allocate an id (atomically, so parallel submitters never collide); the row says 'running' until finished."""
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO candidates (parent_id,kind,worker,strategy,hypothesis,status,created_at) VALUES (?,?,?,?,?,?,?)",
                (parent_id, kind, worker, strategy, hypothesis, "running", time.time()))
            self.db.commit()
            return self.get(cur.lastrowid)

    def save_kernel(self, cand_id: int, files: dict[str, str]):
        kdir = self.cand_dir(cand_id) / "kernel"
        shutil.rmtree(kdir, ignore_errors=True)
        for rel, text in files.items():
            f = kdir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    def finish(self, cand_id: int, *, status: str, geomean_ns=None, result=None, error="", wall_s=0.0,
               profile_summary=None) -> Candidate:
        with self.lock:
            self.db.execute(
                "UPDATE candidates SET status=?, geomean_ns=?, result=?, error=?, wall_s=?, profile_summary=? WHERE id=?",
                (status, geomean_ns, json.dumps(result or {}), error, wall_s, json.dumps(profile_summary or {}), cand_id))
            self.db.commit()
            cand = self.get(cand_id)
        self._mirror(cand)
        return cand

    def set_profile_summary(self, cand_id: int, summary: dict):
        with self.lock:
            self.db.execute("UPDATE candidates SET profile_summary=? WHERE id=?", (json.dumps(summary), cand_id))
            self.db.commit()
        self._mirror(self.get(cand_id))

    def set_meta(self, key: str, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
            self.db.commit()

    def write_log(self, cand_id: int, name: str, text: str | bytes, sub: str = "logs") -> Path:
        d = self.cand_dir(cand_id) / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_bytes(text) if isinstance(text, bytes) else p.write_text(text)
        return p

    def write_brief(self, cand_id: int, text: str):
        (self.cand_dir(cand_id)).mkdir(parents=True, exist_ok=True)
        (self.cand_dir(cand_id) / "brief.md").write_text(text)

    def append_journal(self, text: str):
        with open(self.dir / "JOURNAL.md", "a", encoding="utf-8") as f:    # one write: safe for parallel submitters
            f.write(text.rstrip() + "\n\n")

    def add_note(self, kind: str, text: str, cands: list[int] | None = None) -> str:
        """The orchestrator's own understanding (insight, decision, dead-end, ...), appended to NOTES.md with a timestamp."""
        stamp = time.strftime("%Y-%m-%d %H:%M")
        refs = f" ({', '.join('#%d' % c for c in cands)})" if cands else ""
        entry = f"### {stamp} [{kind}]{refs}\n{text.strip()}"
        with open(self.dir / "NOTES.md", "a", encoding="utf-8") as f:
            f.write(entry + "\n\n")
        return entry

    def notes(self, n: int | None = None) -> list[str]:
        p = self.dir / "NOTES.md"
        if not p.exists():
            return []
        parts = ["### " + b.strip() for b in p.read_text().split("### ") if b.strip()]
        return parts[-n:] if n else parts

    def _mirror(self, c: Candidate):
        d = self.cand_dir(c.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(c.__dict__ | {"code": "", "profile": ""}, indent=2, default=str))
        best = self.best()
        if best and best.id == c.id and (d / "kernel").is_dir():
            shutil.rmtree(self.dir / "best", ignore_errors=True)
            shutil.copytree(d / "kernel", self.dir / "best")

    # ---- reads --------------------------------------------------------------------------
    @staticmethod
    def _cand(row) -> Candidate:
        d = {k: row[k] for k in row.keys() if k in _FIELDS}
        d["result"] = json.loads(d.get("result") or "{}")
        d["profile_summary"] = json.loads(d.get("profile_summary") or "{}")
        return Candidate(**d)

    def get(self, cand_id: int) -> Candidate:
        with self.lock:
            row = self.db.execute("SELECT * FROM candidates WHERE id=?", (cand_id,)).fetchone()
        if row is None:
            raise KeyError(f"no candidate #{cand_id}")
        return self._cand(row)

    def all(self) -> list[Candidate]:
        return self._query()

    def _query(self, where="", args=(), order="id", limit=None) -> list[Candidate]:
        sql = f"SELECT * FROM candidates {where} ORDER BY {order}" + (f" LIMIT {int(limit)}" if limit else "")
        with self.lock:
            return [self._cand(r) for r in self.db.execute(sql, args)]

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def recent(self, n: int) -> list[Candidate]:
        return self._query(order="id DESC", limit=n)[::-1]

    def correct(self) -> list[Candidate]:
        """Correct and benchmarked, fastest first."""
        return self._query("WHERE status='pass' AND geomean_ns IS NOT NULL", order="geomean_ns ASC")

    def best(self) -> Candidate | None:
        rows = self.correct()
        return rows[0] if rows else None

    def seed(self) -> Candidate | None:
        rows = self._query("WHERE kind='seed' AND status='pass'", order="id ASC", limit=1)
        return rows[0] if rows else None

    def seed_ns(self) -> float | None:
        s = self.seed()
        if s and s.geomean_ns:
            return s.geomean_ns
        return self.get_meta("seed_geomean_ns")

    def get_meta(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def files(self, cand_id: int) -> dict[str, str]:
        """The kernel files of a candidate (name -> text); legacy rows return their single submission.py."""
        kdir = self.cand_dir(cand_id) / "kernel"
        if kdir.is_dir():
            return {p.relative_to(kdir).as_posix(): p.read_text() for p in sorted(kdir.rglob("*"))
                    if p.is_file() and p.suffix in TEXT_EXT}
        legacy = self.cand_dir(cand_id) / "submission.py"
        if legacy.exists():
            return {"submission.py": legacy.read_text()}
        c = self.get(cand_id)
        return {"submission.py": c.code} if c.code else {}

    def ancestors(self, cand_id: int) -> list[Candidate]:
        """Parent, grandparent, ... (nearest first)."""
        out, seen = [], set()
        cur = self.get(cand_id).parent_id
        while cur and cur not in seen:
            seen.add(cur)
            try:
                c = self.get(cur)
            except KeyError:
                break
            out.append(c)
            cur = c.parent_id
        return out
