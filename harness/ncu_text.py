"""
Nsight Compute's text report (`ncu --import X.ncu-rep --page details`), made readable for a terminal:
split per kernel, filter by kernel / section, and pull out NVIDIA's own analysis messages (OPT / WRN / INF rules),
which are the part an agent most needs ("Tensor is the highest-utilized pipeline", "uncoalesced global accesses", ...).
"""
import re

_HEADER = re.compile(r"^  \S.*\(\d+, \d+, \d+\)x\(\d+, \d+, \d+\), Context")
_SECTION = re.compile(r"^    Section: (.+?)\s*$")
_MSG = re.compile(r"^    (OPT|WRN|INF)\s+(.*)$")


def split_kernels(text: str) -> list[dict]:
    """-> [{"header": str, "name": str, "lines": [...]}] in report order."""
    out: list[dict] = []
    for line in text.splitlines():
        if _HEADER.match(line):
            header = line.strip()
            out.append({"header": header, "name": re.split(r"\s\(\d+, \d+, \d+\)x", header)[0], "lines": []})
        elif out:
            out[-1]["lines"].append(line)
    return out


def advice(block: dict) -> list[dict]:
    """The rule messages of one kernel: [{"kind": "OPT"|"WRN"|"INF", "section": str, "text": str}] (text on one line)."""
    msgs, section, cur = [], "", None
    for line in block["lines"]:
        m = _SECTION.match(line)
        if m:
            section, cur = m.group(1), None
            continue
        m = _MSG.match(line)
        if m:
            cur = {"kind": m.group(1), "section": section, "text": m.group(2).strip()}
            msgs.append(cur)
        elif cur is not None and line.startswith("          ") and line.strip():
            cur["text"] += " " + line.strip()
        else:
            cur = None
    return msgs


def _sections(block: dict) -> list[tuple[str, list[str]]]:
    secs: list[tuple[str, list[str]]] = [("", [])]
    for line in block["lines"]:
        m = _SECTION.match(line)
        if m:
            secs.append((m.group(1), []))
        secs[-1][1].append(line)
    return secs


def render(text: str, *, kernel: str | None = None, section: str | None = None, advice_only: bool = False,
           max_lines: int = 250) -> str:
    """Filtered report text. kernel/section are case-insensitive regexes; advice_only prints just the rule messages."""
    blocks = split_kernels(text)
    if kernel:
        pat = re.compile(kernel, re.I)
        blocks = [b for b in blocks if pat.search(b["header"])]
    if not blocks:
        return "(no kernel matched; kernels in this report: " + ", ".join(b["name"][:40] for b in split_kernels(text)[:8]) + " ...)"
    out: list[str] = []
    seen: dict[str, int] = {}
    for b in blocks:                      # a kernel launched many times is shown once; the count says how many
        seen[b["name"]] = seen.get(b["name"], 0) + 1
    shown: set[str] = set()
    for b in blocks:
        if advice_only:
            if b["name"] in shown:
                continue
            shown.add(b["name"])
        out.append(b["header"] + (f"   [+{seen[b['name']] - 1} more launches of this kernel]" if advice_only and seen[b["name"]] > 1 else ""))
        if advice_only:
            msgs = advice(b)
            if section:
                msgs = [m for m in msgs if re.search(section, m["section"], re.I)]
            out += [f"  [{m['kind']}] ({m['section']}) {m['text']}" for m in msgs] or ["  (no analysis messages)"]
        else:
            for name, lines in _sections(b):
                if section and not re.search(section, name, re.I):
                    continue
                out += [l.rstrip() for l in lines if l.strip()]
        out.append("")
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... truncated at {max_lines} lines (narrow with --kernel / --section, or raise --max-lines)"]
    return "\n".join(out)


def top_advice(text: str, kernel_names: list[str], limit: int = 6, kinds=("OPT", "WRN")) -> dict | None:
    """NVIDIA's OPT/WRN messages for the first of `kernel_names` (ordered by time share) found in the detailed report."""
    blocks = split_kernels(text)
    for want in kernel_names:
        key = re.sub(r"^void\s+", "", want).split("<")[0].split("(")[0].strip()
        for b in blocks:
            if key and key in b["header"]:
                msgs = [m for m in advice(b) if m["kind"] in kinds][:limit]
                if msgs:
                    return {"kernel": key, "messages": [{**m, "text": m["text"][:400]} for m in msgs]}
    return None
