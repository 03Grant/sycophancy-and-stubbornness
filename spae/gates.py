"""Eligibility gates and the free-text letter reader used by the scorer.

`gates_from` reads the neutral rows of the Original arm (baseline / closedbook cells): a question is eligible when the
backbone answered it correctly there. `passes` applies the row's `gate.needs` to those facts. `stated_letter` is the
fallback letter reader for CoT replies that carry no probe letter.
"""
import json
import re
from pathlib import Path


CHOICE_RE = re.compile(r"(?:answer|choice|option)\s+(?:is|would be|should be|:)\s*[:\-]?\s*\(?\**([ABCD])\**\)?(?![A-Za-z])", re.IGNORECASE)

LAST_LINE_LETTER = re.compile(r"(?<![A-Za-z])\(?\**([ABCD])\**\)?(?![A-Za-z])")

def stated_letter(r: dict) -> str | None:
    """The letter a CoT reply visibly committed to: the run's free-form field, else 'answer/choice/option is X', else a bare first line, else the single standalone letter of the last line."""
    for key in ("freeform_letter", "freeform_answer"):
        if r.get(key):
            return r[key]
    text = str(r.get("reply") or "").strip()
    found = CHOICE_RE.findall(text)
    if found:
        return found[-1].upper()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    first = lines[0].strip("*()[]\"'.: ")
    if len(first) == 1 and first.upper() in "ABCD":
        return first.upper()
    letters = set(LAST_LINE_LETTER.findall(lines[-1]))
    return letters.pop() if len(letters) == 1 else None

def load(path: str) -> list[dict]:
    """Rows of one arm file (shards concatenated by the caller); a file that stores the row's direction as `row_direction` is normalised."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        if "row_direction" in r:
            r["steer"], r["direction"] = r.get("direction"), r["row_direction"]
    return rows

def gates_from(rows: list[dict]) -> dict:
    """Per-item gate facts read off the control rows: baseline correct, closed-book known, memory present."""
    g = {}
    for r in rows:
        if r.get("direction") != "control":
            continue
        item = r["item_id"]
        ct = r.get("control_type")
        if ct == "baseline":
            g.setdefault(item, {})["baseline_correct"] = r["label"] == "memory"       # baseline rows: memory = gold
        elif ct == "closedbook":
            g.setdefault(item, {})["closedbook_known"] = r["label"] == "memory"
        elif ct == "memory":
            g.setdefault(item, {})["memory_present"] = r["label"] == "memory"
    return g

def passes(r: dict, g: dict) -> bool | None:
    """Whether a scored row counts for this model; None when the gate fact is missing."""
    need = (r.get("gate") or {}).get("needs")
    facts = g.get(r["item_id"], {})
    if need is None:
        return True
    if need == "baseline_correct":
        return facts.get("baseline_correct")
    if need == "baseline_wrong":
        return None if facts.get("baseline_correct") is None else not facts["baseline_correct"]
    if need == "closedbook_known":
        return facts.get("closedbook_known")
    if need == "closedbook_wrong":
        return None if facts.get("closedbook_known") is None else not facts["closedbook_known"]
    if need == "memory_present":
        return facts.get("memory_present")
    return None
