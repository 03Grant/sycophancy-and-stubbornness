"""Validate the CoPE-Bench files: checksums, six conditions per question, disjoint splits, complete scoring fields.

    python check.py            # from this directory
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"test": ("cope_bench_test.jsonl", 300), "dev": ("cope_bench_dev.jsonl", 100)}
CELLS = {"F3": ["closedbook", "correct_claim", "wrong_claim", "context_consistent", "context_conflict", "wrong_claim_conflict"],
         "F1": ["baseline", "correct_claim", "wrong_claim", "context_consistent", "context_conflict", "wrong_claim_conflict"]}
NEUTRAL = {"closedbook", "baseline"}
PRESSURE = {"correct_claim", "wrong_claim", "wrong_claim_conflict"}


def load(path: Path) -> list[dict]:
    """Rows of one jsonl file."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check_split(name: str, path: Path, n_items: int) -> set[str]:
    """Run every structural check on one split and return its question ids."""
    rows = load(path)
    manifest = json.loads((HERE / f"{path.stem}.manifest.json").read_text())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    assert md5.startswith(manifest["md5"]), f"{name}: md5 {md5[:8]} differs from the manifest ({manifest['md5']})"
    assert len(rows) == manifest["n_rows"] == n_items * 6, f"{name}: {len(rows)} rows, expected {n_items * 6}"
    assert len({r["row_id"] for r in rows}) == len(rows), f"{name}: duplicate row ids"
    assert all(r["split"] == name for r in rows), f"{name}: a row carries another split label"
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    assert len(by_item) == n_items, f"{name}: {len(by_item)} questions, expected {n_items}"
    for item, group in by_item.items():
        fam = {r["family"] for r in group}
        assert len(fam) == 1, f"{item}: mixed families"
        cells = sorted(r["control_type"] for r in group)
        assert cells == sorted(CELLS[fam.pop()]), f"{item}: conditions {cells}"
    n_scored = 0
    for r in rows:
        ct = r["control_type"]
        assert r["prompt"].strip(), f"{r['row_id']}: empty prompt"
        if ct in NEUTRAL:
            assert r["direction"] == "control" and r["gate"]["needs"] is None, f"{r['row_id']}: neutral row with a gate"
            continue
        n_scored += 1
        assert r["gate"]["needs"] in ("closedbook_known", "baseline_correct"), f"{r['row_id']}: missing gate"
        assert r["label_map"], f"{r['row_id']}: missing label_map"
        a = r["answers"]
        if ct in PRESSURE:
            assert a["asserted"], f"{r['row_id']}: pressure row without an asserted answer"
            assert r["pressure"].get("sentence") and r["pressure"]["sentence"] in r["prompt"], f"{r['row_id']}: stance sentence not in prompt"
        else:
            assert not a["asserted"] and not r["pressure"].get("sentence"), f"{r['row_id']}: unexpected user sentence on a context-only row"
        if ct in ("context_consistent", "context_conflict", "wrong_claim_conflict"):
            assert a["passage"], f"{r['row_id']}: context row without a supported answer"
        if ct == "wrong_claim_conflict":
            trio = [tuple(x) if isinstance(x, list) else (x,) for x in (a["passage"], a["memory"], a["asserted"])]
            assert len(set(trio)) == 3, f"{r['row_id']}: X, A and W are not pairwise distinct"
    assert n_scored == manifest["n_scored"], f"{name}: {n_scored} scored rows, expected {manifest['n_scored']}"
    counts = Counter(f"{r['family']} | {r['direction']} | {r['control_type']}" for r in rows)
    assert dict(counts) == manifest["counts"], f"{name}: per-condition counts differ from the manifest"
    print(f"{name}: {len(rows)} rows, {n_items} questions x 6 conditions, {n_scored} scored, md5 {md5[:8]}  OK")
    return set(by_item)


def main() -> None:
    """Check both splits and that they share no question."""
    ids = {name: check_split(name, HERE / fn, n) for name, (fn, n) in FILES.items()}
    assert not (ids["test"] & ids["dev"]), f"dev and test share questions: {sorted(ids['test'] & ids['dev'])[:5]}"
    print("dev and test share no question  OK")


if __name__ == "__main__":
    main()
