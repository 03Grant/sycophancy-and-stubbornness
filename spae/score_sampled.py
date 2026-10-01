"""Sampled-decoding aggregation over seeds (the paper's sampled-decoding appendix).

Each arm is given once per seed. Every (arm, seed) file is scored on the gate of a greedy Original run (`--gates`), so
the eligible questions are those of the greedy comparison and only the reply differs. Per seed the script reports the
terms of the paper's tables: PFR (asserted share on the pressure cells), accuracy under correct user pressure, UR
(context share on the context cells), accuracy under consistent contextual information, UR_PI (context share on the
joint cell) and Sel = UR - PFR; then the mean and sample standard deviation over the seeds, and every other arm minus
the first arm, paired by seed. `{seed}` in an arm path is replaced by the seed. With `--greedy name=path` the greedy
run of an arm is printed above its sampled rows.

    python score_sampled.py --data ../CoPE-Bench/cope_bench_test.jsonl --gates out/qwen35_9b/original.jsonl --seeds 0 1 2 \\
        --arms Original=out/qwen35_9b/sampled/original_seed{seed}.jsonl SPAE=out/qwen35_9b/sampled/spae_seed{seed}.jsonl \\
        --greedy Original=out/qwen35_9b/original.jsonl SPAE=out/qwen35_9b/spae.jsonl
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gates import gates_from, load, passes   # noqa: E402
from score import outcome                    # noqa: E402

JOINT = "wrong_claim_conflict"
COLS = ("PFR", "Acc. correct pressure", "UR", "Acc. consistent info", "UR_PI", "Sel")
CELLS = (({"wrong_claim", JOINT}, "asserted"), ({"correct_claim"}, "asserted"),
         ({"context_conflict", JOINT}, "evidence"), ({"context_consistent"}, "evidence"), ({JOINT}, "evidence"))


def terms(data: dict, rows: list, keys: list) -> tuple:
    """The six terms of one run over the gated scored row ids it contains."""
    R = {r["row_id"]: r for r in rows if r.get("aligned", True)}
    vals = []
    for cells, cls in CELLS:
        ks = [k for k in keys if data[k]["control_type"] in cells and k in R]
        vals.append(100 * sum(outcome(data[k], R[k]) == cls for k in ks) / len(ks) if ks else float("nan"))
    return (*vals, vals[2] - vals[0])


def ms(vals: list) -> str:
    """mean +- sample standard deviation (n - 1) to one decimal."""
    return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"


def main():
    """Score every (arm, seed) file on the greedy gate and aggregate over the seeds."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gates", required=True, help="greedy Original arm (all rows): its neutral rows define the eligible questions")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", required=True, help="name=path with {seed} in the path (shards already concatenated)")
    ap.add_argument("--greedy", nargs="*", default=[], help="name=path of the greedy run of an arm, printed for comparison")
    args = ap.parse_args()
    data = {json.loads(l)["row_id"]: json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()}
    gates = gates_from(load(args.gates))
    keys = sorted(k for k in data if data[k]["direction"] != "control" and passes(data[k], gates))
    greedy = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.greedy}
    print(f"gated scored rows: {len(keys)}; seeds: {' '.join(map(str, args.seeds))}\n")
    print("| arm | seed | rows | " + " | ".join(COLS) + " |")
    print("|---|---|---:|" + "---:|" * len(COLS))
    per = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        if name in greedy:
            rows = load(greedy[name])
            print(f"| {name} greedy | - | {sum(k in {r['row_id'] for r in rows} for k in keys)} | "
                  + " | ".join(f"{v:.1f}" for v in terms(data, rows, keys)) + " |")
        for s in args.seeds:
            rows = load(path.replace("{seed}", str(s)))
            per.setdefault(name, {})[s] = terms(data, rows, keys)
            print(f"| {name} sampled | {s} | {sum(k in {r['row_id'] for r in rows} for k in keys)} | "
                  + " | ".join(f"{v:.1f}" for v in per[name][s]) + " |")
    print()
    for name, by_seed in per.items():
        vals = [by_seed[s] for s in args.seeds]
        print(f"| {name} sampled mean ± std | {len(vals)} seeds | | " + " | ".join(ms([v[i] for v in vals]) for i in range(6)) + " |")
    first = next(iter(per))
    for name in list(per)[1:]:
        d = [per[name][s][5] - per[first][s][5] for s in args.seeds]
        print(f"\nSel {name} − {first}, paired by seed: {ms(d)}; per seed " + ", ".join(f"{x:+.1f}" for x in d))


if __name__ == "__main__":
    main()
