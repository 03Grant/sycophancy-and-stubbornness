#!/bin/bash
# Qwen3.5-9B rows of the paper on the CoPE-Bench test split: auxiliary call, Original arm, SPAE arm, score.
# Usage: MODEL=<hf id or local path> ./run_qwen35_9b.sh     (defaults: MODEL=Qwen/Qwen3.5-9B, OUT=out/qwen35_9b)
# Every stage appends to its output file and skips rows already on disk, so a stage can be restarted; add
# --shard i/n and a shard-specific --out to a stage to split its rows over n processes (one GPU each) and concatenate
# the shard files afterwards.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M=${MODEL:-Qwen/Qwen3.5-9B}
D=${DATA:-../CoPE-Bench/cope_bench_test.jsonl}
OUT=${OUT:-out/qwen35_9b}
mkdir -p "$OUT"
SCORED=wrong_claim,correct_claim,context_conflict,context_consistent,wrong_claim_conflict
# Paper settings for Qwen3.5-9B: alpha 0.3 (softmax and linear side), delta 0.5, tau 0.7, rho 0.5, every layer.
SPAE="--kernel share --alpha-memory 0.3 --lin-alpha 0.3 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.5 \
      --receivers-memory all --receivers-context after --localiser quote --full-letters --no-think"

# 1. auxiliary call on all 1,800 rows (the neutral rows are included; call 2 ignores what it returns for them)
$PY spae_call1.py --model "$M" --conditions "$D" --prompt-file prompts/call1.txt --no-think --out "$OUT/call1.jsonl"

# 2. Original arm on all rows (its neutral rows define the eligibility gate), SPAE arm on the 1,500 scored rows
$PY spae_two_call.py call2 --model "$M" --conditions "$D" --call1 "$OUT/call1.jsonl" --prompt-file prompts/call1.txt \
    --arm baseline $SPAE --out "$OUT/original.jsonl"
$PY spae_two_call.py call2 --model "$M" --conditions "$D" --call1 "$OUT/call1.jsonl" --prompt-file prompts/call1.txt \
    --arm dual --cells $SCORED $SPAE --out "$OUT/spae.jsonl"

# 3. per-condition table, PFR / UR / Sel (printed as WrongFlip / Update / Selectivity) and paired bootstrap vs Original
$PY score.py --data "$D" --arms Original="$OUT/original.jsonl" SPAE="$OUT/spae.jsonl" | tee "$OUT/score.md"
