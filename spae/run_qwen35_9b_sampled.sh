#!/bin/bash
# Sampled-decoding rows of the paper's appendix for Qwen3.5-9B: the Original and SPAE arms rerun with the reply sampled
# at the backbone's shipped values (temperature 0.7, top-p 0.8, top-k 20) for three seeds. The auxiliary call and the
# eligibility gate are those of the greedy run, so run_qwen35_9b.sh must have produced $OUT/call1.jsonl and
# $OUT/original.jsonl first; only the reply changes. Each seed is one deterministic run (per-row seed = seed * 1000003
# + row number); every stage appends and skips rows already on disk, and --shard i/n with a shard-specific --out splits a
# stage as in run_qwen35_9b.sh.
# Usage: MODEL=<hf id or local path> ./run_qwen35_9b_sampled.sh     (defaults: MODEL=Qwen/Qwen3.5-9B, OUT=out/qwen35_9b, SEEDS="0 1 2")
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M=${MODEL:-Qwen/Qwen3.5-9B}
D=${DATA:-../CoPE-Bench/cope_bench_test.jsonl}
OUT=${OUT:-out/qwen35_9b}
SEEDS=${SEEDS:-0 1 2}
mkdir -p "$OUT/sampled"
SCORED=wrong_claim,correct_claim,context_conflict,context_consistent,wrong_claim_conflict
SPAE="--kernel share --alpha-memory 0.3 --lin-alpha 0.3 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.5 \
      --receivers-memory all --receivers-context after --localiser quote --full-letters --no-think"
SAMPLE="--sample-answer --temperature 0.7 --top-p 0.8 --top-k 20"
for s in $SEEDS; do
    [ -s "$OUT/call1.jsonl" ] && [ -s "$OUT/original.jsonl" ] || { echo "run run_qwen35_9b.sh first ($OUT/call1.jsonl, $OUT/original.jsonl)"; exit 1; }
    # 1. Original arm, reply sampled, on the 1,500 scored rows (the gate comes from the greedy Original at scoring time)
    $PY spae_two_call.py call2 --model "$M" --conditions "$D" --call1 "$OUT/call1.jsonl" --prompt-file prompts/call1.txt \
        --arm baseline --cells $SCORED $SPAE $SAMPLE --seed $s --out "$OUT/sampled/original_seed$s.jsonl"
    # 2. SPAE arm, reply sampled
    $PY spae_two_call.py call2 --model "$M" --conditions "$D" --call1 "$OUT/call1.jsonl" --prompt-file prompts/call1.txt \
        --arm dual --cells $SCORED $SPAE $SAMPLE --seed $s --out "$OUT/sampled/spae_seed$s.jsonl"
done

# 3. per seed, mean and sample standard deviation over the seeds, SPAE minus Original paired by seed
$PY score_sampled.py --data "$D" --gates "$OUT/original.jsonl" --seeds $SEEDS \
    --arms Original="$OUT/sampled/original_seed{seed}.jsonl" SPAE="$OUT/sampled/spae_seed{seed}.jsonl" \
    --greedy Original="$OUT/original.jsonl" SPAE="$OUT/spae.jsonl" | tee "$OUT/sampled/score.md"
