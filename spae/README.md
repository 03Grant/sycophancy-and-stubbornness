# SPAE: Suppressing Pressure, Amplifying Evidence

Reference implementation of SPAE on CoPE-Bench, in the configuration used for **Qwen3.5-9B** in the paper. SPAE uses
two calls to the same frozen model:

1. **Auxiliary call** (`spae_call1.py`): the model receives the complete request and copies two lines out of it
   verbatim, `User pressure:` and `Information:`, each possibly `NONE`. Each
   copied line is located in the request by exact string matching and mapped to token positions through the
   tokenizer offsets; a line that is `NONE` or cannot be found leaves its side inactive.
2. **Answer call** (`spae_two_call.py call2`): the unmodified request is decoded greedily under a patched attention.
   On every softmax layer the pressure keys $M_P$ receive the bias $\log\alpha$ on every query row, and the
   information keys $M_I$ receive, per head and query row, the bias that raises their attention share $s$ to
   $\min(\tau, s+\delta)$ (clipped to 6 nats) on the rows after the span and at the generated positions. On the
   linear-attention layers of the hybrid Qwen stacks the delta-rule write gate is multiplied by $\alpha$ on $M_P$
   and moved towards one by the fraction $\rho$ on $M_I$. The two sets are made disjoint with pressure taking
   priority. Nothing that affects the generation reads a condition label or a reference answer; the row's labels are only copied into the output record for scoring.

The method code was frozen as run; the modules below are the parts of the research code that this pipeline executes.

| file | role |
|---|---|
| `spae_call1.py` | auxiliary call: prompt scaffold, greedy generation, parsing of the two lines |
| `spae_two_call.py` | the runner: `call1` (the auxiliary call with attention diagnostics, same copied lines) and `call2` (Original / SPAE / ablation arms, read-out, per-row record; `--sample-answer` samples the reply) |
| `attention_kernel.py` | `ShareBias`: the patched eager-attention forward and the delta-rule write-gate wrapper |
| `scaffold.py` | chat-template scaffolds, verbatim localiser (copied text to token positions), alignment between the two layouts |
| `generation.py` | generation caps (400 tokens CoT, 32 short answer), receiver-row rule, per-row scoring hook |
| `readout.py` | fixed-position letter read-out for chain-of-thought replies (cut before the stated conclusion, compare option-letter logits) |
| `answer_match.py` | normalisation and phrase matching for short answers |
| `score.py`, `gates.py` | per-condition table, PFR / UR / Sel, paired bootstrap; eligibility gates from the Original arm's neutral rows |
| `score_sampled.py` | sampled decoding: the terms per seed, mean and sample standard deviation over the seeds, SPAE minus Original paired by seed |
| `prompts/call1.txt` | the auxiliary-call wording (with its invented examples), the same for every backbone |
| `run_qwen35_9b.sh` | the full Qwen3.5-9B pipeline on the test split |
| `run_qwen35_9b_sampled.sh` | the sampled-decoding repeat of the Qwen3.5-9B comparison (three seeds), after `run_qwen35_9b.sh` |
| `requirements.txt` | Python dependencies |

## Environment

Python 3.10, PyTorch 2.11 (CUDA 13), transformers 5.12, numpy 2.2; Qwen3.5-9B in bf16 needs about 20 GB of GPU
memory; the paper's runs used NVIDIA H200 GPUs. The kernel replaces `eager_attention_forward` of the model's attention module, so the runner loads the
model with `attn_implementation="eager"`; on Qwen3.5 the transformers implementation of the Gated DeltaNet layers
is wrapped (the optional fused kernels are not required). `--no-think` switches the Qwen3.5 thinking mode off in the
chat template; every backbone is run as an instruct model with the whole prompt as one user turn.

```
pip install -r requirements.txt
```

## Reproducing the Qwen3.5-9B rows

```
MODEL=Qwen/Qwen3.5-9B ./run_qwen35_9b.sh          # or the three stages below by hand
```

```
D=../CoPE-Bench/cope_bench_test.jsonl
SPAE="--kernel share --alpha-memory 0.3 --lin-alpha 0.3 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.5 \
      --receivers-memory all --receivers-context after --localiser quote --full-letters --no-think"

O=out/qwen35_9b
python spae_call1.py --model $MODEL --conditions $D --prompt-file prompts/call1.txt --no-think --out $O/call1.jsonl
python spae_two_call.py call2 --model $MODEL --conditions $D --call1 $O/call1.jsonl --prompt-file prompts/call1.txt \
       --arm baseline $SPAE --out $O/original.jsonl
python spae_two_call.py call2 --model $MODEL --conditions $D --call1 $O/call1.jsonl --prompt-file prompts/call1.txt \
       --arm dual --cells wrong_claim,correct_claim,context_conflict,context_consistent,wrong_claim_conflict $SPAE \
       --out $O/spae.jsonl
python score.py --data $D --arms Original=$O/original.jsonl SPAE=$O/spae.jsonl
```

Every stage appends to its output file and skips the rows already on disk, so it can be interrupted and restarted;
`--shard i/n` with a shard-specific `--out` splits a stage over `n` processes (concatenate the shard files afterwards)
and `--limit N` runs a smoke test. The Original arm is run on all 1,800 rows because its 300 neutral rows define the eligible questions;
the SPAE arm is run on the 1,500 scored rows.

### Flags and the paper's parameters

| paper | flag | Qwen3.5-9B |
|---|---|---|
| $\alpha$, multiplier on the pressure keys (softmax layers) | `--alpha-memory` | 0.3 |
| $\alpha$ on the linear-attention write gate (hybrid stacks) | `--lin-alpha` | 0.3 |
| $\delta$, share increment for the information keys | `--max-transfer` | 0.5 |
| $\tau$, share ceiling for the information keys | `--target-share` | 0.7 |
| $\rho$, write-gate push for the information keys (hybrid stacks) | `--lin-rho` | 0.5 |
| suppression on every query row; amplification after the span and at generated positions | `--receivers-memory all --receivers-context after` | as listed |
| amplification layers | `--layers lo:hi[:step]` (unset = every layer) | every layer |
| localisation by verbatim matching of the copied lines | `--localiser quote` | as listed |
| multiple-choice letter read over the question's full option set | `--full-letters` | as listed |

The runner's own defaults are not the paper setting; the flag string above supplies every parameter. The other backbones
use the same executable and the same auxiliary-call wording, with the parameters of the paper's settings table.

### Sampled decoding

The paper's sampled-decoding appendix repeats the comparison with the reply sampled at the settings each backbone
ships with, for Qwen3.5-9B temperature 0.7, top-p 0.8 and top-k 20, over the seeds 0, 1 and 2; the auxiliary call
and the eligibility gate are those of the greedy run, so only the reply changes.

```
MODEL=Qwen/Qwen3.5-9B ./run_qwen35_9b_sampled.sh      # after run_qwen35_9b.sh; outputs in out/qwen35_9b/sampled/
```

`--sample-answer` switches the reply decode of `call2` to sampling with `--temperature`, `--top-p` and `--top-k`
(unset: the model's generation config) and `--seed`; the generator is re-seeded for every row from the seed and the
row number, so a sampled run is deterministic and can be restarted or sharded like a greedy one. The script runs the
Original and SPAE arms on the 1,500 scored rows for every seed and calls `score_sampled.py`, which gates every
file on the greedy Original run (`--gates`), prints the terms of the paper's tables per seed with the greedy values
above them, then their mean and sample standard deviation over the seeds, and Sel of SPAE minus Original paired by
seed. Sel is computed per seed before averaging.

Arms of `call2` (`--arm`): `baseline` (Original: nothing armed), `dual` (SPAE), `suppress` / `amplify` (one side
only), `random` (the paper's Random control: the same numbers of tokens drawn uniformly at random from the request,
the two sets kept disjoint; `--random-window` places each set as one contiguous window instead). `--oracle value
--oracle-stance` replaces the copied lines by the row's own answer value and stance sentence (the paper's Oracle
control). The remaining flags are development ablations that the paper setting does not use.

### Output records

The auxiliary-call file has one row per request: `stance`, `evidence` (the copied `User pressure:` and `Information:` lines, `""` for `NONE`), `raw`
(the generation), `stance_in_prompt` / `evidence_in_prompt` (verbatim presence after normalisation) and the
diagnostic `route`. The answer-call file has one row per request the stage was run on: `reply`, `answer` (the read-out letter or
the first line), `label` (`context` / `memory` / `both` / `neither` against the row's candidates), `mask_m` /
`mask_e` (the armed token positions) with `mask_*_source` (`quote`, `abstain` when the copy was not found, `none`
when the line was `NONE`), `letter_logits`, `n_gen_tokens`, `hit_cap`, and the attention-share statistics
`share_pre` / `share_post` / `m_share_pre`; rows of a sampled run also carry `sampled_answer`, `seed`, `temperature`,
`top_p` and `top_k`. `score.py` reads `row_id`, `reply`, `answer`, `label` and `aligned` from the scored rows, and `item_id`,
`direction`, `control_type` and `label` from the neutral rows of the first arm to build the gate.

### Scoring output

`score.py` prints, per task family and condition, the shares of replies that name the context answer, the neutral
answer, the user answer or none, then the headline (`WrongFlip` = PFR, `Update` = UR, `Selectivity` = Sel) with a
paired bootstrap against the first arm. Eligible questions are those whose neutral rows the Original arm answers
correctly, so the eligible set is defined by the Original run passed as the first arm.
