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
| `run_qwen35_9b.sh` | the Qwen3.5-9B pipeline on the test split: auxiliary call, Original, SPAE, score |
| `run_qwen35_9b_sampled.sh` | the sampled-decoding repeat of Original and SPAE (three seeds), after `run_qwen35_9b.sh` |
| `run.py` | one entry point for every arm: `--methods all` or a comma list of arm names, greedy or `--seeds` (see below) |
| `baselines/` | the six baselines: `run_baseline.py` (S2A, CAD, AdaCAD), `caa.py`, `juice.py` + `juice_core.py`, `autopasta.py` + `autopasta_core.py`, their shared helpers (`common.py`), prompts (`prompts.json`) and the frozen Qwen3.5-9B artifacts (`artifacts/qwen35_9b/`) |
| `setup.sh`, `requirements.txt` | one-command environment and the pinned dependencies |

## Environment

Python 3.10, PyTorch 2.11 (CUDA 13), transformers 5.12, numpy 2.2; Qwen3.5-9B in bf16 needs about 20 GB of GPU
memory; the paper's runs used NVIDIA H200 GPUs. The kernel replaces `eager_attention_forward` of the model's attention module, so the runner loads the
model with `attn_implementation="eager"`; on Qwen3.5 the transformers implementation of the Gated DeltaNet layers
is wrapped (the optional fused kernels are not required). `--no-think` switches the Qwen3.5 thinking mode off in the
chat template; every backbone is run as an instruct model with the whole prompt as one user turn.

```
./setup.sh                      # creates .venv with the pinned versions (PyTorch 2.11 for CUDA 13); then: source .venv/bin/activate
CUDA=cu126 ./setup.sh           # another PyTorch build (cu126, cu128, cu130 or cpu)
```

`setup.sh` installs PyTorch from the PyTorch index for the chosen CUDA build and then `requirements.txt` (transformers,
numpy, and pastalib with its `datasets` import for AutoPASTA). `pip install -r requirements.txt` alone works too and takes
whichever PyTorch build the package index serves.

## Running the comparison

```
python run.py --model Qwen/Qwen3.5-9B --methods all                     # Original, the six baselines and SPAE, then score.py
python run.py --model Qwen/Qwen3.5-9B --methods original,spae,autopasta  # any subset, by arm name
python run.py --model Qwen/Qwen3.5-9B --methods all --limit 10           # smoke test on ten rows per stage
python run.py --model Qwen/Qwen3.5-9B --methods all --seeds 0 1 2        # the sampled-decoding repeat of every arm
```

Arms: `original`, `spae`, `random`, `oracle` (the two-call runner) and `s2a`, `cad`, `adacad`, `caa`, `juice`, `autopasta`
(the baselines); `all` is the main table. The Original arm is always run first because its neutral rows define the
eligibility gate; the auxiliary call runs once and is shared by `spae`, `random` and `oracle`. Outputs go to
`out/<label>/<arm>.jsonl` and `out/<label>/score.md`; with `--seeds`, to `out/<label>/sampled/<arm>_seed<s>.jsonl` and
`out/<label>/sampled/score.md`. The SPAE parameters, the thinking switch and the sampling values of the five backbones are
read from the settings table in `run.py` (`--spae-flags` overrides them); the fitted baselines read the artifacts of the
backbone from `baselines/artifacts/<label>/` (`--artifacts` overrides, `--variant` picks fit-mix, fit-syco or fit-stub).
Every stage appends to its file and skips rows already on disk, so an interrupted run continues where it stopped.

## Reproducing the Qwen3.5-9B rows by hand

```
MODEL=Qwen/Qwen3.5-9B ./run_qwen35_9b.sh          # Original and SPAE only; or the three stages below by hand
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

## Baselines

The baseline runners take the same CoPE-Bench rows and write the same record fields as the two-call runner (`reply`,
`answer`, `aligned`, plus their own diagnostics), so `score.py` scores every arm alike; on chain-of-thought rows `answer`
is the probe letter read at the fixed suffix and `stated_answer` the letter the reply states in prose. No answer or
condition label reaches any method; the context-free input of CAD and AdaCAD and the passage of the AutoPASTA extraction
prompt are derived from the neutral row of the same question.

| arm | runner | what it does |
|---|---|---|
| S2A | `run_baseline.py --method s2a` | rewrites the request with the published prompt (`prompts.json`, 1,024-token budget) into an unbiased context and a question, answers from those two parts followed by `Answer in an unbiased way.` and the row's own format instruction; a rewrite without both labelled parts falls back to the original request (`metadata.rewrite_parsed`); `--rewrites` reuses the greedy rewrites of an earlier run |
| CAD | `run_baseline.py --method cad` | contrastive decoding against the context-free input with the published weight of 1, at every generated token and at the letter probe |
| AdaCAD | `run_baseline.py --method adacad` | the same with the token-wise Jensen-Shannon divergence (natural log) as the weight (`metadata.max_alpha`) |
| CAA | `caa.py` | a mean-difference vector per layer from the official 1,000 sycophancy pairs (`--pairs`, sycophancy minus truthful, raw scale), added at one layer with one multiplier from the final prompt token onwards; `--stage fit` scans every layer (`--layers` narrows it, e.g. `14-18`) and multipliers +-{0.5, 1, 1.5, 2} on the development split, `--stage test` runs the selected pair |
| JuICE | `juice.py` | scales the o_proj contribution of the selected heads by 1 - s (suppression heads) and 1 + e (enhancement heads) with a capture pass and an inject pass per token; `profile` scales one head at a time by 1 + alpha over the alpha grid on the four profiling rows and records the change of the target token's probability (prefill only), `heads` ranks the profiled heads by their gain sum over the positive and the negative alphas (up to ten each), `fit` runs one of the 36 (s, e) pairs on the held-out development rows, `select` picks the pair, `test` runs it |
| AutoPASTA | `autopasta.py` | `extract` asks the model for the key sentence of the request's added paragraphs with the published extraction prompt, `map` maps it with all-MiniLM-L6-v2 to the closest sentence and its token span, the search stages (`coarse`, `rank`, `fine`, `candidates`, `fit`, `select`) pick the heads on the development split with a prefill-only read-out, `test` generates under pastalib's attention edit (alpha 0.01, `scale_position` exclude) on the selected heads |

The search stages of the three fitted baselines need an Original run of the development split (`--gates`), whose
correctly answered neutral rows define the fitting gate; the objective is the share of fitting rows whose reply names the
row's target answer (`common.hit`). `baselines/artifacts/qwen35_9b/` holds the frozen outcome of the paper's fits for
Qwen3.5-9B: the CAA vectors and selection, the JuICE head lists and selected pairs, the AutoPASTA selections with their
layer ranking and candidate configurations, the mapped spans of every test and development row (`spans_test.jsonl`,
`spans_dev.jsonl`), so the test stages reproduce the paper's inputs without rerunning the extraction, and the Original run
of the development split that gated the fits (`dev_original.jsonl`). The per-configuration development outputs the
selections were read from are not included; `fit` regenerates them.

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
