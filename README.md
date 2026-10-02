# CoPE-Bench and SPAE

Anonymous artifact accompanying the submission "Suppressing Pressure, Amplifying Evidence" (SPAE) on CoPE-Bench.

```
CoPE-Bench/     the benchmark: test split (300 questions, 1,800 rows) and development split (100 questions, 600 rows),
                manifests with checksums, a validator (check.py) and a data card (README.md)
spae/           the method: auxiliary call, answer call with the attention operator, scorer, the auxiliary-call
                wording, the six baselines with their development-split fit (fit.py), and one entry point that runs
                any arm by name or all of them (run.py, README.md); code only, no run outputs
```

Quick start:

```
python CoPE-Bench/check.py                                   # verifies the data files
cd spae && ./setup.sh && source .venv/bin/activate           # one-command environment (PyTorch 2.11 for CUDA 13)
python fit.py --model Qwen/Qwen3.5-9B                        # fits CAA, JuICE and AutoPASTA on the development split
python run.py --model Qwen/Qwen3.5-9B --methods all          # Original, the six baselines and SPAE, then the scores
python run.py --model Qwen/Qwen3.5-9B --methods spae,autopasta --limit 10   # a subset, by arm name; smoke test
python run.py --model Qwen/Qwen3.8-27B --methods all --device-map auto      # a backbone spread over several GPUs
```

The five backbones of the paper (Qwen3.5-4B, Qwen3.5-9B, Qwen2.5-14B-Instruct, gemma-4-26B-A4B-it, Qwen3.8-27B) are
run with the same two commands; their SPAE parameters and decoding settings are read from the table in `spae/run.py`.
Each data row is a complete prompt with its condition, candidate answers and eligibility gate; the method reads only
the prompt. See the two README files for the condition definitions, the row schema, the flags that correspond to the
paper's parameters, and the sampled-decoding repeat.
