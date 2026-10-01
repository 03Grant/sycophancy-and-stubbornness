# CoPE-Bench and SPAE

Anonymous artifact accompanying the submission "Suppressing Pressure, Amplifying Evidence" (SPAE) on CoPE-Bench.

```
CoPE-Bench/     the benchmark: test split (300 questions, 1,800 rows) and development split (100 questions, 600 rows),
                manifests with checksums, a validator (check.py) and a data card (README.md)
spae/           the method: auxiliary call, answer call with the attention operator, scorer, the auxiliary-call
                wording, and the script that reproduces the Qwen3.5-9B rows of the paper (README.md)
```

Quick start:

```
python CoPE-Bench/check.py                       # verifies the data files
pip install -r spae/requirements.txt
cd spae && MODEL=Qwen/Qwen3.5-9B ./run_qwen35_9b.sh   # auxiliary call, Original arm, SPAE arm, score
```

Each data row is a complete prompt with its condition, candidate answers and eligibility gate; the method reads only
the prompt. See the two README files for the condition definitions, the row schema, the flags that correspond to the
paper's parameters, and the sampled-decoding repeat.
