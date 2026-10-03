# CoPE-Bench and SPAE

Code and data for the paper "Suppressing Pressure, Amplifying Evidence: Self-Guided Attention Steering to Mitigate Sycophancy and Stubbornness".

> **Data sources.** CoPE-Bench is derived from two public benchmarks: **NQ-Swap** (Longpre et al., 2021, built on
> Natural Questions) and **TruthfulQA** MC1 (Lin et al., 2022). We redistribute modified questions, passages and options
> from both under their original licenses. See [Data sources and licenses](#data-sources-and-licenses) and cite both
> datasets when using CoPE-Bench.

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

## Data sources and licenses

CoPE-Bench contains no new questions; every row is built from one of two public datasets. The data files keep the
license of their source, the code of this repository is released under the MIT License (`LICENSE`), and the full
notices are in `THIRD_PARTY_NOTICES.md`.

| source | what we use | what we change | license |
|---|---|---|---|
| [NQ-Swap](https://github.com/apple/ml-knowledge-conflicts) (Longpre et al., 2021), built on [Natural Questions](https://ai.google.com/research/NaturalQuestions) | 200 questions with their original and entity-substituted passages | screening, a third answer, 24 stance templates, prompt rendering | questions and passages CC BY-SA 3.0 (Natural Questions); substitution framework Apple sample code license |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) MC1 (Lin et al., 2022) | 200 questions with their full option sets | option screening, one option rendered as a note, 24 stance templates | Apache License 2.0 |

The baselines download two further resources on first use and this repository does not redistribute them: the CAA
sycophancy pairs ([nrimsky/CAA](https://github.com/nrimsky/CAA), MIT; a mixture of Anthropic's model-written
evaluations, CC BY 4.0) and the `sentence-transformers/all-MiniLM-L6-v2` encoder (Apache 2.0). AutoPASTA runs on the
official `pastalib` package (MIT).

If you use CoPE-Bench, please cite NQ-Swap and TruthfulQA together with this submission:

```bibtex
@inproceedings{longpre2021entity,
  title={Entity-Based Knowledge Conflicts in Question Answering},
  author={Longpre, Shayne and Perisetla, Kartik and Chen, Anthony and Ramesh, Nikhil and DuBois, Chris and Singh, Sameer},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={7052--7063},
  year={2021}
}
@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={3214--3252},
  year={2022}
}
```
