"""Shared helpers of the baseline runners: model-config access, option and answer parsing, row filters, jsonl loading and
the fitting objective (`hit`)."""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from answer_match import first_line, phrase_match   # noqa: E402
from readout import parse_freeform                 # noqa: E402


def text_config(cfg):
    """The text sub-config of a multimodal config (dict or object), or the config itself."""
    return cfg.get('text_config', cfg) if isinstance(cfg, dict) else getattr(cfg, 'text_config', cfg)


def decoder_layers(model):
    """The decoder layer list of a plain causal LM or of a multimodal wrapper (model.model.language_model)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, 'language_model') else inner.layers


def model_text_config(model):
    """The text config of a loaded model: the nested text_config of a multimodal wrapper, else the config itself."""
    return text_config(model.config)


def full_attention_layers(cfg):
    """Indices of the layers that carry softmax attention (every layer unless the config lists layer_types)."""
    tc = text_config(cfg)
    n = tc['num_hidden_layers'] if isinstance(tc, dict) else tc.num_hidden_layers
    types = (tc.get('layer_types') if isinstance(tc, dict) else getattr(tc, 'layer_types', None)) or ['full_attention'] * n
    return [i for i, t in enumerate(types) if t == 'full_attention']


def parse_answer(text, letters):
    """The option letter a chain-of-thought reply states in prose (a committed phrase, a bare first line, or a single letter on the last line)."""
    answer = parse_freeform(text, allow_bare=False, letters=letters)
    if answer:
        return answer
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return None
    chars = ''.join(letters)
    first = lines[0].strip("*()[]\"'.: ")
    if first in letters:
        return first
    found = set(re.findall(rf'(?<![A-Za-z])\(?\**([{chars}])\**\)?(?![A-Za-z])', lines[-1]))
    return found.pop() if len(found) == 1 else None


def options(prompt):
    """The option letters of a multiple-choice request, read off its 'A. ' lines."""
    letters = re.findall(r'^([A-M])\. ', prompt, re.M)
    if letters and letters != list('ABCDEFGHIJKLM'[:len(letters)]):
        raise ValueError(f'Invalid option sequence: {letters}')
    return letters


def add_sampling_args(p):
    """CLI switches for sampling the reply instead of greedy decoding (the paper's main runs are greedy)."""
    p.add_argument('--sample-answer', action='store_true', help='sample the reply with temperature/top-p/top-k, seeded per row from --seed')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--rows', type=Path, help='keep only the row ids listed in this file (one per line)')


def add_model_args(p):
    """CLI switches shared by every runner: the model, its label, the memory cap and the thinking switch."""
    p.add_argument('--model', required=True, help='Hugging Face id or local path')
    p.add_argument('--model-label', help='name written into every output row (default: the last path component of --model)')
    p.add_argument('--cap-gib', type=float, help='optional per-process GPU memory cap in GiB')
    p.add_argument('--no-think', action='store_true', help='chat template with enable_thinking=False (thinking-mode models)')


def keep_rows(rows, path):
    """Rows whose row_id is listed in `path` (one id per line), in their original order; all rows without a path."""
    if not path:
        return rows
    keep = {x.strip() for x in Path(path).read_text().splitlines() if x.strip()}
    return [r for r in rows if r['row_id'] in keep]


def load(paths):
    """Rows of one or more jsonl files keyed by row_id (later files overwrite earlier ones)."""
    rows = {}
    for path in ([paths] if isinstance(paths, (str, Path)) else paths):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r['row_id']] = r
    return rows


def hit(record, prediction, key='gold'):
    """The fitting objective: whether a prediction names the answer stored under `answers[key]` (letter rows compare the
    parsed letter, short-answer rows match the first reply line against the alias list)."""
    target = record['answers'].get(key)
    if target is None:
        return False
    if record['instruction_style'] == 'cot_letter':
        return prediction.get('answer') == target
    aliases = target if isinstance(target, list) else [target]
    return phrase_match(first_line(prediction.get('reply') or ''), aliases)
