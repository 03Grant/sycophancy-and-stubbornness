"""Prompt- and decoding-level baselines on CoPE-Bench: S2A, CAD and AdaCAD (and an uninstrumented `original` arm).

No answer or condition label is passed into any method. The context-free input of CAD / AdaCAD is the neutral row of the
same question; the S2A rewrite is a separate greedy generation whose two parts replace the request. Every row records the
reply, the probe letter read at a fixed suffix (chain-of-thought rows) and the method's own diagnostics.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import add_model_args, add_sampling_args, keep_rows, options, parse_answer   # noqa: E402
from readout import ANSWER_SUFFIX, find_commitment_cut, letter_token_ids                # noqa: E402

PROMPTS = json.loads((HERE / 'prompts.json').read_text())
CHAT_KWARGS = {}   # set to {'enable_thinking': False} by --no-think


class Engine:
    """One loaded model with left-padded batching, greedy or per-row-seeded sampled decoding, and the letter probe."""

    def __init__(self, model_path, cap_gib=None, no_think=False):
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cap_gib and self.device == 'cuda':
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(cap_gib * 2**30 / total, .95), 0)
        if no_think:
            CHAT_KWARGS['enable_thinking'] = False
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'
        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation='eager')
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation='eager')
        self.model = self.model.to(self.device).eval()
        self.eos = self.model.generation_config.eos_token_id
        self.eos = set(self.eos if isinstance(self.eos, list) else [self.eos])
        self.eos.discard(None)
        penalty = self.model.generation_config.repetition_penalty or 1.0
        self.repetition = RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None
        self.sampling, self.seed, self.batch = None, 0, 1

    def configure_sampling(self, args):
        """Turn on reply sampling from the CLI switches; greedy when --sample-answer is absent."""
        if getattr(args, 'sample_answer', False):
            self.sampling = {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k}
            self.seed = args.seed
            print(json.dumps({'stage': 'sampling', **self.sampling, 'seed': self.seed}), flush=True)

    def row_seed(self, row_id):
        """Reseed torch for one row so a sampled reply is reproducible row by row (batch size one)."""
        if self.sampling:
            torch.manual_seed(self.seed * 1000003 + int(str(row_id).split(':')[-1]))

    @contextmanager
    def greedy(self):
        """Decode greedily inside the block (auxiliary generations such as the S2A rewrite stay deterministic)."""
        saved, self.sampling = self.sampling, None
        try:
            yield
        finally:
            self.sampling = saved

    def pick(self, logits):
        """Next token per row: argmax, or a temperature, top-k, top-p sample (HF warper order) under sampling."""
        if not self.sampling:
            return logits.argmax(-1)
        l = logits.float() / self.sampling['temperature']
        k = self.sampling['top_k']
        if k and k < l.shape[-1]:
            l = l.masked_fill(l < l.topk(k, dim=-1).values[:, -1:], float('-inf'))
        p = self.sampling['top_p']
        if p < 1:
            sorted_logits, order = l.sort(-1, descending=True)
            cum = sorted_logits.softmax(-1).cumsum(-1)
            drop = cum - sorted_logits.softmax(-1) > p
            l = l.masked_fill(drop.scatter(-1, order, drop), float('-inf'))
        return torch.multinomial(l.softmax(-1), 1)[:, 0]

    def stamp(self, out):
        """Record the decoding settings on an output row."""
        if self.sampling:
            out.update(sampled_answer=True, seed=self.seed, **self.sampling)
        out['batch'] = self.batch
        return out

    def encode(self, prompt):
        """Token ids of one request as a single user turn of the chat template."""
        messages = [{'role': 'user', 'content': prompt}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **CHAT_KWARGS)
        return ids['input_ids'] if hasattr(ids, 'keys') else list(ids)

    def pad(self, sequences):
        """Left-padded id and mask tensors for a batch of token lists."""
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, width - len(seq):] = torch.tensor(seq)
            mask[i, width - len(seq):] = 1
        return ids.to(self.device), mask.to(self.device)

    def trim(self, tokens):
        """Generated ids up to the first end-of-sequence or pad token."""
        out = []
        for t in tokens:
            if t in self.eos or t == self.tok.pad_token_id:
                break
            out.append(t)
        return out

    def ordinary(self, sequences, max_tokens):
        """Plain generation (greedy or sampled) for a batch of token lists."""
        ids, mask = self.pad(sequences)
        gen = {'do_sample': True, **self.sampling} if self.sampling else {'do_sample': False}
        result = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens,
                                     pad_token_id=self.tok.pad_token_id, use_cache=True, logits_to_keep=1, **gen)
        return [self.trim(row.tolist()) for row in result[:, ids.shape[1]:]]

    @staticmethod
    def combine(a, b, method):
        """Contrastive logits: CAD with a fixed weight of 1, AdaCAD with the token-wise Jensen-Shannon divergence."""
        a, b = a.float(), b.float()
        if method == 'cad':
            alpha = torch.ones(a.shape[0], 1, device=a.device)
        else:
            p, q = a.softmax(-1), b.softmax(-1)
            logm = ((p + q) / 2).clamp_min(1e-30).log()
            alpha = .5 * ((p * (p.clamp_min(1e-30).log() - logm)).sum(-1, keepdim=True)
                          + (q * (q.clamp_min(1e-30).log() - logm)).sum(-1, keepdim=True))
        return (1 + alpha) * a - alpha * b, alpha[:, 0]

    @torch.inference_mode()
    def contrastive(self, sequences, bares, max_tokens, method):
        """Decode with two caches, the full request and its context-free counterpart, combining the logits per token."""
        ids, mask = self.pad(sequences)
        bid, bmask = self.pad(bares)
        caches = [None, None]
        outputs = [[] for _ in sequences]
        active = torch.ones(len(sequences), dtype=torch.bool, device=self.device)
        max_alpha = torch.zeros(len(sequences), device=self.device)
        for _ in range(max_tokens):
            values = []
            for j, (tokens, amask) in enumerate([(ids, mask), (bid, bmask)]):
                positions = amask.long().cumsum(-1) - 1
                positions.masked_fill_(amask == 0, 0)
                current = tokens if caches[j] is None else tokens[:, -1:]
                positions = positions[:, -current.shape[1]:]
                pred = self.model(input_ids=current, attention_mask=amask,
                                  position_ids=positions, past_key_values=caches[j],
                                  use_cache=True, logits_to_keep=1)
                caches[j] = pred.past_key_values
                values.append(pred.logits[:, -1, :])
            logits, alpha = self.combine(*values, method)
            max_alpha = torch.maximum(max_alpha, alpha * active)
            if self.repetition is not None:
                logits = self.repetition(ids, logits)
            nxt = self.pick(logits)
            for i, token in enumerate(nxt.tolist()):
                if active[i]:
                    if token in self.eos:
                        active[i] = False
                    else:
                        outputs[i].append(token)
            if not active.any():
                break
            nxt = torch.where(active, nxt, self.tok.pad_token_id)[:, None]
            ids, bid = torch.cat([ids, nxt], 1), torch.cat([bid, nxt], 1)
            mask = torch.cat([mask, active.long()[:, None]], 1)
            bmask = torch.cat([bmask, active.long()[:, None]], 1)
        return outputs, max_alpha.tolist()

    @torch.inference_mode()
    def probe(self, prompt_ids, gen_ids, letters, bare_ids=None, method=None):
        """Letter logits at the fixed suffix after the (trimmed) chain of thought; CAD / AdaCAD combine the two inputs here too."""
        cut, marker, trimmed = find_commitment_cut(self.tok, gen_ids, phrase_fallback=True, letters=letters)
        suffix = self.tok.encode(ANSWER_SUFFIX, add_special_tokens=False)
        tail = gen_ids[:cut] + suffix
        seq = torch.tensor([prompt_ids + tail], device=self.device)
        logits = self.model(input_ids=seq, attention_mask=torch.ones_like(seq), logits_to_keep=1).logits[:, -1, :]
        if method in ('cad', 'adacad'):
            bseq = torch.tensor([bare_ids + tail], device=self.device)
            other = self.model(input_ids=bseq, attention_mask=torch.ones_like(bseq), logits_to_keep=1).logits[:, -1, :]
            logits, _ = self.combine(logits, other, method)
        mapping = letter_token_ids(self.tok, ANSWER_SUFFIX, letters)
        scores = {l: float(logits[0, idx]) for l, idx in mapping.items()}
        return {'probe_answer': max(scores, key=scores.get), 'letter_logits': scores,
                'probe_cut': cut, 'probe_marker': marker, 'probe_trimmed': trimmed}


def without_batch(signature):
    """Signature fields that must match on resume; the batch size only changes the chunking of the same rows."""
    return {k: v for k, v in signature.items() if k != 'batch'}


def split_s2a(text):
    """The two parts of an S2A rewrite, or None when the rewrite did not produce both labelled sections."""
    match = re.search(r'Unbiased text context(?:\s*\([^\n]*\))?\s*:\s*(.*?)'
                      r'Question/Query(?:\s*\([^\n]*\))?\s*:\s*(.+)', text, re.S | re.I)
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def finish_row(e, r, tokens, extra, probe_fn):
    """The output record of one row: ids, reply, the stated letter and (chain-of-thought rows) the probe read-out."""
    reply = e.tok.decode(tokens, skip_special_tokens=True)
    out = {k: r[k] for k in ('row_id', 'item_id', 'family', 'control_type', 'instruction_style')}
    out.update(reply=reply, n_gen_tokens=len(tokens), aligned=True, **extra)
    if r['instruction_style'] == 'cot_letter':
        letters = options(r['prompt'])
        out['stated_answer'] = parse_answer(reply, letters)
        out.update(probe_fn(letters))
        out['answer'] = out.get('probe_answer', out['stated_answer'])
    else:
        out['answer'] = next(iter(reply.strip().splitlines()), '')
    return e.stamp(out)


def main():
    """Run one method over the benchmark rows, appending one record per row (restartable, shardable)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['original', 's2a', 'cad', 'adacad'], required=True)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int)
    p.add_argument('--batch', type=int, default=1)
    add_sampling_args(p)
    p.add_argument('--rewrites', nargs='*', type=Path, help='s2a: reuse the greedy rewrites recorded in these earlier output files')
    args = p.parse_args()
    label = args.model_label or Path(args.model).name
    rows = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    bare_map = {r['item_id']: r['prompt'] for r in rows if r['direction'] == 'control'}
    si, sn = map(int, args.shard.split('/'))
    assert 0 <= si < sn
    rows = [r for r in rows if args.method == 'original' or r['direction'] != 'control']
    rows = keep_rows(rows, args.rows)
    rows = [r for i, r in enumerate(rows) if i % sn == si]
    if args.limit:
        # interleave the two families so that a smoke test covers both answer formats
        groups = [[r for r in rows if r['family'] == family] for family in ('F1', 'F3')]
        rows = [r for i in range(max(map(len, groups))) for g in groups for r in g[i:i + 1]][:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    signature = {'schema': 1, 'model': args.model, 'method': args.method, 'shard': args.shard,
                 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'prompt_sha256': hashlib.sha256((HERE / 'prompts.json').read_bytes()).hexdigest(),
                 'limit': args.limit, 'batch': args.batch}
    cfgpath = args.out.with_suffix('.config.json')
    if cfgpath.exists() and without_batch(json.loads(cfgpath.read_text())) != without_batch(signature):
        raise RuntimeError('Refusing to resume an incompatible configuration')
    cfgpath.write_text(json.dumps(signature, indent=2) + '\n')
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)['row_id'])
    rows = [r for r in rows if r['row_id'] not in done]
    if not rows:
        print('COMPLETE already done', flush=True)
        return
    e = Engine(args.model, args.cap_gib, args.no_think)
    e.configure_sampling(args)
    e.batch = args.batch   # under sampling a chunk shares one stream seeded from its first row
    rewrite_map = {}
    for path in args.rewrites or []:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('metadata', {}).get('rewrite') is not None:
                    rewrite_map[r['row_id']] = r['metadata']['rewrite']
    prepared = [(r, e.encode(r['prompt']), e.encode(bare_map[r['item_id']])) for r in rows]
    prepared.sort(key=lambda x: (x[0]['instruction_style'], len(x[1]), x[0]['row_id']))
    started = time.time()
    completed = 0
    with args.out.open('a', buffering=1) as fh:
        for style in ['short_phrase', 'cot_letter']:
            group = [x for x in prepared if x[0]['instruction_style'] == style]
            for offset in range(0, len(group), args.batch):
                chunk = group[offset:offset + args.batch]
                seqs, bares = [x[1] for x in chunk], [x[2] for x in chunk]
                meta = [{} for _ in chunk]
                if args.method == 's2a':
                    # the rewrite is greedy (deterministic) and may be reused from an earlier run via --rewrites
                    texts = [rewrite_map.get(x[0]['row_id']) for x in chunk]
                    missing = [i for i, t in enumerate(texts) if t is None]
                    if missing:
                        with e.greedy():
                            gens = e.ordinary([e.encode(PROMPTS['s2a-rewrite'].replace('{text}', chunk[i][0]['prompt'])) for i in missing], 1024)
                        for i, gen in zip(missing, gens):
                            texts[i] = e.tok.decode(gen, skip_special_tokens=True)
                    for i, (x, text) in enumerate(zip(chunk, texts)):
                        parts = split_s2a(text)
                        meta[i] = {'rewrite': text, 'rewrite_parsed': parts is not None, 'rewrite_reused': i not in missing}
                        if parts:
                            instr = x[0]['prompt'].rsplit('\n\n', 1)[-1]
                            seqs[i] = e.encode(parts[0] + '\n' + parts[1] + '\n' + PROMPTS['s2a-answer'] + '\n' + instr)
                max_tokens = 400 if style == 'cot_letter' else 32
                e.row_seed(chunk[0][0]['row_id'])
                if args.method in ('cad', 'adacad'):
                    gens, maxima = e.contrastive(seqs, bares, max_tokens, args.method)
                    for m, a in zip(meta, maxima):
                        m['max_alpha'] = a
                else:
                    gens = e.ordinary(seqs, max_tokens)
                for i, ((r, _, _), tokens) in enumerate(zip(chunk, gens)):
                    out = finish_row(e, r, tokens, {'model': label, 'method': args.method, 'metadata': meta[i]},
                                     lambda letters, i=i: e.probe(seqs[i], gens[i], letters, bares[i], args.method))
                    fh.write(json.dumps(out, ensure_ascii=False) + '\n')
                    completed += 1
                fh.flush()
                os.fsync(fh.fileno())
                print(json.dumps({'stage': 'generation', 'completed': completed, 'remaining': len(rows) - completed,
                                  'elapsed_s': round(time.time() - started, 1)}), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
