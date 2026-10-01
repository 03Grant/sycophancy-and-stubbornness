"""AutoPASTA engine: the attention-mask edit of the official pastalib (scale_position `exclude`, alpha 0.01) on the
selected heads, the key-sentence extraction prompt, and the all-MiniLM-L6-v2 mapping of an extracted sentence back to a
token span of the request.

pastalib's module table names Llama, Mistral, Gemma, GPT-J and Phi-3 only, and its hook locates the highlighted span by
substring; here the same `edit_multisection_attention` hook is registered on the softmax-attention modules of any causal
LM, with the span given as token positions, which is what the stored mappings hold."""
import json
import re
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from pastalib.pasta import PASTA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import decoder_layers, model_text_config                     # noqa: E402
from readout import ANSWER_SUFFIX, find_commitment_cut, letter_token_ids  # noqa: E402
from run_baseline import CHAT_KWARGS, PROMPTS, Engine                     # noqa: E402

ALPHA = 0.01
SCALE_POSITION = 'exclude'
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
EXTRACT_MAX_TOKENS = 128
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=\S)')   # a sentence end followed by whitespace


class SpanPASTA(PASTA):
    """pastalib's steerer with a generic model entry and a context manager that takes token ranges."""

    def setup_model(self, model):
        """Register any causal LM: head count from its text config, attention modules from its decoder layers."""
        self.model_name = 'generic'
        self.num_attn_head = model_text_config(model).num_attention_heads
        self.layers = decoder_layers(model)

    @contextmanager
    def steer(self, token_ranges, input_len):
        """Register the pastalib hook on every selected layer for the block; `token_ranges` is one (start, end) per batch row in padded coordinates."""
        ranges = torch.tensor(token_ranges, dtype=torch.long)
        hooks = []
        for layer_idx in self.all_layers_idx:
            fn = partial(self.edit_multisection_attention, head_idx=self.head_config[layer_idx], token_ranges=[ranges], input_len=input_len)
            hooks.append(self.layers[layer_idx].self_attn.register_forward_pre_hook(self.with_mask(fn), with_kwargs=True))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    @staticmethod
    def with_mask(fn):
        """Materialise the causal mask when the model passes none, so that pastalib's edit has a tensor to add to."""
        def hook(module, args, kwargs):
            if kwargs.get('attention_mask') is None:
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                q = hidden.shape[1]
                cache = kwargs.get('past_key_values') or kwargs.get('past_key_value')
                past = cache.get_seq_length(getattr(module, 'layer_idx', 0)) if cache is not None else 0
                kv = past + q
                mask = torch.full((q, kv), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
                mask = torch.triu(mask, diagonal=past + 1)
                kwargs['attention_mask'] = mask[None, None].expand(hidden.shape[0], 1, q, kv).clone()
            return fn(module, args, kwargs)
        return hook


class PastaEngine(Engine):
    """The baseline engine with PASTA steering for generation and the probe, plus the extraction and mapping stages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = None

    def pasta_for(self, heads):
        """A steerer for one head configuration ({layer: [heads]})."""
        return SpanPASTA(self.model, self.tok, {int(k): list(v) for k, v in heads.items()}, alpha=ALPHA, scale_position=SCALE_POSITION)

    @contextmanager
    def steering(self, sequences, spans, heads, input_len=None):
        """Steer a batch: each row's span (token positions of the unpadded sequence) becomes a range in left-padded coordinates; rows without a span are left unedited."""
        width = max(map(len, sequences))
        ranges = []
        for seq, span in zip(sequences, spans):
            pad = width - len(seq)
            ranges.append((pad + min(span), pad + max(span) + 1) if span else (0, 0))
        length = width if input_len is None else input_len
        if not any(spans):
            yield
            return
        with self.pasta_for(heads).steer([r if r != (0, 0) else (0, length) for r in ranges], length):
            yield

    @torch.inference_mode()
    def pasta(self, seqs, max_new, spans, heads):
        """Generate a batch under steering (the highlighted span of the request stays emphasised at every generated token)."""
        with self.steering(seqs, spans, heads):
            return self.ordinary(seqs, max_new)

    @torch.inference_mode()
    def pasta_probe(self, prompt_ids, gen_ids, letters, span, heads):
        """Letter logits at the fixed suffix with the request span still steered (the generated tokens are not downweighted, as during generation)."""
        cut, marker, trimmed = find_commitment_cut(self.tok, gen_ids, phrase_fallback=True, letters=letters)
        seq = prompt_ids + gen_ids[:cut] + self.tok.encode(ANSWER_SUFFIX, add_special_tokens=False)
        with self.steering([seq], [span], heads, input_len=len(prompt_ids)):
            ids = torch.tensor([seq], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[:, -1, :]
        mapping = letter_token_ids(self.tok, ANSWER_SUFFIX, letters)
        scores = {l: float(logits[0, idx]) for l, idx in mapping.items()}
        return {'probe_answer': max(scores, key=scores.get), 'letter_logits': scores,
                'probe_cut': cut, 'probe_marker': marker, 'probe_trimmed': trimmed}

    def extract(self, rows, bare_map):
        """The key sentence the model selects for each row, from the extraction prompt over the request's added paragraphs."""
        out = []
        prompts = [PROMPTS['autopasta-extract'].format(question=r['question'], passage='\n\n'.join(s['text'] for s in segments(r['prompt'], bare_map[r['item_id']]))) for r in rows]
        with self.greedy():
            for start in range(0, len(rows), self.batch):
                gens = self.ordinary([self.encode(p) for p in prompts[start:start + self.batch]], EXTRACT_MAX_TOKENS)
                for r, gen in zip(rows[start:start + self.batch], gens):
                    out.append({'row_id': r['row_id'], 'item_id': r['item_id'], 'question': r['question'], 'prompt': r['prompt'],
                                'segments': segments(r['prompt'], bare_map[r['item_id']]),
                                'key_sentence_raw': self.tok.decode(gen, skip_special_tokens=True).strip(), 'n_gen_tokens': len(gen)})
        return out

    def embed(self, texts):
        """all-MiniLM-L6-v2 sentence embeddings (mean pooling over the attention mask, L2-normalised)."""
        if self.encoder is None:
            from transformers import AutoModel, AutoTokenizer
            self.encoder = (AutoTokenizer.from_pretrained(ENCODER), AutoModel.from_pretrained(ENCODER).to(self.device).eval())
        tok, model = self.encoder
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def map_sentence(self, record):
        """Map an extracted sentence to the closest candidate sentence of the request (cosine of the encoder embeddings) and to its token positions."""
        cands = candidates(record['segments'])
        base = {'row_id': record['row_id'], 'item_id': record['item_id'], 'key_sentence_raw': record['key_sentence_raw'],
                'encoder': ENCODER, 'candidate_count': len(cands)}
        if not cands or not record['key_sentence_raw']:
            return {**base, 'mapped': False, 'highlight_tokens': [], 'selected_source_text': None, 'char_start': None, 'char_end': None, 'cosine_similarity': None}
        emb = self.embed([record['key_sentence_raw']] + [c['text'] for c in cands])
        sims = (emb[1:] @ emb[0]).tolist()
        best = max(range(len(cands)), key=lambda i: (sims[i], -i))
        c = cands[best]
        return {**base, 'mapped': True, 'highlight_tokens': self.span_tokens(record['prompt'], c['start'], c['end']),
                'selected_source_text': c['text'], 'char_start': c['start'], 'char_end': c['end'], 'cosine_similarity': sims[best]}

    def span_tokens(self, prompt, start, end):
        """Token positions of prompt[start:end] in the chat-template encoding of the request."""
        text = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False, **CHAT_KWARGS)
        base = text.find(prompt)
        assert base >= 0
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        lo, hi = base + start, base + end
        return [i for i, (a, b) in enumerate(enc['offset_mapping']) if b > lo and a < hi and b > a]


def segments(prompt, bare):
    """The paragraphs of a request that its neutral counterpart does not contain, with character offsets."""
    have = set(bare.split('\n\n'))
    out, pos = [], 0
    for para in prompt.split('\n\n'):
        if para not in have:
            out.append({'start': pos, 'end': pos + len(para), 'text': para})
        pos += len(para) + 2
    return out


def candidates(segs):
    """Candidate sentences of the added paragraphs: each paragraph split at sentence ends, offsets kept."""
    out = []
    for seg in segs:
        pos = 0
        for piece in SENTENCE_RE.split(seg['text']):
            idx = seg['text'].find(piece, pos)
            out.append({'start': seg['start'] + idx, 'end': seg['start'] + idx + len(piece), 'text': piece})
            pos = idx + len(piece)
    return out
