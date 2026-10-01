"""JuICE head intervention: the selected heads' contributions to o_proj are scaled by 1 + alpha (recorded in a first
forward pass, added in a second one), for generation and for the letter probe."""
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import decoder_layers, model_text_config                     # noqa: E402
from readout import ANSWER_SUFFIX, find_commitment_cut, letter_token_ids  # noqa: E402
from run_baseline import Engine                                           # noqa: E402


class HeadHooks:
    """Per-layer o_proj hooks: `capture` records alpha * (head contribution), `inject` adds the recorded value."""

    def __init__(self, model, specs):
        self.model = model
        self.by_layer = {}
        self.cache = {}
        for layer, head, alpha in specs:
            if alpha:
                self.by_layer.setdefault(int(layer), []).append((int(head), float(alpha)))

    def contribution(self, module, x, entries):
        """Sum over the layer's selected heads of alpha times that head's share of the o_proj output."""
        heads = int(model_text_config(self.model).num_attention_heads)
        dim = module.weight.shape[1] // heads
        assert dim * heads == module.weight.shape[1]
        result = None
        for head, alpha in entries:
            assert 0 <= head < heads
            lo, hi = head * dim, (head + 1) * dim
            part = F.linear(x[..., lo:hi], module.weight[:, lo:hi], bias=None) * alpha
            result = part if result is None else result + part
        return result

    @contextmanager
    def install(self, mode):
        """Register the hooks in `capture` or `inject` mode for the duration of the block."""
        handles = []
        if mode == 'capture':
            self.cache = {}
        try:
            for layer, entries in self.by_layer.items():
                def hook(module, inputs, output, layer=layer, entries=entries):
                    if mode == 'capture':
                        self.cache[layer] = self.contribution(module, inputs[0], entries).detach()
                    else:
                        value = self.cache[layer]
                        assert value.shape == output.shape
                        return output + value
                handles.append(decoder_layers(self.model)[layer].self_attn.o_proj.register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()


class JuiceEngine(Engine):
    """The baseline engine with the two-pass JuICE decode and probe."""

    @torch.inference_mode()
    def juice(self, seqs, max_new, specs):
        """Decode a batch with the head intervention: every step runs a capture pass and an inject pass."""
        ids, mask = self.pad(seqs)
        cache = [None, None]
        hooks = HeadHooks(self.model, specs)
        alive = torch.ones(len(seqs), device=self.device, dtype=torch.bool)
        results = [[] for _ in seqs]
        for step in range(max_new):
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 0)
            for which, mode in enumerate(('capture', 'inject')):
                current = ids if cache[which] is None else ids[:, -1:]
                with hooks.install(mode):
                    out = self.model(input_ids=current, attention_mask=mask, position_ids=positions[:, -current.shape[1]:],
                                     past_key_values=cache[which], use_cache=True, logits_to_keep=1)
                cache[which] = out.past_key_values
            logits = out.logits[:, -1, :]
            if self.repetition is not None:
                logits = self.repetition(ids, logits)
            nxt = self.pick(logits)
            for i, t in enumerate(nxt.tolist()):
                if alive[i]:
                    if t in self.eos:
                        alive[i] = False
                    else:
                        results[i].append(t)
            if not alive.any():
                break
            nxt = torch.where(alive, nxt, self.tok.pad_token_id)[:, None]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, alive.long()[:, None]], 1)
        return results

    @torch.inference_mode()
    def juice_probe(self, prompt, tokens, letters, specs):
        """Letter logits at the fixed suffix under the same two-pass intervention."""
        cut, marker, trimmed = find_commitment_cut(self.tok, tokens, phrase_fallback=True, letters=letters)
        seq = torch.tensor([prompt + tokens[:cut] + self.tok.encode(ANSWER_SUFFIX, add_special_tokens=False)], device=self.device)
        hooks = HeadHooks(self.model, specs)
        for mode in ('capture', 'inject'):
            with hooks.install(mode):
                out = self.model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False, logits_to_keep=1)
        mapping = letter_token_ids(self.tok, ANSWER_SUFFIX, letters)
        logits = {l: float(out.logits[0, -1, t]) for l, t in mapping.items()}
        return {'probe_answer': max(logits, key=logits.get), 'letter_logits': logits,
                'probe_cut': cut, 'probe_marker': marker, 'probe_trimmed': trimmed}
