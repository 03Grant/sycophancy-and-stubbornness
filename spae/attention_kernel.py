"""SPAE attention operator: a patched eager-attention forward with two key sets per row.

Suppression side: the user-pressure keys M receive the additive bias log(alpha) on every query row (prefill included).
Amplification side: the contextual-information keys E receive, per layer, head and query row, the bias that raises
their post-softmax share from s to min(tau, s + delta), clipped to [0, 6] nats, on the allowed receiver rows only.
Hybrid stacks (Qwen3.5 / Qwen3.8): the linear-attention layers have no attention matrix, so the write gate beta_t of
their delta rule is scaled instead (M positions multiplied by lin_alpha, E positions moved towards 1 by lin_rho),
at prefill only. Nothing here reads a label; the row's spans are installed by the runner.
"""
import math
import sys

import torch


def decoder_layers(model):
    """The decoder's layer list, whether the text stack sits directly under `model.model` or behind a multimodal wrapper."""
    node = model.model
    for attr in ("layers", "language_model.layers", "text_model.layers", "model.layers"):
        obj = node
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder layer list on " + type(model).__name__)

class ShareBias:
    """Patched eager attention: a share-controlled additive key bias, restricted by query row and by head."""

    def __init__(self, model):
        # hybrid stacks (linear + full attention) only expose self_attn on the full-attention layers; the hook lives there
        layers = decoder_layers(model)
        self.hooked = [i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")]
        attn = layers[self.hooked[0]].self_attn
        self.module = __import__(type(attn).__module__, fromlist=["x"])
        self.original = self.module.eager_attention_forward
        self.repeat_kv = self.module.repeat_kv
        self.n_layers = len(layers)
        self.device = model.device
        self.reset()
        self.collect = False
        self.collect_upto = 0
        self.step = {}
        # hybrid stacks: the linear-attention layers have no attention matrix to bias, so their delta-rule write gate
        # beta_t is scaled per position instead (stance positions down by lin_alpha, evidence positions towards 1 by lin_rho)
        self.linear = [layer.linear_attn for layer in layers if hasattr(layer, "linear_attn")]
        rules = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule",
                 "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
        for la in self.linear:
            if getattr(la, "_kc_wrapped", False):
                continue
            found = [n for n in rules if hasattr(la, n)]
            for n in found:
                setattr(la, n, self._wrap_rule(getattr(la, n)))
            la._kc_wrapped = bool(found)
        if self.linear and not getattr(self.linear[0], "_kc_wrapped", False):
            # some builds keep the delta rule as a module-level function instead of a method: patch it there
            mod = sys.modules[type(self.linear[0]).__module__]
            for n in rules:
                fn = getattr(mod, n, None)
                if callable(fn) and not getattr(fn, "_kc_wrapped", False):
                    wrapped = self._wrap_rule(fn)
                    wrapped._kc_wrapped = True
                    setattr(mod, n, wrapped)

    def _wrap_rule(self, fn):
        """Delta-rule call with the write gate beta scaled on the installed spans; prefill rows only, generated rows are never a span."""
        def call(*args, **kw):
            want_m = self.lin_alpha != 1.0 and self.has_m
            want_e = self.lin_rho > 0.0 and self.has_e
            if self.mode != "off" and (want_m or want_e):
                beta = kw["beta"] if "beta" in kw else args[4]                    # [rows, seq, heads]
                B, S = beta.shape[0], beta.shape[1]
                if S > 1:
                    beta = beta.clone()
                    if want_m:
                        beta = torch.where(self.m_mask[:B, :S].unsqueeze(-1), beta * self.lin_alpha, beta)
                    if want_e:
                        beta = torch.where(self.e_mask[:B, :S].unsqueeze(-1), beta + (1.0 - beta) * self.lin_rho, beta)
                    if "beta" in kw:
                        kw["beta"] = beta
                    else:
                        args = args[:4] + (beta,) + args[5:]
            return fn(*args, **kw)
        return call

    def reset(self, rows: int = 1, device=None) -> None:
        """Drop every span and restriction, so the next forward pass is unmodified.

        Spans are per-row boolean masks over key positions rather than index lists, so one forward pass can
        carry a whole batch of items with different evidence spans. The running statistics stay on the device
        and are read back once per batch: reading them per layer and per step costs a synchronisation each.
        """
        self.rows = rows
        self.e_mask = None         # BoolTensor [rows, keys]: evidence positions
        self.d_mask = None         # BoolTensor [rows, keys]: competitor positions
        self.ref_mask = None       # BoolTensor [rows, keys]: span whose dynamic gain the control has to match
        self.q_allow = None        # BoolTensor [rows, positions] over absolute query positions, or None for all
        self.head_mask = None      # BoolTensor [rows, layers, heads] for the evidence bias, or None for all
        self.m_head_mask = None    # the same restriction for the suppression bias, or None for all
        self.probe_mask = None     # BoolTensor [rows, keys]: measured only, never boosted
        self.m_mask = None         # BoolTensor [rows, keys]: suppressed positions, biased by log(m_alpha) before E is measured
        self.m_allow = None        # BoolTensor [rows, positions] over absolute query positions for M, or None for all
        self.active = None         # BoolTensor [rows]: rows still generating, or None for all
        self.has_e = False         # cached on the host: testing the mask inside the forward would sync every layer
        self.has_d = False
        self.has_m = False
        self.m_alpha = 1e-9        # multiplier on the suppressed keys; 1e-9 is zero attention at bf16 precision
        self.lin_alpha = 1.0       # linear-attention write gate multiplier on the suppressed positions (1 = untouched)
        self.lin_rho = 0.0         # fraction by which the evidence positions' write gate moves towards 1 (0 = untouched)
        self.row_has_e = None      # BoolTensor [rows]: rows with an empty span are left completely alone
        dev = device or self.device
        z = lambda: torch.zeros(rows, dtype=torch.float64, device=dev)   # noqa: E731
        self.delta_sum, self.delta_n = z(), z()
        self.stats = {k: z() for k in ("n", "e_pre", "e_post", "p_pre", "p_post", "reached", "m_n", "m_pre")}
        self.mode = "off"
        self.alpha = 1.0
        self.target = 0.3
        self.cap = 0.2
        self.gamma = 0.5
        self.beta_max = math.exp(6.0)
        self.min_share = 0.0       # heads whose unbiased share on the span is below this are left alone (0 = all heads)

    def load_batch(self, plans: list, width: int) -> None:
        """Install one batch of per-row spans, already expressed in the padded key coordinates."""
        dev = self.device
        self.reset(len(plans), dev)
        blank = lambda: torch.zeros(len(plans), width, dtype=torch.bool, device=dev)          # noqa: E731
        fields = {"e": blank(), "d": blank(), "ref": blank(), "probe": blank(), "m": blank()}
        used = {k: False for k in fields}
        allow = torch.ones(len(plans), width, dtype=torch.bool, device=dev)
        m_allow = torch.ones(len(plans), width, dtype=torch.bool, device=dev)
        any_allow = any_m_allow = False
        heads = m_heads = None
        for i, plan in enumerate(plans):
            for key in fields:
                span = [k for k in plan.get(key + "_span") or [] if k < width]
                if span:
                    fields[key][i, torch.tensor(span, dtype=torch.long, device=dev)] = True
                    used[key] = True
            if plan.get("q_allow") is not None:
                any_allow = True
                a = plan["q_allow"]
                allow[i, : len(a)] = torch.tensor(a, dtype=torch.bool, device=dev)
                allow[i, len(a):] = False
            if plan.get("m_allow") is not None:
                any_m_allow = True
                a = plan["m_allow"]
                m_allow[i, : len(a)] = torch.tensor(a, dtype=torch.bool, device=dev)
                m_allow[i, len(a):] = False
            if plan.get("head_mask") is not None:
                heads = heads if heads is not None else torch.ones(
                    len(plans), *plan["head_mask"].shape, dtype=torch.bool, device=dev)
                heads[i] = torch.as_tensor(plan["head_mask"], device=dev)
            if plan.get("m_head_mask") is not None:
                m_heads = m_heads if m_heads is not None else torch.ones(
                    len(plans), *plan["m_head_mask"].shape, dtype=torch.bool, device=dev)
                m_heads[i] = torch.as_tensor(plan["m_head_mask"], device=dev)
        self.e_mask, self.has_e = fields["e"], used["e"]
        self.row_has_e = fields["e"].any(-1)
        self.d_mask, self.has_d = (fields["d"], used["d"]) if used["d"] else (None, False)
        self.ref_mask = fields["ref"] if used["ref"] else None
        self.probe_mask = fields["probe"] if used["probe"] else None
        self.q_allow = allow if any_allow else None
        self.m_mask, self.has_m = (fields["m"], used["m"]) if used["m"] else (None, False)
        self.m_allow = m_allow if any_m_allow else None
        self.head_mask = heads
        self.m_head_mask = m_heads
        self.active = torch.ones(len(plans), dtype=torch.bool, device=dev)

    def read_stats(self) -> list:
        """Per-row counters, copied off the device once, after the whole batch has generated."""
        cols = {k: v.tolist() for k, v in self.stats.items()}
        d_sum, d_n = self.delta_sum.tolist(), self.delta_n.tolist()
        return [dict(delta_sum=d_sum[i], delta_n=d_n[i], **{k: cols[k][i] for k in cols})
                for i in range(self.rows)]

    def _beta_dynamic(self, share: torch.Tensor) -> torch.Tensor:
        """Multiplier that raises the span's share to min(target, share + cap); 1 where it already suffices."""
        want = torch.clamp(torch.minimum(share + self.cap, torch.full_like(share, self.target)), max=0.999)
        want = torch.maximum(want, share)                                   # never reduce
        s = share.clamp(1e-6, 0.999)
        beta = (want * (1 - s)) / (s * (1 - want).clamp_min(1e-6))
        return beta.clamp(1.0, self.beta_max)

    def _beta_for_share(self, share: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Multiplier that adds exactly `delta` to a span's post-softmax share."""
        s = share.clamp(1e-6, 0.999)
        want = (s + delta).clamp(1e-6, 0.999)
        return ((want * (1 - s)) / (s * (1 - want)).clamp_min(1e-6)).clamp(1.0, self.beta_max)

    def _beta_transfer(self, s_e: torch.Tensor, s_d: torch.Tensor):
        """Multipliers on E and D that move gamma of D's share onto E, keeping every other position's mass."""
        s_e = s_e.clamp(1e-6, 0.999)
        a = (s_e + self.gamma * s_d).clamp(max=0.999)                       # E's target share
        c = 1 - self.gamma * s_d                                            # D is scaled by (1 - gamma)
        beta_e = (a * (c - s_e)) / (s_e * (1 - a)).clamp_min(1e-6)
        return beta_e.clamp(1.0, self.beta_max), max(1.0 - self.gamma, 1e-6)

    def _forward(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        """Eager attention with the share controller applied before the final softmax."""
        keys = self.repeat_kv(key, module.num_key_value_groups)
        values = self.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : keys.shape[-2]]
        if self.mode != "off" and self.has_m:
            # Suppression is applied first, as a plain key bias on its own set, so that the evidence controller below
            # measures and raises the evidence share on the distribution that no longer reads the suppressed span.
            B, H, Q, K = weights.shape
            m = self.m_mask[:B, :K].view(B, 1, 1, K)
            base_m = torch.softmax(weights.float(), dim=-1)
            s_m = (base_m * m).sum(-1)                                      # [B, H, Q]
            log_m = torch.full_like(s_m, math.log(self.m_alpha))
            if self.m_allow is not None:
                q_abs = torch.arange(K - Q, K, device=weights.device)
                gate = self.m_allow[:B][:, q_abs].view(B, 1, Q)
                log_m = torch.where(gate, log_m, torch.zeros_like(log_m))
            if self.m_head_mask is not None:                                # restrict the layers and heads
                hm = self.m_head_mask[:B, module.layer_idx].view(B, H, 1)
                log_m = torch.where(hm, log_m, torch.zeros_like(log_m))
            live_m = (self.m_mask[:B].any(-1)).to(base_m.dtype)
            if self.active is not None:
                live_m = live_m * self.active[:B].to(base_m.dtype)
            self.stats["m_n"] += (live_m * float(H * Q)).double()
            self.stats["m_pre"] += (s_m * live_m.view(B, 1, 1)).sum((1, 2)).double()
            weights = weights + (log_m.unsqueeze(-1) * m.to(log_m.dtype)).to(weights.dtype)
        if self.mode != "off" and self.has_e:
            B, H, Q, K = weights.shape
            e = self.e_mask[:B, :K].view(B, 1, 1, K)
            base = torch.softmax(weights.float(), dim=-1)                   # [B, H, Q, K]
            s_e = (base * e).sum(-1)                                        # [B, H, Q]
            log_d = None
            if self.mode == "fixed":
                log_e = torch.full_like(s_e, math.log(self.alpha))
            elif self.mode == "dynamic":
                if self.ref_mask is None:
                    beta_e = self._beta_dynamic(s_e)
                else:                                               # matched control: same gain, other content
                    s_ref = (base * self.ref_mask[:B, :K].view(B, 1, 1, K)).sum(-1)
                    gain = (torch.minimum(s_ref + self.cap, torch.full_like(s_ref, self.target)) - s_ref).clamp(min=0)
                    beta_e = self._beta_for_share(s_e, gain)
                log_e = torch.log(beta_e)
            elif not self.has_d:
                log_e = torch.zeros_like(s_e)
            else:
                d = self.d_mask[:B, :K].view(B, 1, 1, K)
                beta_e, beta_d = self._beta_transfer(s_e, (base * d).sum(-1))
                log_e = torch.log(beta_e)
                log_d = torch.full_like(s_e, math.log(beta_d))
            if self.min_share > 0:                                          # only heads that already read the span
                log_e = torch.where(s_e >= self.min_share, log_e, torch.zeros_like(log_e))
            if self.q_allow is not None:                                    # restrict the receiver rows
                q_abs = torch.arange(K - Q, K, device=base.device)
                gate = self.q_allow[:B][:, q_abs].view(B, 1, Q)
                log_e = torch.where(gate, log_e, torch.zeros_like(log_e))
                log_d = None if log_d is None else torch.where(gate, log_d, torch.zeros_like(log_d))
            if self.head_mask is not None:                                  # restrict the heads
                hm = self.head_mask[:B, module.layer_idx].view(B, H, 1)
                log_e = torch.where(hm, log_e, torch.zeros_like(log_e))
                log_d = None if log_d is None else torch.where(hm, log_d, torch.zeros_like(log_d))
            bias = log_e.unsqueeze(-1) * e.to(log_e.dtype)                  # E wins wherever E and D overlap
            if log_d is not None:
                bias = bias + log_d.unsqueeze(-1) * (self.d_mask[:B, :K].view(B, 1, 1, K) & ~e).to(log_e.dtype)
            beta_applied = torch.exp(log_e)
            z = (1 - s_e + beta_applied * s_e).clamp_min(1e-6)
            new_e = (beta_applied * s_e) / z
            # accumulate per row on the device; one read-back per batch instead of a sync per layer and step
            live_row = self.row_has_e[:B].to(base.dtype)
            if self.active is not None:
                live_row = live_row * self.active[:B].to(base.dtype)
            live = live_row.view(B, 1, 1)
            self.delta_sum += ((new_e - s_e) * live).sum((1, 2)).double()
            self.delta_n += (live_row * float(H * Q)).double()
            applied = (beta_applied > 1.0 + 1e-6).to(base.dtype) * live
            st = self.stats
            st["n"] += applied.sum((1, 2)).double()
            st["e_pre"] += (s_e * applied).sum((1, 2)).double()
            st["e_post"] += (new_e * applied).sum((1, 2)).double()
            st["reached"] += ((new_e >= self.target - 0.01).to(base.dtype) * applied).sum((1, 2)).double()
            if self.probe_mask is not None:
                pr = self.probe_mask[:B, :K].view(B, 1, 1, K)
                s_p = (base * pr).sum(-1)
                s_in = (base * (pr & e)).sum(-1)
                new_p = (beta_applied * s_in + (s_p - s_in)) / z
                st["p_pre"] += (s_p * applied).sum((1, 2)).double()
                st["p_post"] += (new_p * applied).sum((1, 2)).double()
            weights = weights + bias.to(weights.dtype)
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32)
        if self.collect:
            self.step[module.layer_idx] = weights[:, :, -1, : self.collect_upto].to(torch.float16)
        output = torch.matmul(weights.to(query.dtype), values)
        return output.transpose(1, 2).contiguous(), None

    def __enter__(self):
        self.module.eager_attention_forward = self._forward
        return self

    def __exit__(self, *_exc):
        self.module.eager_attention_forward = self.original
        return False
