"""SPAE runner: the auxiliary call (call 1) and the answer call (call 2) on CoPE-Bench rows.

Two-call attention intervention with TWO slots: a stance to suppress and evidence to amplify, from one call 1.

Call 1 (one prompt, no route line): the model copies the user pressure and the supplied information out of
the request as two lines, `User pressure:` and `Information:`, each possibly NONE. The copied strings are matched back into the plain prompt (quote
localiser); a copy that cannot be matched leaves that side unarmed (`--attention-fallback` uses the attention the copy
paid to the request instead).

Call 2 answers the plain prompt with both key sets armed in one forward pass: the stance keys are removed first
(log(alpha_memory) on every query row, prefill included), then the evidence share is raised towards the target on the
query rows after the evidence span. The two sets are made disjoint by giving the stance priority. Arms:

    baseline   no bias
    suppress   stance keys only
    amplify    evidence keys only
    dual       both

Nothing that affects the generation reads the row's cell or labels (they are only copied into the output record for
scoring and diagnostics); a slot the model left as NONE simply stays unarmed.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generation as M                                                              # noqa: E402
import scaffold as K                                                                # noqa: E402
from attention_kernel import ShareBias                                              # noqa: E402
from scaffold import contiguous_mask                                                # noqa: E402
from readout import ANSWER_SUFFIX, find_commitment_cut, letter_token_ids, read_letter_logits   # noqa: E402

CUE = "User pressure:"
EVIDENCE_TAG = "Information:"
# The auxiliary-call wording is read from --prompt-file (prompts/call1.txt in the paper setting) and registered under
# the name "file"; call-1 rows record it as `router`, and call2 rebuilds the call-1 layout from the same file.
PROMPTS: dict[str, str] = {}

GEN_TOKENS, BASE_CUE = M.GEN_TOKENS, M.BASE_CUE


class DualWatcher(LogitsProcessor):
    """Collect each decoding step's attention and stop once the Information line is complete."""

    def __init__(self, patch, tok, prompt_len: int, max_steps: int):
        self.patch, self.tok, self.P, self.max_steps = patch, tok, prompt_len, max_steps
        self.steps, self.end = [], None

    def __call__(self, input_ids, scores):
        hooked = getattr(self.patch, "hooked", None) or list(range(self.patch.n_layers))
        if len(self.patch.step) < len(hooked):
            return scores
        self.steps.append(torch.stack([self.patch.step[l] for l in hooked], 1).cpu())
        self.patch.step = {}
        text = self.tok.decode(input_ids[0, self.P:], skip_special_tokens=False)
        done = EVIDENCE_TAG in text and "\n" in text.split(EVIDENCE_TAG, 1)[1].lstrip()   # the Information line is closed
        if done or input_ids.shape[1] - self.P >= self.max_steps:
            self.end = len(self.steps)
            scores[:, :] = -float("inf")
            scores[:, self.tok.eos_token_id] = 0.0
        return scores


def parse_dual(gen_text: str) -> tuple[str, str]:
    """The user-pressure line (everything before the Information tag) and the information line; NONE and blanks become ''."""
    head, _, tail = gen_text.partition(EVIDENCE_TAG)
    stance = head.strip().split("\n")[0].strip()
    evidence = tail.strip().split("\n")[0].strip() if tail else ""
    clean = lambda s: "" if (not s or s.strip().strip('"').upper() in ("NONE", "NONE.")) else s.strip()   # noqa: E731
    return clean(stance), clean(evidence)


def char_to_steps(tok, gen_ids: list[int], lo_char: int, hi_char: int) -> tuple[int, int]:
    """Generated-token index range [lo, hi) whose decoded text overlaps the character range."""
    lo = hi = None
    seen = 0
    for i, t in enumerate(gen_ids):
        piece = tok.decode([t], skip_special_tokens=False)
        nxt = seen + len(piece)
        if lo is None and nxt > lo_char and piece.strip():
            lo = i
        if seen >= hi_char:
            hi = i
            break
        seen = nxt
    if lo is None:
        return 0, 0
    return lo, len(gen_ids) if hi is None else hi


def slot_ranges(tok, gen_ids: list[int]) -> dict:
    """Step ranges of the two copied lines inside the generation."""
    text = tok.decode(gen_ids, skip_special_tokens=False)
    out = {}
    head, _, tail = text.partition(EVIDENCE_TAG)
    s_line = head.strip().split("\n")[0]
    if s_line:
        st = text.find(s_line)
        out["m"] = char_to_steps(tok, gen_ids, st, st + len(s_line))
    if tail:
        e_line = tail.strip().split("\n")[0]
        if e_line:
            st = text.find(e_line, len(head) + len(EVIDENCE_TAG))
            out["e"] = char_to_steps(tok, gen_ids, st, st + len(e_line))
    return out


def evidence_targets(tok, ids, prompt, a, b, record) -> list[int]:
    """Evaluation-only positions of the evidence the intervention should find, per family."""
    ans = record["answers"]
    if record["family"] == "F3" and ans.get("passage"):
        pos = []
        for cand in ans["passage"]:
            pos += K.span_positions(tok, ids, prompt, a, b, cand)
        return sorted(set(pos))
    if record["family"] == "F4" and ans.get("passage"):
        return K.span_positions(tok, ids, prompt, a, b, f'"{ans["passage"][0]}"')
    if record["family"] == "F1" and record["pressure"].get("note"):
        return K.span_positions(tok, ids, prompt, a, b, record["pressure"]["note"])
    if record["family"] == "F1" and record["direction"] == "update":
        return K.span_positions(tok, ids, prompt, a, b, record["pressure"]["sentence"])
    return []


def dual_labels(tok, ids, prompt, a, b, record) -> dict:
    """Stance positions (when the row has a stance) and evidence targets (when it has evidence)."""
    stance = record.get("stance_sentence") or ""
    has_stance = record["pressure"].get("kind") == "user_claim" and bool(stance)
    return {"stance": K.span_positions(tok, ids, prompt, a, b, stance) if has_stance else [],
            "evidence": evidence_targets(tok, ids, prompt, a, b, record),
            "question": K.span_positions(tok, ids, prompt, a, b, record["question"]) if record.get("question") else []}


def coverage(mask: list[int], target: list[int]) -> float | None:
    """Fraction of the target covered by the mask; None when there is no target."""
    if not target:
        return None
    return len(set(mask) & set(target)) / len(target)


def reply_gen_kw(args, record):
    """Generation kwargs for the reply: greedy, or per-row seeded sampling under --sample-answer."""
    if not args.sample_answer:
        return {"do_sample": False}
    torch.manual_seed(args.seed * 1000003 + int(record["row_id"].split(":")[-1]))
    return {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}


@torch.no_grad()
def call1(model, tok, patch, record, args, trailing):
    """Copy the stance and the evidence once, greedily, and localise each from the attention during its copy."""
    ids, a, b = K.build_stage1(tok, record["prompt"], PROMPTS[args.prompt], CUE, trailing)
    P = len(ids)
    patch.reset()
    patch.collect, patch.collect_upto = True, P
    watcher = DualWatcher(patch, tok, P, args.route_steps)
    if args.sample:
        # ordinary sampling with the model's own generation defaults (temperature / top_p), seeded per row for a
        # reproducible repeat; greedy stays the method's setting
        torch.manual_seed(args.seed * 1000003 + int(record["row_id"].split(":")[-1]))
        gen_kw = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    else:
        gen_kw = {"do_sample": False}
    out = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=args.route_steps + 1,
                         pad_token_id=tok.eos_token_id, logits_processor=LogitsProcessorList([watcher]), **gen_kw)
    patch.collect = False
    e = watcher.end or len(watcher.steps)
    gen_ids = [int(t) for t in out[0, P:P + e].tolist() if t != tok.eos_token_id]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    stance, evidence = parse_dual(text)
    ranges = slot_ranges(tok, gen_ids)
    masks, scores = {}, {}
    A = torch.stack(watcher.steps[:e], 0).float().numpy()[:, 0] if watcher.steps else None   # [steps, L, H, P]
    for slot, span in (("m", stance), ("e", evidence)):
        vec = np.zeros(P, np.float32)
        if A is not None and span and slot in ranges and ranges[slot][1] > ranges[slot][0]:
            lo, hi = ranges[slot]
            step_scores = A.reshape(A.shape[0], -1, P).mean(1)[lo:hi]
            if len(step_scores):
                vec = step_scores.max(0)
            vec[:a] = 0.0
            vec[b:] = 0.0
        masks[slot] = contiguous_mask(vec, args.frac, args.min_len) if span else []
        scores[slot] = np.round(vec[a:b], 5).tolist()
    labels = dual_labels(tok, ids, record["prompt"], a, b, record)
    route = "BOTH" if stance and evidence else "MEMORY" if stance else "CONTEXT" if evidence else "NONE"
    expected = {"hold": "MEMORY", "update": "CONTEXT", "both": "BOTH", "control": "NONE"}.get(record["direction"])
    return {"row_id": record["row_id"], "item_id": record["item_id"], "source": record["source"],
            "family": record["family"], "direction": record["direction"], "control_type": record.get("control_type"),
            "route": route, "route_expected": expected, "router": args.prompt,
            "stance": stance, "evidence": evidence, "raw": text,
            "stance_in_prompt": bool(stance) and K.quote_in_prompt(stance, record["prompt"]),
            "evidence_in_prompt": bool(evidence) and K.quote_in_prompt(evidence, record["prompt"]),
            "mask_m": masks["m"], "mask_e": masks["e"],
            "hit_m": coverage(masks["m"], labels.get("stance") or []), "hit_e": coverage(masks["e"], labels.get("evidence") or []),
            "request_span": [a, b], "prompt_len": P, "label_keys": labels,
            "score_m": scores["m"], "score_e": scores["e"]}


def arm_kernel(patch, mask_m: list[int], mask_e: list[int], P: int, width: int, args) -> dict:
    """Install both key sets for one row; the stance keys are removed on every query row, the evidence share is raised."""
    patch.reset()
    if not mask_m and not mask_e:
        return {"kernel": "off", "receivers": None}
    dev = args.device_obj
    band = args.layers
    if band and "=" in band:
        # per answer-format bands, e.g. "cot=0:1,short=0.5:1" (the format is read off the request's instruction, not a label)
        table = dict(kv.split("=") for kv in band.split(","))
        band = table.get("cot" if args.style == "cot_letter" else "short")
    def layer_mask(spec):
        """A [layers, heads] mask from "lo:hi[:step]"; a third field is a stride, so an all-softmax stack can be
        biased on the same fraction of its depth as the full-attention layers of a hybrid one (Gemma "3:30:4"
        mirrors Qwen's 3,7,...)."""
        parts = spec.split(":")
        lo, hi = (float(x) for x in parts[:2])
        step = int(parts[2]) if len(parts) > 2 else 1
        L = args.n_layers
        lo, hi = (int(round(lo * L)), int(round(hi * L))) if hi <= 1.0 else (int(lo), int(hi))
        m = torch.zeros(L, args.n_heads, dtype=torch.bool)
        m[lo:hi:step] = True
        return m

    band_m = args.layers_m
    if band_m and "=" in band_m:
        table = dict(kv.split("=") for kv in band_m.split(","))
        band_m = table.get("cot" if args.style == "cot_letter" else "short")
    head_mask = layer_mask(band) if (band and mask_e) else None      # the evidence bias only; the stance keys use m_mask
    m_head_mask = layer_mask(band_m) if (band_m and mask_m) else None
    allow_e = M.after_allow(mask_e, P, width, dev) if (mask_e and args.receivers_context in ("after", "prefill")) else None
    if allow_e is not None and args.receivers_context == "prefill":
        allow_e[P:] = False                        # bias the prompt rows downstream of the span only; generation reads the biased KV unbiased
    if allow_e is not None and args.first_rows:
        allow_e[P + args.first_rows:] = False      # bias the prompt rows after the span and only the first K generated rows
    allow_m = M.after_allow(mask_m, P, width, dev) if (mask_m and args.receivers_memory == "after") else None
    patch.load_batch([{"e_span": sorted(mask_e), "d_span": [], "ref_span": [], "probe_span": [],
                       "m_span": sorted(mask_m),
                       "q_allow": allow_e.tolist() if allow_e is not None else None,
                       "m_allow": allow_m.tolist() if allow_m is not None else None,
                       "head_mask": head_mask, "m_head_mask": m_head_mask}], width)
    patch.m_alpha = args.alpha_memory if args.alpha_memory > 0 else 1e-9
    patch.lin_alpha = args.lin_alpha if mask_m else 1.0          # hybrid stacks only; no-op on a dense model
    patch.lin_rho = args.lin_rho if mask_e else 0.0
    patch.min_share = args.min_share
    if mask_e and args.kernel == "share":
        patch.mode = "dynamic"
        patch.target, patch.cap = args.target_share, args.max_transfer
    else:
        patch.mode = "fixed"
        patch.alpha = args.alpha_context if mask_e else 1.0
    return {"kernel": patch.mode, "receivers": {"m": args.receivers_memory, "e": args.receivers_context},
            "target_share": patch.target if patch.mode == "dynamic" else None,
            "max_transfer": patch.cap if patch.mode == "dynamic" else None,
            "layers": band, "layers_m": band_m, "min_share": args.min_share,
            "lin_alpha": patch.lin_alpha, "lin_rho": patch.lin_rho}


def answer_names_span(tok, record: dict, g: list[int], span: str, letter_ids, suffix_ids, model, args) -> bool:
    """Whether a reply already names the copied span: the letter's option text overlaps the span (MCQ), or the
    first line of the reply contains it (short phrase). Uses only the call-1 copy, never the row's labels."""
    if not span:
        return False
    reply = tok.decode(g, skip_special_tokens=True)
    if record["instruction_style"] == "cot_letter":
        cut, _, _ = find_commitment_cut(tok, g)
        measure = K.plain_prompt_ids(tok, record["prompt"] + (BASE_CUE[record["instruction_style"]] if K.PLAIN else "")) + g[:cut] + suffix_ids
        logits = read_letter_logits(model, tok, [measure], letter_ids, model.device, 1, args.last_only)[0]
        letter = max(logits, key=logits.get)
        opts = record.get("choices") or []
        i = ord(letter) - ord("A") if len(letter) == 1 else -1
        if not (0 <= i < len(opts)):
            return False
        o, sp = K.normalise(opts[i]), K.normalise(span)
        return bool(o) and (f" {o} " in f" {sp} " or f" {sp} " in f" {o} ")
    from answer_match import first_line, phrase_match
    return phrase_match(first_line(reply), [span])


def slot_mask(tok, prompt_cue: str, plain: list[int], span: str, fallback: list[int], carry, args) -> tuple[list[int], str]:
    """Plain-prompt positions of one slot: the copied text matched back (any length), else the carried attention mask.

    A copy that is not found verbatim in the prompt is something the model invented; by default the slot is then left
    unarmed rather than localised from attention, because an amplified invention is worse than no intervention.
    """
    if not span:
        return [], "none"
    q = M.quote_span_plain(tok, prompt_cue, span, plain) if args.localiser == "quote" else []
    if q and args.occurrences == "other":
        # arm the other verbatim occurrences of the copied value (where it is used, e.g. an answer option), and only
        # fall back to the copied location itself when the value occurs nowhere else
        others = [k for k in all_occurrences(tok, prompt_cue, span, plain) if k not in set(q)]
        q = others if others else q
    elif q and (args.all_occurrences or args.occurrences == "all"):
        q = sorted(set(q) | set(all_occurrences(tok, prompt_cue, span, plain)))
    if q:
        return q, "quote"
    if not args.attention_fallback:
        return [], "abstain"
    fb = carry(fallback)
    return fb, ("attention" if fb else "none")


def char_span_positions(tok, plain: list[int], prompt: str, start: int, length: int) -> list[int]:
    """Plain-prompt token positions whose character offsets overlap prompt[start:start+length]."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    pos = K.find_slice(plain, body)
    if pos < 0:
        pos = K.find_slice(plain, body[1:-1])
        pos = pos - 1 if pos >= 0 else -1
    if pos < 0:
        return []
    return [pos + i for i, (s, e) in enumerate(offs) if e > start and s < start + length]


def all_occurrences(tok, prompt_cue: str, span: str, plain: list[int]) -> list[int]:
    """Plain-prompt positions of every verbatim occurrence of the copied span's first line (word-bounded), not only the first."""
    q = (span or "").split("\n")[0].strip().strip('"').strip()
    if len(q) < 2 or q.upper() == "NONE":
        return []
    pos = []
    bare = q.rstrip(".!?,;:").strip()
    unlabelled = re.sub(r"^[A-Za-z][\w' ]{0,24}:\s*", "", bare).strip()     # 'Note: X' / 'Background: X' -> X (the label is the request's own)
    for cand in {q, bare, unlabelled}:                                 # a copied sentence keeps its full stop; an option line has none
        if len(cand) < 2:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", prompt_cue):
            pos += K.span_positions(tok, plain, prompt_cue, 0, len(plain), cand, m.start())
    return sorted(set(pos))


def stance_referent_spans(stance: str, prompt: str) -> list[str]:
    """Texts the copied stance sentence points at as its answer: the option lines of any letters it names, and any
    quoted or trailing value it asserts. Built from the request and the call-1 copy only (no labels)."""
    if not stance:
        return []
    options = {m.group(1): m.group(2).strip() for m in re.finditer(r"(?m)^([A-H])\.\s+(.+?)\s*$", prompt)}
    refs = []
    named = set(re.findall(r"(?<![A-Za-z])(?:option|answer is|it is|it's|be|choose|pick|say|says|said|is)\s+\(?([A-H])\)?(?![A-Za-z])", stance))
    named |= set(re.findall(r"(?<![A-Za-z])\(?([B-H])\)?(?=[\s.,;:!?]|$)", stance))      # a bare 'A' is usually the article
    for L in named:
        if L in options:
            refs.append(options[L])
    return refs


LABEL_RE = re.compile(r"^\s*[A-Z][A-Za-z ]{1,24}:\s*")          # 'Note:', 'Background:', 'For reference:', 'Passage:'
OPTION_LINE_RE = re.compile(r"(?m)^([A-H])\.\s+(.+?)\s*$")


def apply_shape_rules(tok, prompt_cue: str, plain: list[int], row1: dict, mask_m: list[int], mask_e: list[int], src_e: str):
    """Request-shape knowledge applied to the two copies (no labels): a stance copy that is a labelled line, an option
    line or the closing format instruction is not a stance; an evidence copy is matched with its label stripped, and
    when the request has lettered options every option line whose text contains the copied value is armed too."""
    info = {}
    stance = (row1.get("stance") or "").split("\n")[0].strip()
    last_line = prompt_cue.rstrip().splitlines()[-1].strip() if prompt_cue.strip() else ""
    if mask_m and stance:
        st = stance.strip('"').strip()
        if LABEL_RE.match(st) or OPTION_LINE_RE.match(st) or (last_line and (st in last_line or last_line in st)):
            mask_m, info["stance_dropped"] = [], True
    ev = (row1.get("evidence") or "").split("\n")[0].strip().strip('"').strip()
    if ev:
        value = LABEL_RE.sub("", ev).strip().rstrip(".").strip()
        if value and value != ev.rstrip("."):
            pos = M.quote_span_plain(tok, prompt_cue, value, plain)
            if pos:
                mask_e, src_e, info["label_stripped"] = pos, "quote", True
        vn = K.normalise(value)
        extra = []
        for m in OPTION_LINE_RE.finditer(prompt_cue):
            on = K.normalise(m.group(2))
            if vn and on and (f" {vn} " in f" {on} " or f" {on} " in f" {vn} "):
                extra += K.span_positions(tok, plain, prompt_cue, 0, len(plain), m.group(0).strip(), m.start())
        if extra:
            mask_e = sorted(set(mask_e) | set(extra)); info["options_added"] = len(extra)
            if src_e in ("none", "abstain"):
                src_e = "quote"
    return mask_m, mask_e, src_e, info


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Character bounds of the sentence containing [start, end): back to the previous terminator, colon, newline or quote mark; forward to the next terminator, newline or quote mark."""
    left = start
    while left > 0:
        c = text[left - 1]
        if c in '\n"' or c == ":" or (c in ".!?" and text[left] == " "):
            break
        left -= 1
    while left < start and text[left] == " ":
        left += 1
    right = end
    while right < len(text):
        c = text[right]
        if c in '\n"':
            break
        right += 1
        if c in ".!?" and (right >= len(text) or text[right] in ' \n"'):
            break
    return left, right


def expand_to_sentence(tok, prompt_cue: str, plain: list[int], span: str, mask: list[int]) -> tuple[list[int], str]:
    """Replace a verbatim-located evidence span by the sentence around it; the original mask when the sentence cannot be located."""
    q = (span or "").split("\n")[0].strip()
    m = re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", prompt_cue) if len(q) >= 2 else None
    if not m:
        return mask, ""
    lo, hi = sentence_bounds(prompt_cue, m.start(), m.end())
    sent = prompt_cue[lo:hi].strip()
    if len(sent) <= len(q):
        return mask, sent
    pos = M.quote_span_plain(tok, prompt_cue, sent, plain)
    return (pos if pos else mask), sent


def passage_text(prompt: str, value: str) -> str:
    """The supplied passage of a passage-QA prompt: the paragraph holding the value, minus its label ('Passage:', 'I found this passage:') and quotation marks; '' when no paragraph holds it."""
    for para in prompt.split("\n\n"):
        if para.startswith("Question:") or not re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", para):
            continue
        body = re.sub(r"^[^:\n]{0,60}:\s*", "", para.strip()).strip()
        if len(body) > 1 and body[0] == '"' and body[-1] == '"':
            body = body[1:-1].strip()
        return body
    return ""


def passage_sentence(passage: str, value: str) -> str:
    """The sentence of the passage that holds the value (split on sentence-final punctuation only), else the passage."""
    for sent in re.split(r"(?<=[.!?])\s+", passage):
        if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", sent):
            return sent.strip()
    return passage


def oracle_spans(record: dict, kind: str) -> list[str]:
    """Gold evidence spans of one row at the requested granularity, as text to be matched back into the prompt.

    F3 (passage QA): value = the supported answer string; sentence = the passage sentence containing it; passage = the
    whole passage. F1 (MCQ with a note): note = the note line; value = the note without its label; option = the option
    line the note supports; note_option = both. Rows without evidence give no span.
    """
    prompt, ans = record["prompt"], record["answers"]
    if record["family"] == "F3":
        vals = [v for v in (ans.get("passage") or []) if v and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", prompt)]
        if not vals:
            return []
        if kind in ("value", "note"):
            return [vals[0]]
        pas = passage_text(prompt, vals[0])
        if kind == "passage":
            return [pas] if pas else []
        if kind in ("sentence", "option", "note_option"):
            return [passage_sentence(pas, vals[0])] if pas else []
        return []
    note = record.get("pressure", {}).get("note") or ""
    if not note:
        return []
    if kind in ("note", "sentence", "passage"):
        return [note]
    if kind == "value":
        return [(note.split(": ", 1)[1] if ": " in note else note).rstrip(".")]
    letters = [a for a in (ans.get("passage") or []) if isinstance(a, str) and len(a) == 1]
    opts = []
    for L in letters:
        i = ord(L) - ord("A")
        if 0 <= i < len(record.get("choices") or []):
            opts.append(f"{L}. {record['choices'][i]}")
    if kind == "option":
        return opts
    if kind == "note_option":
        return [note] + opts
    return []


@torch.no_grad()
def call2(model, tok, patch, record, row1, args, trailing, rng, letter_ids, suffix_ids):
    """Answer the plain prompt under the armed key sets and record the reply."""
    router = row1.get("router") or "file"
    ids1, a, b = K.build_stage1(tok, record["prompt"], PROMPTS.get(router) or PROMPTS["file"], CUE, trailing)
    cue = BASE_CUE[record["instruction_style"]] if K.PLAIN else ""
    plain = K.plain_prompt_ids(tok, record["prompt"] + cue)
    P = len(plain)
    base = {"row_id": record["row_id"], "item_id": record["item_id"], "source": record["source"],
            "family": record["family"], "direction": record["direction"], "control_type": record.get("control_type"),
            "gate": record.get("gate"), "arm": args.arm, "route": row1.get("route"), "route_expected": row1.get("route_expected"),
            "stance": row1.get("stance"), "evidence": row1.get("evidence"),
            "instruction_style": record["instruction_style"], "base_cue": cue,
            "memory": record["memory"], "context": record["context"], "asserted": record.get("asserted"),
            "label_map": record["label_map"], "answers": record["answers"]}
    al = K.align(ids1, a, b, plain)
    if al is None:
        return {**base, "aligned": False}
    shift, lo, hi = al
    carry = lambda ks: sorted({k + shift for k in ks if lo <= k < hi})                       # noqa: E731
    labels = {k: carry(v) for k, v in (row1.get("label_keys") or {}).items()}
    mask_m, src_m = slot_mask(tok, record["prompt"] + cue, plain, row1.get("stance") or "", row1.get("mask_m") or [], carry, args)
    mask_e, src_e = slot_mask(tok, record["prompt"] + cue, plain, row1.get("evidence") or "", row1.get("mask_e") or [], carry, args)
    if row1.get("stance_spans"):
        # a word-selection call 1 gives character spans of the chosen words: arm exactly those tokens
        pos = set()
        for s0, e0 in row1["stance_spans"]:
            pos |= set(char_span_positions(tok, plain, record["prompt"] + cue, int(s0), int(e0) - int(s0)))
        mask_m, src_m = sorted(pos), ("spans" if pos else "abstain")
    if row1.get("evidence_spans"):
        # a word-selection call 1 gives character spans of the chosen words: arm exactly those tokens (no verbatim search)
        pos = set()
        for s0, e0 in row1["evidence_spans"]:
            pos |= set(char_span_positions(tok, plain, record["prompt"] + cue, int(s0), int(e0) - int(s0)))
        mask_e, src_e = sorted(pos), ("spans" if pos else "abstain")
    if args.oracle_stance:
        # localisation oracle for the suppression side: the stance keys come from the row's own stance_sentence
        st = record.get("stance_sentence") or ""
        mask_m = sorted(M.quote_span_plain(tok, record["prompt"] + cue, st, plain)) if st else []
        src_m = ("oracle" if mask_m else "oracle_miss") if st else "none"
        base["stance"] = st
        base["oracle_stance"] = True
    if args.oracle:
        # localisation oracle: the evidence keys come from the record's own labels instead of the call-1 copy
        spans = oracle_spans(record, args.oracle)
        mask_e = sorted({k for sp in spans for k in M.quote_span_plain(tok, record["prompt"] + cue, sp, plain)})
        if args.occurrences == "other":
            others = sorted({k for sp in spans for k in all_occurrences(tok, record["prompt"] + cue, sp, plain)} - set(mask_e))
            mask_e = others if others else mask_e
        elif args.all_occurrences or args.occurrences == "all":
            mask_e = sorted(set(mask_e) | {k for sp in spans for k in all_occurrences(tok, record["prompt"] + cue, sp, plain)})
        src_e = "oracle" if mask_e else "oracle_miss"
        base["evidence"] = " || ".join(spans)
        base["oracle"] = args.oracle
    sentence = ""
    if args.expand_sentence and mask_e and src_e == "quote":
        mask_e, sentence = expand_to_sentence(tok, record["prompt"] + cue, plain, row1.get("evidence") or "", mask_e)
    base["evidence_sentence"] = sentence
    shape = {}
    if args.shape_rules:
        mask_m, mask_e, src_e, shape = apply_shape_rules(tok, record["prompt"] + cue, plain, row1, mask_m, mask_e, src_e)
    base["shape"] = shape
    referent_dropped = False
    if args.drop_stance_referent and mask_e and row1.get("stance"):
        # an evidence copy that is the very option the stance asserts is the stance's content, not supplied material
        ev_text = K.normalise((row1.get("evidence") or "").split("\n")[0])
        for ref in stance_referent_spans(row1["stance"], record["prompt"]):
            rn = K.normalise(ref)
            if rn and ev_text and (f" {rn} " in f" {ev_text} " or f" {ev_text} " in f" {rn} "):
                mask_e, src_e, referent_dropped = [], "referent", True
                break
    base["referent_dropped"] = referent_dropped
    overlap = len(set(mask_m) & set(mask_e))
    if args.overlap_neither and overlap and overlap >= 0.5 * min(len(mask_m), len(mask_e)):
        # the two single-slot copies name the same text: one of them is wrong and we cannot tell which, so arm neither
        mask_m, mask_e = [], []
        base["overlap_dropped"] = True
    mask_e = [k for k in mask_e if k not in set(mask_m)]          # the stance has priority; the sets stay disjoint
    if src_e == "attention" and len(mask_e) < args.min_len:
        mask_e = []
    if args.arm == "baseline":
        mask_m, mask_e = [], []
    elif args.arm == "suppress":
        mask_e = []
    elif args.arm == "amplify":
        mask_m = []
    elif args.arm == "random":
        # localisation control (the paper's Random arm): the same number of tokens drawn uniformly at random from the
        # request, the two slots kept disjoint; --random-window places each set as one contiguous window instead
        req0, req1 = a + shift, b + shift
        def elsewhere(L, taken=frozenset()):
            """Draw L request positions outside `taken`: scattered tokens, or one random window under --random-window."""
            L = max(L, args.min_len)
            if args.random_window:
                start = rng.randrange(req0, max(req0 + 1, req1 - L))
                return list(range(start, min(req1, start + L)))
            pool = [k for k in range(req0, req1) if k not in taken]
            return sorted(rng.sample(pool, min(L, len(pool))))
        mask_m = elsewhere(len(mask_m)) if mask_m else []
        mask_e = [k for k in elsewhere(len(mask_e), frozenset(mask_m)) if k not in set(mask_m)] if mask_e else []
        base["random_mode"] = "window" if args.random_window else "tokens"
    gen_len = GEN_TOKENS[record["instruction_style"]]
    width = P + gen_len + len(suffix_ids) + 8
    args.style = record["instruction_style"]
    gated_kept = None
    if args.gate and mask_e:
        # amplify only when needed: a decode under the stance suppression alone (plain when there is no stance) whose
        # answer already names the copied span is kept as is; otherwise the evidence keys are armed on top
        arm_kernel(patch, mask_m, [], P, width, args)
        out = model.generate(torch.tensor([plain], device=model.device), max_new_tokens=gen_len,
                             pad_token_id=tok.eos_token_id, **reply_gen_kw(args, record))
        g0 = [int(t) for t in out[0, P:].tolist()]
        if tok.eos_token_id in g0:
            g0 = g0[: g0.index(tok.eos_token_id)]
        span_text = (base.get("evidence") or "").split(" || ")[0]
        gated_kept = answer_names_span(tok, record, g0, span_text, letter_ids, suffix_ids, model, args)
        if gated_kept:
            mask_e = []
    base["gated_kept"] = gated_kept
    armed = arm_kernel(patch, mask_m, mask_e, P, width, args)

    if gated_kept:
        g = g0
    else:
        out = model.generate(torch.tensor([plain], device=model.device), max_new_tokens=gen_len,
                             pad_token_id=tok.eos_token_id, **reply_gen_kw(args, record))
        g = [int(t) for t in out[0, P:].tolist()]
        if tok.eos_token_id in g:
            g = g[: g.index(tok.eos_token_id)]
    reply = tok.decode(g, skip_special_tokens=True)
    probe, logits, cut_reason = None, None, None
    if record["instruction_style"] == "cot_letter":
        # --full-letters: read over the item's own option letters (the paper setting); without it the probe reads A-D only
        letters = [chr(65 + i) for i in range(len(record.get("choices") or []))] if args.full_letters and record.get("choices") else None
        if letters and "".join(letters) not in args.letter_cache:
            args.letter_cache["".join(letters)] = letter_token_ids(tok, ANSWER_SUFFIX, letters)
        ids_for_read = args.letter_cache["".join(letters)] if letters else letter_ids
        cut, cut_reason, _ = find_commitment_cut(tok, g, letters=letters) if letters else find_commitment_cut(tok, g)
        measure = plain + g[:cut] + suffix_ids
        logits = read_letter_logits(model, tok, [measure], ids_for_read, model.device, 1, args.last_only)[0]
        probe = max(logits, key=logits.get)
    st = {k: float(v.sum()) for k, v in patch.stats.items()}
    d_sum, d_n = float(patch.delta_sum.sum()), float(patch.delta_n.sum())
    n = st["n"]
    share = {"share_pre": round(st["e_pre"] / n, 5) if n else None,
             "share_post": round(st["e_post"] / n, 5) if n else None,
             "m_share_pre": round(st["m_pre"] / st["m_n"], 5) if st.get("m_n") else None,
             "delta_share": round(d_sum / d_n, 5) if d_n else None}
    patch.reset()
    return {**base, "aligned": True, "mask_m": mask_m, "mask_e": mask_e, "mask_m_len": len(mask_m), "mask_e_len": len(mask_e),
            "mask_m_source": src_m, "mask_e_source": src_e, "overlap": overlap,
            "hit_m": coverage(mask_m, labels.get("stance") or []) if mask_m else None,
            "hit_e": coverage(mask_e, labels.get("evidence") or []) if mask_e else None,
            "prompt_len": P, "reply": reply, "n_gen_tokens": len(g), "hit_cap": len(g) >= gen_len,
            "letter_logits": logits, "cut_reason": cut_reason, **armed, **share,
            **M.score_row(record, reply, probe)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["call1", "call2"])
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--call1", type=Path, help="call-1 rows (call2 only)")
    p.add_argument("--arm", choices=["baseline", "suppress", "amplify", "dual", "random"], default="dual")
    p.add_argument("--random-window", action="store_true",
                   help="--arm random: place each token set as one contiguous window of the request instead of drawing its positions one by one")
    p.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template (Qwen3 / Qwen3.5 thinking switch)")
    p.add_argument("--prompt-file", type=Path, required=True,
                   help="the auxiliary-call wording (prompts/call1.txt); call2 needs it too, to rebuild the call-1 layout")
    p.add_argument("--sample", action="store_true", help="call1: sample instead of greedy (model generation defaults unless overridden)")
    p.add_argument("--sample-answer", action="store_true", help="call2: sample the reply instead of greedy, seeded per row from --seed; temperature/top-p/top-k as for --sample")
    p.add_argument("--temperature", type=float, help="--sample / --sample-answer: default the model's generation_config, else 0.7")
    p.add_argument("--top-p", type=float, help="--sample / --sample-answer: default the model's generation_config, else 0.9")
    p.add_argument("--top-k", type=int, help="--sample / --sample-answer: default the model's generation_config, else 50")
    p.add_argument("--alpha-memory", type=float, default=0.0, help="multiplier on the stance keys (0 = remove)")
    p.add_argument("--alpha-context", type=float, default=4.0, help="fixed kernel: multiplier on the evidence keys")
    p.add_argument("--kernel", choices=["fixed", "share"], default="share")
    p.add_argument("--target-share", type=float, default=0.5)
    p.add_argument("--max-transfer", type=float, default=0.3)
    p.add_argument("--receivers-memory", choices=["all", "after"], default="all")
    p.add_argument("--lin-alpha", type=float, default=1.0,
                   help="hybrid stacks: multiply the linear-attention write gate of the stance positions (1 = off, 0 = never written)")
    p.add_argument("--lin-rho", type=float, default=0.0,
                   help="hybrid stacks: move the linear-attention write gate of the evidence positions towards 1 by this fraction (0 = off)")
    p.add_argument("--receivers-context", choices=["all", "after", "prefill"], default="after")
    p.add_argument("--expand-sentence", action="store_true",
                   help="widen the evidence span to the sentence that contains the copied value before arming it")
    p.add_argument("--localiser", choices=["quote", "attention"], default="quote")
    p.add_argument("--shape-rules", action="store_true", help="call2: apply request-shape knowledge (labelled lines, option lines, format line) to the two copies")
    p.add_argument("--overlap-neither", action="store_true", help="call2: when the stance and evidence copies cover the same text, arm neither slot")
    p.add_argument("--drop-stance-referent", action="store_true", help="call2: leave the evidence slot unarmed when its copy is the option the copied stance asserts")
    p.add_argument("--occurrences", choices=["first", "all", "other"], default="first",
                   help="call2: which verbatim occurrences of the copied evidence to arm: the copied one, all of them, or the others (fallback: the copied one)")
    p.add_argument("--all-occurrences", action="store_true", help="call2: arm every verbatim occurrence of the copied evidence in the request, not only the first")
    p.add_argument("--first-rows", type=int, default=0, help="evidence bias on the first K generated rows only (plus the prompt rows after the span)")
    p.add_argument("--layers", help="evidence bias only in these layers, 'lo:hi[:step]' as indices or as fractions of the depth")
    p.add_argument("--layers-m", help="the same restriction for the stance suppression; default is every layer")
    p.add_argument("--min-share", type=float, default=0.0, help="evidence bias only in heads whose unbiased share on the span is at least this")
    p.add_argument("--gate", action="store_true", help="call2: decode unbiased first and keep that reply when its answer already names the copied span")
    p.add_argument("--attention-fallback", action="store_true",
                   help="localise a slot from call-1 attention when its copy is not found verbatim (default: leave the slot unarmed)")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--gen-tokens", type=int,
                   help="override the CoT generation cap (default 400), to test whether truncation drives a result")
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--route-steps", type=int, default=160)
    p.add_argument("--families", nargs="+", help="restrict to these families (F1 / F3 / F4)")
    p.add_argument("--cells", help="comma list of control_type cells to keep")
    p.add_argument("--oracle-stance", action="store_true",
                   help="call2: take the stance span from the row's stance_sentence instead of the call-1 copy")
    p.add_argument("--oracle", choices=["value", "sentence", "passage", "note", "option", "note_option"],
                   help="call2: take the evidence keys from the record's labels at this granularity instead of the call-1 copy")
    p.add_argument("--directions", help="comma list of row directions to keep")
    p.add_argument("--shard", help="i/n")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--last-only", action="store_true")
    p.add_argument("--full-letters", action="store_true", help="call2: read the CoT letter over the item's full option set instead of A-D")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--no-think", action="store_true", help="chat template with enable_thinking=False (thinking-mode models)")
    p.add_argument("--device")
    args = p.parse_args()
    if args.no_thinking:
        K.CHAT_KWARGS["enable_thinking"] = False
    PROMPTS["file"] = args.prompt_file.read_text().strip()
    args.prompt = "file"
    K.PLAIN = args.no_chat_template
    if args.no_think:
        K.CHAT_KWARGS = {"enable_thinking": False}

    records = [json.loads(l) for l in args.conditions.read_text().splitlines() if l.strip()]
    if args.families:
        records = [r for r in records if r["family"] in args.families]
    if args.directions:
        keep = set(args.directions.split(","))
        records = [r for r in records if r.get("direction") in keep]
    if args.cells:
        keep = set(args.cells.split(","))
        records = [r for r in records if r.get("control_type") in keep]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        records = [r for j, r in enumerate(records) if j % n == i]
    records = records[: args.limit] if args.limit else records
    rows1 = {}
    if args.stage == "call2" and args.call1:
        rows1 = {json.loads(l)["row_id"]: json.loads(l) for l in args.call1.read_text().splitlines() if l.strip()}
        records = [r for r in records if r["row_id"] in rows1]
    elif args.stage == "call2":
        assert args.oracle, "call2 needs --call1 unless --oracle is set"
        rows1 = {r["row_id"]: {"router": args.prompt, "stance": "", "evidence": "", "mask_m": [], "mask_e": [],
                               "label_keys": {"stance": [], "evidence": [], "question": []}} for r in records}
    done = set()
    if args.out.exists():
        done = {json.loads(l)["row_id"] for l in args.out.read_text().splitlines() if l.strip()}
    records = [r for r in records if r["row_id"] not in done]
    print(f"{args.stage} arm={args.arm}: {len(records)} rows to run ({len(done)} already on disk)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model = model.to(device).eval()
    if args.sample or args.sample_answer:
        gc = model.generation_config
        args.temperature = args.temperature if args.temperature is not None else (gc.temperature if gc.temperature not in (None, 1.0) or gc.do_sample else 0.7)
        args.top_p = args.top_p if args.top_p is not None else (gc.top_p if gc.top_p is not None else 0.9)
        args.top_k = args.top_k if args.top_k is not None else (gc.top_k if gc.top_k is not None else 50)
        print(f"sampling: temperature {args.temperature} top_p {args.top_p} top_k {args.top_k} seed {args.seed}", flush=True)
    trailing = 0 if K.PLAIN else K.trailing_turn_tokens(tok)
    letter_ids = letter_token_ids(tok, ANSWER_SUFFIX)
    args.letter_cache = {}
    suffix_ids = tok.encode(ANSWER_SUFFIX, add_special_tokens=False)
    rng = random.Random(args.seed)
    patch = ShareBias(model)
    args.device_obj = model.device
    text_cfg = getattr(model.config, "text_config", model.config)
    if args.gen_tokens:
        GEN_TOKENS["cot_letter"] = args.gen_tokens
    args.n_layers, args.n_heads = patch.n_layers, text_cfg.num_attention_heads
    t0 = time.time()
    with patch, args.out.open("a") as fh:
        for i, rec in enumerate(records):
            if args.stage == "call1":
                row = call1(model, tok, patch, rec, args, trailing)
                if args.sample:
                    row.update({"sampled": True, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p})
                info = (f"route={row['route']} exp={row['route_expected']} hit_m={row['hit_m']} hit_e={row['hit_e']} "
                        f"stance={row['stance'][:40]!r} evidence={row['evidence'][:40]!r}")
            else:
                row = call2(model, tok, patch, rec, rows1[rec["row_id"]], args, trailing, rng, letter_ids, suffix_ids)
                if args.sample_answer:
                    row.update({"sampled_answer": True, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})
                info = (f"m={row.get('mask_m_len')}/{row.get('mask_m_source')} e={row.get('mask_e_len')}/{row.get('mask_e_source')} "
                        f"label={row.get('label')} reply={row.get('reply', '')[:40]!r}")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(records) - 1:
                print(f"[{i + 1}/{len(records)}] {time.time() - t0:.0f}s {rec['row_id']} {rec['family']}:{rec.get('control_type')} {info}", flush=True)


if __name__ == "__main__":
    main()
