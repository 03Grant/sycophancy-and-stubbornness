"""Decoding constants and the two helpers the runner shares with the answer call.

`GEN_TOKENS` fixes the greedy generation caps (400 new tokens for chain-of-thought multiple choice, 32 for short
answers). `quote_span_plain` localises a copied line in the plain prompt (first occurrence, word-bounded; then the
normalised match of `scaffold.quote_positions`). `after_allow` is the receiver-row rule: every generated position and
the prompt rows after the span, the span's own rows excluded. `score_row` reads the reply against the context and
memory answers (letter probe for CoT rows, normalised phrase match otherwise).
"""
import re

import torch

import scaffold as K
from answer_match import ending_position, first_line, juice_normalise, phrase_match
from readout import parse_freeform


GEN_TOKENS = {"cot_letter": 400, "short_phrase": 32, "short_phrase_cued": 32, "completion": 64}

BASE_CUE = {"cot_letter": "\nLet's think step by step.", "short_phrase": "\nAnswer:",
            "short_phrase_cued": "", "completion": ""}   # _cued rows carry the answer cue in the prompt

def quote_span_plain(tok, prompt: str, span: str, plain: list[int]) -> list[int]:
    """Plain-prompt positions of the copied span: its first line matched verbatim at a word boundary (quotation marks kept), else the normalised match of scaffold.quote_positions; empty when neither finds it."""
    q = (span or "").split("\n")[0].strip()
    if len(q) < 2 or q.upper() == "NONE":
        return []
    m = re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", prompt)
    if m:
        pos = K.span_positions(tok, plain, prompt, 0, len(plain), q, m.start())
        if pos:
            return pos
    return K.quote_positions(tok, prompt, span, plain, [0, len(plain)])

def after_allow(mask: list[int], P: int, width: int, device) -> "torch.Tensor":
    """Query rows that may be intervened on: everything that can causally read the whole span, minus the span.

    Leaving the span's own rows alone is what lets the transfer raise the receiver's share without corrupting
    the source's representation.
    """
    allow = torch.zeros(width, dtype=torch.bool, device=device)
    allow[P:] = True                                    # every generated position
    allow[max(mask) + 1:P] = True                       # prefill rows downstream of the whole span
    for k in mask:
        allow[k] = False
    return allow

def score_row(record: dict, reply: str, probe: str | None) -> dict:
    """Unified context/memory scoring: the letter probe for the CoT rows, the normalised phrase matcher otherwise."""
    if record["instruction_style"] == "cot_letter":
        ctx, mem = probe in record["context"], probe in record["memory"]
        label = "both" if ctx and mem else "context" if ctx else "memory" if mem else "neither"
        return {"answer": probe, "match_context": ctx, "match_memory": mem, "label": label,
                "freeform_answer": parse_freeform(reply)}
    return score(record, reply)


def juice_score(record: dict, reply: str) -> dict:
    """The reference protocol's reading: normalised containment of any alias in the first `keep_words` words."""
    kept = " ".join(reply.split()[: record["keep_words"]])
    n = juice_normalise(kept)
    hit = lambda aliases: any(juice_normalise(a) and juice_normalise(a) in n for a in aliases)   # noqa: E731
    return {"juice_output": kept, "juice_acc": hit(record["context"]), "juice_memory": hit(record["memory"])}

def score(record: dict, reply: str) -> dict:
    """Parse the reply for one record and decide which target it matches."""
    kind = record["answer_type"]
    if kind == "phrase":
        answer = first_line(reply)
        ctx, mem = phrase_match(answer, record["context"]), phrase_match(answer, record["memory"])
    elif kind == "ending":
        answer = first_line(reply)
        pc, pm = ending_position(reply, record["context"]), ending_position(reply, record["memory"])
        ctx = pc is not None and (pm is None or pc <= pm)
        mem = pm is not None and (pc is None or pm < pc)
    else:
        m = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", reply)
        answer = m.group(1) if m else ""
        ctx, mem = answer in record["context"], answer in record["memory"]
    label = "both" if ctx and mem else "context" if ctx else "memory" if mem else "neither"
    out = {"answer": answer, "match_context": ctx, "match_memory": mem, "label": label}
    if record.get("keep_words"):
        out.update(juice_score(record, reply))
    return out
