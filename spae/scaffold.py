"""Chat-template scaffolds for the two calls and the verbatim localiser (copied text -> token positions).

`build_stage1` assembles the auxiliary call (instruction + "User request:" + the request in one user turn, the
assistant turn opened and the cue prefilled); `plain_prompt_ids` is the answer call's unmodified prompt. `align`
carries request positions from the first layout onto the second. `span_positions` / `quote_positions` map a copied
string back to the token positions it covers through the tokenizer's offset mapping. `contiguous_mask` is only used by
the attention-based fallback localiser, which the paper setting does not enable.
"""
import re

import numpy as np

from answer_match import normalise


CHAT_KWARGS: dict = {}     # extra chat-template kwargs, e.g. {"enable_thinking": False} for thinking-mode models

def template_ids(tok, content: str, gen: bool) -> list[int]:
    """Token ids of one user turn through the chat template."""
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=gen, tokenize=True, **CHAT_KWARGS)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)

def trailing_turn_tokens(tok) -> int:
    """How many template tokens follow the user content when the turn is closed without a generation prompt."""
    probe = template_ids(tok, "zzqx", gen=False)
    text = [tok.decode([t]) for t in probe]
    hits = [i for i, t in enumerate(text) if "zzqx" in t or "qx" in t]
    if not hits:                                             # tokenisers whose per-token decode hides the probe: count the template's tail
        s = tok.apply_chat_template([{"role": "user", "content": "zzqx"}], add_generation_prompt=False, tokenize=False, **CHAT_KWARGS)
        return len(tok.encode(s[s.rfind("zzqx") + 4:], add_special_tokens=False))
    return len(probe) - hits[-1] - 1

PLAIN = False        # set by --no-chat-template: base models get the scaffold as plain text

def build_stage1(tok, prompt: str, instruction: str, cue: str, trailing: int,
                 pre: str = "", post: str = "") -> tuple[list[int], int, int]:
    """Call-1 token ids: instruction header + request in a user turn, then the cue prefilled as the assistant turn.

    `pre` and `post` wrap the request inside the user turn (for wordings that delimit it with tags); the
    returned span still covers the request itself, so the carry to call 2 is unaffected.
    """
    header_text = instruction + "\n\nUser request:\n\n" + pre
    if PLAIN:
        ids = tok.encode(header_text + prompt + post + "\n\n" + cue, add_special_tokens=True)
        a = len(tok.encode(header_text, add_special_tokens=True))
        b = len(tok.encode(header_text + prompt, add_special_tokens=True))
        return ids, a, b
    ids = template_ids(tok, header_text + prompt + post, gen=True) + tok.encode(cue, add_special_tokens=False)
    a = len(template_ids(tok, header_text, gen=False)) - trailing
    b = len(template_ids(tok, header_text + prompt, gen=False)) - trailing
    return ids, a, b

def plain_prompt_ids(tok, prompt: str) -> list[int]:
    """Token ids of the plain prompt: chat template with the assistant turn opened, or bare text."""
    return tok.encode(prompt, add_special_tokens=True) if PLAIN else template_ids(tok, prompt, gen=True)

def find_slice(hay: list[int], needle: list[int]) -> int:
    """Start index of needle inside hay, or -1."""
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1

def span_positions(tok, ids: list[int], prompt: str, a: int, b: int, needle: str, search_from: int = 0) -> list[int]:
    """Positions inside ids[a:b] covered by the first occurrence of needle in the prompt text, by char offsets."""
    start = prompt.find(needle, search_from)
    if start < 0 or not needle:
        return []
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    pos = find_slice(ids[a:b], body)
    if pos < 0:                                   # boundary tokens differ; match on the inner run instead
        pos = find_slice(ids[a:b], body[1:-1])
        pos = pos - 1 if pos >= 0 else -1
    if pos < 0:
        return []
    return [a + pos + i for i, (s, e) in enumerate(offs) if e > start and s < start + len(needle)]

def quote_in_prompt(quote: str, prompt: str) -> bool:
    """Whether the copied phrase actually occurs in the prompt, after normalisation and with any markup the model added stripped."""
    q = normalise(re.sub(r"</?[A-Za-z][^>]*>", " ", quote).strip().strip('"').strip())
    return bool(q) and f" {q} " in f" {normalise(re.sub(r'</?[A-Za-z][^>]*>', ' ', prompt))} "

def quote_positions(tok, prompt: str, quote: str, plain: list[int], req: list[int]) -> list[int]:
    """Plain-prompt positions covered by the copied text, by normalised string match (the copy-based localiser)."""
    q = re.sub(r"</?[A-Za-z][^>]*>", " ", quote).strip().strip('"').strip()
    q = q.split("\n")[0].strip().rstrip(".").strip()
    if len(q) < 3:
        return []
    start = prompt.find(q)
    if start < 0:
        # fall back to a case-insensitive search on collapsed whitespace
        low = re.sub(r"\s+", " ", prompt.lower())
        s2 = low.find(re.sub(r"\s+", " ", q.lower()))
        if s2 < 0:
            return []
        # map the collapsed offset back approximately by walking the original text
        idx, j = 0, 0
        while idx < len(prompt) and j < s2:
            if not (prompt[idx].isspace() and idx + 1 < len(prompt) and prompt[idx + 1].isspace()):
                j += 1
            idx += 1
        start = idx
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    pos = find_slice(plain, body)
    if pos < 0:
        pos = find_slice(plain, body[1:-1])
        pos = pos - 1 if pos >= 0 else -1
    if pos < 0:
        return []
    return [pos + i for i, (s, e) in enumerate(offs) if e > start and s < start + len(q) and req[0] <= pos + i < req[1]]

def align(ids1: list[int], a: int, b: int, plain: list[int]):
    """Offset carrying call-1 request positions onto the plain prompt, checked token by token."""
    for lo_t, hi_t in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)):
        lo, hi = a + lo_t, b - hi_t
        pos = find_slice(plain, ids1[lo:hi])
        if pos >= 0:
            return pos - lo, lo, hi
    return None


def contiguous_mask(score: np.ndarray, frac: float, min_len: int) -> list[int]:
    """Positions of the run around the peak that stay above frac * peak; empty if shorter than min_len."""
    if score.size == 0 or not np.isfinite(score).any():
        return []
    p = int(np.nanargmax(score))
    thr = frac * score[p]
    lo, hi = p, p
    while lo - 1 >= 0 and score[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < len(score) and score[hi + 1] >= thr:
        hi += 1
    run = list(range(lo, hi + 1))
    return run if len(run) >= min_len else []
