"""Fixed-position letter read-out for chain-of-thought replies.

The reply is cut just before it states a conclusion (`find_commitment_cut`), the answer suffix is appended and
the logits of the option letters at that position are compared (`read_letter_logits`); the argmax letter is the
answer. `parse_freeform` recovers the letter the model wrote out itself, for diagnostics.
"""

import hashlib
import inspect
import re

import torch


ANSWER_SUFFIX = "\nAnswer:"
# Four-option default. An item may offer up to
# 13 options (TruthfulQA MC1); the runner passes the item's own letter set (ALL_LETTERS[:n]) under --full-letters.
LETTERS = ["A", "B", "C", "D"]
ALL_LETTERS = list("ABCDEFGHIJKLM")


def letter_class(letters) -> str:
    """Regex character class for a leading letter set, e.g. [ABCD]."""
    letters = list(letters)
    assert letters == ALL_LETTERS[: len(letters)], letters
    return f"[{''.join(letters)}]"

# ---------------------------------------------------------------------------
# Where to stop the reasoning trace.
#
# The suffix that carries the measurement is appended after the model's reasoning. It has to land
# after the reasoning is finished but before the model has written a letter, or the read is a copy
# of text already on the page rather than a decision being made.
# ---------------------------------------------------------------------------

# A thinking model closes its reasoning explicitly and that tag is the exact boundary. Inside the
# block the model routinely proposes and retracts answers ("so the answer is B ... wait, no"), so a
# phrase found before the tag belongs to the reasoning and must not end it.
THINK_CLOSE_RE = re.compile(r"</think>")

# Phrase-level fallback, for models that do not reason inside a thinking block at all.
#
# It must stay OFF for a thinking model. Inside the block the model constantly floats and withdraws
# candidates -- "so perhaps the answer is D, but that is the crater count, not the age ... wait" --
# and every such phrase is a hypothesis, not a decision; cutting on one of them would discard reasoning
# that still revises the letter. A generation that never closed the block simply ran out of budget
# mid-thought: nothing in it is a commitment either, so the whole trace is kept and only a trailing
# answer sentence is trimmed.
def commitment_patterns(letters=LETTERS) -> list[str]:
    """The commitment phrases, with the letter class sized to the item's option count."""
    cls = letter_class(letters)
    return [
        # A bare "answer:" also appears mid-reasoning, so it only counts when a letter follows it.
        rf"\banswer\s*[:\-]\s*\(?{cls}\b",
        r"\b(?:the|my|final)\s+(?:correct\s+)?answer\s+is\b",
        r"\bfinal answer\b",
        r"\bi(?:'| a)?m going with\b",
        r"\bi'?ll go with\b",
        r"\bi would go with\b",
        rf"\*\*\s*{cls}\s*[.\*]",
    ]


COMMITMENT_PATTERNS = commitment_patterns()
COMMITMENT_RE = re.compile("|".join(COMMITMENT_PATTERNS), re.IGNORECASE)

# Two ways a commitment phrase can belong to something other than a decision being made now.
# Attribution: the model restates someone else's claim -- "the user believes the answer is C".
# Without this the stance conditions are cut far earlier than the neutral ones and their reasoning
# lengths stop being comparable. Only non-first-person subjects count, or a genuine "I think the
# answer is B" would be skipped as well.
ATTRIBUTION_RE = re.compile(
    r"\b(?:user|users|you|your|yours|they|them|their|he|him|his|she|her|someone|somebody|"
    r"professor|teacher|person|student|prompt|problem|question|passage|text|stem|option|choice|"
    r"statement|claim|suggestion)\b",
    re.IGNORECASE,
)
# Retrospect: the model recalls a guess it made earlier -- "the initial thought was that ...".
RETROSPECT_RE = re.compile(
    r"\b(?:initial|initially|earlier|previously|originally|at first|first thought|"
    r"had thought|was thinking|used to think)\b",
    re.IGNORECASE,
)
OWNERSHIP_WINDOW = 60

# Whichever rule set the cut, the kept text must not end on a sentence that already states an
# answer, or the suffix simply completes a conclusion the model has written out.
TRAILING_COMMITMENT_PATTERNS = [
    r"\b(?:the|my|final|correct)\s+answer\s+(?:is|should\s+be|would\s+be|must\s+be|seems)",
    r"\bfinal answer\b",
    r"\bso,?\s+(?:it'?s|the answer|option)\b",
    r"\bi'?(?:ll|m| will| am)?\s*(?:go|going)\s+with\b",
    r"\bi'?m leaning (?:towards?|toward)\b",
    r"\bthe\s+(?:correct\s+)?(?:option|choice)\s+(?:is|should\s+be|would\s+be)",
    r"\bthat\s+(?:would\s+be|is|must\s+be)\s+option\b",
]
TRAILING_COMMITMENT_RE = re.compile("|".join(TRAILING_COMMITMENT_PATTERNS), re.IGNORECASE)
# The measured context must never end on an option letter, whatever the sentence around it means:
# the suffix is appended right after it and the probe would read a token already on the page. This
# also covers a verdict the phrase list does not name ("actually it should be D") and a generation
# that stopped mid-enumeration on a bare "B". Trimming an ordinary elimination sentence ("that rules
# out option C.") is the accepted cost; the sentence bound keeps it from running away, and the same
# rule applies to every arm.
TRAILING_LETTER_RE = re.compile(r"\b([ABCD])\b[\s.,):\*\-\"\']*$")


def trailing_letter_re(letters=LETTERS):
    """A lone option letter closing the text, sized to the item's option count."""
    return re.compile(rf"\b({letter_class(letters)})\b[\s.,):\*\-\"\']*$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?][\'\")\]]*\s+|\n\s*\n)")
MAX_TRAILING_SENTENCES = 4

# Used only to recover what the model itself said, for an agreement check against the fixed-position read.
FREEFORM_RE = re.compile(
    r"(?:answer\s*(?:is)?\s*[:\-]?\s*|final answer\s*[:\-]?\s*|\*\*)\(?([ABCD])\)?\b",
    re.IGNORECASE,
)


def freeform_re(letters=LETTERS):
    """The stated-answer pattern, sized to the item's option count."""
    return re.compile(
        rf"(?:answer\s*(?:is)?\s*[:\-]?\s*|final answer\s*[:\-]?\s*|\*\*)\(?({letter_class(letters)})\)?\b",
        re.IGNORECASE,
    )


_REGEX_CACHE: dict[str, tuple] = {}


def letter_regexes(letters=None) -> tuple:
    """(commitment, trailing-letter, freeform, bare-letter) regexes for one letter set; the A-D set is the module default."""
    letters = LETTERS if letters is None else list(letters)
    key = "".join(letters)
    if key not in _REGEX_CACHE:
        if letters == LETTERS:
            _REGEX_CACHE[key] = (COMMITMENT_RE, TRAILING_LETTER_RE, FREEFORM_RE, BARE_LETTER_RE)
        else:
            _REGEX_CACHE[key] = (
                re.compile("|".join(commitment_patterns(letters)), re.IGNORECASE),
                trailing_letter_re(letters),
                freeform_re(letters),
                re.compile(rf"^[\s\*\(\[\"\']*({letter_class(letters)})\b"),
            )
    return _REGEX_CACHE[key]


def letter_token_ids(tokenizer, suffix: str, letters=None) -> dict[str, int]:
    """Resolve the single token each letter becomes after the suffix, and fail loudly if it is not single."""
    base = tokenizer.encode(suffix, add_special_tokens=False)
    ids = {}
    for letter in (LETTERS if letters is None else letters):
        full = tokenizer.encode(suffix + " " + letter, add_special_tokens=False)
        if full[: len(base)] != base:
            raise ValueError(f"suffix {suffix!r} retokenizes when {letter!r} is appended; choose another suffix")
        if len(full) != len(base) + 1:
            raise ValueError(
                f"{letter!r} is not a single token after {suffix!r} "
                f"(got {tokenizer.convert_ids_to_tokens(full[len(base):])}); choose another suffix"
            )
        ids[letter] = full[-1]
    return ids


def find_commitment_match(text: str, phrase_fallback: bool, letters=None):
    """The end of the thinking block if there is one, otherwise the first phrase the model owns."""
    think = THINK_CLOSE_RE.search(text)
    if think:
        return think
    if not phrase_fallback:
        return None
    commitment_re = letter_regexes(letters)[0]
    for match in commitment_re.finditer(text):
        window = text[max(0, match.start() - OWNERSHIP_WINDOW): match.start()]
        if ATTRIBUTION_RE.search(window) or RETROSPECT_RE.search(window):
            continue
        return match
    return None


def char_to_token(tokenizer, gen_ids: list[int], target: int, upper: int) -> int:
    """Largest token count at or below upper whose decoding stops at or before target characters."""
    low, high = 0, upper
    while low < high:
        mid = (low + high + 1) // 2
        if len(tokenizer.decode(gen_ids[:mid], skip_special_tokens=False)) <= target:
            low = mid
        else:
            high = mid - 1
    return low


def last_sentence_start(text: str) -> int:
    """Character index at which the final sentence of the kept text begins."""
    stripped = len(text.rstrip())
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        if match.end() < stripped:
            start = match.end()
    return start


def trim_trailing_commitment(tokenizer, gen_ids: list[int], cut: int, letters=None) -> int:
    """Drop trailing sentences that already state an answer, so the suffix cannot complete one."""
    trailing_letter = letter_regexes(letters)[1]
    for _ in range(MAX_TRAILING_SENTENCES):
        if cut <= 0:
            return 0
        text = tokenizer.decode(gen_ids[:cut], skip_special_tokens=False)
        start = last_sentence_start(text)
        tail = text[start:]
        if not (TRAILING_COMMITMENT_RE.search(tail) or trailing_letter.search(tail.strip())):
            break
        if start == 0:
            return 0
        cut = char_to_token(tokenizer, gen_ids, start, cut)
    return cut


def find_commitment_cut(tokenizer, gen_ids: list[int], phrase_fallback: bool = False,
                        letters=None) -> tuple[int, str, int]:
    """Cut before the model states a conclusion; return the token index, the marker and tokens trimmed."""
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    match = find_commitment_match(full_text, phrase_fallback, letters)
    if match is None:
        cut, reason = len(gen_ids), "none"
    else:
        # Cutting on the marker's first character rather than on a token index matters: for a
        # pattern like "**A." the containing token would keep the letter in the measured context.
        cut = char_to_token(tokenizer, gen_ids, match.start(), len(gen_ids))
        marker = match.group(0).lower()
        # decode() is not guaranteed to be an exact prefix for every tokenizer; back off only if the
        # marker itself leaked into the tail, never on an unrelated phrase earlier in the reasoning.
        for _ in range(4):
            tail = tokenizer.decode(gen_ids[:cut], skip_special_tokens=False).lower()[-(len(marker) + 4):]
            if cut == 0 or marker not in tail:
                break
            cut -= 1
        reason = match.group(0).strip()
    before = cut
    cut = trim_trailing_commitment(tokenizer, gen_ids, cut, letters)
    return cut, reason, before - cut


# For a reply without a chain of thought the whole generation is the answer, so it is usually a bare "C" with no
# carrier phrase for FREEFORM_RE to match. Only the leading letter counts: a letter further in
# belongs to a sentence the direct instruction asked the model not to write.
BARE_LETTER_RE = re.compile(r"^[\s\*\(\[\"\']*([ABCD])\b")


def parse_freeform(text: str, allow_bare: bool = False, letters=None) -> str | None:
    """Recover the letter the model stated on its own, for the agreement check."""
    _, _, freeform, bare = letter_regexes(letters)
    matches = freeform.findall(text)
    if matches:
        return matches[-1].upper()
    if allow_bare:
        match = bare.match(text)
        if match:
            return match.group(1).upper()
    return None


def rollout_key(record: dict, rollout: int) -> str:
    """Stable identity of one rollout, used to resume an interrupted run."""
    return f"{record['item_id']}|{record['condition']}|{record['stance_position']}|{record['stance_frame']}|{rollout}"


def batch_seed(global_seed: int, keys: list[str]) -> int:
    """Derive a reproducible seed from the batch contents so a batch can be replayed exactly."""
    digest = hashlib.sha256((str(global_seed) + "".join(keys)).encode()).hexdigest()
    return int(digest[:8], 16)


def build_prompt_ids(tokenizer, prompt: str, use_chat_template: bool, prefill: str = "") -> list[int]:
    """Tokenize one prompt, opening an assistant turn so the CoT continues inside it."""
    if not use_chat_template:
        return tokenizer.encode(prompt, add_special_tokens=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    # Transformers versions differ on whether this returns ids or a BatchEncoding.
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    encoded = list(encoded)
    # The chat template of a thinking model opens <think> for the model, so an instruction alone
    # cannot stop it reasoning. Prefilling the assistant turn with the closing tag is what makes a
    # no-CoT arm possible: the model resumes after a thinking block that contains nothing.
    if prefill:
        encoded += tokenizer.encode(prefill, add_special_tokens=False)
    return encoded


def left_pad(sequences: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad so that position -1 is the real last token of every row."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
        mask[i, width - len(seq):] = 1
    return ids.to(device), mask.to(device)


def supports_logits_to_keep(model) -> bool:
    """Whether the forward pass can skip logits for every position but the last."""
    return "logits_to_keep" in inspect.signature(model.forward).parameters


@torch.no_grad()
def read_letter_logits(model, tokenizer, sequences: list[list[int]], letter_ids: dict[str, int],
                       device, micro_batch: int, last_only: bool) -> list[dict[str, float]]:
    """One forward pass per sequence batch; return the four letter logits at the final position."""
    out = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(sequences), micro_batch):
        chunk = sequences[start:start + micro_batch]
        ids, mask = left_pad(chunk, pad_id, device)
        # Materialising logits for every position costs batch x length x vocab floats and all but
        # the last row is discarded, which on a large vocabulary is the peak allocation of the run.
        kwargs = {"logits_to_keep": 1} if last_only else {}
        logits = model(input_ids=ids, attention_mask=mask, **kwargs).logits[:, -1, :].float()
        for row in logits:
            out.append({letter: row[token_id].item() for letter, token_id in letter_ids.items()})
    return out
