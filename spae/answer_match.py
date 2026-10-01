"""Pure text matching shared by every scorer: normalisation, answer extraction, target matching.

Pure Python (no torch), shared by the runner and the scorer.
"""

import re
import string
import unicodedata
from string import ascii_uppercase

NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                     "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"})
ARTICLES = {"a", "an", "the"}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and articles, spell numbers as digits, collapse spaces."""
    text = text.lower().replace("-", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = [NUMBER_WORDS.get(w, w) for w in text.split() if w not in ARTICLES]
    return " ".join(words)


def first_line(reply: str) -> str:
    """The answer proper: the first non-empty line, without a leading 'Answer:' label."""
    for line in reply.strip().splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^(?:answer|a)\s*[:\-]\s*", "", line, flags=re.IGNORECASE).strip()
    return ""


def phrase_match(answer: str, aliases: list[str]) -> bool:
    """Containment of any normalised alias in the normalised answer (or the reverse for one-word answers)."""
    a = normalise(answer)
    if not a:
        return False
    for alias in aliases:
        n = normalise(alias)
        if n and (f" {n} " in f" {a} " or (len(a.split()) <= 2 and f" {a} " in f" {n} ")):
            return True
    return False


def ending_position(reply: str, aliases: list[str]) -> int | None:
    """Character offset of the earliest alias in the reply, as whole words, or None."""
    best = None
    for alias in aliases:
        n = normalise(alias)
        if not n:
            continue
        m = re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", normalise(reply))
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def juice_normalise(text: str) -> str:
    """The reference implementation's normaliser: lowercase, strip diacritics and non-word characters, collapse spaces.

    Deliberately not `normalise`: that one drops articles, which would erase a one-letter answer such as "A".
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Letter answers -------------------------------------------------------
# `readout.py` reads a letter from a fixed position with a logit probe, which needs the letter
# set up front. These recover the letter the model itself wrote, for any option count, so a source
# with more (or fewer) than four options needs no special case.

def letter_class(n_options: int) -> str:
    """The regex character class covering exactly the letters an n-option item offers."""
    if not 2 <= n_options <= len(ascii_uppercase):
        raise ValueError(f"an MCQ needs 2..{len(ascii_uppercase)} options, got {n_options}")
    return f"[A-{ascii_uppercase[n_options - 1]}]" if n_options > 1 else "[A]"


def parse_letter(text: str, n_options: int, allow_bare: bool = False) -> str | None:
    """The last letter the model stated behind an answer phrase, or a leading bare letter."""
    cls = letter_class(n_options)
    carried = re.findall(
        rf"(?:answer\s*(?:is)?\s*[:\-]?\s*|final answer\s*[:\-]?\s*|\*\*)\(?({cls})\)?\b",
        text, re.IGNORECASE)
    if carried:
        return carried[-1].upper()
    if allow_bare:
        m = re.match(rf"^[\s\*\(\[\"\']*({cls})\b", text)
        if m:
            return m.group(1).upper()
    return None


# --- Commitment ------------------------------------------------------------
# A reasoning model's reply is mostly deliberation, and deliberation restates the prompt. Matching a
# target inside it scores the model for quoting the question. Every reply must therefore be cut down
# to what the model actually committed to, and a reply that never committed has to be visible as such
# rather than scored off its thinking.

COMMIT_MARKERS = (
    r"</think>",
    r"\banswer\s*[:\-]",
    r"\b(?:the|my|final)\s+(?:correct\s+)?answer\s+is\b",
    r"\bfinal answer\b",
)
COMMIT_RE = re.compile("|".join(COMMIT_MARKERS), re.IGNORECASE)


def commit_cut(reply: str) -> tuple[str, bool]:
    """The text after the model stopped deliberating, and whether it ever did."""
    last = None
    for m in COMMIT_RE.finditer(reply):
        last = m
    if last is None:
        return reply, False
    return reply[last.end():].strip(), True
