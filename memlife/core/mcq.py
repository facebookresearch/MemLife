"""Multiple-choice formatting and answer parsing, for A--H and for multi-select.

Every answer -- single or multi -- is represented as the **canonical set string**: the chosen
letters, deduplicated, sorted, comma-joined with no spaces (``"C"``, ``"E,F"``). Gold is stored
the same way, so exact-match scoring is a plain string comparison and single-select behaves
exactly as it always did.

The parser is deliberately strict. A permissive predecessor fell through to "the first bare
A--E character anywhere in the response", which on truncated generations resolved to whatever
letter happened to appear first and produced a 51 %-"A" artifact. A miss must be visible as a
miss: ``None`` is a real outcome to count, never a reason to guess.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from . import sampling as _sp

#: A--H. EgoLife-EQA's frequency-counting questions carry DAY 1..DAY 7 + NONE.
LETTERS = "ABCDEFGH"

_L = f"[{LETTERS}]"
#: The separator must be REPEATABLE: "E, F, and G" puts both a comma and the word "and"
#: between two letters. At least one space or punctuation mark is required, so "EF" is not a
#: set -- and the word boundary in :func:`_letters_in` would reject it anyway.
_SEP = r"(?:[\s,;/+&]+(?:and[\s,;/+&]+)?)"
_SET = rf"{_L}(?:{_SEP}{_L})*"
_TRAIL = r"[)\]\"'*_`.,!]*"

# TWO priority classes, because neither anchor alone is right and both single-rule attempts
# were wrong in practice:
#   class 0 -- a line-initial DECLARATION, bare or labelling a restated option;
#   class 1 -- a loose mention that merely ends its line ("Based on the memory, ANSWER: C").
# Class dominates, recency breaks ties WITHIN a class. Both halves are load-bearing:
#   "ANSWER: A ... ANSWER: B) turning off the stove"         -> B  (later declaration wins)
#   "ANSWER: B\nExplanation: this contradicts answer: C"      -> B  (prose loses to a decl.)
# Ranking purely by position gives C in the second case; purely by class gives A in the first.
_DECL_RE = re.compile(
    rf"^[ \t]*(?:[-*+>]\s*|\d+[.)]\s*)?[*_`]*\s*ANSWER\s*:\s*[*_`(\[\"']*\s*"
    rf"({_SET})\s*{_TRAIL}\s*$",
    re.MULTILINE | re.IGNORECASE)
_LOOSE_RE = re.compile(rf"ANSWER\s*:\s*[*_`(\[\"']*\s*({_SET})\s*{_TRAIL}\s*$",
                       re.MULTILINE | re.IGNORECASE)
#: "ANSWER: B) turning off the stove" -- the letter labels the option it then restates.
#: SINGLE letter only: on "ANSWER: B, D" the comma would otherwise read as the label separator
#: and silently drop D.
_LABELLED_RE = re.compile(rf"^[ \t]*[*_`]*\s*ANSWER\s*:\s*[*_`]*\s*({_L})\s*[).:]\s+\S",
                          re.MULTILINE | re.IGNORECASE)
_BARE_RE = re.compile(rf"^\s*[*_`]*\s*({_SET})\s*[.):]?\s*[*_`]*\s*$", re.IGNORECASE)


def canon(letters: Iterable[str]) -> str:
    """``{'F', 'E'}`` -> ``'E,F'``. The one canonical form, shared by loaders and the parser."""
    return ",".join(sorted({str(x).strip().upper() for x in letters if str(x).strip()}))


def format_options(choices: Sequence[str]) -> str:
    """``["red", "blue"]`` -> ``"A. red\\nB. blue"``.

    Choices arrive **unprefixed**; loaders must strip any leading ``"A. "`` first, or options
    are double-prefixed.
    """
    n = len(choices)
    if not 2 <= n <= len(LETTERS):
        raise ValueError(f"expected 2..{len(LETTERS)} choices, got {n}")
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def strip_letter_prefix(choice: str) -> str:
    return re.sub(rf"^\s*[{LETTERS}]\s*[.)]\s*", "", choice)


def _letters_in(blob: str, valid: set[str]) -> str | None:
    """Every valid letter in a matched span, canonicalised; ``None`` if any is out of range.

    **Word-bounded, and that is not cosmetic.** A bare character class reads the separator
    words as answers: "ANSWER: E and F" returns "A,D,E,F" because AND contains an A and a D.

    Out-of-range is a NO-PARSE, not a silent drop: if the model answers "E, I" on an 8-way
    question it has not answered the question, and scoring the E alone would sometimes hit
    gold and inflate accuracy.
    """
    got = re.findall(rf"\b[{LETTERS}]\b", blob.upper())
    if not got or any(g not in valid for g in got):
        return None
    return canon(got)


def parse_answer(text: str, n_choices: int, *, thinking: bool = False) -> str | None:
    """Extract the answer as a canonical set string, or ``None``.

    ``n_choices`` is required. Defaulting it to the permissive maximum would let a call site
    that forgot to thread it accept ``E`` on a 4-way benchmark -- landing in the accuracy
    denominator as a wrong answer instead of in the no-parse count.

    Pass ``thinking=True`` from any call site whose generation used a thinking profile. Only
    the caller knows: the chat template seeds the opening tag, so a truncated trace has neither
    tag and the text alone cannot say. With ``thinking=True`` a missing terminator means the
    generation was cut mid-reasoning, and a tentative "my best guess is ANSWER: A" that the
    model was in the middle of rejecting must be a no-parse.
    """
    if not 2 <= n_choices <= len(LETTERS):
        raise ValueError(f"n_choices must be 2..{len(LETTERS)}, got {n_choices}")
    if not text:
        return None
    if thinking:
        i = _sp.find_think_close(text, last=True)
        if i >= 0:
            text = text[i:]
        elif (any(o in text for o in _sp.THINK_OPEN)
              or not (_DECL_RE.search(text) or _LABELLED_RE.search(text))):
            return None
        # else: the caller already stripped the trace. Composing action_region() with
        # thinking=True used to return None for every parse, silently zeroing a whole run.
    valid = set(LETTERS[:n_choices])

    best: tuple[int, int, str] | None = None
    for cls, rx in ((0, _DECL_RE), (0, _LABELLED_RE), (1, _LOOSE_RE)):
        for m in rx.finditer(text):
            cand = (-cls, m.start(), m.group(1))
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is not None:
        return _letters_in(best[2], valid)

    m = _BARE_RE.match(text)
    return _letters_in(m.group(1), valid) if m else None


def letter_to_index(letter: str) -> int:
    return LETTERS.index(letter.upper())
