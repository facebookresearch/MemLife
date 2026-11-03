# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Speech transcripts: load, slice to a time window, render for a prompt.

A prepared transcript is a JSON list of ``{start, end, speaker, text}`` with times in seconds
relative to its own video file. ``memlife/data/prepare.py`` writes them; everything downstream
reads them through here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import io as mio

#: Rough budget for a rendered transcript block, at ~4 characters per token.
DEFAULT_MAX_TOKENS = 1024

#: Rendered for a genuinely silent window. Stated explicitly rather than left blank: an empty
#: quote block reads to the model as missing input rather than as silence.
NO_SPEECH = "(no speech)"


@dataclass(frozen=True)
class Turn:
    start: float          #: seconds, relative to the video file
    end: float
    text: str
    speaker: str = ""

    def render(self) -> str:
        return f"[{self.speaker}] {self.text}" if self.speaker else self.text


def load(path: Path, *, expected: bool = True) -> list[Turn]:
    """Read a prepared transcript.

    ``expected`` comes from the dataset's ``has_asr``. Where a dataset HAS audio, a missing file
    is an error rather than silence: a path drift during preparation would otherwise turn the
    speech channel off for the whole run, every segment would render ``(no speech)``, and
    nothing downstream would say so. An EMPTY file is a recording preparation found silent
    (a few EgoLife runs and one SuperMemory session are), so it loads as no turns.
    """
    p = Path(path)
    if not p.exists():
        if expected:
            raise FileNotFoundError(
                f"transcript missing: {p}. The dataset declares has_asr=true, so this is a "
                f"preparation bug, not silence.")
        return []
    return [Turn(float(r["start"]), float(r["end"]), t, r.get("speaker", ""))
            for r in mio.read_json(p)
            if (t := (r.get("text") or "").strip())]


def save(path: Path, turns: Iterable[Turn], *, allow_empty: bool = False) -> int:
    rows = [{"start": t.start, "end": t.end, "speaker": t.speaker, "text": t.text}
            for t in turns if t.text.strip()]
    if not rows and not allow_empty:
        raise ValueError(
            f"refusing to write an empty transcript to {path}: it would load as a silent "
            f"recording. Pass allow_empty=True if this recording really is silent.")
    mio.write_json(Path(path), rows)
    return len(rows)


def slice_window(turns: Sequence[Turn], t_start: float, t_end: float) -> list[Turn]:
    """Turns overlapping ``[t_start, t_end)``.

    Overlap, not containment: a sentence spanning a segment boundary is evidence for both
    segments, and requiring containment silently deletes speech at every boundary.
    """
    return [t for t in turns if t.start < t_end and t.end > t_start]


def render(turns: Sequence[Turn], max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Render as ``[speaker] text`` lines, truncated on a line boundary.

    When turns exist but not even the first line fits, that line is character-truncated rather
    than dropped: returning ``(no speech)`` there would tell the model a very talkative segment
    was silent.
    """
    turns = [t for t in turns if t.text.strip()]
    if not turns:
        return NO_SPEECH
    budget = max_tokens * 4
    out: list[str] = []
    used = 0
    for t in turns:
        line = t.render()
        if used + len(line) + 1 > budget:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out) if out else turns[0].render()[:budget]


# ------------------------------------------------------------------------------- SRT parsing

_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
                       r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})")
_SPEAKER = re.compile(r"^\s*([A-Z][A-Za-z0-9_]{0,20})\s*[:：]\s*(.+)$")
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def parse_srt(text: str, *, offset_s: float = 0.0) -> list[Turn]:
    """Parse an SRT file into turns, shifting every time by ``offset_s``.

    EgoLife cues are bilingual, a Chinese line and an English line; the English one is kept
    (about 4% of cues have no CJK-free line and keep their text as is). A leading ``Name:`` on
    the cue text becomes the speaker tag, which is what lets the writer prompt ask for people
    to be named.
    """
    out: list[Turn] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        # Locate the timing line rather than assuming it is line 2: the cue index is optional.
        i = next((j for j, ln in enumerate(lines) if _SRT_TIME.search(ln)), None)
        if i is None:
            continue
        g = [int(x) for x in _SRT_TIME.search(lines[i]).groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0 + offset_s
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0 + offset_s
        text_lines = lines[i + 1:]
        body = " ".join([ln for ln in text_lines if not _CJK.search(ln)] or text_lines)
        if not body:
            continue
        sm = _SPEAKER.match(body)
        speaker, body = (sm.group(1), sm.group(2).strip()) if sm else ("", body)
        out.append(Turn(start, end, body, speaker))
    out.sort(key=lambda t: (t.start, t.end))
    return out
