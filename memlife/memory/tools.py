# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The reader's tools, and how their results are rendered back to the model.

:class:`Tools` owns everything a call needs -- the store, the stream's video timeline, the
transcripts, the read-time budgets -- so the reader stays a loop and the tools stay pure.

**Window-to-video resolution is ours, never the model's.** The model names an ABSOLUTE window;
we pick the covering video by greatest overlap, convert to file-local coordinates and clamp.
The model never learns that files exist, so it cannot ask for a window spanning two of them and
silently get frames from one.

**Every tool takes ``now``, and none may skip it.** Causality is enforced in one place, at the
index, and a window lying entirely in the future raises rather than returning ``[]`` -- ``[]``
reads as "nothing happened", which is a different claim.

**Retrieved rows carry no transcript**, and that is the deployed design rather than an
omission. Speech reaches the reader in one place only: the window it explicitly re-watches with
``fetch_video``, where it arrives in full. A 200-character fragment repeated on every one of 32
hits is worse than either silence or completeness -- the descriptions already incorporate what
was said, and the fragments crowd them out.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core import asr as masr
from ..core import frames as mframes
from ..data.schema import Stream, VideoRef
from .store import Entry, MemoryStore


class ToolError(ValueError):
    """A call that cannot be honoured. Counted as a model error, never as empty memory."""


@dataclass
class ToolConfig:
    top_k: int = 32
    fetch_memory_limit: int = 64
    #: MemLife-V. False means the reader has no pixel surface at all.
    source_video: bool = False
    fetch_video_frames: int = 30
    max_fetch_video_frames: int = 60
    #: The transcript beside ``fetch_video``'s frames, in characters. The writer's own budget,
    #: so a deliberately re-watched window is never truncated.
    asr_render_chars_video: int = 4096
    embed_device: str = "cpu"


def _cap(text: str, n: int | None) -> str:
    """Truncate to ``n`` characters, saying so. A silent cut looks like a short transcript."""
    if n is None or len(text) <= n:
        return text
    return text[:n] + f" [+{len(text) - n} chars]"


def resolve_window(videos: tuple[VideoRef, ...], a: float, b: float
                   ) -> tuple[VideoRef, float, float]:
    """Absolute window -> ``(video, file_t0, file_t1)``, by greatest overlap, clamped."""
    best, best_ov = None, 0.0
    for v in videos:
        ov = min(b, v.abs_end) - max(a, v.abs_start)
        if ov > best_ov:
            best, best_ov = v, ov
    if best is None:
        raise ToolError(f"no video covers [{a:.1f}, {b:.1f}]")
    t0 = max(0.0, min(best.duration_s, best.to_file(a)))
    t1 = max(0.0, min(best.duration_s, best.to_file(b)))
    if t1 <= t0:
        raise ToolError(f"[{a:.1f}, {b:.1f}] clamps to an empty window inside {best.video_id}")
    return best, t0, t1


class Tools:
    """One question's view of one stream."""

    def __init__(self, store: MemoryStore, stream: Stream, turns_by_video: dict[str, list],
                 *, now: float, cfg: ToolConfig):
        self.store = store
        self.stream = stream
        self.turns = turns_by_video
        self.now = float(now)
        self.cfg = cfg

    # -------------------------------------------------------------------- helpers

    def _anchor(self, raw) -> tuple[float, float] | None:
        """Validate a model-authored window. ``None`` means 'everything up to now'."""
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ToolError(f"time_anchor must be [t_start, t_end], got {raw!r}")
        try:
            a, b = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            raise ToolError(f"time_anchor values must be numbers, got {raw!r}") from None
        if b <= a:
            raise ToolError(f"time_anchor {[a, b]} is empty or inverted")
        if a >= self.now:
            raise ToolError(
                f"time_anchor {[a, b]} starts at or after the current time {self.now:.1f}; "
                f"you can only ask about the past.")
        return a, b

    def _full_window(self) -> tuple[float, float]:
        return (0.0, self.now)

    # -------------------------------------------------------------------- the tools

    def search(self, query: str, time_anchor=None, top_k: int | None = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("search_memory needs a non-empty `query`")
        anchor = self._anchor(time_anchor) or self._full_window()
        k = self.cfg.top_k if top_k is None else max(1, min(int(top_k), self.cfg.top_k))
        hits = self.store.search(query, k, now=self.now, anchor=anchor,
                                 device=self.cfg.embed_device)
        return {"tool": "search_memory", "query": query, "time_anchor": list(anchor),
                # The EFFECTIVE budget, after clamping the model's request. MemOpt's
                # retrievability replay needs it to decide whether a candidate would have been
                # returned, and recovering it from the hit count is wrong whenever the window
                # held fewer entries than the budget.
                "top_k": k,
                "hits": [self._hit_row(e, s) for e, s in hits]}

    def fetch_memory(self, time_anchor=None) -> dict:
        # A missing window is the full causal window, not an error: burning a round on "you
        # must pass time_anchor" costs a tenth of the budget.
        anchor = self._anchor(time_anchor) or self._full_window()
        entries, dropped = self.store.in_window(
            anchor, now=self.now, limit=self.cfg.fetch_memory_limit, sample="even")
        return {"tool": "fetch_memory", "time_anchor": list(anchor), "n_dropped": dropped,
                "hits": [self._hit_row(e, None) for e in entries]}

    def fetch_video(self, time_anchor, num_frames: int | None = None) -> tuple[dict, list]:
        if not self.cfg.source_video:
            raise ToolError("fetch_video is not available: this reader has no source-video "
                            "access")
        # fetch_video KEEPS requiring a window: "sample 30 frames from the whole history" is
        # not a meaningful request, and defaulting it would return the first 30 frames of a
        # multi-day recording.
        anchor = self._anchor(time_anchor)
        if anchor is None:
            raise ToolError("fetch_video needs an explicit time_anchor, e.g. [t_start, t_end]")
        # Clip to the causal window before touching pixels: a window straddling `now` must not
        # return future frames just because it started in the past.
        a, b = anchor[0], min(anchor[1], self.now)
        if b <= a:
            raise ToolError(f"time_anchor {list(anchor)} lies entirely in the future")
        n = self.cfg.fetch_video_frames if num_frames is None else int(num_frames)
        n = max(1, min(n, self.cfg.max_fetch_video_frames))
        video, t0, t1 = resolve_window(self.stream.videos, a, b)
        imgs = mframes.sample_frames(video.path, t0, t1, n)
        # NOT stripped, unlike a hit row would be: this is a window the model explicitly asked
        # about, so "(no speech)" is an answer -- it distinguishes a silent stretch from audio
        # we simply did not show.
        text = masr.render(masr.slice_window(self.turns.get(video.video_id, []), t0, t1))
        # Report what was SERVED. resolve_window clamps to one file, so a request spanning a
        # gap between two recordings yields frames from only part of it -- labelling those with
        # the full request lets the model conclude nothing happened in the part it never saw.
        return ({"tool": "fetch_video",
                 "time_anchor": [round(video.to_abs(t0), 1), round(video.to_abs(t1), 1)],
                 "requested": [round(a, 1), round(b, 1)], "num_frames": len(imgs),
                 "video_id": video.video_id, "transcript": text}, imgs)

    # -------------------------------------------------------------------- rows

    @staticmethod
    def _hit_row(e: Entry, score: float | None) -> dict:
        return {"id": e.segment_id, "video_id": e.video_id,
                "t_start": round(e.abs_start, 1), "t_end": round(e.abs_end, 1),
                "score": None if score is None else round(score, 4),
                "text": e.description}

    # -------------------------------------------------------------------- dispatch

    def call(self, name: str, args: dict) -> tuple[dict, list]:
        """Execute one parsed action. Returns ``(result, frames)``."""
        if not isinstance(args, dict):
            raise ToolError(f"args must be an object, got {type(args).__name__}")
        if name == "search_memory":
            return self.search(args.get("query", ""), args.get("time_anchor"),
                               args.get("top_k")), []
        if name == "fetch_memory":
            return self.fetch_memory(args.get("time_anchor")), []
        if name == "fetch_video":
            return self.fetch_video(args.get("time_anchor"), args.get("num_frames"))
        raise ToolError(f"unknown tool {name!r}")


# ------------------------------------------------------------------------ rendering

def render(result: dict, *, frames_attached: bool = False, n_frames_dropped: int = 0,
           asr_chars: int = 4096) -> str:
    """Render a tool result for the model. A pure function of the result dict."""
    name = result["tool"]

    if name == "fetch_video":
        head = (f"fetch_video(time_anchor={result['time_anchor']}, "
                f"num_frames={result['num_frames']}, video_id={result['video_id']})")
        asr = _cap(result.get("transcript") or "", asr_chars)
        tail = f"\n  asr: {asr}" if asr else ""
        if frames_attached:
            return head + "\n  [frames attached]" + tail
        return (head + " (frames dropped)" + tail
                + f"\n  [{n_frames_dropped or result['num_frames']} frames omitted]")

    hits = result.get("hits", [])
    if name == "search_memory":
        lines = [f"search_memory(query={result['query']!r}) -> {len(hits)} hits, in time order"]
    else:
        lines = [f"fetch_memory(time_anchor={result['time_anchor']}) -> {len(hits)} hits"]
    for h in hits:
        score = f"|score={h['score']}" if h["score"] is not None else ""
        lines.append(f"  [{h['id']}|video={h['video_id']}|"
                     f"t={h['t_start']}-{h['t_end']}{score}] {h['text']}")
    # An enumeration that hit its cap MUST say so: "the last 64 of 500" rendered as a bare list
    # is how a model concludes nothing happened in the part it was never shown.
    if result.get("n_dropped"):
        lines.append(f"  [{result['n_dropped']} more entries in this window not shown; they "
                     f"are spread between the ones above -- narrow the window to see every one]")
    return "\n".join(lines)


# ------------------------------------------------------------------------ action parsing

def parse_action(text: str, *, final_round: bool = False):
    """Parse one action into ``("tool_name", {...})`` or ``("answer", "<raw>")``, or None.

    **A tool call outranks an answer, except on the final round.** A response like
    ``ANSWER: A - actually, let me verify.\\nTOOL: {...}`` states an intention to keep looking;
    taking the A commits to a letter the model was in the middle of retracting. On the final
    round there is no next round, so the letter wins.

    Among answers the LAST wins: ``ANSWER: A ... on reflection ANSWER: C`` resolves to C.

    The JSON is extracted by brace matching rather than a regex: a query containing ``}`` is
    normal, and a non-greedy regex truncates the object mid-string.
    """
    import json

    answer = None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ANSWER:"):
            answer = s[len("ANSWER:"):].strip()
            continue
        if not s.upper().startswith("TOOL:"):
            continue
        body = s[len("TOOL:"):].strip()
        j = body.find("{")
        if j < 0:
            continue
        depth, in_str, esc = 0, False, False
        for k, ch in enumerate(body[j:], start=j):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(body[j:k + 1])
                    except json.JSONDecodeError:
                        break
                    nm, args = d.get("name"), d.get("args") or {}
                    if isinstance(nm, str) and isinstance(args, dict):
                        if not (final_round and answer is not None):
                            return nm, args
                    break
    return ("answer", answer) if answer is not None else None
