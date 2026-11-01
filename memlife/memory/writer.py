"""The MemLife writer. One description per 30 s segment.

**Every segment is independent.** A description is a function of that segment's frames and
transcript and nothing else -- no predecessor, no running summary, no question. Three
consequences follow, and they are the reason the design is shaped this way:

* the whole build is **embarrassingly parallel**, so it shards by video and by segment;
* a description does not depend on the stream clock, so retiming a stream (SuperMemory-LVQA)
  needs no rewrite;
* during MemOpt the writer is the policy and each segment is one training example, with no
  sequential dependency to unroll.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core import asr as masr
from ..core import frames as mframes
from . import prompts
from .segments import Segment


@dataclass
class WriterConfig:
    n_frames: int = 8
    max_tokens: int = 512
    #: Sampling overrides from ``writer.decoding``. Empty for the role's own profile.
    over: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames}")


def build_messages(segment: Segment, turns, *, cfg: WriterConfig, sample=None) -> list[dict]:
    """The exact conversation the writer sees: frames, then transcript, then the instruction.

    ``sample`` overrides how a window becomes frames, and exists for one caller: the MemOpt
    rollout dataset, which needs this function's TEXT and the POSITION of each image but hands
    the trainer frame paths instead of pixels. Reimplementing the assembly there would let the
    training prompt drift from the deployed one silently -- this seam makes "train == deploy"
    true by construction rather than by review.
    """
    if sample is None:
        sample = mframes.sample_frames
    transcript = masr.render(masr.slice_window(turns, segment.t_start, segment.t_end))
    content: list[dict] = []
    # `image_pil` is the part type vLLM accepts for an in-memory PIL image.
    for im in sample(segment.path, segment.t_start, segment.t_end, cfg.n_frames,
                     allow_pad=False):
        content.append({"type": "image_pil", "image_pil": im})
    content.append({"type": "text", "text": prompts.WRITER_USER.format(transcript=transcript)})
    return [{"role": "system", "content": prompts.WRITER_SYSTEM},
            {"role": "user", "content": content}]


def write(engine, segments: list[Segment], turns, *, cfg: WriterConfig,
          batch_size: int = 8) -> list[str]:
    """Describe a list of segments. Returns descriptions aligned to ``segments``.

    Batched, because the segments are independent -- which is the writer's whole shape.
    """
    out: list[str] = []
    for i in range(0, len(segments), batch_size):
        chunk = segments[i:i + batch_size]
        msgs = [build_messages(s, turns, cfg=cfg) for s in chunk]
        texts = engine.generate_text(msgs, "writer", max_tokens=cfg.max_tokens, **cfg.over)
        for s, t in zip(chunk, texts):
            t = t.strip()
            if not t:
                raise RuntimeError(
                    f"{s.segment_id}: the writer returned an empty description. Storing it "
                    f"would put an empty row in the index that matches nothing and is "
                    f"invisible in aggregate counts.")
            out.append(t)
    return out
