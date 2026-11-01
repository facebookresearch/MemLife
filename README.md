# MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories

Source code and the EgoLife-EQA benchmark for our paper ([arXiv:2609.40195](https://arxiv.org/abs/2609.40195)).

**MemLife** is an episodic memory system for long-horizon egocentric video: it writes the stream
down as it is recorded and answers questions by reading that memory back. **MemOpt** is the
reinforcement-learning framework that trains the memory writer on what the reader needs.

## Overview

**Writer.** Every 30-second segment becomes one first-person memory entry, written from eight
frames and the segment's speech. Entries do not depend on any question or on each other, so a
whole recording is written in parallel and never rewritten.

**Reader.** A time-indexed agentic reader that takes up to ten rounds of tool calls:

| Tool | What it does |
|---|---|
| `search_memory(query, time_anchor, top_k)` | semantic search within a time interval |
| `fetch_memory(time_anchor)` | every entry in an interval, in time order |
| `fetch_video(time_anchor, num_frames)` | re-watch the source footage (**MemLife-V** only) |

Every tool enforces a causal cutoff, so memory written after the question was asked is never
visible.

**MemOpt.** The writer never sees a question, so MemOpt trains it with GRPO on three
signals (**FIRM**) scored by a frozen evaluator:

- **Faithfulness** (per token): is each token supported by the segment?
- **Informativeness**: does the entry still entail the answer-relevant key fact?
- **Retrievability**: would the reader's own cached searches still retrieve it?

They are multiplied per token, `x[i,t] = R_faith[i,t] · R_inf[i] · R_ret[i]`, and normalised
within each group in a length-fair way (see [`memlife/memopt/advantage.py`](memlife/memopt/advantage.py)).

## Project Structure

```
MemLife/
├── configs/
│   ├── default.yaml          # Every setting behind the reported numbers
│   ├── model/                # Qwen3.5-9B, Qwen3.6-27B
│   └── dataset/              # the four benchmarks, plus custom.yaml for your own recordings
├── datasets/egolife_eqa/     # EgoLife-EQA questions and evidence (released with this work)
├── memlife/
│   ├── core/                 # Retrieval index, answer parsing, frames, ASR, LLM engine
│   ├── data/                 # Dataset loaders and video preparation
│   ├── memory/               # MemLife: prompts, writer, store, reader and its tools, build CLI
│   ├── eval/                 # Evaluation loop, scoring, report
│   ├── memopt/               # MemOpt: FIRM rewards, advantage, key facts, replay table, trainer
│   └── demo.py               # Memory and questions for your own recordings
└── scripts/
    ├── build_memory.sh       # Write a memory store (multi-GPU)
    ├── run_eval.sh           # Answer a benchmark from a store, then report
    ├── serve_judge.sh        # Serve the frozen MemOpt evaluator with vLLM
    └── train_memopt.sh       # One MemOpt epoch, end to end
```

## Installation

```bash
pip install -e .             # MemLife
pip install -e '.[train]'    # MemOpt (adds verl)
```

Tested with Python 3.11, vLLM 0.21 and transformers 5.9 for inference, and with vLLM 0.18 and
verl 0.9.0.dev0 for training. All paths are relative to `$MEMLIFE_ROOT`, which defaults to the
repository root.

## Data

| Dataset | Questions | Source |
|---|---|---|
| [SuperMemory-VQA](https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/SuperMemory-VQA) | 4,771 | ten subjects; its train split is MemOpt's training data |
| [EgoLifeQA](https://egolife-ai.github.io/) | 500 | one seven-day recording |
| SuperMemory-LVQA | 623 | SuperMemory-VQA test questions over all ten subjects' memory; nothing extra to download |
| **EgoLife-EQA** | 100 | questions about recurring events in EgoLife; shipped in [`datasets/egolife_eqa/`](datasets/egolife_eqa/) |

Each benchmark is prepared into the same layout, documented in
[`memlife/data/prepared.py`](memlife/data/prepared.py):

```
data/<dataset>/
├── videos.json                  # the recording timeline
├── qa.json                      # the questions
└── transcripts/<video_id>.json  # speech, where available
```

Timestamps are seconds on a per-subject clock, and `query_abs_s` is the moment a question is
asked, which sets the reader's causal cutoff.

```bash
# SuperMemory-VQA: re-encode the videos, then convert the questions and transcripts
python -m memlife.data.prepare transcode --src raw/SuperMemory-VQA/data/video --dst data/supermemory/videos
python -m memlife.data.prepare supermemory --src raw/SuperMemory-VQA

# EgoLife (EgoLifeQA and EgoLife-EQA): merge the ~30 s clips into continuous recordings, and
# convert the transcripts and EgoLifeQA questions
python -m memlife.data.prepare egolife --src raw/EgoLife
```

`raw/` holds each release as downloaded from Hugging Face. SuperMemory-LVQA and EgoLife-EQA need
no extra step: they reuse these recordings, and EgoLife-EQA's questions ship in this repository.

## Usage

### Your own recordings

Turn a folder of videos into memory, then ask it questions:

```bash
python -m memlife.demo build --videos my_videos/ --out runs/demo
python -m memlife.demo ask --out runs/demo \
    "What was I holding while sitting at the dining table with a checkered tablecloth?" \
    --choices "A shopping cart" "A film clapperboard" "A whiteboard" "A smartphone"
```

```
  round 1: search_memory(query='dining table checkered tablecloth sitting holding') -> 9 entries
ANSWER: B (A film clapperboard)
```

`build` re-encodes every `*.mp4` (needs ffmpeg), places the videos back-to-back in file-name
order on one clock, and writes one memory entry per 30-second segment to `runs/demo/store/`.
To give the writer speech, pass `--transcripts DIR` with one `<video name>.json` per video, a
list of `{"start", "end", "text"}` in seconds from the start of that file.

`ask` answers as of the end of the recording, or as of `--at SECONDS`. Add `--source-video`
to let the reader re-watch footage (MemLife-V). To write memory with a writer you trained with
MemOpt (see below), add `--set model.local_path=runs/memopt/writer_eN` to `build`.

### MemLife and MemLife-V

To reproduce the benchmarks, build a memory store once, then read it with or without access to
the source video. The build takes a store name; the eval takes a name for its predictions, then
the store to read:

```bash
SHARDS=8 scripts/build_memory.sh egolifeqa test base                          # store "base"
scripts/run_eval.sh egolifeqa test memlife   base                             # MemLife
scripts/run_eval.sh egolifeqa test memlife_v base --set reader.source_video=true   # MemLife-V
```

Use `supermemory`, `supermemory_lvqa` or `egolife_eqa` in place of `egolifeqa` for the other
benchmarks. SuperMemory-LVQA shares SuperMemory-VQA's store, and EgoLife-EQA shares EgoLifeQA's;
building them under the same store name writes only the subjects not yet written.

The report prints accuracy, retrieval recall, and the parse-failure count. Stores
and predictions record the configuration that produced them, and resuming into a directory made
with different settings is refused. Any setting in [`configs/default.yaml`](configs/default.yaml)
can be overridden with `--set key=value`.

### Training MemOpt

MemOpt trains Qwen3.5-9B on the SuperMemory-VQA train split. We used one 8-GPU node for training
and one for the evaluator.

```bash
# 1. Serve the frozen evaluator; more replicas speed up reward scoring
MODEL=Qwen/Qwen3.5-9B scripts/serve_judge.sh 8100 &
export MEMLIFE_JUDGE_ENDPOINTS=http://<host>:8100       # comma-separated for several replicas

# 2. Three epochs of one run; epoch N saves its writer to runs/memopt/writer_eN
for e in 1 2 3; do scripts/train_memopt.sh $e; done

# 3. Score every epoch on the validation split
for e in 1 2 3; do
  SHARDS=8 scripts/build_memory.sh supermemory val memopt_e$e --set model.local_path=runs/memopt/writer_e$e
  scripts/run_eval.sh supermemory val memlife_memopt_e$e memopt_e$e
done
```

Each epoch rebuilds the memory with the current writer, records the frozen reader's searches,
and runs GRPO, resuming from the previous epoch's checkpoint. `python -m memlife.memopt.train
--preflight` checks the verl integration in seconds.

Pick the epoch with the best validation accuracy and recall (epoch 3 in the paper), then
evaluate it on the test split exactly as above:

```bash
E=3
SHARDS=8 scripts/build_memory.sh supermemory test memopt_e$E --set model.local_path=runs/memopt/writer_e$E
scripts/run_eval.sh supermemory test memlife_memopt   memopt_e$E
scripts/run_eval.sh supermemory test memlife_v_memopt memopt_e$E --set reader.source_video=true
```

Give each writer its own store name: a store is matched to its writer by model id, which a
local checkpoint path does not change.

## Citation

```bibtex
@article{xiong2026memlife,
  title={MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories},
  author={Xiong, Guangzhi and Zhang, Xinyuan and Yang, Xiao and Yun, Hyokun and Zhang, Kai and Kuo, Shiun-Zu and Ha, Hyeonjeong and Chen, Xilun and Sun, Kai and Liang, Lucas and others},
  journal={arXiv preprint arXiv:2609.40195},
  year={2026}
}
```

## License

MemLife is CC BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. The EgoLife-EQA
annotations are released under the same license; the EgoLife recordings remain subject to the
EgoLife project's terms.
