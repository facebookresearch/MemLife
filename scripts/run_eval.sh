#!/usr/bin/env bash
# Answer a benchmark from a built store, then print the numbers.
#
#   scripts/run_eval.sh <dataset> <split> <tag> <store-tag> [extra memlife args...]
#
#   scripts/run_eval.sh egolifeqa test memlife   base                              # MemLife
#   scripts/run_eval.sh egolifeqa test memlife_v base --set reader.source_video=true   # MemLife-V
#
# `tag` names this run's predictions; `store-tag` names the memory it reads. MemLife and
# MemLife-V read the SAME store -- they differ only in whether the reader may re-watch video --
# so give them one store tag and two run tags.
#
# Resume is per question, so the sharded pass and the single pass after it cooperate: the
# second picks up whatever the shards left, and it is also what marks the run complete (a
# sharded worker never can -- it only ever sees its own slice).
set -euo pipefail

DATASET=${1:?dataset}; SPLIT=${2:?split}; TAG=${3:?tag}; STORE=${4:?store-tag}; shift 4
SHARDS=${SHARDS:-1}
# Worker i gets the i-th VISIBLE GPU (see scripts/build_memory.sh).
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((SHARDS - 1)))}"

if [[ $SHARDS -gt 1 ]]; then
  for ((i = 0; i < SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} \
    python -m memlife.eval --dataset "$DATASET" --split "$SPLIT" --tag "$TAG" \
        --store-tag "$STORE" --shard "$i" --num-shards "$SHARDS" "$@" &
  done
  wait
fi
python -m memlife.eval --dataset "$DATASET" --split "$SPLIT" --tag "$TAG" \
    --store-tag "$STORE" "$@"

python -m memlife.eval.report --dataset "$DATASET" --split "$SPLIT" --tag "$TAG"
