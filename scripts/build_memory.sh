#!/usr/bin/env bash
# Build a MemLife memory store.
#
#   scripts/build_memory.sh <dataset> <split> <tag> [extra memlife args...]
#   scripts/build_memory.sh egolifeqa test base
#   SHARDS=8 scripts/build_memory.sh supermemory train ep0 --set model.local_path=/ckpt/ep0
#
# Two stages, because they parallelise differently:
#   write   GPU. Segments are independent, so this shards over videos and streams; SHARDS
#           workers are launched, worker i pinned to GPU i.
#   index   CPU only, one process -- it reads every part and writes one index, so sharding it
#           would make the workers race on the store directory.
set -euo pipefail

DATASET=${1:?dataset}; SPLIT=${2:?split}; TAG=${3:?tag}; shift 3
SHARDS=${SHARDS:-1}
# Worker i gets the i-th VISIBLE GPU. Under a scheduler CUDA_VISIBLE_DEVICES is already set to
# the whole allocation, so it must be indexed, not inherited -- or every worker lands on one GPU.
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((SHARDS - 1)))}"

pids=()
for ((i = 0; i < SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[i]} \
  python -m memlife.memory.build --dataset "$DATASET" --split "$SPLIT" --tag "$TAG" \
      --stage write --shard "$i" --num-shards "$SHARDS" "$@" &
  pids+=($!)
done
# Per worker: a bare `wait` returns 0 whatever the workers did, so a failed write would reach
# the index stage and surface only as "no part matching this config".
for p in "${pids[@]}"; do wait "$p"; done

python -m memlife.memory.build --dataset "$DATASET" --split "$SPLIT" --tag "$TAG" \
    --stage index "$@"
