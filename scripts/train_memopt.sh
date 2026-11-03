#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# One MemOpt epoch, end to end.
#
#   scripts/train_memopt.sh <epoch> [extra memlife args...]
#   scripts/train_memopt.sh 1 && scripts/train_memopt.sh 2 && scripts/train_memopt.sh 3
#
# The three epochs are ONE training run, chained: each call resumes the same checkpoint
# directory (optimizer state and all, KL reference = the base model) and trains up to epoch
# <epoch>. Between them, the memory bank and the reader traces are rebuilt with the writer the
# previous epoch produced, because a stale bank would score candidates against competitors that
# no longer exist. The writer epoch N produces is merged to runs/memopt/writer_eN, under
# $MEMLIFE_ROOT (default: the current directory).
#
# Prerequisites, in this order:
#   1. prepared SuperMemory-VQA under data/supermemory/      (see README "Data")
#   2. an evaluator service, with $MEMLIFE_JUDGE_ENDPOINTS pointing at it
#      (scripts/serve_judge.sh)
#   3. verl installed
set -euo pipefail

EPOCH=${1:?epoch number, 1-based}; shift
BASE=${BASE:-Qwen/Qwen3.5-9B}
DATASET=${DATASET:-supermemory}
SPLIT=${SPLIT:-train}
NGPUS=${NGPUS:-8}
# Absolute, like memlife.paths: the reward workers run in other processes, and the frame URLs
# the evaluator is sent must be absolute.
OUT=$(realpath -m "${MEMLIFE_ROOT:-$PWD}/runs/memopt")
RUN_DIR=$OUT/${DATASET}_${SPLIT}
CKPT=$OUT/ckpt
TAG=e$EPOCH
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPUS - 1)))}"

# The reader and the evaluator are the base model throughout, as at deployment; only the
# memory bank is written by the writer under training -- greedily, as it is deployed.
ARGS=(--dataset "$DATASET" "$@")
WRITER_ARGS=("${ARGS[@]}")
if (( EPOCH > 1 )); then
  WRITER_ARGS+=(--set "model.local_path=$OUT/writer_e$((EPOCH - 1))")
fi

# -- 1. the memory bank the reader will read, written by the CURRENT writer ------------------
SHARDS=$NGPUS scripts/build_memory.sh "$DATASET" "$SPLIT" "$TAG" "${WRITER_ARGS[@]}"

# -- 2. the reader's actions over the training questions -------------------------------------
# These are what retrievability replays. The reader is frozen throughout training; only its
# memory changes, which is why this is re-run per epoch and not per step.
SHARDS=$NGPUS scripts/run_eval.sh "$DATASET" "$SPLIT" "$TAG" "$TAG" "${ARGS[@]}"

# -- 3. key facts: Q(c) and the target fact per pair -----------------------------------------
# Independent of the writer, so this is a no-op after the first epoch (it resumes per pair).
# One worker per GPU, then one pass that merges their files into keyfacts.jsonl.
pids=()
for ((i = 0; i < NGPUS; i++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[i]} \
  python -m memlife.memopt.keyfacts --split "$SPLIT" --shard "$i" --num-shards "$NGPUS" \
      "${ARGS[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python -m memlife.memopt.keyfacts --split "$SPLIT" "${ARGS[@]}"

# -- 4. the replay table ---------------------------------------------------------------------
python -m memlife.memopt.trace --split "$SPLIT" \
    --pred-tag "$TAG" --store-tag "$TAG" --out "$RUN_DIR/pack_$TAG" "${ARGS[@]}"

# -- 5. the rollout dataset -------------------------------------------------------------------
# Neither the frames nor the rows depend on the writer, so later epochs only re-check the frames.
# The rows staying fixed matters: the resumed trainer locates its epoch by dividing by the
# dataset length. Frames are extracted by one CPU worker per GPU first; the last pass writes the
# parquet.
pids=()
for ((i = 0; i < NGPUS; i++)); do
  python -m memlife.memopt.data --split "$SPLIT" --shard "$i" --num-shards "$NGPUS" \
      "${ARGS[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python -m memlife.memopt.data --split "$SPLIT" "${ARGS[@]}"

# -- 6. GRPO ----------------------------------------------------------------------------------
# Every value below is read from configs/ via memlife.config, so the training arm is described
# in one place rather than duplicated here.
eval "$(python - "${ARGS[@]}" <<'PY'
import shlex, sys
from memlife import config as mcfg
a = mcfg.add_args(__import__("argparse").ArgumentParser()).parse_args(sys.argv[1:])
c = mcfg.from_args(a)
t = c["train"]
out = {
    "MAX_PROMPT": t["max_prompt_length"], "MAX_RESP": t["max_response_length"],
    "ROLLOUT_N": t["rollout_n"], "BATCH": t["train_batch_size"],
    "MINI": t["ppo_mini_batch_size"], "MAX_TOK": t["ppo_max_token_len_per_gpu"],
    "LR": t["lr"], "WD": t["weight_decay"],
    "CLIP": t["clip_ratio"], "KL": t["kl_loss_coef"], "KLTYPE": t["kl_loss_type"],
    "ENT": t["entropy_coeff"], "GRAD_CLIP": t["grad_clip"],
    "TEMP": t["temperature"], "TOPP": t["top_p"], "TOPK": t["top_k"], "SEED": c["seed"],
}
print("\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in out.items()))
PY
)"

export MEMLIFE_PACK="$RUN_DIR/pack_$TAG"
export MEMLIFE_KEYFACTS="$RUN_DIR/keyfacts.jsonl"
export MEMLIFE_FRAMES="$RUN_DIR/frames"
# MUST equal data.max_response_length below: the token-level reward array is padded to it.
export MEMLIFE_MAX_RESPONSE_LENGTH="$MAX_RESP"

python -m memlife.memopt.train --preflight

# enable_thinking=False: the deployed writer prompt is non-thinking, and Qwen3.5's template
# otherwise opens a <think> block, so the policy would be trained on a different prompt.
# reward.num_workers is the number of concurrent faithfulness audits (memlife/memopt/reward.py).
# save_freq only has to be positive: verl always saves the run's last step, which is the end of
# this epoch. The remaining actor/ref/rollout settings only decide how the step fits in memory:
# each rollout engine sharded over two GPUs and the frozen reference copy kept on CPU (otherwise
# the engine cannot get its memory back when it wakes after the update), no
# torch.compile (minutes of CPU on the first step, then nothing gained), padding removed.
python -m memlife.memopt.train \
    algorithm.adv_estimator=memopt \
    data.train_files="$RUN_DIR/rollouts.parquet" \
    data.val_files="$RUN_DIR/rollouts.parquet" \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESP" \
    data.seed="$SEED" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$BASE" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOK" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOK" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOK" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.weight_decay="$WD" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
    actor_rollout_ref.actor.clip_ratio="$CLIP" \
    actor_rollout_ref.actor.entropy_coeff="$ENT" \
    actor_rollout_ref.actor.grad_clip="$GRAD_CLIP" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL" \
    actor_rollout_ref.actor.kl_loss_type="$KLTYPE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$TEMP" \
    actor_rollout_ref.rollout.top_p="$TOPP" \
    actor_rollout_ref.rollout.top_k="$TOPK" \
    reward_model.enable=False \
    reward.num_workers=4 \
    reward.custom_reward_function.path=pkg://memlife.memopt.reward \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.use_v1=False \
    "trainer.logger=[console]" \
    trainer.project_name=memlife \
    trainer.experiment_name=memopt \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    trainer.total_epochs="$EPOCH" \
    trainer.save_freq=1000000 \
    trainer.default_local_dir="$CKPT" \
    ${TRAINER_ARGS:-}

# -- 7. the writer for the next epoch's bank, and for evaluation ------------------------------
STEP=$(cat "$CKPT/latest_checkpointed_iteration.txt")
python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/global_step_$STEP/actor" --target_dir "$OUT/writer_e$EPOCH"
