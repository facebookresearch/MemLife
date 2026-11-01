#!/usr/bin/env bash
# Serve the MemOpt evaluator: the frozen model that scores faithfulness and informativeness.
#
#   MODEL=Qwen/Qwen3.5-9B scripts/serve_judge.sh 8100
#   export MEMLIFE_JUDGE_ENDPOINTS=http://$(hostname):8100
#
# `--allowed-local-media-path` is required: the faithfulness audit passes frames as file://
# URLs into the extracted frame tree, so the evaluator scores the same pixels the writer saw
# rather than a second, independently resized copy.
#
# The 8-image limit matches memory.frames_per_segment. Lower and every call is
# rejected; higher and nothing breaks, but nothing uses it either.
set -euo pipefail

PORT=${1:-8100}
MODEL=${MODEL:?set MODEL to the evaluator checkpoint}
FRAMES=${MEMLIFE_FRAMES:-${MEMLIFE_ROOT:-$PWD}/runs}

# --max-num-seqs and gdn_prefill_backend: see configs/model/qwen3.5-9b.yaml. vLLM's defaults
# reserve ~25 GiB of recurrent state, and pick a GDN kernel whose JIT compile hangs engine init.
exec vllm serve "$MODEL" \
    --served-model-name judge \
    --port "$PORT" \
    --dtype bfloat16 \
    --allowed-local-media-path "$FRAMES" \
    --limit-mm-per-prompt '{"image": 8}' \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --gpu-memory-utilization 0.60
