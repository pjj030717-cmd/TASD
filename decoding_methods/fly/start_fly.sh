#!/bin/bash
set -e

export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRAFT_MODEL="/data/jjpan/experiments/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
CONFIG_PATH="/data/jjpan/experiments/decoding_methods/fly/FLy_Qwen2.5_72b.json"

fly --model hf \
    --model_args pretrained=${DRAFT_MODEL},config_path=${CONFIG_PATH} \
    --tasks humaneval_instruct \
    --batch_size 1 \
    --apply_chat_template \
    --confirm_run_unsafe_code
