#!/usr/bin/env bash
# Qwen3.5-9B LoRA + MMK12, BPM from Qwen3-VL-32B-Thinking (image-in / text-out).
# Run from the repository root. Reuses the 0.8B recipe's loss, data and LoRA wiring.
#
# Teacher research (official tokenizer.json and chat_template.json):
#   https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking/tree/main
#   https://huggingface.co/Qwen/Qwen3.5-9B/tree/main
# Both use BPE with ByteLevel pre-tokenizers and decoders. Base vocabulary sizes
# are 151643 (teacher) and 248044 (student), excluding added tokens; the vocabulary
# sets differ, not just their IDs. The teacher accepts MMK12's images and has a
# dedicated Thinking checkpoint, unlike the old recipe's 4B-Instruct teacher.
#
# Thinking: the student explicitly receives enable_thinking=true. The teacher's
# official Thinking template always pre-fills <think>\n; enable_thinking=true is
# also passed to its template. BPM scores the student's sampled response, rather
# than generating a separate teacher answer. Keep teacher_generation_suffix empty:
# BPM normalizes a duplicated opening <think> at the prompt/response seam itself.
# An Instruct checkpoint is not an equivalent replacement for this teacher.
#
# Default single-node layout: 4 student GPUs (two TP=2 rollout replicas, FSDP2
# actor across all 4), 4 teacher GPUs (two TP=2 replicas). Full-size memory usage
# and convergence need validation on the target hardware; tune token budgets and
# parallelism for its memory capacity. Both engines use native vLLM.
#
# Prepare data once (raw shards: https://huggingface.co/datasets/FanqingM/MMK12):
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path /path/to/raw/mmk12 --local_save_dir ~/data/mmk12
# Use the converted train.parquet/test.parquet, not the raw dataset shards.
# Dependencies and base environment knobs are documented in the 0.8B recipe.
# STUDENT_PATH / TEACHER_PATH accept Hub IDs or local checkpoint directories;
# TRAIN_FILE / VAL_FILE default to ~/data/mmk12/{train,test}.parquet.
# Caller Hydra overrides come last and win, including the thinking settings.
set -euo pipefail

export STUDENT_PATH=${STUDENT_PATH:-Qwen/Qwen3.5-9B}
export TEACHER_PATH=${TEACHER_PATH:-Qwen/Qwen3-VL-32B-Thinking}
export TRAIN_GPUS=${TRAIN_GPUS:-4}
export TEACHER_GPUS=${TEACHER_GPUS:-4}
export TEACHER_TP=${TEACHER_TP:-2}
# Leave headroom for the teacher's different tokenization and image expansion.
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}
STUDENT_TP=${STUDENT_TP:-2}

for name in TRAIN_GPUS STUDENT_TP; do
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got '${!name}'" >&2
        exit 1
    fi
done
if (( TRAIN_GPUS % STUDENT_TP != 0 )); then
    echo "TRAIN_GPUS (${TRAIN_GPUS}) must be a multiple of STUDENT_TP (${STUDENT_TP})" >&2
    exit 1
fi

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_5_0_8b_bpm_lora_mmk12_qwen3_vl.sh" \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    +bpm.teacher_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${STUDENT_TP}" \
    trainer.experiment_name=qwen3_5_9b_lora_mmk12_bpm_qwen3_vl_32b_thinking \
    "$@"
