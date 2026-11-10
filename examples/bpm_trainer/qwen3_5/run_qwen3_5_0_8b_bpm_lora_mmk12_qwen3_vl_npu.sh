#!/usr/bin/env bash
# Ascend NPU variant of run_qwen3_5_0_8b_bpm_lora_mmk12_qwen3_vl.sh: same recipe, plus the
# vllm-ascend settings of the Qwen3-Omni NPU recipes in examples/gspo_trainer.
#
# Layout on one 16-NPU Ascend 910C node: student on TRAIN_GPUS=8, teacher on TEACHER_GPUS=8
# (eight TP=1 replicas of Qwen3-VL-4B-Instruct on vllm-ascend; pass TEACHER_TP=2 for four TP=2
# replicas). For a two-node layout, start a Ray cluster first (ray start --head on node 1,
# ray start --address=<head_ip>:<port> on node 2), set TRAIN_GPUS / TEACHER_GPUS to 16, and
# run this script on the head node.
#
# Both roles run verl's native vLLM rollout here, so the vllm-ascend settings go to
# engine_kwargs.vllm on the student rollout and on the teacher's inference block.
#
# All paths and knobs are the base recipe's environment variables.
set -euo pipefail

# Same vllm-ascend weight-format setting as the Qwen3-Omni NPU recipes; Ray workers inherit it.
export VLLM_ASCEND_ENABLE_NZ=0
export TRAIN_GPUS=${TRAIN_GPUS:-8}
export TEACHER_GPUS=${TEACHER_GPUS:-8}

# vllm-ascend's AscendSampler does not implement "processed_logprobs" yet (its own default is
# "raw_logprobs"): on NPU the processed path yields non-finite sampled-token logprobs. Force
# vLLM's raw mode for the student rollout.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_5_0_8b_bpm_lora_mmk12_qwen3_vl.sh" \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,16,64,128,512,1024,2048,3072,4096]" \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE \
    ++distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    ++distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE \
    trainer.logger='["console","tensorboard"]' \
    "$@"
