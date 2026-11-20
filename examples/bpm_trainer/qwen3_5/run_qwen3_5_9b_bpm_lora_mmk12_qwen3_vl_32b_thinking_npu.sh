#!/usr/bin/env bash
# Ascend NPU entrypoint for Qwen3.5-9B / Qwen3-VL-32B-Thinking BPM on MMK12.
# Run from the repository root in the project's Ascend training environment with
# compatible torch-npu, CANN, vLLM and vllm-ascend builds supporting both models.
# Model support: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html
#
# Defaults: one 16-NPU node, 8 student NPUs + 8 teacher NPUs. Student rollout uses
# TP=2 (four replicas); the FSDP2 actor spans the 8 student NPUs. Teacher TP=2
# gives four 32B replicas. For an 8-NPU node, set TRAIN_GPUS=4 TEACHER_GPUS=4;
# memory capacity must still accommodate the configured context and batches.
# The *_GPUS names are verl's resource-count fields and count NPUs here.
#
# Paths, MMK12 conversion, thinking behavior and other knobs are documented in
# the sibling GPU launcher. Both thinking settings are inherited explicitly.
# The full-size training recipe has not been validated on NPU hardware.
set -euo pipefail

export VLLM_ASCEND_ENABLE_NZ=0
export TRAIN_GPUS=${TRAIN_GPUS:-8}
export TEACHER_GPUS=${TEACHER_GPUS:-8}

# Match the existing NPU recipes; AscendSampler needs raw sampled-token logprobs.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_5_9b_bpm_lora_mmk12_qwen3_vl_32b_thinking.sh" \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,16,64,128,512,1024,2048,3072,4096]" \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE \
    ++distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    ++distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE \
    trainer.logger='["console","tensorboard"]' \
    trainer.experiment_name=qwen3_5_9b_lora_mmk12_bpm_qwen3_vl_32b_thinking_npu \
    "$@"
