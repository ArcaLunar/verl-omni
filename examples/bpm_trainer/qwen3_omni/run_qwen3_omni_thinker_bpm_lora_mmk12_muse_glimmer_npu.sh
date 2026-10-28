#!/usr/bin/env bash
# Ascend NPU variant of run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh: same recipe,
# plus the vllm-ascend settings of the Qwen3-Omni NPU recipes in examples/gspo_trainer.
#
# Layout on one 16-NPU Ascend 910C node: student on TRAIN_GPUS=8, teacher on TEACHER_GPUS=8
# (four TP=2 replicas of Muse-Glimmer-30B on vllm-ascend). For a two-node layout, start a Ray
# cluster first (ray start --head on node 1, ray start --address=<head_ip>:<port> on node 2),
# set TRAIN_GPUS / TEACHER_GPUS to 16 and run this script on the head node.
#
# All paths and knobs are the base recipe's environment variables.
set -euo pipefail

# Same vllm-ascend weight-format setting as the Qwen3-Omni NPU recipes.
export VLLM_ASCEND_ENABLE_NZ=0
export TRAIN_GPUS=${TRAIN_GPUS:-8}
export TEACHER_GPUS=${TEACHER_GPUS:-8}

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh" \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,16,64,128,512,1024,2048,3072,4096]" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode=PIECEWISE \
    trainer.logger='["console","tensorboard"]' \
    "$@"
