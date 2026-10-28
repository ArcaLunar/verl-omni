#!/usr/bin/env bash
# Ascend NPU variant of run_bpm_qwen3_omni_thinker_lora_smoke.sh: tiny Qwen3-Omni Thinker
# (LoRA) student on 2 NPUs, tiny Muse-Glimmer teacher on 1 NPU served by vllm-ascend.
#
# Override via env: ASCEND_RT_VISIBLE_DEVICES (3 devices), plus the base smoke's variables.
set -euo pipefail

export DEVICE_NAME=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ASCEND_ENABLE_NZ=0
export NUM_GPUS=${NUM_GPUS:-2}
export TEACHER_GPUS=${TEACHER_GPUS:-1}

ASCEND_SET_ENV="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
# The Ascend env scripts read unset variables (e.g. ZSH_VERSION); as in the other NPU smokes,
# source them with nounset off.
set +u
if [[ -f "${ASCEND_SET_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_SET_ENV}"
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_bpm_qwen3_omni_thinker_lora_smoke.sh" \
    trainer.device=npu \
    actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    "$@"
