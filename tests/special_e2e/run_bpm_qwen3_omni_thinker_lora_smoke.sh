#!/usr/bin/env bash
# BPM cross-tokenizer distillation e2e smoke: tiny Qwen3-Omni Thinker (LoRA) student,
# tiny Muse-Glimmer teacher served by verl's native vLLM replica.
#
# The two tiny models carry differently trained tokenizers, so the run exercises BPM's
# cross-tokenizer alignment end to end: teacher prompt re-rendering, top-k teacher
# prefill, CPU target construction in the agent loop, the TransferQueue payload, the
# actor-side BPM loss, and the row-kind census.
#
# GPUs: NUM_GPUS for the student (FSDP2 needs at least 2 to shard the LoRA sync),
# plus TEACHER_GPUS for the teacher pool, i.e. 3 by default.
#
# Requires: verl, verl-omni, vllm-omni installed.
# Override via env: NUM_GPUS, TEACHER_GPUS, STUDENT_PATH, TEACHER_PATH, DATA_DIR,
#                   TOTAL_TRAIN_STEPS, LOG_FILE
set -euo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_USE_EXTERNAL_MODULES=verl_omni

NUM_GPUS=${NUM_GPUS:-2}
TEACHER_GPUS=${TEACHER_GPUS:-1}
STUDENT_PATH=${STUDENT_PATH:-${HOME}/models/tiny-random/Qwen3-Omni}
TEACHER_PATH=${TEACHER_PATH:-${HOME}/models/tiny-random/Muse-Glimmer}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
LOG_FILE=${LOG_FILE:-$(mktemp -t bpm_smoke.XXXXXX.log)}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ── Build tiny models (no Hub access) and dummy data ──────────────────────────
python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" --output-dir "${STUDENT_PATH}" --force
python3 "${REPO_ROOT}/tests/special_e2e/build_muse_glimmer_tiny_random.py" --output-dir "${TEACHER_PATH}" --force
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" --local_save_dir "${DATA_DIR}"
fi

# ── Run V1 training with BPM (2 steps) ────────────────────────────────────────
python3 -m verl_omni.trainer.main_omni \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=4 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${STUDENT_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.trainer.omni.bpm.agent_loop.BPMAgentLoopManagerTQ \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.distillation_loss.loss_mode=bpm \
    distillation.distillation_loss.topk=16 \
    distillation.distillation_loss.use_task_rewards=false \
    distillation.distillation_loss.use_policy_gradient=false \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.teacher_models.teacher_model._target_=verl.workers.config.DistillationTeacherModelConfig \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_PATH}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3 \
    distillation.teacher_models.teacher_model.inference.max_model_len=1024 \
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False \
    bpm.beta=0.5 \
    bpm.teacher_generation_suffix="' to=user<|message|>'" \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=bpm-qwen3-omni-thinker-lora-e2e \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

# The tiny tokenizers segment differently, so spanning (N:1) rows must have been trained.
for metric in distillation/loss distillation/bpm_rows_one_to_one; do
    if ! grep -q "${metric}" "${LOG_FILE}"; then
        echo "BPM smoke failed: ${metric} missing from ${LOG_FILE}" >&2
        exit 1
    fi
done
n_to_one=$(grep -oE "bpm_rows_n_to_one:[0-9.eE+-]+" "${LOG_FILE}" | tail -n 1 | cut -d: -f2)
if ! awk -v share="${n_to_one:-0}" 'BEGIN { exit !(share > 0) }'; then
    echo "BPM smoke failed: no N:1 rows were trained (bpm_rows_n_to_one=${n_to_one:-missing})" >&2
    exit 1
fi
echo "BPM Qwen3-Omni Thinker + Muse-Glimmer teacher e2e smoke passed (log: ${LOG_FILE})."
