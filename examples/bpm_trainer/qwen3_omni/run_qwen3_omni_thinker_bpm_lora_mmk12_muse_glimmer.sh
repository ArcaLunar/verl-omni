#!/usr/bin/env bash
# Qwen3-Omni Thinker + LoRA on MMK12, distilled on-policy from Muse-Glimmer-30B with BPM
# (cross-tokenizer, arXiv:2607.22334). Pure distillation: no task reward in the loss.
#
# Layout on one 8-GPU node: student rollout + actor on TRAIN_GPUS, the teacher on
# TEACHER_GPUS (TEACHER_TP GPUs per vLLM replica).
#
# The teacher scores each student response as its answer to the user: Muse-Glimmer's
# generation prompt stops at "<|start|>assistant", so the recipe appends the answer
# channel header " to=user<|message|>" (bpm.teacher_generation_suffix).
#
# Data preparation (run once):
#   pip install math-verify
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> --local_save_dir ~/data/mmk12
#
# Runtime dependencies (all Ray worker nodes): math-verify (reward logging), qwen-vl-utils.
#
# Paths (env overrides):
#   STUDENT_PATH  Qwen3-Omni-30B-A3B-Instruct checkpoint
#   TEACHER_PATH  Muse-Glimmer-30B checkpoint (its tokenizer and chat template are read from it)
#   TRAIN_FILE / VAL_FILE  MMK12 parquet files
set -euo pipefail

# Make verl_omni available to Ray workers
export VERL_USE_EXTERNAL_MODULES=verl_omni

STUDENT_PATH=${STUDENT_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TEACHER_PATH=${TEACHER_PATH:-"$HOME/models/meta-models/Muse-Glimmer-30B"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}
TRAIN_GPUS=${TRAIN_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-2}
TOPK=${TOPK:-64}
BETA=${BETA:-0.0}
WHITESPACE_MASK=${WHITESPACE_MASK:-all}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-16384}

for name in TRAIN_GPUS TEACHER_GPUS TEACHER_TP TOPK; do
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got '${!name}'" >&2
        exit 1
    fi
done
if (( TEACHER_GPUS % TEACHER_TP != 0 )); then
    echo "TEACHER_GPUS (${TEACHER_GPUS}) must be a multiple of TEACHER_TP (${TEACHER_TP})" >&2
    exit 1
fi

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${STUDENT_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.trainer.omni.bpm.agent_loop.BPMAgentLoopManagerTQ \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.distillation_loss.loss_mode=bpm \
    distillation.distillation_loss.topk="${TOPK}" \
    distillation.distillation_loss.use_task_rewards=false \
    distillation.distillation_loss.use_policy_gradient=false \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.teacher_models.teacher_model._target_=verl.workers.config.DistillationTeacherModelConfig \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_PATH}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8 \
    distillation.teacher_models.teacher_model.inference.max_model_len="${TEACHER_MAX_MODEL_LEN}" \
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False \
    bpm.beta="${BETA}" \
    bpm.whitespace_mask="${WHITESPACE_MASK}" \
    bpm.teacher_modalities="['image']" \
    bpm.teacher_generation_suffix="' to=user<|message|>'" \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=bpm \
    trainer.experiment_name=qwen3_omni_thinker_lora_mmk12_bpm_muse_glimmer \
    trainer.n_gpus_per_node="${TRAIN_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    "$@"
