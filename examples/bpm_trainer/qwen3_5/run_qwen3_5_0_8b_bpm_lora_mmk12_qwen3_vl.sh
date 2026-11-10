#!/usr/bin/env bash
# Qwen3.5-0.8B (LoRA) on MMK12, distilled on-policy from Qwen3-VL-4B-Instruct with BPM
# (cross-tokenizer, arXiv:2607.22334). Pure distillation: no task reward in the loss.
#
# Layout on one 8-GPU node: student rollout + actor on TRAIN_GPUS, the teacher on
# TEACHER_GPUS (TEACHER_TP GPUs per vLLM replica).
#
# The two roles run on different engines. The student has no vllm-omni pipeline (vLLM-Omni
# only ships `qwen3_omni` / `qwen3_tts`), so it uses verl's native vLLM rollout. The teacher
# is a plain vision-language model outside verl-omni's omni registry, so it runs on verl's
# native vLLM teacher replica too (the `_target_` override below is what lets a non-vllm_omni
# teacher coexist with a top-k loss).
#
# The teacher re-renders the prompt with its own chat template and, being a VLM, gets the
# sample's images: Qwen3-VL and Qwen3.5 both use Qwen2VL-style media placeholders, so the
# teacher tokenizes `<|vision_start|><|image_pad|><|vision_end|>` and lets vLLM expand it.
#
# Seam: Qwen3.5's generation prompt pre-fills "<think>\n" while Qwen3-VL's stops at
# "<|im_start|>assistant\n", so BPM's seam normalization appends "<think>\n" to the teacher
# prompt and the teacher scores the student's own response from the same byte prefix. Keep
# the default chat-template kwargs (thinking on): passing enable_thinking=false changes what
# the student generates and breaks the seam.
#
# Vocabulary: the student tokenizer (Qwen3.5, 248k) is the finer one and the teacher's
# (Qwen3-VL, 152k) is coarser, so a given top-k covers more teacher mass per row than with a
# 248k teacher. Raise TOPK if distillation/bpm_q_other stays high.
#
# Data preparation (run once):
#   pip install math-verify
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> --local_save_dir ~/data/mmk12
#
# Runtime dependencies (all Ray worker nodes): math-verify (reward logging), qwen-vl-utils, and --
# for the student's Gated-DeltaNet layers -- flash-linear-attention plus causal-conv1d (without
# them Transformers falls back to a much slower torch path).
#
# Paths (env overrides; a Hugging Face ID and a local checkpoint dir are both accepted):
#   STUDENT_PATH  default Qwen/Qwen3.5-0.8B
#   TEACHER_PATH  default Qwen/Qwen3-VL-4B-Instruct (its tokenizer, processor and chat template are read
#                 from it; Qwen/Qwen3-VL-4B-Thinking is a drop-in alternative)
#   TRAIN_FILE / VAL_FILE  default the single train/test parquet files written by the data preparation above
set -euo pipefail

# Make verl_omni available to Ray workers (also activates the qwen3_5 adapter registration)
export VERL_USE_EXTERNAL_MODULES=verl_omni

STUDENT_PATH=${STUDENT_PATH:-Qwen/Qwen3.5-0.8B}
TEACHER_PATH=${TEACHER_PATH:-Qwen/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}
TRAIN_GPUS=${TRAIN_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-1}
TOPK=${TOPK:-64}
BETA=${BETA:-0.0}
WHITESPACE_MASK=${WHITESPACE_MASK:-all}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-16384}
# Qwen3.5-0.8B is a hybrid Gated-DeltaNet + full-attention VLM (18 of its 24 decoder layers are
# linear attention), so the linear-attention projections are listed too. `visual` is excluded:
# LoRA leaves the frozen vision tower untouched.
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']"}

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
    actor_rollout_ref.model.exclude_modules=".*visual.*" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
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
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=bpm \
    trainer.experiment_name=qwen3_5_0_8b_lora_mmk12_bpm_qwen3_vl \
    trainer.n_gpus_per_node="${TRAIN_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    "$@"
