# Cross-Tokenizer On-Policy Distillation (BPM) Recipes

Last updated: 10/30/2026

Recipes for [BPM](https://github.com/verl-project/verl-omni/blob/main/docs/algo/bpm.md): on-policy distillation from a teacher whose
tokenizer differs from the student's. The student samples its own responses; the teacher
scores them after re-rendering the prompt in its own chat template, and its next-token
distribution is mapped onto the student vocabulary by byte-prefix marginalization.

| Recipe | Student / teacher | Platform | Launcher |
| --- | --- | --- | --- |
| MMK12 LoRA, pure distillation | Qwen3-Omni-30B-A3B-Instruct Thinker / Muse-Glimmer-30B | GPU | [run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh](https://github.com/verl-project/verl-omni/blob/main/examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh) |
| MMK12 LoRA, pure distillation | same | Ascend NPU | [run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer_npu.sh](https://github.com/verl-project/verl-omni/blob/main/examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer_npu.sh) |

## Qwen3-Omni Thinker from Muse-Glimmer-30B on MMK12

The loss is the BPM target alone (`use_task_rewards=false`); the MMK12 reward is computed
for logging only. Rollouts sample one response per prompt at temperature 1.0, as in the
paper.

Layout on one 8-GPU node: student rollout and actor on `TRAIN_GPUS=4`, the teacher on
`TEACHER_GPUS=4` as two vLLM replicas at `TEACHER_TP=2`. The teacher runs on verl's native
vLLM replica, because Muse-Glimmer is not in verl-omni's omni model registry; it needs a
vLLM build that serves `MuseGlimmerForConditionalGeneration` (vLLM 0.30 does).

Muse-Glimmer's generation prompt ends at `<|start|>assistant` and the model then picks a
channel: `to=self` for reasoning, `to=user` for the answer. The recipe sets
`bpm.teacher_generation_suffix=' to=user<|message|>'`, so the teacher scores each student
response as its answer to the user, exactly as the chat template renders an assistant
answer.

On Ascend NPU (910C, 16 NPUs per node) use the `_npu` launcher: it defaults to 8 NPUs for
the student and 8 for the teacher (four TP=2 replicas on vllm-ascend), and adds the
vllm-ascend settings of the Qwen3-Omni NPU recipes (`VLLM_ASCEND_ENABLE_NZ=0`,
`weight_nz_mode=0`, piecewise graphs). All other environment variables are shared.

Data preparation and dependencies follow the
[MMK12 GSPO recipe](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/README.md):

```bash
pip install math-verify qwen-vl-utils
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path <path_to_raw_mmk12> --local_save_dir ~/data/mmk12

STUDENT_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
TEACHER_PATH=/path/to/Muse-Glimmer-30B \
bash examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh
```

Environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOPK` | `64` | Teacher top-k log-probs per position. |
| `BETA` | `0.0` | `bpm.beta`; `0` is forward KL, the paper's best setting. |
| `WHITESPACE_MASK` | `all` | `bpm.whitespace_mask`; `indentation` keeps the lone-space rows Qwen tokenizers emit before numbers. |
| `TEACHER_MAX_MODEL_LEN` | `16384` | Teacher context; must hold the teacher-expanded prompt plus the response. |

Any Hydra override passed on the command line comes last and wins.

The two tokenizers are both byte-level BPE but share only part of their vocabulary
(Qwen3-Omni 151k tokens, Muse-Glimmer 202k; about half of Muse-Glimmer's tokens have a
byte-identical Qwen3-Omni token). Muse-Glimmer keeps up to three digits in one token where
Qwen splits every digit, and merges Chinese text differently, so on our math and Chinese
samples roughly one row in eight is not 1:1 and BPM's 1:N and N:1 targets are in play.

## Validation

- CPU: `tests/trainer/omni/bpm/test_bpm_recipe_on_cpu.py` composes this launcher's
  overrides and checks the teacher, loss and agent-loop wiring.
- Smoke on 3 devices with a tiny random student and teacher: `tests/special_e2e/run_bpm_qwen3_omni_thinker_lora_smoke.sh` (GPU) and
  `tests/special_e2e/run_bpm_qwen3_omni_thinker_lora_npu_smoke.sh` (Ascend NPU).
- The full-size recipe has not been run yet.
