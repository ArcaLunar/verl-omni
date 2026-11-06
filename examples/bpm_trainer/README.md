# Cross-Tokenizer On-Policy Distillation (BPM) Recipes

Last updated: 11/02/2026

Recipes for [BPM](https://github.com/verl-project/verl-omni/blob/main/docs/algo/bpm.md): on-policy distillation from a teacher whose
tokenizer differs from the student's. The student samples its own responses; the teacher
scores them after re-rendering the prompt in its own chat template, and its next-token
distribution is mapped onto the student vocabulary by byte-prefix marginalization.

| Recipe | Student / teacher | Platform | Launcher |
| --- | --- | --- | --- |
| MMK12 LoRA, pure distillation | Qwen3-Omni-30B-A3B-Instruct Thinker / Qwen3.5-35B-A3B | GPU | [run_qwen3_omni_thinker_bpm_lora_mmk12_qwen3_5.sh](https://github.com/verl-project/verl-omni/blob/main/examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_qwen3_5.sh) |
| MMK12 LoRA, pure distillation | same | Ascend NPU | [run_qwen3_omni_thinker_bpm_lora_mmk12_qwen3_5_npu.sh](https://github.com/verl-project/verl-omni/blob/main/examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_qwen3_5_npu.sh) |

## Qwen3-Omni Thinker from Qwen3.5-35B-A3B on MMK12

The loss is the BPM target alone (`use_task_rewards=false`); the MMK12 reward is computed
for logging only. Rollouts sample one response per prompt at temperature 1.0, as in the
paper.

Layout on one 8-GPU node: student rollout and actor on `TRAIN_GPUS=4`, the teacher on
`TEACHER_GPUS=4` as two vLLM replicas at `TEACHER_TP=2`. The teacher runs on verl's native
vLLM replica, because Qwen3.5-35B-A3B is a vision-language model outside verl-omni's omni
model registry; it needs a vLLM build that serves `Qwen3_5MoeForConditionalGeneration`
(vLLM 0.30 does), and transformers 5.13+ recognizes the `qwen3_5_moe` config, so the
teacher-loop `HFModelConfig` loads without extra patching.

Qwen3.5's generation prompt ends at `<|im_start|>assistant\n<think>\n`: the template
pre-fills the thinking tag. BPM's seam normalization strips that pre-filled `<think>` when
the student response opens with its own `<think>`, so the teacher scores the response from
the same byte prefix. Keep the default chat-template kwargs (thinking on): passing
`enable_thinking=false` pre-fills an empty `<think>...</think>` block and breaks the seam.
The recipe therefore leaves `bpm.teacher_generation_suffix` at its default `""`.

On Ascend NPU (910C, 16 NPUs per node) use the `_npu` launcher: it defaults to 8 NPUs for
the student and 8 for the teacher (four TP=2 replicas on vllm-ascend), and adds the
vllm-ascend settings of the Qwen3-Omni NPU recipes (`VLLM_ASCEND_ENABLE_NZ=0`,
`weight_nz_mode=0`, piecewise graphs). It also forces
`actor_rollout_ref.rollout.logprobs_mode=raw_logprobs`: vllm-ascend's `AscendSampler` does
not implement `processed_logprobs` yet (its own default is `raw_logprobs`), and on NPU the
processed path yields non-finite sampled-token logprobs that vllm-omni's AR scheduler
rejects. All other environment variables are shared.

Data preparation and dependencies follow the
[MMK12 GSPO recipe](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/README.md). The
converter reads every raw `train-*.parquet` / `test-*.parquet` shard (the Hub layout is three train shards and
one test shard) and writes one `train.parquet` and one `test.parquet`, which is what `TRAIN_FILE` / `VAL_FILE`
default to:

```bash
pip install math-verify qwen-vl-utils
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path <path_to_raw_mmk12> --local_save_dir ~/data/mmk12

bash examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_qwen3_5.sh
```

The raw shards themselves are not usable as `TRAIN_FILE` / `VAL_FILE`: they still carry the
`id / question / answer / subject / image` schema, which lacks the `prompt` and `reward_model` columns the RL
dataset reads.

`STUDENT_PATH` / `TEACHER_PATH` default to the Hugging Face Hub IDs `Qwen/Qwen3-Omni-30B-A3B-Instruct` and
`Qwen/Qwen3.5-35B-A3B`; both accept a local checkpoint directory instead. Hub IDs are downloaded into the
standard Hub cache on every node that loads them (~60 GB for the student and ~70 GB for the teacher in bf16), so
point `HF_HOME` (or `HF_HUB_CACHE`) at a shared or sufficiently large disk before the first run.

Environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STUDENT_PATH` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Student checkpoint: Hub ID or local dir. |
| `TEACHER_PATH` | `Qwen/Qwen3.5-35B-A3B` | Teacher checkpoint: Hub ID or local dir; its tokenizer, processor and chat template are read from it. |
| `TRAIN_FILE` | `~/data/mmk12/train.parquet` | Training parquet. Several shards work as a quoted glob (`"$HOME/raw/train-*.parquet"`) or an explicit Hydra list (`"['a.parquet','b.parquet']"`); a directory is rejected. |
| `VAL_FILE` | `~/data/mmk12/test.parquet` | Validation parquet; same forms as `TRAIN_FILE`. |
| `TOPK` | `64` | Teacher top-k log-probs per position. Qwen3.5's 248k vocabulary covers less mass per top-k than a 152k teacher, so `128` is worth trying if `distillation/bpm_q_other` is high. |
| `BETA` | `0.0` | `bpm.beta`; `0` is forward KL, the paper's best setting. |
| `WHITESPACE_MASK` | `all` | `bpm.whitespace_mask`; `indentation` keeps the lone-space rows Qwen tokenizers emit before numbers. |
| `TEACHER_MAX_MODEL_LEN` | `16384` | Teacher context; must hold the teacher-expanded prompt plus the response. |

Any Hydra override passed on the command line comes last and wins.

The two tokenizers are both byte-level BPE with the same Qwen pre-tokenizer family but
different vocabularies: Qwen3-Omni has 151,643 BPE tokens, Qwen3.5 has 248,044, and 131,617
tokens are byte-identical (86.8% of Qwen3-Omni's vocabulary, 53.1% of Qwen3.5's). The
Qwen3.5 tokenizer is NOT a superset: it drops about 20k Qwen3 tokens and adds about 116k
new merges, so the vocabularies are genuinely different and BPM's byte mapping is what
makes the pair work. On a 10-response MMK12-style sample, 91.4% of student response rows
are 1:1 and 2.6% are split (1:N/N:1), with 4.5% all-whitespace rows withheld under
`whitespace_mask=all` and none under `indentation`.

## Validation

- CPU: `tests/trainer/omni/bpm/test_bpm_recipe_on_cpu.py` composes this launcher's
  overrides and checks the teacher, loss and agent-loop wiring.
- Smoke on 3 devices with a tiny random student and a tiny random Qwen3.5-MoE teacher:
  `tests/special_e2e/run_bpm_qwen3_omni_thinker_lora_smoke.sh` (GPU) and
  `tests/special_e2e/run_bpm_qwen3_omni_thinker_lora_npu_smoke.sh` (Ascend NPU). The tiny
  teacher is built by `tests/special_e2e/build_qwen3_5_tiny_random.py` (text-only
  `Qwen3_5MoeForCausalLM`, byte-level tokenizer on a different corpus than the student's,
  attention-only tiny layers so vLLM skips its hybrid SSM KV-cache path: tiny
  Gated-DeltaNet dimensions cannot satisfy that path's SSM/attention page-size alignment).
  Because the random tiny teacher's softmax is nearly uniform, the smoke scores its whole
  vocabulary as top-k (the trained-teacher recipe keeps `TOPK=64`); otherwise the realized
  teacher tokens never reach the top-k, so spanning rows become `TEACHER_TOKEN_MISSING`.
- The full-size recipe has not been run yet.
