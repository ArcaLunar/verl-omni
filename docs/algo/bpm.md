# Cross-Tokenizer On-Policy Distillation (BPM)

Last updated: 10/30/2026.

## Background

[On-policy distillation](omni_opd.md) needs the teacher's next-token distribution on
the student's own rollouts. When teacher and student use different tokenizers, a
teacher position and a student position no longer refer to the same text prefix, and
their distributions live on different vocabularies.

Byte-Prefix Marginalization ([arXiv:2607.22334](https://arxiv.org/abs/2607.22334))
re-expresses the teacher distribution on the student vocabulary through the bytes both
tokenizers decode to. Each teacher token `w` is routed to `phi(w)`, the longest student
token whose bytes prefix `bytes(w)`; mass routed to the same student token is summed, and
mass with no student prefix goes to an explicit residual cell, so the target is defined
for every student token, follows byte content, and sums to one.

Per student response position the target is built according to how the two
segmentations relate there (Table 2 of the paper, ratios teacher:student):

| Relation | Example | Target |
| --- | --- | --- |
| 1:1, 1:N head | `at` → `at`; `ate` → `at`·`e` | scatter of the aligned teacher row through `phi` |
| 1:N interior | the `e` above | teacher mass continuing the realized prefix, renormalized |
| N:1, M:N | `</think>` spanning `</`·`think`·`>` | realized-path chain `prod q_k(w_k) * pi(r)`, moved off the head cell |
| stop | the realized stop token | teacher stop mass, trained under forward KL |

The student is trained on `D_beta(t || p~)`, where `p~` is the student distribution
projected onto the target's explicit cells plus one complement cell: forward KL at
`beta=0`, generalized JSD for `0 < beta < 1`, skew reverse KL at `beta=1`. The loss is
normalized by all response tokens, including rows withheld from the loss.

## How it runs in verl-omni

BPM plugs into verl's on-policy distillation path without changing verl:

1. **Agent loop.** `BPMAgentLoopManagerTQ` runs workers whose teacher step re-renders the
   prompt with the teacher's own chat template (a VLM processor renders its own media
   placeholders), re-tokenizes the student response with the teacher tokenizer, and asks
   the teacher server for top-k prompt log-probs. The BPM target is built on CPU, off the
   training critical path, and packed into the `teacher_ids` / `teacher_logprobs` fields
   verl already carries to the actor.
2. **Actor.** `distillation_loss.loss_mode=bpm` is registered as a logits-level
   distillation loss. A registered actor-loss override (`omni_algos`) installs
   `bpm_distillation_ppo_loss` on the actor worker group; its logits stage computes
   `D_beta`, and verl's `distillation_ppo_loss` aggregates the result.

Compared with the official implementation, the teacher returns top-k log-probs instead
of hidden states, so the trainer never loads the teacher's `lm_head`. Teacher mass
outside the top-k goes to the complement cell; a row that needs the probability of a
realized teacher token missing from the top-k is withheld from the loss.

## Configuration

Distillation settings required by BPM:

| Key | Value |
| --- | --- |
| `distillation.enabled` | `true` |
| `distillation.distillation_loss.loss_mode` | `bpm` |
| `distillation.distillation_loss.topk` | teacher top-k, e.g. `64` |
| `distillation.distillation_loss.use_policy_gradient` | `false` |
| `distillation.distillation_loss.loss_max_clamp` | `null` |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` |
| `+actor_rollout_ref.rollout.agent.agent_loop_manager_class` | `verl_omni.trainer.omni.bpm.agent_loop.BPMAgentLoopManagerTQ` |

The actor-loss builder rejects runs that would change the objective (another
`loss_agg_mode`, a loss clamp, or policy-gradient mode). Exactly one teacher is supported.

A teacher outside the omni model registry (for example Muse-Glimmer) runs on verl's
native vLLM replica:

```bash
distillation.teacher_models.teacher_model._target_=verl.workers.config.DistillationTeacherModelConfig
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False
```

The `bpm` group (`verl_omni/trainer/config/omni/bpm/bpm.yaml`, dataclass `BPMConfig`):

| Key | Default | Meaning |
| --- | --- | --- |
| `beta` | `0.5` | Divergence axis; the paper's main results use `0` (forward KL). |
| `rkl_lambda` | `0.1` | Skew of the reverse-KL endpoint, used when `beta=1`. |
| `whitespace_mask` | `all` | All-whitespace student tokens withheld from the loss: `all` (the paper's mask), `indentation`, or `off`. |
| `stop_bridge` | `bridge` | `bridge` trains the realized stop row under forward KL; `skip` drops it. |
| `teacher_modalities` | `[image, video]` | Input modalities the teacher accepts; other samples are not distilled. |
| `teacher_mm_processor_kwargs` | `{}` | Media processor kwargs sent with each teacher request, e.g. `max_pixels`. |
| `teacher_chat_template_kwargs` | `{}` | Extra arguments for the teacher's chat template. |
| `teacher_generation_suffix` | `""` | Text appended to the teacher's generation prompt, where the assistant answer starts (e.g. Muse-Glimmer's ` to=user<\|message\|>` channel header). |
| `teacher_tokenizer_path` | `null` | Local or Hub tokenizer path; defaults to the teacher model path. |
| `loss_row_chunk` | `2048` | Rows per checkpointed chunk in the loss. |

### Whitespace mask on Qwen tokenizers

The paper masks every student token made only of spaces or tabs, to stop indentation
noise from collapsing generated code. Qwen-family tokenizers also emit a lone space
token before every number, so on math text the same rule withholds the row before each
number (6–14% of rows on our math and LaTeX samples, against the paper's "rare outside code").
`whitespace_mask=indentation` withholds only whitespace inside a line's leading
indentation.

## Metrics

The actor logs `distillation/bpm_rows_<kind>`, the share of response tokens of each row
kind, and `distillation/bpm_q_other`, the mean complement mass on trained rows. Compare the
`one_to_one` / `one_to_n_*` / `n_to_one` / `m_to_n` shares with Table 6 of the paper;
large `byte_mismatch`, `teacher_token_missing` or `unsupported_input` shares point at a
tokenizer, top-k or data problem.

## Usage

Recipe:
[`examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/bpm_trainer/README.md),
Qwen3-Omni-30B-A3B-Instruct Thinker (LoRA) distilled from Muse-Glimmer-30B on MMK12.

GPU smoke: `tests/special_e2e/run_bpm_qwen3_omni_thinker_lora_smoke.sh` builds a tiny
Qwen3-Omni student and a tiny Muse-Glimmer teacher with differently trained tokenizers and
runs two steps on 3 GPUs.

## Limitations

- The target uses the teacher's top-k, not its full vocabulary.
- Megatron students need `tensor_model_parallel_size=1`; FSDP students have no such limit.
- One teacher; single-turn agent loops (the teacher scores the final output).
