# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BPM distillation loss on the actor.

Per row the student distribution is projected onto the target's explicit cells plus one
complement cell (``p~`` of the paper's appendix) and compared with the packed BPM target
under ``D_beta`` (Eq. 7): forward KL at ``beta=0``, generalized JSD for ``0 < beta < 1``
and skew reverse KL at ``beta=1``. Stop-bridge and forced-delta rows always train under
forward KL. Aggregation over all response tokens gives Eq. 9.

The loss plugs into verl's distillation path: ``"bpm"`` is registered as a logits-level
loss (``use_topk=True``), so the engine hands the student logits to
:func:`bpm_distillation_ppo_loss`; everything else is verl's ``distillation_ppo_loss``.
Divergence formulas follow the official BPM implementation
(``slime_plugins/bpm/loss/bpm_kernels.py``).
"""

from functools import partial
from typing import Any, Callable

import torch
import torch.utils.checkpoint
from tensordict import TensorDict
from verl.trainer.distillation import distillation_ppo_loss
from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding

from verl_omni.trainer.omni.bpm.alignment import RowKind
from verl_omni.trainer.omni.bpm.encoding import unpack_targets
from verl_omni.trainer.omni.omni_algos import register_omni_distillation_actor_loss
from verl_omni.workers.config.omni import BPMConfig

__all__ = ["bpm_distillation_ppo_loss", "bpm_logits_loss", "bpm_row_losses", "build_bpm_actor_loss"]

_DIVERGENCE_KINDS = (
    RowKind.ONE_TO_ONE,
    RowKind.ONE_TO_N_HEAD,
    RowKind.ONE_TO_N_INTERIOR,
    RowKind.N_TO_ONE,
    RowKind.M_TO_N,
)
_FSDP_STRATEGIES = {"fsdp", "fsdp2", "veomni"}
_EPS = 1e-12


def _xlogy_ratio(weight: torch.Tensor, log_weight: torch.Tensor, log_ref: torch.Tensor) -> torch.Tensor:
    """``sum weight * (log_weight - log_ref)`` over cells, with ``0 * log 0 = 0``."""
    return torch.where(weight > 0, weight * (log_weight - log_ref), torch.zeros_like(weight)).sum(-1)


def _chunk_row_losses(
    logits: torch.Tensor,
    cell_ids: torch.Tensor,
    cell_probs: torch.Tensor,
    other: torch.Tensor,
    kinds: torch.Tensor,
    beta: float,
    rkl_lambda: float,
    train_stop: bool,
) -> torch.Tensor:
    logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    explicit = cell_probs > 0
    cell_logp = torch.gather(logits, -1, cell_ids) - log_z  # [n, W]
    # a finite sentinel keeps the gradient of rows without explicit cells free of NaN
    sentinel = torch.finfo(cell_logp.dtype).min
    log_p_in = torch.logsumexp(torch.where(explicit, cell_logp, torch.full_like(cell_logp, sentinel)), dim=-1)
    log_p_in = log_p_in.clamp(max=-torch.finfo(cell_logp.dtype).eps)
    log_p_other = torch.log(-torch.expm1(log_p_in))  # log(1 - p_in), stable as p_in -> 1

    target = torch.cat([torch.where(explicit, cell_probs.float(), 0.0), other.float().unsqueeze(-1)], dim=-1)
    log_p = torch.cat([torch.where(explicit, cell_logp, 0.0), log_p_other.unsqueeze(-1)], dim=-1)
    student = torch.where(torch.cat([explicit, torch.ones_like(explicit[:, :1])], dim=-1), log_p.exp(), 0.0)
    log_target = torch.log(target.clamp_min(_EPS))

    forward_kl = _xlogy_ratio(target, log_target, log_p)
    if beta == 0.0:
        divergence = forward_kl
    elif beta < 1.0:
        mixture = beta * target + (1.0 - beta) * student
        log_mixture = torch.log(mixture.clamp_min(_EPS))
        divergence = beta * _xlogy_ratio(target, log_target, log_mixture) + (1.0 - beta) * _xlogy_ratio(
            student, log_p, log_mixture
        )
    else:
        skewed = (1.0 - rkl_lambda) * target + rkl_lambda * student
        divergence = _xlogy_ratio(student, log_p, torch.log(skewed.clamp_min(_EPS)))

    zeros = torch.zeros_like(forward_kl)
    content = torch.isin(kinds, torch.tensor(_DIVERGENCE_KINDS, device=kinds.device, dtype=kinds.dtype))
    losses = torch.where(content, divergence, zeros)
    losses = torch.where(kinds == RowKind.FORCED_DELTA, forward_kl, losses)
    if train_stop:
        losses = torch.where(kinds == RowKind.STOP_BRIDGE, forward_kl, losses)
    return losses


def bpm_row_losses(
    logits: torch.Tensor,
    cell_ids: torch.Tensor,
    cell_probs: torch.Tensor,
    other: torch.Tensor,
    kinds: torch.Tensor,
    bpm_config: BPMConfig,
) -> torch.Tensor:
    """Per-row BPM divergence between the student and the packed target.

    Args:
        logits: ``(N, V)`` student logits over the full vocabulary.
        cell_ids: int64 ``(N, W)`` explicit target ids; cells with probability 0 are padding.
        cell_probs: float ``(N, W)`` target probabilities of the explicit cells.
        other: float ``(N,)`` complement mass.
        kinds: int ``(N,)`` :class:`RowKind` per row; rows outside the trained kinds get loss 0.
        bpm_config: BPM settings (divergence axis, stop handling, chunk size).

    Returns:
        torch.Tensor: float32 ``(N,)`` per-row loss.
    """
    chunk = partial(
        _chunk_row_losses,
        beta=bpm_config.beta,
        rkl_lambda=bpm_config.rkl_lambda,
        train_stop=bpm_config.stop_bridge == "bridge",
    )
    parts = []
    for start in range(0, logits.shape[0], bpm_config.loss_row_chunk):
        rows = slice(start, start + bpm_config.loss_row_chunk)
        args = (logits[rows], cell_ids[rows], cell_probs[rows], other[rows], kinds[rows])
        if torch.is_grad_enabled() and logits.requires_grad:
            # recompute the fp32 copy of the logits in backward instead of keeping it
            parts.append(torch.utils.checkpoint.checkpoint(chunk, *args, use_reentrant=False))
        else:
            parts.append(chunk(*args))
    if not parts:
        return logits.new_zeros((0,), dtype=torch.float32)
    return torch.cat(parts)


def _split_teacher_fields(data: TensorDict, strategy: str, data_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay the packed teacher fields out like the student logits on this rank."""
    packed_ids, packed_probs = data["teacher_ids"], data["teacher_logprobs"]
    if strategy in _FSDP_STRATEGIES:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

        packed_ids, packed_probs = packed_ids.values().unsqueeze(0), packed_probs.values().unsqueeze(0)
        if get_ulysses_sequence_parallel_world_size() > 1:
            packed_ids, packed_probs = slice_input_tensor(packed_ids, dim=1), slice_input_tensor(packed_probs, dim=1)
        return packed_ids, packed_probs
    if strategy == "megatron":
        from megatron.core import parallel_state
        from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError("BPM needs full-vocabulary student logits; use tensor_model_parallel_size=1.")
        preprocess = preprocess_thd_engine if data_format == "thd" else preprocess_bshd_engine
        return preprocess(packed_ids, pre_process=True)[0], preprocess(packed_probs, pre_process=True)[0]
    raise NotImplementedError(f"BPM does not support actor strategy {strategy!r}.")


def bpm_logits_loss(
    config: ActorConfig,
    bpm_config: BPMConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Logits-processor stage: per-token BPM loss and diagnostics.

    Args:
        config: Actor config; ``strategy`` selects the sequence layout.
        bpm_config: BPM settings.
        data: Micro-batch with nested ``teacher_ids`` / ``teacher_logprobs`` packed by
            :func:`verl_omni.trainer.omni.bpm.encoding.pack_targets`.
        student_logits: ``(B, T_local, V)`` student logits on this rank.
        data_format: ``"thd"`` or ``"bshd"`` (Megatron only).

    Returns:
        dict[str, torch.Tensor]: ``(B, T_local)`` tensors ``distillation_losses``, ``bpm_row_kind``
        and ``bpm_q_other``.
    """
    packed_ids, packed_probs = _split_teacher_fields(data, config.strategy, data_format)
    if packed_ids.shape[:2] != student_logits.shape[:2]:
        raise ValueError(
            f"BPM targets {tuple(packed_ids.shape[:2])} do not match student logits {tuple(student_logits.shape[:2])}."
        )
    kinds, other, cell_ids, cell_probs = unpack_targets(packed_ids, packed_probs)
    batch, length, vocab = student_logits.shape
    losses = bpm_row_losses(
        student_logits.reshape(batch * length, vocab),
        cell_ids.reshape(batch * length, -1),
        cell_probs.reshape(batch * length, -1),
        other.reshape(-1),
        kinds.reshape(-1),
        bpm_config,
    )
    return {
        "distillation_losses": losses.view(batch, length),
        "bpm_row_kind": kinds.to(torch.float32),
        "bpm_q_other": other.to(torch.float32).detach(),
    }


@register_distillation_loss(DistillationLossSettings(names=["bpm"], use_topk=True))  # type: ignore[arg-type]
def compute_bpm_distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Final stage: response-aligned BPM losses and the row-kind census.

    Returns:
        tuple: float ``(bsz, resp_len)`` losses (zero on untrained rows, which still count in
        the token-mean normalizer) and metrics.
    """
    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    kinds = no_padding_2_padding(model_output["bpm_row_kind"], data).round().long()
    other = no_padding_2_padding(model_output["bpm_q_other"], data)
    response_mask = data["response_mask"]
    response_mask = (response_mask.to_padded_tensor(0) if response_mask.is_nested else response_mask).bool()
    num_tokens = response_mask.sum().clamp_min(1)

    metrics: dict[str, Any] = {}
    for kind in RowKind:
        share = ((kinds == kind) & response_mask).sum() / num_tokens
        metrics[f"distillation/bpm_rows_{kind.name.lower()}"] = share.item()
    trained = response_mask & torch.isin(kinds, torch.tensor(_DIVERGENCE_KINDS, device=kinds.device))
    if trained.any():
        metrics["distillation/bpm_q_other"] = other[trained].mean().item()
    return losses, metrics


def bpm_distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    bpm_config: BPMConfig,
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Actor loss for ``loss_mode=bpm``: BPM in the logits processor, verl's distillation loss after."""
    if student_logits is not None:
        return bpm_logits_loss(config, bpm_config, data, student_logits, data_format)
    return distillation_ppo_loss(config, distillation_config, model_output, data, dp_group)


@register_omni_distillation_actor_loss("bpm")
def build_bpm_actor_loss(config) -> Callable:
    """Build the actor loss for ``loss_mode=bpm`` from the full trainer config.

    Raises:
        ValueError: If the run would normalize or clamp the loss differently from the paper.
    """
    actor_config: ActorConfig = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
    bpm_config: BPMConfig = omega_conf_to_dataclass(config.bpm)
    loss_config = distillation_config.distillation_loss
    if actor_config.loss_agg_mode != "token-mean":
        raise ValueError(
            "BPM normalizes by all response tokens (Eq. 9); set actor_rollout_ref.actor.loss_agg_mode=token-mean, "
            f"got {actor_config.loss_agg_mode!r}."
        )
    if loss_config.loss_max_clamp is not None:
        raise ValueError("BPM losses must not be clamped; set distillation.distillation_loss.loss_max_clamp=null.")
    if loss_config.use_policy_gradient:
        raise ValueError("BPM is a supervised loss; set distillation.distillation_loss.use_policy_gradient=false.")
    return partial(
        bpm_distillation_ppo_loss,
        config=actor_config,
        distillation_config=distillation_config,
        bpm_config=bpm_config,
    )
