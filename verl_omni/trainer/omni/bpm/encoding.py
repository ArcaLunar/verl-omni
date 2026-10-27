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
"""Pack BPM targets into the ``teacher_ids`` / ``teacher_logprobs`` sample fields.

verl already carries these two fields from the agent loop to the actor, one row per
sequence position. BPM reuses them, aligned with the student sequence: row ``g`` holds
the target for the token at ``g + 1``, column 0 holds the row kind (in ``teacher_ids``)
and the complement mass (in ``teacher_logprobs``), and columns ``1:`` hold the explicit
student ids and their target probabilities (probabilities, not log-probabilities).
Cells with zero probability are padding: only positive cells are explicit, everything
else belongs to the complement.
"""

import torch

from verl_omni.trainer.omni.bpm.targets import SampleTargets

__all__ = ["pack_targets", "unpack_targets"]


def pack_targets(targets: SampleTargets, prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out one sample's targets over its student sequence.

    Args:
        targets: Targets for the ``R`` student response positions.
        prompt_len: Student prompt length ``P``; must be positive.

    Returns:
        tuple: int32 ``(P + R, 1 + W)`` ids and float32 ``(P + R, 1 + W)`` probabilities.
        Rows outside ``[P - 1, P + R - 1)`` are zero, i.e. kind ``NONE``.

    Raises:
        ValueError: If ``prompt_len`` is not positive.
    """
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}.")
    num_rows, width = targets.ids.shape
    ids = torch.zeros((prompt_len + num_rows, 1 + width), dtype=torch.int32)
    probs = torch.zeros((prompt_len + num_rows, 1 + width), dtype=torch.float32)
    rows = slice(prompt_len - 1, prompt_len - 1 + num_rows)
    ids[rows, 0] = torch.from_numpy(targets.kinds.astype("int32"))
    ids[rows, 1:] = torch.from_numpy(targets.ids.astype("int32"))
    probs[rows, 0] = torch.from_numpy(targets.other)
    probs[rows, 1:] = torch.from_numpy(targets.probs)
    return ids, probs


def unpack_targets(
    packed_ids: torch.Tensor, packed_probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed rows into kinds, complement mass, explicit ids and probabilities.

    Args:
        packed_ids: int ``(..., 1 + W)`` packed ids.
        packed_probs: float ``(..., 1 + W)`` packed probabilities.

    Returns:
        tuple: kinds ``(...)``, complement ``(...)``, ids ``(..., W)`` (int64) and probs ``(..., W)``.
    """
    return packed_ids[..., 0], packed_probs[..., 0], packed_ids[..., 1:].long(), packed_probs[..., 1:]
