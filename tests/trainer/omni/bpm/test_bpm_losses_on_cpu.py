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
"""BPM actor loss: D_beta on the projected student, row-kind routing, verl wiring."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl_omni.trainer.omni.bpm.alignment import RowKind
from verl_omni.trainer.omni.bpm.encoding import pack_targets
from verl_omni.trainer.omni.bpm.losses import (
    bpm_distillation_ppo_loss,
    bpm_logits_loss,
    bpm_row_losses,
    compute_bpm_distillation_loss,
)
from verl_omni.trainer.omni.bpm.targets import SampleTargets
from verl_omni.workers.config.omni import BPMConfig

VOCAB = 6


def _rows(kinds, cells, width=3):
    """Pack ``[(kind, {id: prob}, other), ...]`` into (ids, probs, other, kinds) tensors."""
    ids = torch.zeros((len(cells), width), dtype=torch.long)
    probs = torch.zeros((len(cells), width))
    other = torch.zeros(len(cells))
    for row, (target, rest) in enumerate(cells):
        for column, (token_id, prob) in enumerate(target.items()):
            ids[row, column], probs[row, column] = token_id, prob
        other[row] = rest
    return ids, probs, other, torch.tensor(kinds, dtype=torch.int32)


def _projected(logits, target):
    """Reference p~ and t over (explicit cells..., complement) for one row."""
    probs = torch.softmax(logits.double(), -1)
    explicit = list(target)
    student = torch.stack([probs[i] for i in explicit] + [1 - sum(probs[i] for i in explicit)])
    return student, explicit


def _reference(logits, target, other, beta, rkl_lambda=0.1):
    student, explicit = _projected(logits, target)
    teacher = torch.tensor([target[i] for i in explicit] + [other], dtype=torch.float64)

    def kl(a, b):
        mask = a > 0
        return (a[mask] * (a[mask].log() - b[mask].log())).sum()

    if beta == 0:
        return kl(teacher, student)
    if beta == 1:
        return kl(student, (1 - rkl_lambda) * teacher + rkl_lambda * student)
    mixture = beta * teacher + (1 - beta) * student
    return beta * kl(teacher, mixture) + (1 - beta) * kl(student, mixture)


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.3, 1.0])
def test_row_divergence_matches_reference_and_gradient(beta):
    torch.manual_seed(0)
    logits = torch.randn(2, VOCAB, requires_grad=True)
    cells = [({1: 0.6, 3: 0.3}, 0.1), ({2: 0.9}, 0.1)]
    ids, probs, other, kinds = _rows([RowKind.ONE_TO_ONE, RowKind.M_TO_N], cells)
    losses = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=beta))

    reference_logits = logits.detach().clone().requires_grad_(True)
    reference = torch.stack([_reference(reference_logits[r], c, o, beta) for r, (c, o) in enumerate(cells)])
    torch.testing.assert_close(losses.double(), reference, atol=1e-5, rtol=1e-5)
    losses.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-5, rtol=1e-5)


def test_stop_and_forced_delta_rows_train_forward_kl_regardless_of_beta():
    torch.manual_seed(1)
    logits = torch.randn(3, VOCAB)
    cells = [({5: 0.8}, 0.2), ({2: 1.0}, 0.0), ({1: 0.5}, 0.5)]
    ids, probs, other, kinds = _rows([RowKind.STOP_BRIDGE, RowKind.FORCED_DELTA, RowKind.ONE_TO_ONE], cells)
    losses = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=0.5))
    assert losses[0].item() == pytest.approx(_reference(logits[0], *cells[0], beta=0).item(), abs=1e-5)
    # forced delta: forward KL to a one-hot is the cross-entropy of the realized token
    assert losses[1].item() == pytest.approx(-torch.log_softmax(logits[1], -1)[2].item(), abs=1e-5)
    assert losses[2].item() == pytest.approx(_reference(logits[2], *cells[2], beta=0.5).item(), abs=1e-5)

    skipped = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=0.5, stop_bridge="skip"))
    assert skipped[0].item() == 0.0


def test_untrained_rows_have_zero_loss_and_finite_gradient():
    logits = torch.randn(4, VOCAB, requires_grad=True)
    cells = [({}, 0.0), ({}, 0.0), ({1: 1.0}, 0.0), ({}, 0.0)]
    kinds = [RowKind.NONE, RowKind.WHITESPACE_MASKED, RowKind.TEACHER_TOKEN_MISSING, RowKind.UNSUPPORTED_INPUT]
    ids, probs, other, kinds = _rows(kinds, cells)
    losses = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=0.5))
    assert losses.tolist() == [0.0, 0.0, 0.0, 0.0]
    losses.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() == 0


def test_padding_cells_are_not_explicit_and_chunking_is_exact():
    torch.manual_seed(2)
    logits = torch.randn(5, VOCAB)
    cells = [({3: 0.7}, 0.3)] * 5
    ids, probs, other, kinds = _rows([RowKind.ONE_TO_ONE] * 5, cells)
    base = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=0.5))
    # padding slots carry id 0 with probability 0; moving them to id 4 must not change anything
    padded_elsewhere = torch.where(probs > 0, ids, torch.full_like(ids, 4))
    torch.testing.assert_close(bpm_row_losses(logits, padded_elsewhere, probs, other, kinds, BPMConfig(beta=0.5)), base)
    chunked = bpm_row_losses(logits, ids, probs, other, kinds, BPMConfig(beta=0.5, loss_row_chunk=2))
    torch.testing.assert_close(chunked, base)


def _sample_targets(kinds, cells, width):
    targets = SampleTargets(
        kinds=np.array(kinds, dtype=np.int8),
        ids=np.zeros((len(kinds), width), dtype=np.int64),
        probs=np.zeros((len(kinds), width), dtype=np.float32),
        other=np.zeros((len(kinds),), dtype=np.float32),
    )
    for row, (target, rest) in enumerate(cells):
        for column, (token_id, prob) in enumerate(target.items()):
            targets.ids[row, column], targets.probs[row, column] = token_id, prob
        targets.other[row] = rest
    return targets


def _micro_batch():
    """Two sequences (prompt 2 + response 2, prompt 3 + response 1) with packed BPM targets."""
    width = 2
    first = _sample_targets([RowKind.ONE_TO_ONE, RowKind.STOP_BRIDGE], [({1: 0.9}, 0.1), ({5: 0.7}, 0.3)], width)
    second = _sample_targets([RowKind.WHITESPACE_MASKED], [({}, 0.0)], width)
    packed = [pack_targets(first, prompt_len=2), pack_targets(second, prompt_len=3)]

    def nested(tensors):
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    data = TensorDict(
        {
            "prompts": nested([torch.tensor([7, 8]), torch.tensor([7, 8, 9])]),
            "responses": nested([torch.tensor([1, 5]), torch.tensor([2])]),
            "response_mask": nested([torch.ones(2, dtype=torch.long), torch.ones(1, dtype=torch.long)]),
            "teacher_ids": nested([p[0] for p in packed]),
            "teacher_logprobs": nested([p[1] for p in packed]),
        },
        batch_size=[2],
    )
    return data


def test_logits_stage_and_final_stage_line_up_with_responses():
    torch.manual_seed(3)
    data = _micro_batch()
    student_logits = torch.randn(1, 8, VOCAB)  # 4 + 4 packed tokens, as the FSDP engine passes them
    actor = SimpleNamespace(strategy="fsdp2")
    outputs = bpm_distillation_ppo_loss(
        actor, None, BPMConfig(beta=0.0), data=data, student_logits=student_logits, data_format="thd"
    )
    assert set(outputs) == {"distillation_losses", "bpm_row_kind", "bpm_q_other"}
    assert all(value.shape == (1, 8) for value in outputs.values())
    # row P - 1 = 1 of the first sequence predicts its first response token
    expected = -torch.log_softmax(student_logits[0, 1], -1)[1] * 0.9
    expected += 0.1 * torch.log(torch.tensor(0.1) / (1 - torch.softmax(student_logits[0, 1], -1)[1]))
    expected += 0.9 * torch.log(torch.tensor(0.9))
    assert outputs["distillation_losses"][0, 1].item() == pytest.approx(expected.item(), abs=1e-5)

    offsets = torch.tensor([0, 4, 8])
    model_output = {k: torch.nested.nested_tensor_from_jagged(v[0], offsets) for k, v in outputs.items()}
    losses, metrics = compute_bpm_distillation_loss(actor, None, model_output, data)
    assert losses.shape == (2, 2)
    assert losses[0, 0].item() == pytest.approx(expected.item(), abs=1e-5)
    assert losses[1].tolist() == [0.0, 0.0]  # masked whitespace row and padding
    assert metrics["distillation/bpm_rows_one_to_one"] == pytest.approx(1 / 3)
    assert metrics["distillation/bpm_rows_stop_bridge"] == pytest.approx(1 / 3)
    assert metrics["distillation/bpm_rows_whitespace_masked"] == pytest.approx(1 / 3)
    assert metrics["distillation/bpm_q_other"] == pytest.approx(0.1, abs=1e-6)


def test_logits_stage_rejects_misaligned_targets_and_unknown_strategy():
    data = _micro_batch()
    with pytest.raises(ValueError, match="do not match"):
        bpm_logits_loss(SimpleNamespace(strategy="fsdp"), BPMConfig(), data, torch.randn(1, 7, VOCAB), "thd")
    with pytest.raises(NotImplementedError, match="strategy"):
        bpm_logits_loss(SimpleNamespace(strategy="torchtitan"), BPMConfig(), data, torch.randn(1, 8, VOCAB), "thd")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"beta": 1.5}, "beta"),
        ({"beta": 1.0, "rkl_lambda": 0.0}, "rkl_lambda"),
        ({"stop_bridge": "floor"}, "stop_bridge"),
        ({"teacher_modalities": ["image", "depth"]}, "teacher_modalities"),
        ({"loss_row_chunk": 0}, "loss_row_chunk"),
    ],
)
def test_bpm_config_validation(override, message):
    with pytest.raises(ValueError, match=message):
        BPMConfig(**override)
