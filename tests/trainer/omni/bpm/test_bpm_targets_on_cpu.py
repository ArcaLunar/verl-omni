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
"""BPM target construction: the paper's worked examples, every row kind, and mass conservation."""

import numpy as np
import pytest
import torch

from verl_omni.trainer.omni.bpm.alignment import TRAINED_KINDS, RowKind, align_sample
from verl_omni.trainer.omni.bpm.encoding import pack_targets, unpack_targets
from verl_omni.trainer.omni.bpm.targets import build_sample_targets, target_width

STUDENT_STOP, TEACHER_STOP = 900, 901  # the make_bridge fixture's stop ids


def _build(bridge, student_ids, teacher_ids, rows, teacher_rows, mask_whitespace_rows=True):
    alignment = align_sample(bridge, student_ids, teacher_ids, mask_whitespace_rows=mask_whitespace_rows)
    return build_sample_targets(bridge, alignment, *teacher_rows(rows))


def _row(targets, position):
    cells = {}
    for student_id, prob in zip(targets.ids[position].tolist(), targets.probs[position].tolist(), strict=True):
        if prob > 0:
            cells[student_id] = cells.get(student_id, 0.0) + prob
    return cells, float(targets.other[position])


def test_worked_example_table_4(make_bridge, teacher_rows):
    bridge = make_bridge({1: "a", 2: "at", 3: "ate", 4: "bit"}, {11: "at", 12: "ate", 13: "a", 14: "bit"})
    targets = _build(bridge, [2], [11], [{11: 0.5, 12: 0.2, 13: 0.2, 14: 0.1}, {TEACHER_STOP: 1.0}], teacher_rows)
    cells, other = _row(targets, 0)
    assert targets.kinds[0] == RowKind.ONE_TO_ONE
    assert cells == pytest.approx({1: 0.2, 2: 0.5, 3: 0.2, 4: 0.1})
    assert other == pytest.approx(0.0, abs=1e-6)


def test_scatter_pools_shared_prefix_and_routes_unmatched_mass_to_complement(make_bridge, teacher_rows):
    # Fig. 1b: teacher tokens absent from the student vocab pool onto their longest student prefix
    bridge = make_bridge({1: "bell", 2: "x"}, {11: "bell", 12: "bells", 13: "zz", 14: "x"})
    rows = [{11: 0.17, 12: 0.09, 13: 0.04, 14: 0.7}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1], [11], rows, teacher_rows)
    cells, other = _row(targets, 0)
    assert cells == pytest.approx({1: 0.26, 2: 0.7})
    assert other == pytest.approx(0.04, abs=1e-6)


def test_spanning_think_close_gets_realized_path_chain(make_bridge, teacher_rows):
    # Appendix D: student </think> spans teacher </ · think · >; chain = 0.7 * 0.9 * 0.95
    bridge = make_bridge(
        {1: "</think>", 2: "</", 3: "\n\n", 4: "x"},
        {11: "</", 12: "think", 13: ">\n\n", 14: ">", 15: "x"},
    )
    rows = [{11: 0.7, 15: 0.3}, {12: 0.9, 15: 0.1}, {13: 0.9, 14: 0.05, 15: 0.05}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1, 3], [11, 12, 13], rows, teacher_rows)
    cells, other = _row(targets, 0)
    assert targets.kinds[0] == RowKind.N_TO_ONE
    # first-token scatter alone would leave </think> with no mass and train the student against it
    assert cells[1] == pytest.approx(0.7 * 0.9 * 0.95)
    assert cells[2] == pytest.approx(0.7 - 0.7 * 0.9 * 0.95)
    assert cells[4] == pytest.approx(0.3)
    assert other == pytest.approx(0.0, abs=1e-6)


def test_counterfactual_longer_student_token_on_aligned_row(make_bridge, teacher_rows):
    # realized "1" + "2"; the student could have emitted "12", which spans two teacher tokens
    bridge = make_bridge({1: "1", 2: "2", 3: "12"}, {11: "1", 12: "2"})
    rows = [{11: 0.8, 12: 0.2}, {12: 0.5, 11: 0.5}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1, 2], [11, 12], rows, teacher_rows)
    cells, other = _row(targets, 0)
    assert cells == pytest.approx({1: 0.8 - 0.4, 3: 0.4, 2: 0.2})
    assert other == pytest.approx(0.0, abs=1e-6)


def test_interior_conditional_and_forced_delta(make_bridge, teacher_rows):
    # teacher "10" splits into student "1" + "0"; the "0" row conditions on the realized "1"
    bridge = make_bridge({1: "1", 2: "0", 3: "2"}, {11: "10", 12: "1", 13: "12", 14: "2"})
    rows = [{11: 0.5, 12: 0.2, 13: 0.2, 14: 0.1}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1, 2], [11], rows, teacher_rows)
    head, head_other = _row(targets, 0)
    assert head == pytest.approx({1: 0.9, 3: 0.1}) and head_other == pytest.approx(0.0, abs=1e-6)
    cells, other = _row(targets, 1)
    assert targets.kinds[1] == RowKind.ONE_TO_N_INTERIOR
    assert cells == pytest.approx({2: 0.5 / 0.9, 3: 0.2 / 0.9})
    assert other == pytest.approx(0.2 / 0.9, abs=1e-6)

    delta = _build(bridge, [1, 2], [11], [{11: 0.6, 14: 0.4}, {TEACHER_STOP: 1.0}], teacher_rows)
    assert delta.kinds[1] == RowKind.FORCED_DELTA
    assert _row(delta, 1) == ({2: 1.0}, 0.0)


def test_m_to_n_row_conditions_then_moves_chain_mass(make_bridge, teacher_rows):
    # teacher "ab" · "cd", student "a" · "bc" · "d"
    bridge = make_bridge({1: "a", 2: "bc", 3: "d", 4: "b", 5: "c"}, {21: "ab", 22: "cd", 23: "a", 24: "c"})
    rows = [{21: 0.6, 23: 0.4}, {22: 0.7, 24: 0.3}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1, 2, 3], [21, 22], rows, teacher_rows)
    assert list(targets.kinds) == [RowKind.ONE_TO_N_HEAD, RowKind.M_TO_N, RowKind.ONE_TO_N_INTERIOR]
    cells, other = _row(targets, 1)
    assert cells == pytest.approx({2: 0.6})  # chain P(abc | a) = 0.6 * 1.0, taken from the "b" head
    assert other == pytest.approx(0.4, abs=1e-6)
    cells, other = _row(targets, 2)
    assert cells == pytest.approx({3: 0.7}) and other == pytest.approx(0.3, abs=1e-6)


def test_stop_mass_routes_to_realized_stop_and_bridge_row(make_bridge, teacher_rows):
    bridge = make_bridge({1: "x"}, {11: "x"})
    rows = [{11: 0.8, TEACHER_STOP: 0.2}, {TEACHER_STOP: 0.9, 11: 0.1}]
    targets = _build(bridge, [1, STUDENT_STOP], [11], rows, teacher_rows)
    assert _row(targets, 0)[0] == pytest.approx({1: 0.8, STUDENT_STOP: 0.2})
    assert targets.kinds[1] == RowKind.STOP_BRIDGE
    cells, other = _row(targets, 1)
    assert cells == pytest.approx({STUDENT_STOP: 0.9}) and other == pytest.approx(0.1, abs=1e-6)

    truncated = _build(bridge, [1], [11], rows, teacher_rows)  # no realized stop: use the turn-end id
    assert _row(truncated, 0)[0] == pytest.approx({1: 0.8, STUDENT_STOP: 0.2})

    no_stop_mass = _build(bridge, [1, STUDENT_STOP], [11], [rows[0], {11: 1.0}], teacher_rows)
    assert no_stop_mass.kinds[1] == RowKind.NONE


def test_rows_needing_a_missing_realized_teacher_token_are_excluded(make_bridge, teacher_rows):
    bridge = make_bridge({1: "1", 2: "0", 3: "x"}, {11: "10", 12: "1", 13: "x"})
    # realized teacher "10" is outside the top-k: the interior row cannot be conditioned
    targets = _build(bridge, [3, 1, 2], [13, 11], [{13: 1.0}, {12: 0.7, 13: 0.3}, {TEACHER_STOP: 1.0}], teacher_rows)
    assert list(targets.kinds) == [RowKind.ONE_TO_ONE, RowKind.ONE_TO_N_HEAD, RowKind.TEACHER_TOKEN_MISSING]
    assert _row(targets, 1)[0] == pytest.approx({1: 0.7, 3: 0.3})  # scatter needs no realized factor
    assert _row(targets, 2) == ({}, 0.0)

    span_bridge = make_bridge({1: "</think>", 2: "</"}, {11: "</", 12: "think", 13: ">", 15: "y"})
    span = _build(span_bridge, [1], [11, 12, 13], [{11: 1.0}, {15: 1.0}, {13: 1.0}, {TEACHER_STOP: 1.0}], teacher_rows)
    assert span.kinds[0] == RowKind.TEACHER_TOKEN_MISSING


def test_chain_does_not_cross_a_teacher_control_token(make_bridge, teacher_rows):
    # a teacher control token between "</" and "think" breaks the realized path
    bridge = make_bridge({1: "</think>", 2: "</"}, {11: "</", 12: "think", 13: ">"})
    rows = [{11: 1.0}, {TEACHER_STOP: 1.0}, {12: 1.0}, {13: 1.0}, {TEACHER_STOP: 1.0}]
    targets = _build(bridge, [1], [11, TEACHER_STOP, 12, 13], rows, teacher_rows)
    assert targets.kinds[0] == RowKind.TEACHER_TOKEN_MISSING


def test_identical_tokenizers_reproduce_teacher_distribution(make_bridge, teacher_rows):
    vocab = {1: "a", 2: "b", 3: "c"}
    bridge = make_bridge(vocab, vocab)
    rows = [{1: 0.6, 2: 0.3, 3: 0.1}, {2: 0.5, 3: 0.5}, {TEACHER_STOP: 0.7, 1: 0.3}]
    targets = _build(bridge, [1, 2, STUDENT_STOP], [1, 2], rows, teacher_rows)
    assert _row(targets, 0)[0] == pytest.approx(rows[0])
    assert _row(targets, 1)[0] == pytest.approx(rows[1])
    assert _row(targets, 2)[0] == pytest.approx({STUDENT_STOP: 0.7})


def _greedy_segment(text: bytes, vocab: dict[int, bytes]) -> list[int]:
    by_bytes = {data: token_id for token_id, data in vocab.items()}
    ids, start = [], 0
    while start < len(text):
        end = max(e for e in range(start + 1, len(text) + 1) if text[start:e] in by_bytes)
        ids.append(by_bytes[text[start:end]])
        start = end
    return ids


def _random_segment(text: bytes, vocab: dict[int, bytes], rng) -> list[int]:
    by_bytes = {data: token_id for token_id, data in vocab.items()}
    ids, start = [], 0
    while start < len(text):
        ends = [e for e in range(start + 1, len(text) + 1) if text[start:e] in by_bytes]
        end = int(rng.choice(ends))
        ids.append(by_bytes[text[start:end]])
        start = end
    return ids


@pytest.mark.parametrize("seed", range(20))
def test_trained_rows_are_distributions(make_bridge, seed):
    rng = np.random.default_rng(seed)
    alphabet = ["a", "b", "c"]
    pieces = {"".join(rng.choice(alphabet, size=int(rng.integers(2, 4)))) for _ in range(12)}
    student = {i + 1: p for i, p in enumerate(sorted(set(alphabet) | set(list(pieces)[:6])))}
    teacher = {i + 101: p for i, p in enumerate(sorted(set(alphabet) | set(list(pieces)[6:])))}
    bridge = make_bridge(student, teacher)
    text = "".join(rng.choice(alphabet, size=30)).encode()
    student_ids = _random_segment(text, bridge.student_token_bytes, rng) + [STUDENT_STOP]
    teacher_ids = _greedy_segment(text, bridge.teacher_token_bytes)

    vocab = sorted(teacher) + [TEACHER_STOP]
    probs = rng.dirichlet(np.ones(len(vocab)), size=len(teacher_ids) + 1)
    ids = np.tile(np.array(vocab), (len(teacher_ids) + 1, 1))
    alignment = align_sample(bridge, student_ids, teacher_ids, mask_whitespace_rows=True)
    targets = build_sample_targets(bridge, alignment, ids, np.log(probs))

    # the full teacher vocabulary is given, so no row lacks a teacher factor
    assert set(targets.kinds.tolist()) <= set(TRAINED_KINDS) | {RowKind.UNSYNCED}
    assert targets.kinds[-1] == RowKind.STOP_BRIDGE
    trained = np.isin(targets.kinds, list(TRAINED_KINDS))
    assert (targets.probs >= 0).all() and (targets.other >= 0).all()
    totals = targets.probs.sum(axis=1) + targets.other
    np.testing.assert_allclose(totals[trained], 1.0, atol=1e-5)
    used = targets.ids[targets.probs > 0]
    assert set(used.tolist()) <= set(student) | {STUDENT_STOP}


def test_pack_and_unpack_align_rows_with_the_student_sequence(make_bridge, teacher_rows):
    bridge = make_bridge({1: "x", 2: "y"}, {11: "x", 12: "y"})
    targets = _build(
        bridge, [1, 2, STUDENT_STOP], [11, 12], [{11: 0.9, 12: 0.1}, {12: 1.0}, {TEACHER_STOP: 0.5}], teacher_rows
    )
    packed_ids, packed_probs = pack_targets(targets, prompt_len=4)
    width = target_width(2)
    assert packed_ids.shape == (7, 1 + width) and packed_ids.dtype == torch.int32
    kinds, other, ids, probs = unpack_targets(packed_ids, packed_probs)
    # row P - 1 + k predicts student response token k; the rest carry no target
    assert kinds.tolist() == [0, 0, 0, RowKind.ONE_TO_ONE, RowKind.ONE_TO_ONE, RowKind.STOP_BRIDGE, 0]
    assert ids.dtype == torch.int64 and ids.shape == (7, width)
    assert probs[3, :2].tolist() == pytest.approx([0.9, 0.1])
    assert other[5].item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="prompt_len"):
        pack_targets(targets, prompt_len=0)
