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
"""BPM row classification against the teacher segmentation (Table 2 of the paper)."""

from verl_omni.trainer.omni.bpm.alignment import RowKind, align_sample

STUDENT_STOP = 900  # the make_bridge fixture's student stop id


def _kinds(alignment):
    return [row.kind for row in alignment.rows]


def test_one_to_one_and_one_to_n_head_interior(make_bridge):
    # teacher "10" splits into student "1" + "0"
    bridge = make_bridge({1: "1", 2: "0", 3: "x"}, {11: "10", 13: "x"})
    alignment = align_sample(bridge, [3, 1, 2, STUDENT_STOP], [13, 11], whitespace_mask="all")
    assert _kinds(alignment) == [
        RowKind.ONE_TO_ONE,
        RowKind.ONE_TO_N_HEAD,
        RowKind.ONE_TO_N_INTERIOR,
        RowKind.STOP_BRIDGE,
    ]
    interior = alignment.rows[2]
    assert (interior.teacher_index, interior.within, interior.offset, interior.length) == (1, 1, 2, 1)
    assert alignment.realized_stop_id == STUDENT_STOP
    assert alignment.teacher_stop_row == 2


def test_spanning_rows_n_to_one_and_m_to_n(make_bridge):
    # </think> spans three teacher tokens; "bc" starts inside "ab" and spans past it
    bridge = make_bridge(
        {1: "</think>", 2: "a", 3: "bc", 4: "d"},
        {11: "</", 12: "think", 13: ">", 21: "ab", 22: "cd"},
    )
    alignment = align_sample(bridge, [1, 2, 3, 4], [11, 12, 13, 21, 22], whitespace_mask="all")
    assert _kinds(alignment) == [
        RowKind.N_TO_ONE,
        RowKind.ONE_TO_N_HEAD,
        RowKind.M_TO_N,
        RowKind.ONE_TO_N_INTERIOR,
    ]
    assert alignment.rows[2].teacher_index == 3 and alignment.rows[2].within == 1


def test_interior_row_without_sync_end_is_unsynced(make_bridge):
    # "b" sits inside teacher "abc", whose end (offset 3) is not a student boundary
    bridge = make_bridge({1: "a", 2: "b", 3: "cd"}, {11: "abc", 12: "d"})
    alignment = align_sample(bridge, [1, 2, 3], [11, 12], whitespace_mask="all")
    assert _kinds(alignment) == [RowKind.ONE_TO_N_HEAD, RowKind.UNSYNCED, RowKind.M_TO_N]


def test_whitespace_mask_byte_mismatch_and_tokens_after_stop(make_bridge):
    bridge = make_bridge({1: "x", 2: "  ", 3: "y"}, {11: "x", 12: "  ", 13: "y", 14: "z"})
    masked = align_sample(bridge, [1, 2, 3, STUDENT_STOP, 1], [11, 12, 13], whitespace_mask="all")
    assert _kinds(masked) == [
        RowKind.ONE_TO_ONE,
        RowKind.WHITESPACE_MASKED,
        RowKind.ONE_TO_ONE,
        RowKind.STOP_BRIDGE,
        RowKind.NONE,
    ]
    unmasked = align_sample(bridge, [1, 2, 3], [11, 12, 13], whitespace_mask="off")
    assert unmasked.rows[1].kind == RowKind.ONE_TO_ONE

    mismatch = align_sample(bridge, [1, 3, STUDENT_STOP], [11, 14], whitespace_mask="all")
    assert _kinds(mismatch) == [RowKind.BYTE_MISMATCH, RowKind.BYTE_MISMATCH, RowKind.NONE]


def test_truncated_multibyte_tail_is_unsynced_without_stop_row(make_bridge):
    # the student stopped mid-character: the teacher text drops the incomplete bytes
    bridge = make_bridge({1: "a", 2: b"\xe4\xb8"}, {11: "a"})
    alignment = align_sample(bridge, [1, 2], [11], whitespace_mask="all")
    assert _kinds(alignment) == [RowKind.ONE_TO_ONE, RowKind.UNSYNCED]
    assert alignment.realized_stop_id is None


def test_whitespace_mask_modes_separate_indentation_from_spaces_before_numbers(make_bridge):
    # Qwen-style segmentation: a lone " " before a number, and indentation split over two tokens
    vocab = {1: "x", 2: " =", 3: " ", 4: "1", 5: "\n", 6: "  ", 7: "y"}
    bridge = make_bridge(vocab, {10 + i: text for i, text in vocab.items()})
    student = [1, 2, 3, 4, 5, 6, 6, 7]  # "x = 1\n    y"
    teacher = [10 + i for i in student]
    masked = {
        mode: [
            i
            for i, row in enumerate(align_sample(bridge, student, teacher, whitespace_mask=mode).rows)
            if row.kind == RowKind.WHITESPACE_MASKED
        ]
        for mode in ("all", "indentation", "off")
    }
    assert masked == {"all": [2, 5, 6], "indentation": [5, 6], "off": []}
