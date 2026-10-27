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
"""Byte alignment of one student response with its teacher re-tokenization.

Both segmentations decode to one byte stream. Offsets shared by both are sync points;
each student position is classified by the segmentation relation of Table 2 of the
paper (ratios written teacher:student). Adapted from the official BPM implementation
(``slime_plugins/bpm/loss/bpm_megatron.py``).
"""

from bisect import bisect_right
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge

__all__ = ["RowKind", "RowPlan", "SampleAlignment", "TRAINED_KINDS", "align_sample"]


class RowKind(IntEnum):
    """Target construction for one student response position."""

    # no target: prompt, padding, tokens after the stop, extra control tokens
    NONE = 0
    # student token equals the aligned teacher token
    ONE_TO_ONE = 1
    # student token starts at a teacher boundary and ends inside that teacher token
    ONE_TO_N_HEAD = 2
    # student token starts and ends inside one teacher token
    ONE_TO_N_INTERIOR = 3
    # student token starts at a teacher boundary and spans past that teacher token
    N_TO_ONE = 4
    # student token starts inside a teacher token and spans past it
    M_TO_N = 5
    # realized student stop token, trained toward the teacher's stop mass
    STOP_BRIDGE = 6
    # interior position with a single possible continuation (one-hot target)
    FORCED_DELTA = 7
    # all-whitespace student token, withheld from the loss
    WHITESPACE_MASKED = 8
    # student and teacher byte streams differ
    BYTE_MISMATCH = 9
    # a teacher probability the target needs is not in the teacher's top-k
    TEACHER_TOKEN_MISSING = 10
    # position that neither starts on nor reaches a sync point
    UNSYNCED = 11
    # the prompt carries a modality the teacher does not accept
    UNSUPPORTED_INPUT = 12


# Rows that contribute to the loss.
TRAINED_KINDS = frozenset(
    {
        RowKind.ONE_TO_ONE,
        RowKind.ONE_TO_N_HEAD,
        RowKind.ONE_TO_N_INTERIOR,
        RowKind.N_TO_ONE,
        RowKind.M_TO_N,
        RowKind.STOP_BRIDGE,
        RowKind.FORCED_DELTA,
    }
)


@dataclass
class RowPlan:
    """Alignment of one student response position.

    For content rows, ``teacher_index`` is the governing teacher content token (the one the
    student token starts in), ``within`` the bytes of it already emitted before this row,
    and ``offset``/``length`` the student token's span in the content stream.
    """

    kind: RowKind
    student_id: int = -1
    teacher_index: int = -1
    within: int = 0
    offset: int = 0
    length: int = 0


@dataclass
class SampleAlignment:
    """Alignment of one response.

    Attributes are positional: ``rows[k]`` describes student response position ``k``;
    ``teacher_ids[n]`` is the n-th teacher content token, ``teacher_offsets[n]`` its byte
    offset (with a trailing end offset), and ``teacher_rows[n]`` its position in the teacher
    response, i.e. the teacher row that predicts it. ``teacher_stop_row`` predicts the stop.
    """

    rows: list[RowPlan]
    content_bytes: bytes
    teacher_ids: list[int]
    teacher_offsets: list[int]
    teacher_rows: list[int]
    teacher_stop_row: int
    realized_stop_id: Optional[int]

    def teacher_token_end(self, teacher_index: int) -> int:
        """Return the byte offset where teacher content token ``teacher_index`` ends."""
        return self.teacher_offsets[teacher_index + 1]


def _student_content(bridge: TokenizerBridge, response_ids: list[int]) -> tuple[list[int], Optional[int]]:
    """Return response positions of content tokens before the first stop, and that stop position."""
    content_positions = []
    for position, token_id in enumerate(response_ids):
        if token_id in bridge.student_stop_ids:
            return content_positions, position
        if token_id not in bridge.student_control_ids:
            content_positions.append(position)
    return content_positions, None


def align_sample(
    bridge: TokenizerBridge,
    student_response_ids: list[int],
    teacher_response_ids: list[int],
    *,
    mask_whitespace_rows: bool,
) -> SampleAlignment:
    """Classify every student response position against the teacher segmentation.

    Args:
        bridge: Tokenizer bridge of the pair.
        student_response_ids: Sampled student response ids, including the stop token if any.
        teacher_response_ids: Teacher tokenization of the student response text (no stop token).
        mask_whitespace_rows: Withhold all-whitespace student tokens from the loss.

    Returns:
        SampleAlignment: Per-position row plans plus the teacher-side offsets they refer to.
    """
    rows = [RowPlan(RowKind.NONE) for _ in student_response_ids]
    content_positions, stop_position = _student_content(bridge, student_response_ids)

    teacher_ids, teacher_rows = [], []
    for position, token_id in enumerate(teacher_response_ids):
        if token_id not in bridge.teacher_control_ids:
            teacher_ids.append(token_id)
            teacher_rows.append(position)
    teacher_offsets = [0]
    known = all(t in bridge.teacher_token_bytes for t in teacher_ids)
    known = known and all(student_response_ids[p] in bridge.student_token_bytes for p in content_positions)
    if known:
        for token_id in teacher_ids:
            teacher_offsets.append(teacher_offsets[-1] + len(bridge.teacher_token_bytes[token_id]))
        content_bytes = b"".join(bridge.student_token_bytes[student_response_ids[p]] for p in content_positions)
        teacher_bytes = b"".join(bridge.teacher_token_bytes[t] for t in teacher_ids)
    else:
        content_bytes = teacher_bytes = b""

    alignment = SampleAlignment(
        rows=rows,
        content_bytes=content_bytes,
        teacher_ids=teacher_ids,
        teacher_offsets=teacher_offsets,
        teacher_rows=teacher_rows,
        teacher_stop_row=len(teacher_response_ids),
        realized_stop_id=None if stop_position is None else student_response_ids[stop_position],
    )
    # The teacher text may drop a truncated multi-byte character at the end of the response.
    if not known or not content_bytes.startswith(teacher_bytes):
        for position in content_positions:
            rows[position] = RowPlan(RowKind.BYTE_MISMATCH, student_id=student_response_ids[position])
        return alignment

    synced_len = len(teacher_bytes)
    teacher_index_at = {offset: index for index, offset in enumerate(teacher_offsets[:-1])}
    student_offsets = [0]
    for position in content_positions:
        student_offsets.append(student_offsets[-1] + len(bridge.student_token_bytes[student_response_ids[position]]))
    sync_points = set(student_offsets) & set(teacher_offsets)
    for content_index, position in enumerate(content_positions):
        student_id = student_response_ids[position]
        start, end = student_offsets[content_index], student_offsets[content_index + 1]
        length = end - start
        if end > synced_len:
            kind, governing, within = RowKind.UNSYNCED, -1, 0
        elif mask_whitespace_rows and student_id in bridge.student_whitespace_ids:
            kind, governing, within = RowKind.WHITESPACE_MASKED, -1, 0
        elif start in teacher_index_at:
            governing, within = teacher_index_at[start], 0
            teacher_end = teacher_offsets[governing + 1]
            if end == teacher_end:
                kind = RowKind.ONE_TO_ONE
            elif end < teacher_end:
                kind = RowKind.ONE_TO_N_HEAD
            else:
                kind = RowKind.N_TO_ONE
        else:
            governing = bisect_right(teacher_offsets, start) - 1
            within = start - teacher_offsets[governing]
            teacher_end = teacher_offsets[governing + 1]
            if end > teacher_end:
                kind = RowKind.M_TO_N
            elif teacher_end in sync_points:
                kind = RowKind.ONE_TO_N_INTERIOR
            else:
                kind = RowKind.UNSYNCED
        rows[position] = RowPlan(kind, student_id, governing, within, start, length)

    if stop_position is not None and synced_len == len(content_bytes):
        rows[stop_position] = RowPlan(RowKind.STOP_BRIDGE, student_id=student_response_ids[stop_position])
    return alignment
