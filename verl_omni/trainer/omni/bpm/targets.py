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
"""BPM targets on the student vocabulary, built from teacher top-k rows.

Per student response position the target is a sparse distribution over explicit student
tokens; the remaining mass is the complement cell (the paper's ``∅``). Constructions:

- boundary rows: ``t(u) = sum_{phi(w)=u} q(w)`` (Eq. 4), teacher stop mass routed to the
  student stop token;
- interior rows: the conditional scatter given the realized prefix (Eq. 5);
- spanning rows and longer counterfactual student tokens: the realized-path chain
  ``prod_k q_k(w_k) * pi(r)`` (Eq. 6), moved off the head cell with telescoping
  differences between nested candidates;
- the stop row: the teacher's stop mass on the realized student stop token (Eq. 8).

Teacher mass outside the top-k lands in the complement. Rows that need the probability of
a realized teacher token missing from the top-k are excluded. Adapted from the official
BPM implementation (``slime_plugins/bpm/loss/bpm_megatron.py``, ``bpm_kernels.py``).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from verl_omni.trainer.omni.bpm.alignment import RowKind, SampleAlignment
from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge

__all__ = ["MAX_CHAIN_CANDIDATES", "SampleTargets", "build_sample_targets", "target_width"]

# Nested spanning candidates considered per row (the official implementation's cap).
MAX_CHAIN_CANDIDATES = 8

_BOUNDARY_KINDS = frozenset({RowKind.ONE_TO_ONE, RowKind.ONE_TO_N_HEAD, RowKind.N_TO_ONE})


def target_width(topk: int) -> int:
    """Return the number of explicit target cells per row for a teacher top-``topk``."""
    return topk + MAX_CHAIN_CANDIDATES


@dataclass
class SampleTargets:
    """Sparse targets for one student response.

    ``kinds`` is int8 ``(R,)``; ``ids`` int64 ``(R, W)`` and ``probs`` float32 ``(R, W)``
    hold the explicit cells (unused cells have probability 0); ``other`` float32 ``(R,)``
    is the complement mass, so each trained row sums to one.
    """

    kinds: np.ndarray
    ids: np.ndarray
    probs: np.ndarray
    other: np.ndarray


class _TeacherRows:
    """Teacher top-k rows of one response; row ``j`` predicts teacher response token ``j``."""

    def __init__(self, bridge: TokenizerBridge, topk_ids: np.ndarray, topk_logprobs: np.ndarray):
        self.bridge = bridge
        self.ids = np.asarray(topk_ids, dtype=np.int64)
        logprobs = np.asarray(topk_logprobs, dtype=np.float64)
        self.probs = np.where(self.ids >= 0, np.exp(logprobs), 0.0)

    def contains(self, row: int, token_id: int) -> bool:
        return bool(np.any(self.ids[row] == token_id))

    def _mass(self, row: int, keep) -> float:
        token_bytes = self.bridge.teacher_token_bytes
        total = 0.0
        for token_id, prob in zip(self.ids[row].tolist(), self.probs[row].tolist(), strict=True):
            data = token_bytes.get(token_id)
            if data is not None and keep(data):
                total += prob
        return total

    def prefix_mass(self, row: int, prefix: bytes) -> float:
        return self._mass(row, lambda data: data.startswith(prefix))

    def exact_mass(self, row: int, target: bytes) -> float:
        return self._mass(row, lambda data: data == target)

    def stop_mass(self, row: int) -> float:
        return float(self.probs[row][np.isin(self.ids[row], self.bridge.teacher_stop_id_array)].sum())


class _TargetBuilder:
    def __init__(self, bridge: TokenizerBridge, alignment: SampleAlignment, rows: _TeacherRows):
        self.bridge = bridge
        self.alignment = alignment
        self.rows = rows
        stop_id = alignment.realized_stop_id
        self.stop_target = bridge.student_turn_end_id if stop_id is None else stop_id

    def scatter_rows(self, teacher_rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Eq. 4 for many teacher rows at once, teacher stop mass routed to the student stop token.

        Returns:
            tuple: int64 ``(B, K)`` student ids and float64 ``(B, K)`` masses; duplicate student
            ids are merged and unused cells have mass 0 and id 0.
        """
        ids, probs = self.rows.ids[teacher_rows], self.rows.probs[teacher_rows]
        phi = self.bridge.phi
        in_range = (ids >= 0) & (ids < len(phi))
        student_ids = np.where(in_range, phi[np.clip(ids, 0, len(phi) - 1)], -1)
        student_ids = np.where(np.isin(ids, self.bridge.teacher_stop_id_array), self.stop_target, student_ids)
        probs = np.where(student_ids >= 0, probs, 0.0)

        order = np.argsort(student_ids, axis=1, kind="stable")
        sorted_ids = np.take_along_axis(student_ids, order, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        starts = np.concatenate(
            [np.zeros((len(teacher_rows), 1), dtype=bool), sorted_ids[:, 1:] != sorted_ids[:, :-1]], axis=1
        )
        groups = np.cumsum(starts, axis=1)  # [B, K] merged-cell index per entry
        row_index = np.broadcast_to(np.arange(len(teacher_rows))[:, None], groups.shape)
        merged_ids = np.zeros_like(sorted_ids)
        merged_probs = np.zeros_like(sorted_probs)
        merged_ids[row_index, groups] = sorted_ids
        np.add.at(merged_probs, (row_index, groups), sorted_probs)
        merged_ids = np.where((merged_probs > 0.0) & (merged_ids >= 0), merged_ids, 0)
        return merged_ids, merged_probs

    def conditional(
        self, teacher_index: int, within: int, remaining_len: int
    ) -> Optional[tuple[dict[int, float], float]]:
        """Eq. 5: teacher mass continuing the realized prefix, routed by its next student token.

        Returns the normalized target and the prefix mass ``pi(r_pre)``, or ``None`` when the
        realized teacher token is not in the top-k.
        """
        alignment = self.alignment
        teacher_row = alignment.teacher_rows[teacher_index]
        realized_id = alignment.teacher_ids[teacher_index]
        if not self.rows.contains(teacher_row, realized_id):
            return None
        prefix = self.bridge.teacher_token_bytes[realized_id][:within]
        token_bytes = self.bridge.teacher_token_bytes
        target: dict[int, float] = {}
        prefix_mass = 0.0
        for token_id, prob in zip(
            self.rows.ids[teacher_row].tolist(), self.rows.probs[teacher_row].tolist(), strict=True
        ):
            data = token_bytes.get(token_id)
            if data is None or not data.startswith(prefix):
                continue
            prefix_mass += prob
            student_id = self.bridge.longest_student_prefix(data[within:], max_len=remaining_len)
            if student_id >= 0:
                target[student_id] = target.get(student_id, 0.0) + prob
        return {student_id: mass / prefix_mass for student_id, mass in target.items()}, prefix_mass

    def chain_mass(self, teacher_index: int, data: bytes) -> Optional[float]:
        """Eq. 6: probability the teacher emits ``data`` from the start of ``teacher_index``
        along its realized segmentation; ``None`` if a needed factor is unavailable."""
        alignment = self.alignment
        mass, consumed, index = 1.0, 0, teacher_index
        while True:
            if index >= len(alignment.teacher_ids):
                return None
            teacher_row = alignment.teacher_rows[index]
            if index > teacher_index and teacher_row != alignment.teacher_rows[index - 1] + 1:
                return None  # a removed control token sits between the factors
            realized_id = alignment.teacher_ids[index]
            if not self.rows.contains(teacher_row, realized_id):
                return None
            token = self.bridge.teacher_token_bytes[realized_id]
            remaining = data[consumed:]
            if len(remaining) <= len(token):
                return mass * self.rows.prefix_mass(teacher_row, remaining)
            mass *= self.rows.exact_mass(teacher_row, token)
            consumed += len(token)
            index += 1

    def candidate_masses(
        self, teacher_index: int, start: int, within: int, realized: Optional[tuple[int, int]]
    ) -> Optional[list[tuple[int, float]]]:
        """Chain masses of nested student tokens at ``start`` that span past the governing
        teacher token. ``realized`` must be among them; ``None`` if its chain is unavailable."""
        alignment = self.alignment
        teacher_end = alignment.teacher_token_end(teacher_index)
        candidates = self.bridge.student_prefix_candidates(
            alignment.content_bytes, start, teacher_end - start, MAX_CHAIN_CANDIDATES
        )
        if realized is not None and realized not in candidates:
            candidates = [c for c in candidates if c[1] < realized[1]][: MAX_CHAIN_CANDIDATES - 1] + [realized]
        head = self.bridge.teacher_token_bytes[alignment.teacher_ids[teacher_index]][:within]
        masses = []
        for student_id, length in candidates:
            mass = self.chain_mass(teacher_index, head + alignment.content_bytes[start : start + length])
            if mass is None:
                # Deeper candidates, the realized token included, share the missing factor.
                if realized is not None:
                    return None
                break
            masses.append((student_id, mass))
        return masses


def _move_chain_mass(target: dict[int, float], head_id: int, masses: list[tuple[int, float]]) -> None:
    """Move chain mass off the head cell onto nested candidates c_1 < ... < c_m:
    ``head -= P_1``, ``c_i += P_i - P_{i+1}``, ``c_m += P_m``. A head of ``-1`` draws on the
    complement, which absorbs the change when the row is finalized."""
    if not masses:
        return
    if head_id >= 0:
        target[head_id] = max(0.0, target.get(head_id, 0.0) - masses[0][1])
    for index, (student_id, mass) in enumerate(masses):
        deeper = masses[index + 1][1] if index + 1 < len(masses) else 0.0
        target[student_id] = target.get(student_id, 0.0) + max(0.0, mass - deeper)


def _read_row(targets: SampleTargets, position: int) -> dict[int, float]:
    target: dict[int, float] = {}
    for student_id, prob in zip(targets.ids[position].tolist(), targets.probs[position].tolist(), strict=True):
        if prob > 0.0:
            target[student_id] = target.get(student_id, 0.0) + prob
    return target


def _write_row(targets: SampleTargets, position: int, target: dict[int, float]) -> None:
    width = targets.ids.shape[1]
    cells = sorted(((p, s) for s, p in target.items() if p > 0.0), reverse=True)[:width]
    total = sum(p for p, _ in cells)
    scale = 1.0 / total if total > 1.0 else 1.0
    targets.ids[position] = 0
    targets.probs[position] = 0.0
    for column, (prob, student_id) in enumerate(cells):
        targets.ids[position, column] = student_id
        targets.probs[position, column] = prob * scale
    targets.other[position] = max(0.0, 1.0 - total * scale)


def build_sample_targets(
    bridge: TokenizerBridge,
    alignment: SampleAlignment,
    teacher_topk_ids: np.ndarray,
    teacher_topk_logprobs: np.ndarray,
) -> SampleTargets:
    """Build BPM targets for every student response position of one sample.

    Args:
        bridge: Tokenizer bridge of the pair.
        alignment: Output of :func:`align_sample` for this response.
        teacher_topk_ids: int ``(L + 1, K)`` teacher top-k ids; row ``j < L`` predicts teacher
            response token ``j``, row ``L`` predicts the stop. Negative ids are empty slots.
        teacher_topk_logprobs: float ``(L + 1, K)`` log-probabilities matching ``teacher_topk_ids``.

    Returns:
        SampleTargets: Kinds, explicit cells and complement mass per student position.
    """
    topk = np.asarray(teacher_topk_ids).shape[1]
    num_rows = len(alignment.rows)
    targets = SampleTargets(
        kinds=np.array([row.kind for row in alignment.rows], dtype=np.int8),
        ids=np.zeros((num_rows, target_width(topk)), dtype=np.int64),
        probs=np.zeros((num_rows, target_width(topk)), dtype=np.float32),
        other=np.zeros((num_rows,), dtype=np.float32),
    )
    builder = _TargetBuilder(bridge, alignment, _TeacherRows(bridge, teacher_topk_ids, teacher_topk_logprobs))

    boundary_positions = [p for p, row in enumerate(alignment.rows) if row.kind in _BOUNDARY_KINDS]
    if boundary_positions:
        teacher_rows = [alignment.teacher_rows[alignment.rows[p].teacher_index] for p in boundary_positions]
        merged_ids, merged_probs = builder.scatter_rows(teacher_rows)
        targets.ids[boundary_positions, :topk] = merged_ids
        targets.probs[boundary_positions, :topk] = merged_probs
        targets.other[boundary_positions] = np.clip(1.0 - merged_probs.sum(axis=1), 0.0, 1.0)

    for position, row in enumerate(alignment.rows):
        target: Optional[dict[int, float]] = None
        if row.kind in _BOUNDARY_KINDS:
            realized = (row.student_id, row.length) if row.kind == RowKind.N_TO_ONE else None
            masses = builder.candidate_masses(row.teacher_index, row.offset, 0, realized)
            if masses is not None and not masses:
                continue  # the batched scatter is already the exact target
            if masses is not None:
                target = _read_row(targets, position)
                head_id = int(bridge.phi[alignment.teacher_ids[row.teacher_index]])
                _move_chain_mass(target, head_id, masses)
        elif row.kind == RowKind.ONE_TO_N_INTERIOR:
            remaining_len = alignment.teacher_token_end(row.teacher_index) - row.offset
            result = builder.conditional(row.teacher_index, row.within, remaining_len)
            if result is not None:
                target = result[0]
                if set(target) == {row.student_id} and abs(target[row.student_id] - 1.0) < 1e-9:
                    targets.kinds[position] = RowKind.FORCED_DELTA
                    target = {row.student_id: 1.0}
        elif row.kind == RowKind.M_TO_N:
            remaining_len = alignment.teacher_token_end(row.teacher_index) - row.offset
            result = builder.conditional(row.teacher_index, row.within, remaining_len)
            masses = builder.candidate_masses(row.teacher_index, row.offset, row.within, (row.student_id, row.length))
            if result is not None and masses is not None:
                target, prefix_mass = result
                realized_token = bridge.teacher_token_bytes[alignment.teacher_ids[row.teacher_index]]
                head_id = bridge.longest_student_prefix(realized_token[row.within :], max_len=remaining_len)
                _move_chain_mass(target, head_id, [(s, m / prefix_mass) for s, m in masses])
        elif row.kind == RowKind.STOP_BRIDGE:
            stop_mass = builder.rows.stop_mass(alignment.teacher_stop_row)
            if stop_mass > 0.0:
                target = {row.student_id: stop_mass}
            else:
                targets.kinds[position] = RowKind.NONE
                continue
        else:
            continue

        if target is None:
            targets.kinds[position] = RowKind.TEACHER_TOKEN_MISSING
            targets.ids[position] = 0
            targets.probs[position] = 0.0
            targets.other[position] = 0.0
        else:
            _write_row(targets, position, target)
    return targets
