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
"""Per-tokenizer-pair precomputation for BPM.

Builds the byte representation of both vocabularies and the longest-prefix map
``phi(w) = argmax_{u: bytes(u) prefix of bytes(w)} |bytes(u)|`` (Eq. 3 of the paper),
with ``-1`` standing for the paper's ``⊥``.

Adapted from the official BPM implementation (``slime_plugins/bpm/loss/bpm_phi.py`` and
``slime_plugins/bpm/backend/alignment.py``).
"""

from typing import Optional

import numpy as np

from verl_omni.trainer.omni.bpm.stop_tokens import chat_template_turn_end_ids, detect_stop_token_ids

__all__ = ["TokenizerBridge", "is_byte_level_tokenizer", "tokenizer_token_bytes"]

_WHITESPACE_BYTES = frozenset(b" \t")


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE alphabet: byte value -> printable character."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + shift)
            shift += 1
    return dict(zip(printable, (chr(c) for c in chars), strict=True))


_BYTE_DECODER = {char: byte for byte, char in _bytes_to_unicode().items()}


def is_byte_level_tokenizer(tokenizer) -> bool:
    """Return whether ``tokenizer`` is a byte-level BPE tokenizer (assumption A1 of the paper)."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    return "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(backend.decoder)


def tokenizer_token_bytes(tokenizer, skip_ids: set[int]) -> dict[int, bytes]:
    """Map every content token id of a byte-level tokenizer to its raw bytes.

    Added tokens (e.g. ``<think>``) render as their literal UTF-8 content. Ids in
    ``skip_ids`` and pieces outside the byte-level alphabet are left out.

    Args:
        tokenizer: Byte-level BPE tokenizer.
        skip_ids: Control and stop token ids that carry no response bytes.

    Returns:
        dict[int, bytes]: Token id to non-empty byte string.
    """
    added_contents = {int(token_id): token.content for token_id, token in tokenizer.added_tokens_decoder.items()}
    token_bytes: dict[int, bytes] = {}
    for piece, token_id in tokenizer.get_vocab().items():
        token_id = int(token_id)
        if token_id in skip_ids:
            continue
        if token_id in added_contents:
            data = added_contents[token_id].encode("utf-8")
        else:
            try:
                data = bytes(_BYTE_DECODER[char] for char in piece)
            except KeyError:
                continue
        if data:
            token_bytes[token_id] = data
    return token_bytes


def _control_ids(tokenizer) -> set[int]:
    ids = set()
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


class TokenizerBridge:
    """Byte maps, the ``phi`` routing table and stop sets for one (student, teacher) pair.

    Args:
        student_token_bytes: Student content token id to bytes (no control or stop ids).
        teacher_token_bytes: Teacher content token id to bytes (no control or stop ids).
        student_stop_ids: Student assistant-turn stop ids.
        teacher_stop_ids: Teacher assistant-turn stop ids.
        student_control_ids: Student ids that carry no response bytes; includes the stop ids.
        teacher_control_ids: Teacher ids that carry no response bytes; includes the stop ids.
        student_turn_end_id: Student stop token that receives teacher stop mass when a response
            has no realized stop token.
        teacher_vocab_size: Size of the teacher id space; ``phi`` has this length.
    """

    def __init__(
        self,
        *,
        student_token_bytes: dict[int, bytes],
        teacher_token_bytes: dict[int, bytes],
        student_stop_ids: set[int],
        teacher_stop_ids: set[int],
        student_control_ids: set[int],
        teacher_control_ids: set[int],
        student_turn_end_id: int,
        teacher_vocab_size: int,
    ):
        if not student_stop_ids or not teacher_stop_ids:
            raise ValueError("Both student and teacher need at least one stop token id.")
        if student_turn_end_id not in student_stop_ids:
            raise ValueError(f"student_turn_end_id={student_turn_end_id} is not a student stop id.")
        self.student_token_bytes = student_token_bytes
        self.teacher_token_bytes = teacher_token_bytes
        self.student_stop_ids = frozenset(student_stop_ids)
        self.teacher_stop_ids = frozenset(teacher_stop_ids)
        self.student_control_ids = frozenset(student_control_ids | student_stop_ids)
        self.teacher_control_ids = frozenset(teacher_control_ids | teacher_stop_ids)
        self.student_turn_end_id = int(student_turn_end_id)
        self.teacher_stop_id_array = np.array(sorted(self.teacher_stop_ids), dtype=np.int64)

        self._student_id_by_bytes: dict[bytes, int] = {}
        for token_id in sorted(student_token_bytes):
            self._student_id_by_bytes.setdefault(student_token_bytes[token_id], token_id)
        self._student_prefixes = {data[:end] for data in self._student_id_by_bytes for end in range(1, len(data) + 1)}
        self.max_student_token_len = max(len(data) for data in self._student_id_by_bytes)
        self.student_whitespace_ids = frozenset(
            token_id for token_id, data in student_token_bytes.items() if set(data) <= _WHITESPACE_BYTES
        )

        self.phi = np.full(teacher_vocab_size, -1, dtype=np.int64)
        for token_id, data in teacher_token_bytes.items():
            if token_id < teacher_vocab_size:
                self.phi[token_id] = self.longest_student_prefix(data)

    @classmethod
    def from_tokenizers(
        cls,
        student_tokenizer,
        teacher_tokenizer,
        *,
        student_model_path: Optional[str] = None,
        teacher_model_path: Optional[str] = None,
        student_template_owner=None,
        teacher_template_owner=None,
    ) -> "TokenizerBridge":
        """Build the bridge from two Hugging Face byte-level BPE tokenizers.

        Args:
            student_tokenizer: Student tokenizer.
            teacher_tokenizer: Teacher tokenizer.
            student_model_path: Student model directory, read for ``generation_config`` eos ids.
            teacher_model_path: Teacher model directory, read for ``generation_config`` eos ids.
            student_template_owner: Object owning the student chat template (e.g. a processor).
            teacher_template_owner: Object owning the teacher chat template (e.g. a processor).

        Returns:
            TokenizerBridge: The bridge for this tokenizer pair.

        Raises:
            ValueError: If either tokenizer is not byte-level BPE.
        """
        for name, tokenizer in (("student", student_tokenizer), ("teacher", teacher_tokenizer)):
            if not is_byte_level_tokenizer(tokenizer):
                raise ValueError(f"BPM requires byte-level BPE tokenizers; the {name} tokenizer is not one.")
        student_stop_ids = detect_stop_token_ids(student_tokenizer, student_model_path, student_template_owner)
        teacher_stop_ids = detect_stop_token_ids(teacher_tokenizer, teacher_model_path, teacher_template_owner)
        student_control_ids = _control_ids(student_tokenizer) | student_stop_ids
        teacher_control_ids = _control_ids(teacher_tokenizer) | teacher_stop_ids
        turn_end_ids = chat_template_turn_end_ids(student_tokenizer, student_template_owner) & student_stop_ids
        student_turn_end_id = min(turn_end_ids) if turn_end_ids else int(student_tokenizer.eos_token_id)
        return cls(
            student_token_bytes=tokenizer_token_bytes(student_tokenizer, student_control_ids),
            teacher_token_bytes=tokenizer_token_bytes(teacher_tokenizer, teacher_control_ids),
            student_stop_ids=student_stop_ids,
            teacher_stop_ids=teacher_stop_ids,
            student_control_ids=student_control_ids,
            teacher_control_ids=teacher_control_ids,
            student_turn_end_id=student_turn_end_id,
            teacher_vocab_size=max(int(i) for i in teacher_tokenizer.get_vocab().values()) + 1,
        )

    def longest_student_prefix(self, data: bytes, max_len: Optional[int] = None) -> int:
        """Return the student token with the longest byte string that prefixes ``data``.

        Args:
            data: Byte string to segment.
            max_len: Upper bound on the matched token length, in bytes.

        Returns:
            int: Student token id, or ``-1`` when no student token prefixes ``data``.
        """
        limit = len(data) if max_len is None else min(len(data), max_len)
        for end in range(limit, 0, -1):
            token_id = self._student_id_by_bytes.get(data[:end])
            if token_id is not None:
                return token_id
        return -1

    def student_prefix_candidates(
        self, stream: bytes, start: int, min_len_exclusive: int, max_candidates: int
    ) -> list[tuple[int, int]]:
        """List student tokens that prefix ``stream[start:]`` and are longer than ``min_len_exclusive``.

        Args:
            stream: Realized response bytes.
            start: Byte offset where the candidate tokens begin.
            min_len_exclusive: Candidates must be strictly longer than this many bytes.
            max_candidates: Maximum number of candidates returned.

        Returns:
            list[tuple[int, int]]: ``(student_id, byte_length)`` pairs in increasing length.
        """
        candidates = []
        end_limit = min(len(stream), start + self.max_student_token_len)
        for end in range(start + 1, end_limit + 1):
            piece = stream[start:end]
            if piece not in self._student_prefixes:
                break
            token_id = self._student_id_by_bytes.get(piece)
            if token_id is not None and end - start > min_len_exclusive:
                candidates.append((token_id, end - start))
                if len(candidates) >= max_candidates:
                    break
        return candidates
