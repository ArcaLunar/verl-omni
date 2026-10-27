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
"""Synthetic tokenizer pairs for the BPM CPU tests."""

import numpy as np
import pytest

from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge

STUDENT_STOP = 900
TEACHER_STOP = 901


def _encode(vocab: dict[int, str | bytes]) -> dict[int, bytes]:
    return {token_id: text if isinstance(text, bytes) else text.encode("utf-8") for token_id, text in vocab.items()}


@pytest.fixture
def make_bridge():
    """Build a bridge from ``{id: text}`` vocabularies; stop ids are 900 (student) and 901 (teacher)."""

    def factory(student: dict[int, str | bytes], teacher: dict[int, str | bytes]) -> TokenizerBridge:
        return TokenizerBridge(
            student_token_bytes=_encode(student),
            teacher_token_bytes=_encode(teacher),
            student_stop_ids={STUDENT_STOP},
            teacher_stop_ids={TEACHER_STOP},
            student_control_ids=set(),
            teacher_control_ids=set(),
            student_turn_end_id=STUDENT_STOP,
            teacher_vocab_size=1000,
        )

    return factory


@pytest.fixture
def teacher_rows():
    """Turn ``[{teacher_id: prob}, ...]`` into padded top-k ``(ids, logprobs)`` arrays."""

    def factory(rows: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        topk = max(len(row) for row in rows)
        ids = np.full((len(rows), topk), -1, dtype=np.int64)
        logprobs = np.full((len(rows), topk), -np.inf)
        for index, row in enumerate(rows):
            for column, (token_id, prob) in enumerate(row.items()):
                ids[index, column] = token_id
                logprobs[index, column] = np.log(prob)
        return ids, logprobs

    return factory
