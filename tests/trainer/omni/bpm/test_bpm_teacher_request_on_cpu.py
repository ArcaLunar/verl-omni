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
"""BPM teacher prefill request: response text, think seam, template rendering, row slicing."""

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from verl_omni.trainer.omni.bpm.alignment import RowKind, align_sample
from verl_omni.trainer.omni.bpm.teacher_request import (
    build_teacher_request,
    normalize_think_seam,
    student_response_text,
    teacher_response_rows,
)
from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge

STUDENT_STOP = 900  # the make_bridge fixture's student stop id

TEACHER_TEMPLATE = (
    "{% for m in messages %}<|user|>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% if enable_thinking %}<think>\n{% endif %}{% endif %}"
)


def _teacher_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=280,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )
    backend.train_from_iterator(["solve 2 + 2 = 4, the answer is 4"] * 20, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
    tokenizer.add_tokens(["<think>", "</think>"])
    tokenizer.chat_template = TEACHER_TEMPLATE
    return tokenizer


@pytest.mark.parametrize(
    ("prompt", "response", "expected"),
    [
        ("<|assistant|><think>\n", "<think>\nhmm</think>4", "<|assistant|>"),
        ("<|assistant|>", "hmm</think>4", "<|assistant|><think>\n"),
        ("<|assistant|><think>\n", "hmm</think>4", "<|assistant|><think>\n"),
        ("<|assistant|>", "4", "<|assistant|>"),
    ],
)
def test_normalize_think_seam(prompt, response, expected):
    assert normalize_think_seam(prompt, response) == expected


def test_student_response_text_stops_at_stop_and_trims_cut_character(make_bridge):
    bridge = make_bridge({1: "ab", 2: b"\xe4\xb8", 3: b"\xad", 4: b"\xff"}, {})
    assert student_response_text(bridge, [1, 2, 3, STUDENT_STOP, 1]) == "ab中"
    assert student_response_text(bridge, [1, 2]) == "ab"  # cut by the length limit
    assert student_response_text(bridge, [4, 1]) is None  # invalid UTF-8 mid-response
    assert student_response_text(bridge, [77]) is None  # unknown id


def test_build_teacher_request_round_trips_response_bytes():
    tokenizer = _teacher_tokenizer()
    messages = [{"role": "user", "content": "solve 2 + 2"}]
    response = "the answer</think>4"
    request = build_teacher_request(
        messages, response, tokenizer=tokenizer, chat_template_kwargs={"enable_thinking": False}
    )
    # the bare </think> in the response gets a matching <think> on the prompt side
    assert tokenizer.decode(request.prompt_ids) == "<|user|>solve 2 + 2<|assistant|><think>\n"
    assert tokenizer.decode(request.response_ids) == response
    assert request.teacher_ids[-1] == tokenizer.eos_token_id
    assert request.teacher_ids == request.prompt_ids + request.response_ids + [tokenizer.eos_token_id]

    bridge = TokenizerBridge.from_tokenizers(tokenizer, tokenizer)
    student_ids = tokenizer.encode(response, add_special_tokens=False)
    alignment = align_sample(bridge, student_ids, request.response_ids, mask_whitespace_rows=True)
    assert all(row.kind == RowKind.ONE_TO_ONE for row in alignment.rows)


def test_teacher_response_rows_are_taken_from_the_end():
    # 3 prompt tokens expanded to 5 by the engine, 2 response tokens, 1 stop: 8 rows
    num_rows, response_len = 8, 2
    ids = [[row * 10, None] for row in range(num_rows)]
    logprobs = [[-0.1, None] for _ in range(num_rows)]
    row_ids, row_logprobs = teacher_response_rows(ids, logprobs, response_len)
    assert row_ids.tolist() == [[40, -1], [50, -1], [60, -1]]  # rows N-4 .. N-2; row N-1 is padding
    assert np.isneginf(row_logprobs[:, 1]).all()
    with pytest.raises(ValueError, match="needs more"):
        teacher_response_rows(ids[:3], logprobs[:3], response_len)
