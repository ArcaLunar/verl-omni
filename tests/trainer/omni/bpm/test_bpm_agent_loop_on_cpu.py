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
"""BPM teacher step of the agent loop, against a fake teacher server."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from verl_omni.trainer.omni.bpm.agent_loop import BPMTeacher
from verl_omni.trainer.omni.bpm.alignment import TRAINED_KINDS, RowKind
from verl_omni.trainer.omni.bpm.encoding import unpack_targets
from verl_omni.trainer.omni.bpm.targets import target_width
from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge
from verl_omni.workers.config.omni import BPMConfig

TOPK = 4
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CORPUS = ["The answer is 42 because 6 times 7 is 42.", "Look at the picture: a red cube."] * 30


def _tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    )
    backend.train_from_iterator(CORPUS, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
    tokenizer.chat_template = TEMPLATE
    return tokenizer


class FakeTeacherClient:
    """Answers prefill requests like verl's vLLM server: one top-k row per prefill token."""

    def __init__(self, expanded_media_tokens: int = 0):
        self.calls = []
        self.expanded_media_tokens = expanded_media_tokens

    async def generate(self, request_id, *, prompt_ids, sampling_params, **media):
        self.calls.append({"prompt_ids": prompt_ids, "sampling_params": sampling_params, **media})
        # the engine expands media placeholders, so it sees more prompt tokens than were sent
        sequence = [0] * (self.expanded_media_tokens if media.get("image_data") else 0) + list(prompt_ids)
        rows_ids, rows_logprobs = [], []
        for position in range(len(sequence) - 1):
            realized = sequence[position + 1]
            fillers = [t for t in range(3, 3 + TOPK) if t != realized][: TOPK - 1]
            rows_ids.append([realized] + fillers)
            rows_logprobs.append([math.log(0.85)] + [math.log(0.05)] * (TOPK - 1))
        rows_ids.append([0] * TOPK)
        rows_logprobs.append([0.0] * TOPK)
        return SimpleNamespace(extra_fields={"prompt_ids": rows_ids, "prompt_logprobs": rows_logprobs})


@pytest.fixture(scope="module")
def tokenizers_pair():
    return _tokenizer(320), _tokenizer(290)


def _teacher(tokenizers_pair, client, **config):
    student, teacher = tokenizers_pair
    return BPMTeacher(
        bridge=TokenizerBridge.from_tokenizers(student, teacher),
        teacher_tokenizer=teacher,
        teacher_processor=None,
        client=client,
        bpm_config=BPMConfig(**config),
        topk=TOPK,
    )


def _student_sample(student):
    prompt_ids = student.apply_chat_template([{"role": "user", "content": "Why 42?"}], add_generation_prompt=True)
    response_ids = student.encode("The answer is 42.", add_special_tokens=False)
    return list(prompt_ids), response_ids + [student.convert_tokens_to_ids("<|im_end|>")]


@pytest.mark.asyncio
async def test_packed_targets_cover_the_student_sequence(tokenizers_pair):
    student, teacher = tokenizers_pair
    client = FakeTeacherClient(expanded_media_tokens=5)
    bpm = _teacher(tokenizers_pair, client, teacher_chat_template_kwargs={"enable_thinking": False})
    prompt_ids, response_ids = _student_sample(student)
    raw_prompt = np.array([{"role": "user", "content": "Why 42?"}], dtype=object)

    packed_ids, packed_probs = await bpm.packed_targets(prompt_ids, response_ids, raw_prompt, {"images": ["img"]})
    total = len(prompt_ids) + len(response_ids)
    assert packed_ids.shape == (total, 1 + target_width(TOPK)) == packed_probs.shape
    kinds, other, ids, probs = unpack_targets(packed_ids, packed_probs)
    response_kinds = kinds[len(prompt_ids) - 1 : total - 1].tolist()
    assert set(response_kinds) <= set(TRAINED_KINDS)
    assert response_kinds[-1] == RowKind.STOP_BRIDGE
    assert kinds[: len(prompt_ids) - 1].eq(0).all() and kinds[-1] == 0

    row = len(prompt_ids) - 1
    assert (probs[row].sum() + other[row]).item() == pytest.approx(1.0, abs=1e-5)
    # the stop row trains toward the teacher's mass on its eos, the token after the response
    assert probs[total - 2].sum().item() == pytest.approx(0.85, abs=1e-6)

    call = client.calls[0]
    assert call["sampling_params"] == {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": TOPK}
    assert call["image_data"] == ["img"] and call["video_data"] is None
    assert teacher.decode(call["prompt_ids"]).startswith("<|im_start|>user\nWhy 42?")


@pytest.mark.asyncio
async def test_unsupported_modality_and_invalid_text_skip_the_teacher(tokenizers_pair):
    student, _ = tokenizers_pair
    client = FakeTeacherClient()
    bpm = _teacher(tokenizers_pair, client)
    prompt_ids, response_ids = _student_sample(student)
    messages = [{"role": "user", "content": "Listen."}]

    packed_ids, _ = await bpm.packed_targets(prompt_ids, response_ids, messages, {"audios": ["clip"]})
    kinds = unpack_targets(packed_ids, packed_ids.float())[0]
    assert set(kinds[len(prompt_ids) - 1 : -1].tolist()) == {RowKind.UNSUPPORTED_INPUT}
    assert client.calls == []

    bad_byte = next(i for i, d in bpm.bridge.student_token_bytes.items() if d == b"\xff")
    packed_ids, _ = await bpm.packed_targets(prompt_ids, [bad_byte] + response_ids, messages, None)
    kinds = unpack_targets(packed_ids, packed_ids.float())[0]
    assert set(kinds[len(prompt_ids) - 1 : -1].tolist()) == {RowKind.BYTE_MISMATCH}
    assert client.calls == []
