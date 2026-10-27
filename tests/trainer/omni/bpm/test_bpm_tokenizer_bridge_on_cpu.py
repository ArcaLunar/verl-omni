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
"""BPM tokenizer bridge: byte maps, phi, prefix queries and stop detection."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from verl_omni.trainer.omni.bpm.stop_tokens import detect_stop_token_ids
from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge, is_byte_level_tokenizer

# Byte maps and expected phi from the official BPM golden baseline (tests/golden/golden_core.json).
GOLDEN_TEACHER_HEX = {
    1: "3132",
    2: "313230",
    3: "39",
    4: "3133",
    5: "33",
    6: "31",
    10: "e4b8ad",
    11: "e4b8ade69687",
    12: "e69687",
    20: "f09f9880",
    30: "61",
    31: "6162",
}
GOLDEN_STUDENT_HEX = {
    100: "31",
    101: "32",
    102: "39",
    103: "33",
    104: "30",
    110: "e4b8ad",
    111: "e4b8ade69687",
    113: "f09f9880",
}
GOLDEN_PHI = {1: 100, 2: 100, 3: 102, 4: 100, 5: 103, 6: 100, 10: 110, 11: 111, 12: -1, 20: 113, 30: -1, 31: -1}

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _hex_map(hex_map):
    return {token_id: bytes.fromhex(value) for token_id, value in hex_map.items()}


def _byte_level_tokenizer(corpus: list[str], vocab_size: int = 300) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    )
    backend.train_from_iterator(corpus, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
    tokenizer.add_tokens(["<think>", "</think>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def test_phi_matches_official_golden(make_bridge):
    bridge = make_bridge(_hex_map(GOLDEN_STUDENT_HEX), _hex_map(GOLDEN_TEACHER_HEX))
    for teacher_id, student_id in GOLDEN_PHI.items():
        assert bridge.phi[teacher_id] == student_id, teacher_id
    assert bridge.phi[999] == -1  # ids outside the teacher vocabulary map to ⊥


def test_longest_student_prefix_and_length_cap(make_bridge):
    bridge = make_bridge({1: "a", 2: "ab", 3: "abc", 4: "b"}, {})
    assert bridge.longest_student_prefix(b"abcd") == 3
    assert bridge.longest_student_prefix(b"abcd", max_len=2) == 2
    assert bridge.longest_student_prefix(b"x") == -1
    assert bridge.longest_student_prefix(b"") == -1


def test_prefix_candidates_are_nested_and_longer_than_threshold(make_bridge):
    bridge = make_bridge({1: "a", 2: "ab", 3: "abc", 4: "abcde", 5: "b"}, {})
    assert bridge.student_prefix_candidates(b"abcdef", 0, 1, 8) == [(2, 2), (3, 3), (4, 5)]
    assert bridge.student_prefix_candidates(b"abcdef", 0, 1, 2) == [(2, 2), (3, 3)]
    assert bridge.student_prefix_candidates(b"xab", 1, 0, 8) == [(1, 1), (2, 2)]


def test_whitespace_ids_cover_only_space_and_tab_runs(make_bridge):
    bridge = make_bridge({1: "  ", 2: "\t ", 3: "\n", 4: " x"}, {})
    assert bridge.student_whitespace_ids == {1, 2}


def test_bridge_from_byte_level_tokenizers_round_trips_bytes():
    corpus = ["def foo(x):\n    return x + 1\n", "数学 与 代码 😀", "The answer is 42."] * 20
    student = _byte_level_tokenizer(corpus, vocab_size=300)
    teacher = _byte_level_tokenizer(corpus, vocab_size=280)
    bridge = TokenizerBridge.from_tokenizers(student, teacher)

    text = "<think>数学 😀</think>def foo(x):\n    return 42"
    for tokenizer, token_bytes in ((student, bridge.student_token_bytes), (teacher, bridge.teacher_token_bytes)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert b"".join(token_bytes[i] for i in ids) == text.encode("utf-8")

    im_end = student.convert_tokens_to_ids("<|im_end|>")
    assert im_end in bridge.student_stop_ids
    assert bridge.student_turn_end_id == im_end
    assert student.eos_token_id in bridge.student_control_ids
    assert student.eos_token_id not in bridge.student_token_bytes
    assert bridge.student_token_bytes[student.convert_tokens_to_ids("<think>")] == b"<think>"
    # every teacher content token maps to a student token whose bytes prefix it
    for teacher_id, data in bridge.teacher_token_bytes.items():
        student_id = bridge.phi[teacher_id]
        assert student_id >= 0 and data.startswith(bridge.student_token_bytes[student_id])


def test_detect_stop_token_ids_reads_template_turn_end():
    tokenizer = _byte_level_tokenizer(["hello world"] * 10)
    stop_ids = detect_stop_token_ids(tokenizer)
    assert stop_ids == {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}


def test_detect_stop_token_ids_adds_generation_config_eos(tmp_path):
    tokenizer = _byte_level_tokenizer(["hello world"] * 10)
    tokenizer.chat_template = None
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    (tmp_path / "generation_config.json").write_text(f'{{"eos_token_id": [{im_start}, {tokenizer.eos_token_id}]}}')
    assert detect_stop_token_ids(tokenizer) == {tokenizer.eos_token_id}
    assert detect_stop_token_ids(tokenizer, model_path=str(tmp_path)) == {tokenizer.eos_token_id, im_start}
    assert detect_stop_token_ids(tokenizer, model_path=str(tmp_path / "missing")) == {tokenizer.eos_token_id}


def test_non_byte_level_tokenizer_is_rejected():
    backend = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[UNK]")
    assert not is_byte_level_tokenizer(tokenizer)
    with pytest.raises(ValueError, match="byte-level"):
        TokenizerBridge.from_tokenizers(tokenizer, tokenizer)
