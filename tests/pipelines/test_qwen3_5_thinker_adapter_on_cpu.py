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
"""CPU tests for the Qwen3.5 dense-VLM training adapter."""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_5.thinker_training_adapter import Qwen3_5ThinkerAdapter

_MODEL_CONFIG = SimpleNamespace(trust_remote_code=False)


def _patch_loader(monkeypatch, owner, returned, recorder=None):
    """Replace ``owner.from_pretrained`` with a stub returning ``returned``."""

    def fake_from_pretrained(*args, **kwargs):
        if recorder is not None:
            recorder["args"] = args
            recorder["kwargs"] = kwargs
        return returned

    monkeypatch.setattr(owner, "from_pretrained", fake_from_pretrained)


def test_adapter_registers_the_qwen3_5_architecture():
    resolved = OmniModelBase.get_class_by_name("Qwen3_5ForConditionalGeneration", "thinker")
    assert resolved is Qwen3_5ThinkerAdapter


def test_auto_model_class_owns_the_architecture():
    assert MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES["qwen3_5"] == "Qwen3_5ForConditionalGeneration"
    assert Qwen3_5ThinkerAdapter.auto_model_class is AutoModelForImageTextToText


def test_nothing_is_stripped():
    assert Qwen3_5ThinkerAdapter.get_strip_modules(_MODEL_CONFIG) == []


def test_configure_model_keeps_the_module_graph():
    module = nn.Module()
    module.model = nn.Module()
    module.model.visual = nn.Linear(2, 2)
    module.lm_head = nn.Linear(2, 2)

    configured = Qwen3_5ThinkerAdapter.configure_model(module, _MODEL_CONFIG)

    assert configured is module
    assert hasattr(module, "model") and hasattr(module, "lm_head")


def test_tokenizer_reads_the_jinja_template_beside_the_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "chat_template.jinja").write_text("{{ jinja_template }}")
    tokenizer = SimpleNamespace(chat_template=None)
    _patch_loader(monkeypatch, AutoTokenizer, tokenizer)

    configured = Qwen3_5ThinkerAdapter.configure_tokenizer(str(tmp_path), _MODEL_CONFIG)

    assert configured is tokenizer
    assert tokenizer.chat_template == "{{ jinja_template }}"


def test_tokenizer_reads_the_json_template_beside_the_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "chat_template.json").write_text('{"chat_template": "{{ json_template }}"}')
    tokenizer = SimpleNamespace(chat_template=None)
    _patch_loader(monkeypatch, AutoTokenizer, tokenizer)

    Qwen3_5ThinkerAdapter.configure_tokenizer(str(tmp_path), _MODEL_CONFIG)

    assert tokenizer.chat_template == "{{ json_template }}"


def test_tokenizer_keeps_the_template_it_already_has(tmp_path, monkeypatch):
    tokenizer = SimpleNamespace(chat_template="{{ from_tokenizer_config }}")
    _patch_loader(monkeypatch, AutoTokenizer, tokenizer)

    Qwen3_5ThinkerAdapter.configure_tokenizer(str(tmp_path), _MODEL_CONFIG)

    assert tokenizer.chat_template == "{{ from_tokenizer_config }}"


def test_tokenizer_without_any_template_fails_closed(tmp_path, monkeypatch):
    _patch_loader(monkeypatch, AutoTokenizer, SimpleNamespace(chat_template=None))

    with pytest.raises(FileNotFoundError, match="chat template"):
        Qwen3_5ThinkerAdapter.configure_tokenizer(str(tmp_path), _MODEL_CONFIG)


def test_processor_completes_its_tokenizer_template(tmp_path, monkeypatch):
    (tmp_path / "chat_template.jinja").write_text("{{ processor_template }}")
    tokenizer = SimpleNamespace(chat_template=None)
    processor = SimpleNamespace(tokenizer=tokenizer)
    recorder: dict = {}
    _patch_loader(monkeypatch, AutoProcessor, processor, recorder)

    configured = Qwen3_5ThinkerAdapter.configure_processor(str(tmp_path), _MODEL_CONFIG)

    assert configured is processor
    assert tokenizer.chat_template == "{{ processor_template }}"
    assert recorder["kwargs"]["trust_remote_code"] is False


def test_processor_without_a_tokenizer_is_returned_unchanged(tmp_path, monkeypatch):
    processor = SimpleNamespace()
    _patch_loader(monkeypatch, AutoProcessor, processor)

    assert Qwen3_5ThinkerAdapter.configure_processor(str(tmp_path), _MODEL_CONFIG) is processor
