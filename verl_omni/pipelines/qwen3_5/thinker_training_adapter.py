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
"""Qwen3.5 dense vision-language training adapter.

Qwen3.5 checkpoints (``Qwen3_5ForConditionalGeneration``) are single-stage
vision-language models: the language model is ``model.language_model``, the vision
tower is ``model.visual``, and both classes already declare their
``_no_split_modules``. There is no talker/codec stage to strip and Transformers
already maps ``qwen3_5`` to ``Qwen3VLProcessor``, so the adapter is the registry
entry plus the tokenizer/processor loaders. Those loaders keep a chat template on
the tokenizer: verl's dataset renders prompts through it, and the BPM teacher
re-renders the same messages with its own template, so a missing template turns
into a late, opaque failure inside ``apply_chat_template``.
"""

import json
import os
from typing import Any

from transformers import AutoModelForImageTextToText

from verl_omni.pipelines.model_base import OmniModelBase

__all__ = ["Qwen3_5ThinkerAdapter"]

# Chat templates Qwen ships next to the tokenizer, in priority order. Transformers
# 5.14 resolves ``chat_template.jinja`` on its own; the fallback below covers older
# layouts and checkpoints whose tokenizer config carries no template.
_CHAT_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")


def _checkpoint_chat_template(model_path: str) -> str | None:
    """Return the chat template shipped beside the checkpoint, or ``None``.

    Args:
        model_path: Local path to the model checkpoint.

    Returns:
        Optional[str]: The template text; the JSON form nests it under ``chat_template``.
    """
    for name in _CHAT_TEMPLATE_FILES:
        path = os.path.join(model_path, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".json"):
            with open(path) as f:
                template = json.load(f).get("chat_template")
        else:
            with open(path) as f:
                template = f.read()
        if template:
            return template
    return None


def _ensure_chat_template(tokenizer, model_path: str) -> None:
    """Set ``tokenizer.chat_template`` from the checkpoint when it is unset.

    Args:
        tokenizer: Tokenizer (or processor tokenizer) to complete in place.
        model_path: Local path to the model checkpoint.

    Raises:
        FileNotFoundError: If the tokenizer has no template and none ships with the checkpoint.
    """
    if getattr(tokenizer, "chat_template", None):
        return
    template = _checkpoint_chat_template(model_path)
    if template is None:
        raise FileNotFoundError(
            "Qwen3.5 needs a chat template: the tokenizer has none and neither "
            f"{_CHAT_TEMPLATE_FILES[0]} nor {_CHAT_TEMPLATE_FILES[1]} exists under {model_path}."
        )
    tokenizer.chat_template = template


@OmniModelBase.register("Qwen3_5ForConditionalGeneration", stage="thinker")
class Qwen3_5ThinkerAdapter(OmniModelBase):
    """Training adapter for the single-stage Qwen3.5 dense VLM.

    The trained stage is the language model itself. Qwen3.5 has no talker, codec or
    MTP submodule in the Transformers graph (MTP checkpoint keys are ignored on
    load), so nothing is stripped and the base :meth:`configure_model` suffices.
    """

    # ``AutoModelForMultimodalLM`` does not own this architecture; the image-text-to-text
    # auto class resolves ``qwen3_5`` to ``Qwen3_5ForConditionalGeneration``.
    auto_model_class = AutoModelForImageTextToText

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Return submodules to delete before FSDP wrapping: none.

        Args:
            model_config: The ``OmniModelConfig`` (unused).

        Returns:
            list[str]: Always empty; Qwen3.5 is a single-stage VLM.
        """
        return []

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3.5 tokenizer with a usable chat template.

        The tokenizer backend must stay the checkpoint's byte-level BPE: BPM's
        ``TokenizerBridge`` assumes both sides are byte-level so it can map tokens
        through shared bytes.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The tokenizer, with ``chat_template`` set.

        Raises:
            FileNotFoundError: If no chat template can be found.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        _ensure_chat_template(tokenizer, model_path)
        return tokenizer

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the ``Qwen3VLProcessor`` that renders image prompts.

        The processor is the stock Transformers one. Unlike Qwen3-Omni, verl-omni's
        BPM recipe keeps verl's ``single_turn_agent``, so the omni-specific bindings
        (``get_rope_index``, ``dedup_pad_tokens``) are not needed here.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The processor, whose tokenizer carries the chat template.

        Raises:
            FileNotFoundError: If the processor tokenizer has no chat template.
        """
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            _ensure_chat_template(tokenizer, model_path)
        return processor
