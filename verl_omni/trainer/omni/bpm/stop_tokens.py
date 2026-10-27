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
"""Assistant-turn stop-token detection.

A model's declared ``eos_token_id`` is not always the token that ends an assistant turn
(e.g. GLM-Z1 ends turns with ``<|user|>``), so the stop set is the union of the tokenizer
eos, the generation-config eos and the chat template's turn-end delimiters.

Adapted from the official BPM implementation (``slime_plugins/bpm/backend/special_tokens.py``).
"""

import logging
from typing import Any, Optional

__all__ = ["chat_template_turn_end_ids", "detect_stop_token_ids"]

logger = logging.getLogger(__name__)

# A string no chat template contains; special tokens rendered after it end the turn.
_TURN_END_SENTINEL = "ⓨⓨⓨ"


def _as_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value if v is not None}
    return {int(value)}


def _as_id_list(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token_id) for token_id in value]


def _generation_config_eos_ids(model_path: Optional[str]) -> set[int]:
    if not model_path:
        return set()
    from transformers import GenerationConfig

    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except OSError:
        logger.info("No generation_config found at %s; skipping its eos ids.", model_path)
        return set()
    return _as_id_set(getattr(generation_config, "eos_token_id", None))


def chat_template_turn_end_ids(tokenizer, template_owner=None) -> set[int]:
    """Return the special tokens a chat template emits right after an assistant message.

    Args:
        tokenizer: Tokenizer that defines the special-token ids.
        template_owner: Object whose ``apply_chat_template`` renders the conversation, e.g. a
            VLM processor. Defaults to ``tokenizer``.

    Returns:
        set[int]: Turn-end token ids; empty if the template cannot render a plain conversation.
    """
    owner = template_owner if template_owner is not None else tokenizer
    if getattr(owner, "chat_template", None) is None:
        return set()
    messages = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": _TURN_END_SENTINEL},
    ]
    try:
        rendered = _as_id_list(owner.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    except Exception as exc:  # noqa: BLE001 -- templates may reject this conversation shape
        logger.info("Chat template turn-end detection skipped: %r", exc)
        return set()
    sentinel_ids = tokenizer.encode(_TURN_END_SENTINEL, add_special_tokens=False)
    if not rendered or not sentinel_ids or sentinel_ids[-1] not in rendered:
        return set()
    end = len(rendered) - 1 - rendered[::-1].index(sentinel_ids[-1])
    special_ids = set(tokenizer.all_special_ids) | {int(i) for i in getattr(tokenizer, "added_tokens_decoder", {})}
    return {token_id for token_id in rendered[end + 1 :] if token_id in special_ids}


def detect_stop_token_ids(tokenizer, model_path: Optional[str] = None, template_owner=None) -> set[int]:
    """Return every token id that terminates an assistant turn.

    Args:
        tokenizer: Model tokenizer.
        model_path: Model directory holding ``generation_config.json``; optional.
        template_owner: Object owning the chat template (e.g. a processor); defaults to ``tokenizer``.

    Returns:
        set[int]: Union of tokenizer eos, generation-config eos and chat-template turn-end ids.
    """
    stop_ids = _as_id_set(tokenizer.eos_token_id)
    stop_ids |= _generation_config_eos_ids(model_path)
    stop_ids |= chat_template_turn_end_ids(tokenizer, template_owner)
    return {token_id for token_id in stop_ids if token_id >= 0}
