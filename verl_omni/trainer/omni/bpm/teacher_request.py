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
"""Teacher-side prefill request for one student response.

The prompt is re-rendered with the teacher's own chat template (a VLM processor renders
its own media placeholders), and the student response text is re-tokenized by the teacher.
Prompt and response are tokenized separately so the response segmentation is the teacher's
canonical one. Adapted from the official BPM implementation
(``slime_plugins/bpm/teacher/bpm_teacher_request.py``).
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge

__all__ = [
    "TeacherRequest",
    "build_teacher_request",
    "normalize_think_seam",
    "student_response_text",
    "teacher_response_rows",
]

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass
class TeacherRequest:
    """Teacher prefill ids: ``prompt_ids + response_ids + [stop_id]``."""

    prompt_ids: list[int]
    response_ids: list[int]
    stop_id: int

    @property
    def teacher_ids(self) -> list[int]:
        """Full teacher prefill sequence."""
        return self.prompt_ids + self.response_ids + [self.stop_id]


def student_response_text(bridge: TokenizerBridge, student_response_ids: list[int]) -> Optional[str]:
    """Decode the student response content (before the first stop) to text.

    A multi-byte character cut by the length limit is dropped from the end.

    Args:
        bridge: Tokenizer bridge of the pair.
        student_response_ids: Sampled student response ids.

    Returns:
        Optional[str]: Response text, or ``None`` if the bytes are not valid UTF-8 elsewhere.
    """
    pieces = []
    for token_id in student_response_ids:
        if token_id in bridge.student_stop_ids:
            break
        if token_id not in bridge.student_control_ids:
            data = bridge.student_token_bytes.get(token_id)
            if data is None:
                return None
            pieces.append(data)
    data = b"".join(pieces)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.end == len(data):
            return data[: exc.start].decode("utf-8")
        return None


def normalize_think_seam(prompt_text: str, response_text: str) -> str:
    """Balance the ``<think>`` tag at the teacher's prompt/response seam.

    A teacher template that pre-fills ``<think>`` would see it twice when the student
    generated its own; a teacher template that does not would see a bare ``</think>``.
    Only the prompt changes; byte alignment covers the response alone.

    Args:
        prompt_text: Teacher-rendered prompt.
        response_text: Student response text.

    Returns:
        str: The adjusted prompt text.
    """
    stripped = prompt_text.rstrip()
    prompt_opens = stripped.endswith(_THINK_OPEN)
    response_opens = response_text.lstrip().startswith(_THINK_OPEN)
    if prompt_opens and response_opens:
        return stripped[: -len(_THINK_OPEN)]
    if not prompt_opens and not response_opens and _THINK_CLOSE in response_text:
        return prompt_text + _THINK_OPEN + "\n"
    return prompt_text


def build_teacher_request(
    messages: list[dict[str, Any]],
    response_text: str,
    *,
    tokenizer,
    template_owner=None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
) -> TeacherRequest:
    """Render the prompt with the teacher template and tokenize prompt and response.

    Media placeholders stay unexpanded in ``prompt_ids``; the inference engine expands them
    when the media are passed alongside the request.

    Args:
        messages: Raw prompt messages of the sample, in chat format.
        response_text: Student response text from :func:`student_response_text`.
        tokenizer: Teacher tokenizer.
        template_owner: Object owning the teacher chat template (e.g. a VLM processor).
            Defaults to ``tokenizer``.
        chat_template_kwargs: Extra keyword arguments for ``apply_chat_template``.
        tools: Tool schemas passed to the chat template.

    Returns:
        TeacherRequest: Teacher prefill ids.

    Raises:
        ValueError: If the teacher tokenizer has no eos token.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("The teacher tokenizer needs an eos_token_id to close the prefill sequence.")
    owner = template_owner if template_owner is not None else tokenizer
    template_kwargs = dict(chat_template_kwargs or {})
    if tools is not None:
        template_kwargs["tools"] = tools
    prompt_text = owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    prompt_text = normalize_think_seam(prompt_text, response_text)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    return TeacherRequest(prompt_ids=prompt_ids, response_ids=response_ids, stop_id=int(tokenizer.eos_token_id))


def teacher_response_rows(
    prompt_topk_ids: list[list[Optional[int]]],
    prompt_topk_logprobs: list[list[Optional[float]]],
    response_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the rows that predict the teacher response tokens and the stop.

    The engine returns one row per prefill token, where row ``i`` predicts token ``i + 1``
    and the last row is padding. Rows are taken from the end, so media placeholder expansion
    in the prompt does not shift them.

    Args:
        prompt_topk_ids: Per-row top-k teacher ids of the whole prefill sequence.
        prompt_topk_logprobs: Matching log-probabilities.
        response_len: Number of teacher response tokens (without the stop).

    Returns:
        tuple: int64 ``(response_len + 1, K)`` ids (``-1`` for empty slots) and float64
        log-probabilities; the last row predicts the stop.

    Raises:
        ValueError: If the engine returned fewer rows than the response needs.
    """
    num_rows = len(prompt_topk_ids)
    if num_rows < response_len + 2:
        raise ValueError(f"Teacher returned {num_rows} rows; a response of {response_len} tokens needs more.")
    selected = slice(num_rows - response_len - 2, num_rows - 1)
    ids = np.array([[-1 if i is None else i for i in row] for row in prompt_topk_ids[selected]], dtype=np.int64)
    logprobs = np.array(
        [[-np.inf if p is None else p for p in row] for row in prompt_topk_logprobs[selected]], dtype=np.float64
    )
    return ids, logprobs
