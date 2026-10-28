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
"""Teacher side of BPM in the agent loop.

For each student response the teacher prefills the prompt (re-rendered with the teacher's
own template, media included) plus the re-tokenized response, returns top-k log-probs,
and the BPM target is built on CPU and written into the ``teacher_ids`` /
``teacher_logprobs`` fields that verl carries to the actor.

Select it with ``actor_rollout_ref.rollout.agent.agent_loop_manager_class=
verl_omni.trainer.omni.bpm.agent_loop.BPMAgentLoopManagerTQ``.
"""

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig

from verl_omni.trainer.omni.bpm.alignment import RowKind, align_sample
from verl_omni.trainer.omni.bpm.encoding import pack_targets
from verl_omni.trainer.omni.bpm.targets import SampleTargets, build_sample_targets, target_width
from verl_omni.trainer.omni.bpm.teacher_request import (
    build_teacher_request,
    student_response_text,
    teacher_response_rows,
)
from verl_omni.trainer.omni.bpm.tokenizer_bridge import TokenizerBridge
from verl_omni.workers.config.omni import BPMConfig

__all__ = ["BPMAgentLoopManagerTQ", "BPMAgentLoopWorkerTQ", "BPMTeacher"]

logger = logging.getLogger(__name__)

# multi_modal_data key -> teacher modality name
_MODALITY_OF = {"images": "image", "videos": "video", "audios": "audio"}


def _skipped_targets(num_rows: int, width: int, kind: RowKind) -> SampleTargets:
    return SampleTargets(
        kinds=np.full((num_rows,), kind, dtype=np.int8),
        ids=np.zeros((num_rows, width), dtype=np.int64),
        probs=np.zeros((num_rows, width), dtype=np.float32),
        other=np.zeros((num_rows,), dtype=np.float32),
    )


def _as_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    # non-tensor batch values arrive as numpy object arrays
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    return list(raw_prompt)


class BPMTeacher:
    """Builds the packed BPM target of one sample from a cross-tokenizer teacher.

    Args:
        bridge: Tokenizer bridge of the (student, teacher) pair.
        teacher_tokenizer: Teacher tokenizer.
        teacher_processor: Teacher processor owning the chat template, or ``None``.
        client: Teacher inference client exposing ``generate``.
        bpm_config: BPM settings.
        topk: Teacher top-k log-probs per position.
    """

    def __init__(
        self,
        *,
        bridge: TokenizerBridge,
        teacher_tokenizer,
        teacher_processor,
        client,
        bpm_config: BPMConfig,
        topk: int,
    ):
        self.bridge = bridge
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_processor = teacher_processor
        self.client = client
        self.bpm_config = bpm_config
        self.topk = topk
        self.width = target_width(topk)

    async def packed_targets(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        raw_prompt: Any,
        multi_modal_data: Optional[dict[str, Any]],
    ):
        """Return ``(teacher_ids, teacher_logprobs)`` packed over the student sequence."""
        targets = await self._targets(response_ids, raw_prompt, multi_modal_data or {})
        return pack_targets(targets, prompt_len=len(prompt_ids))

    async def _targets(
        self, response_ids: list[int], raw_prompt: Any, multi_modal_data: dict[str, Any]
    ) -> SampleTargets:
        modalities = {_MODALITY_OF[key] for key, value in multi_modal_data.items() if key in _MODALITY_OF and value}
        if not modalities <= set(self.bpm_config.teacher_modalities):
            return _skipped_targets(len(response_ids), self.width, RowKind.UNSUPPORTED_INPUT)
        response_text = student_response_text(self.bridge, response_ids)
        if response_text is None:
            return _skipped_targets(len(response_ids), self.width, RowKind.BYTE_MISMATCH)

        request = build_teacher_request(
            _as_messages(raw_prompt),
            response_text,
            tokenizer=self.teacher_tokenizer,
            template_owner=self.teacher_processor,
            chat_template_kwargs=self.bpm_config.teacher_chat_template_kwargs,
            generation_suffix=self.bpm_config.teacher_generation_suffix,
        )
        output = await self.client.generate(
            request_id=uuid4().hex,
            prompt_ids=request.teacher_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": self.topk},
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            mm_processor_kwargs=self.bpm_config.teacher_mm_processor_kwargs or None,
        )
        topk_ids, topk_logprobs = teacher_response_rows(
            output.extra_fields["prompt_ids"], output.extra_fields["prompt_logprobs"], len(request.response_ids)
        )
        return await asyncio.to_thread(self._build, response_ids, request.response_ids, topk_ids, topk_logprobs)

    def _build(self, response_ids, teacher_response_ids, topk_ids, topk_logprobs) -> SampleTargets:
        alignment = align_sample(
            self.bridge,
            list(response_ids),
            teacher_response_ids,
            whitespace_mask=self.bpm_config.whitespace_mask,
        )
        return build_sample_targets(self.bridge, alignment, topk_ids, topk_logprobs)


def _build_bpm_teacher(worker) -> BPMTeacher:
    """Create the BPM teacher of an agent-loop worker from its config, tokenizer and clients."""
    distillation_config: DistillationConfig = omega_conf_to_dataclass(worker.config.distillation)
    loss_mode = distillation_config.distillation_loss.loss_mode
    if loss_mode != "bpm":
        raise ValueError(f"BPMAgentLoopManagerTQ requires distillation_loss.loss_mode=bpm, got {loss_mode!r}.")
    if len(distillation_config.teacher_models) != 1:
        raise NotImplementedError("BPM supports exactly one teacher model.")
    teacher_key, teacher_config = next(iter(distillation_config.teacher_models.items()))
    bpm_config: BPMConfig = omega_conf_to_dataclass(worker.config.bpm)
    teacher_path = bpm_config.teacher_tokenizer_path or teacher_config.model_path
    trust_remote_code = worker.config.actor_rollout_ref.model.get("trust_remote_code", False)
    teacher_tokenizer = hf_tokenizer(teacher_path, trust_remote_code=trust_remote_code)
    teacher_processor = hf_processor(teacher_path, trust_remote_code=trust_remote_code)
    bridge = TokenizerBridge.from_tokenizers(
        worker.tokenizer,
        teacher_tokenizer,
        student_model_path=worker.model_config.local_path,
        teacher_model_path=teacher_path,
        student_template_owner=worker.processor,
        teacher_template_owner=teacher_processor,
    )
    logger.info(
        "BPM bridge ready: %d student / %d teacher content tokens, student stops %s, teacher stops %s",
        len(bridge.student_token_bytes),
        len(bridge.teacher_token_bytes),
        sorted(bridge.student_stop_ids),
        sorted(bridge.teacher_stop_ids),
    )
    return BPMTeacher(
        bridge=bridge,
        teacher_tokenizer=teacher_tokenizer,
        teacher_processor=teacher_processor,
        client=worker.teacher_client[teacher_key],
        bpm_config=bpm_config,
        topk=distillation_config.distillation_loss.topk,
    )


@ray.remote
class BPMAgentLoopWorkerTQ(AgentLoopWorkerTQ.__ray_actor_class__):
    """TransferQueue agent-loop worker whose teacher step produces BPM targets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bpm_teacher = _build_bpm_teacher(self) if self.distillation_enabled else None

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write the packed BPM target into ``output.extra_fields`` (upstream signature)."""
        if self.bpm_teacher is None or validate:
            return
        raw_prompt = (sample_kwargs or {}).get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("BPM needs the raw prompt messages; the sample has no raw_prompt.")
        teacher_ids, teacher_logprobs = await self.bpm_teacher.packed_targets(
            prompt_ids, response_ids, raw_prompt, output.multi_modal_data
        )
        output.extra_fields["teacher_ids"] = teacher_ids
        output.extra_fields["teacher_logprobs"] = teacher_logprobs


class BPMAgentLoopManagerTQ(AgentLoopManagerTQ):
    """``AgentLoopManagerTQ`` that runs :class:`BPMAgentLoopWorkerTQ` workers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # workers are spawned later in create(), after this override
        self.agent_loop_workers_class = BPMAgentLoopWorkerTQ
