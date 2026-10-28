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
"""Configuration for BPM cross-tokenizer on-policy distillation (``distillation_loss.loss_mode=bpm``)."""

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["BPMConfig"]


@dataclass
class BPMConfig(BaseConfig):
    """BPM target and divergence settings, read when ``distillation.distillation_loss.loss_mode`` is ``bpm``."""

    # Teacher tokenizer/processor path; defaults to the teacher model path.
    teacher_tokenizer_path: Optional[str] = None

    # Extra keyword arguments for the teacher's apply_chat_template, e.g. {"enable_thinking": true}.
    teacher_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    # Text appended to the teacher's generation prompt, for a template that leaves the start of
    # the assistant answer (e.g. a channel header) to the model.
    teacher_generation_suffix: str = ""

    # Media processor kwargs sent with each teacher request (resolution, fps, ...).
    teacher_mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    # Input modalities the teacher accepts; samples carrying any other modality are not distilled.
    teacher_modalities: list[str] = field(default_factory=lambda: ["image", "video"])

    # All-whitespace (space/tab) student tokens withheld from the loss (Sec. 4.3 of the paper):
    # "all" as in the paper, "indentation" only inside a line's leading whitespace, or "off".
    whitespace_mask: str = "all"

    # Divergence axis: 0 = forward KL, (0, 1) = generalized JSD, 1 = skew reverse KL.
    beta: float = 0.5

    # Skew of the reverse-KL endpoint (beta = 1): KL(p || (1 - lambda) t + lambda p).
    rkl_lambda: float = 0.1

    # Supervision of the realized stop row: "bridge" trains it under forward KL, "skip" drops it.
    stop_bridge: str = "bridge"

    # Rows per checkpointed chunk in the loss; bounds the fp32 copy of the student logits.
    loss_row_chunk: int = 2048

    def __post_init__(self):
        valid_modalities = {"image", "video", "audio"}
        unknown = set(self.teacher_modalities) - valid_modalities
        if unknown:
            raise ValueError(
                f"Invalid teacher_modalities: {sorted(unknown)}. Must be within {sorted(valid_modalities)}"
            )
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"Invalid beta: {self.beta}. Must be in [0, 1]")
        if self.beta == 1.0 and not 0.0 < self.rkl_lambda < 1.0:
            raise ValueError(f"Invalid rkl_lambda: {self.rkl_lambda}. Must be in (0, 1) when beta is 1")
        valid_whitespace_masks = {"all", "indentation", "off"}
        if self.whitespace_mask not in valid_whitespace_masks:
            raise ValueError(
                f"Invalid whitespace_mask: {self.whitespace_mask}. Must be one of {sorted(valid_whitespace_masks)}"
            )
        valid_stop_bridges = {"bridge", "skip"}
        if self.stop_bridge not in valid_stop_bridges:
            raise ValueError(f"Invalid stop_bridge: {self.stop_bridge}. Must be one of {sorted(valid_stop_bridges)}")
        if self.loss_row_chunk <= 0:
            raise ValueError(f"Invalid loss_row_chunk: {self.loss_row_chunk}. Must be positive")
