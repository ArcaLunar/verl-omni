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
"""Build a tiny local Qwen3.5-MoE teacher checkpoint with random weights (fully offline).

The BPM smoke test distills a tiny Qwen3-Omni Thinker from this tiny Qwen3.5-MoE teacher.
Qwen3.5-MoE is a hybrid Gated-DeltaNet + Gated-Attention MoE, so the weights are created
through `transformers`' own `Qwen3_5MoeForCausalLM` instead of hand-written tensor shapes:
transformers 5.13+ registers `qwen3_5_moe_text` and vLLM 0.30 serves
`Qwen3_5MoeForCausalLM`, so the saved checkpoint loads on both sides. The text-only
checkpoint is enough for the smoke because its data is text-only; the multimodal
`Qwen3_5MoeForConditionalGeneration` teacher path is exercised by the MMK12 recipe.

The tiny config uses attention-only layers (`layer_types` all `full_attention`). vLLM
takes its hybrid SSM KV-cache path only when the config actually contains non-attention
layers (`ModelConfig.is_hybrid` checks `layer_types`), and tiny random Gated-DeltaNet
dimensions cannot satisfy that path's SSM/attention page-size alignment: newer vLLM
rejects VllmConfig with "Cannot align ssm_page_size and attn_page_size". Attention-only
layers keep the Qwen3.5-MoE + MoE + tokenizer shape of the teacher while avoiding the
hybrid cache, and the real MMK12 recipe still runs the full hybrid Qwen3.5-35B-A3B teacher.

The byte-level BPE tokenizer is trained in process with Qwen3.5's special tokens and
Qwen3.5's pre-tokenizer regex, on a different corpus than the student's
(``build_qwen3_omni_tiny_random.py``), so the two segment the same text differently and
the smoke exercises BPM's cross-tokenizer alignment.

Usage:
    python tests/special_e2e/build_qwen3_5_tiny_random.py \
        --output-dir ~/models/tiny-random/Qwen3.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/Qwen3.5")

# The Qwen3.5 tokenizer's added-token set (ids 248044-248076 on the released model).
_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
]

# Qwen-style turn layout: `<|im_start|>` / `<|im_end|>` delimit turns and the generation
# prompt opens the assistant turn. Text-only smoke data, so content is a string (list
# content renders media placeholders the way the released template does).
_CHAT_TEMPLATE = (
    "{%- macro render(content) -%}"
    "{%- if content is string -%}{{- content -}}"
    "{%- else -%}{%- for part in content -%}"
    "{%- if part['type'] == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}"
    "{%- elif part['type'] == 'video' -%}{{- '<|vision_start|><|video_pad|><|vision_end|>' -}}"
    "{%- elif part['type'] == 'text' -%}{{- part['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- endmacro -%}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + render(message['content']) + '<|im_end|>' + '\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)

# The released Qwen3.5 pre-tokenizer: Qwen's word/number/symbol splits plus ByteLevel.
_PRE_TOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Deliberately different from the student's corpus: other merges, other segmentation.
_CORPUS = [
    "A teacher model reads the student's answer and scores every next token.",
    "Compute 17 * 3 = 51, then 51 - 9 = 42. Therefore the result is 42.",
    "The area is 128 square units, and 360 degrees make a full turn.",
    "\\frac{1}{2} \\pi r^2 = 8\\pi when r = 4, so the shaded region is 8\\pi.",
    "<|im_start|>user<|im_end|><|im_start|>assistant<|im_end|>",
    " ".join(str(i) for i in range(0, 1000, 7)),
]


def _build_tiny_tokenizer(*, vocab_size: int = 1024) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(_PRE_TOKENIZE_PATTERN), behavior="isolated"),
            ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )
    tokenizer.train_from_iterator(_CORPUS, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        model_max_length=4096,
        chat_template=_CHAT_TEMPLATE,
    )


def _build_tiny_config(tokenizer) -> Qwen3_5MoeTextConfig:
    """Qwen3.5-MoE shrunk to 4 attention-only text layers (see module docstring)."""
    return Qwen3_5MoeTextConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        layer_types=["full_attention"] * 4,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype=torch.bfloat16,
    )


def build(output_dir: str, force: bool) -> None:
    """Write a tiny random Qwen3.5-MoE checkpoint and tokenizer to ``output_dir``."""
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to rebuild it.")
        shutil.rmtree(output_dir)
    torch.manual_seed(0)
    tokenizer = _build_tiny_tokenizer()
    config = _build_tiny_config(tokenizer)
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)
    config = model.config

    os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    files = {
        "generation_config.json": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": stop_ids,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
        },
    }
    for name, content in files.items():
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(content, f, indent=2)
    num_params = sum(t.numel() for t in model.state_dict().values())
    print(f"Saved tiny Qwen3.5-MoE ({num_params / 1e6:.2f}M params) to {output_dir}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    args = parser.parse_args()
    build(os.path.expanduser(args.output_dir), args.force)


if __name__ == "__main__":
    main()
