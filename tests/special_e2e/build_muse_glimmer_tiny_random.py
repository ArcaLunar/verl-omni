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
"""Build a tiny local Muse-Glimmer checkpoint with random weights (fully offline).

The BPM smoke test distills a tiny Qwen3-Omni Thinker from this tiny Muse-Glimmer
teacher. transformers has no Muse-Glimmer model class yet (vLLM ships its own config
and model), so the weights are written directly under the checkpoint's tensor names
with the shapes vLLM derives from the config.

The byte-level BPE tokenizer is trained in process with Muse-Glimmer's special
tokens and a pre-tokenizer that groups up to three digits, on a different corpus
than the student's (``build_qwen3_omni_tiny_random.py``), so the two segment the
same text differently and the smoke exercises BPM's cross-tokenizer alignment.

Usage:
    python tests/special_e2e/build_muse_glimmer_tiny_random.py \
        --output-dir ~/models/tiny-random/Muse-Glimmer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/Muse-Glimmer")

_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad|>",
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "<|image|>",
    "<|video|>",
    "<|patch|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_frame_separator|>",
]

# Same turn layout as the released template: an assistant answer is addressed to the
# user (``to=user``), and the generation prompt stops before that channel header.
_CHAT_TEMPLATE = (
    "{%- macro render(content) -%}"
    "{%- if content is string -%}{{- content -}}"
    "{%- else -%}{%- for part in content -%}"
    "{%- if part['type'] == 'image' -%}{{- '<|patch|>' -}}"
    "{%- elif part['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif part['type'] == 'text' -%}{{- part['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- endmacro -%}"
    "{{- bos_token -}}"
    "{%- if messages[0]['role'] != 'system' -%}"
    "{{- '<|start|>system<|message|>You are a helpful AI assistant.<|eot|>' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'assistant' -%}"
    "{{- '<|start|>assistant to=user<|message|>' + render(message['content']) + '<|eot|>' -}}"
    "{%- else -%}"
    "{{- '<|start|>' + message['role'] + '<|message|>' + render(message['content']) + '<|eot|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}"
)

# Splits like the released tokenizer on the points that matter here: up to three digits
# per piece (the student splits every digit) and a leading space kept on words.
_PRE_TOKENIZE_PATTERN = r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# Deliberately different from the student's corpus: other merges, other segmentation.
_CORPUS = [
    "A teacher model reads the student's answer and scores every next token.",
    "Compute 17 * 3 = 51, then 51 - 9 = 42. Therefore the result is 42.",
    "The area is 128 square units, and 360 degrees make a full turn.",
    "<|start|>user<|message|>How much is 6 times 7?<|eot|><|start|>assistant to=user<|message|>42<|eot|>",
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
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad|>",
        model_max_length=4096,
        chat_template=_CHAT_TEMPLATE,
    )


def _build_tiny_config(tokenizer) -> dict:
    """Muse-Glimmer shrunk to 4 text layers (3 sliding + 1 global) and 2 vision layers."""
    vision_hidden, merge = 64, 2
    return {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "dtype": "bfloat16",
        "image_token_id": tokenizer.convert_tokens_to_ids("<|patch|>"),
        "video_token_id": tokenizer.convert_tokens_to_ids("<|video|>"),
        "text_config": {
            "model_type": "muse_glimmer_text",
            "vocab_size": len(tokenizer),
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "hidden_activation": "silu",
            "max_position_embeddings": 4096,
            "sliding_window": 512,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "final_logit_softcapping": 20.0,
            "tie_word_embeddings": False,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_size": vision_hidden,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_kernel_size": merge,
            "pos_emb_height": 8,
            "pos_emb_width": 8,
            "max_position_embeddings": 64,
            "output_dim": vision_hidden * merge**2,  # pixel-shuffle output
            "adapter_dim": 128,
            "layer_types": ["window_attention", "full_attention"],
        },
    }


def _tiny_weights(config: dict) -> dict[str, torch.Tensor]:
    """Random weights under the released checkpoint's tensor names."""
    text, vision = config["text_config"], config["vision_config"]
    hidden, vocab, inter = text["hidden_size"], text["vocab_size"], text["intermediate_size"]
    q_dim = text["num_attention_heads"] * text["head_dim"]
    kv_dim = text["num_key_value_heads"] * text["head_dim"]
    v_hidden, v_inter = vision["hidden_size"], vision["intermediate_size"]
    patch_dim = vision["patch_temporal"] * 3 * vision["patch_size"] ** 2

    shapes = {
        "lm_head.weight": (vocab, hidden),
        "model.language_model.embed_tokens.weight": (vocab, hidden),
        "model.language_model.norm.weight": (hidden,),
        "model.vision_adapter.fc1.weight": (vision["adapter_dim"], vision["output_dim"]),
        "model.vision_adapter.fc2.weight": (vision["adapter_dim"], vision["adapter_dim"]),
        "model.vision_projection.weight": (hidden, vision["adapter_dim"]),
        "model.vision_tower.ln_pre.weight": (v_hidden,),
        "model.vision_tower.ln_pre.bias": (v_hidden,),
        "model.vision_tower.ln_post.weight": (v_hidden,),
        "model.vision_tower.ln_post.bias": (v_hidden,),
        "model.vision_tower.patch_embedder.patch_embedding.weight": (v_hidden, patch_dim),
        "model.vision_tower.patch_embedder.position_embedding_table.weight": (
            vision["pos_emb_height"] * vision["pos_emb_width"],
            v_hidden,
        ),
    }
    for i in range(text["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm"):
            shapes[f"{prefix}.{norm}.weight"] = (hidden,)
        shapes[f"{prefix}.post_feedforward_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (q_dim, hidden)
        shapes[f"{prefix}.self_attn.gate_proj.weight"] = (q_dim, hidden)  # attention output gate
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv_dim, hidden)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv_dim, hidden)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden, q_dim)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (inter, hidden)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (inter, hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, inter)
    for i in range(vision["num_hidden_layers"]):
        prefix = f"model.vision_tower.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "proj"):
            shapes[f"{prefix}.attn.{proj}.weight"] = (v_hidden, v_hidden)
            shapes[f"{prefix}.attn.{proj}.bias"] = (v_hidden,)
        shapes[f"{prefix}.mlp.fc1.weight"] = (v_inter, v_hidden)
        shapes[f"{prefix}.mlp.fc1.bias"] = (v_inter,)
        shapes[f"{prefix}.mlp.fc2.weight"] = (v_hidden, v_inter)
        shapes[f"{prefix}.mlp.fc2.bias"] = (v_hidden,)
        for norm in ("norm1", "norm2"):
            shapes[f"{prefix}.{norm}.weight"] = (v_hidden,)
            shapes[f"{prefix}.{norm}.bias"] = (v_hidden,)

    weights = {}
    for name, shape in shapes.items():
        if name.endswith(".bias"):
            tensor = torch.zeros(shape)
        elif len(shape) == 1:
            tensor = torch.ones(shape)
        else:
            tensor = torch.randn(shape) * 0.02
        weights[name] = tensor.to(torch.bfloat16)
    return weights


def _processor_config() -> dict:
    """The released processor settings with small media token caps."""
    shared = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "merge_size": 2,
        "patch_size": 14,
        "resample": 1,
        "rescale_factor": 1 / 255,
        "temporal_patch_size": 2,
    }
    return {
        "image_processor": {**shared, "image_processor_type": "MuseGlimmerImageProcessor", "max_image_tokens": 256},
        "video_processor": {
            **shared,
            "video_processor_type": "MuseGlimmerVideoProcessor",
            "do_sample_frames": True,
            "fps": 2.0,
            "max_video_frame_tokens": 16,
            "num_frames": 8,
            "return_metadata": True,
        },
        "processor_class": "MuseGlimmerProcessor",
    }


def build(output_dir: str, force: bool) -> None:
    """Write a tiny random Muse-Glimmer checkpoint and tokenizer to ``output_dir``."""
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to rebuild it.")
        shutil.rmtree(output_dir)
    torch.manual_seed(0)
    tokenizer = _build_tiny_tokenizer()
    config = _build_tiny_config(tokenizer)
    weights = _tiny_weights(config)

    os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_file(weights, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot|>")]
    files = {
        "config.json": config,
        "generation_config.json": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": stop_ids,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "processor_config.json": _processor_config(),
    }
    for name, content in files.items():
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(content, f, indent=2)
    num_params = sum(t.numel() for t in weights.values())
    print(f"Saved tiny Muse-Glimmer ({num_params / 1e6:.2f}M params) to {output_dir}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    args = parser.parse_args()
    build(os.path.expanduser(args.output_dir), args.force)


if __name__ == "__main__":
    main()
