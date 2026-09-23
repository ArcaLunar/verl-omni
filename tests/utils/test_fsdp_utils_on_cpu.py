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
"""CPU tests for diffusion FSDP LoRA param collection."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from verl_omni.utils.fsdp_utils import collect_lora_params


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, x):
        return self.transformer_blocks[0](x)


def _peft_dit():
    model = get_peft_model(
        _TinyDiT(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"),
    )
    model.eval()
    return model


def test_fsdp2_non_layered_collects_lora_without_child_units(monkeypatch):
    """SD3.5 v1 uses FSDP2 + layered_summon=False; child-unit PEFT lookup is empty."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_iter_fsdp2_submodules", lambda _: iter(()))

    params = collect_lora_params(
        module,
        layered_summon=False,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_diffusers_falls_back_when_prefix_walker_is_empty(monkeypatch):
    """Qwen-Image e2e: layered_summon=True; prefix walker skips non-block FSDP units."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()

    def _version(m):
        return 2 if m is module else 0

    monkeypatch.setattr(fsdp_utils, "fsdp_version", _version)

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_collects_fsdp_leaf_lora_when_peft_dump_is_empty(monkeypatch):
    """Qwen-Image-Edit: named targets + FSDP1 leaf wrap; PEFT prefix dump is {}."""
    from collections import OrderedDict
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    class _Adapter(nn.Module):
        def __init__(self, fill: float):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 4), fill))

    class _LoraA(nn.Module):
        def __init__(self):
            super().__init__()
            self.default = _Adapter(1.0)
            self.old = _Adapter(2.0)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = _LoraA()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([_Block()])

    module = _Tiny()

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda m: 1 if m is module else 0)
    monkeypatch.setattr(fsdp_utils, "_peft_lora_params_to_cpu", lambda *args, **kwargs: OrderedDict())
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert list(params) == ["transformer_blocks.0.attn.lora_A.weight"]
    torch.testing.assert_close(params["transformer_blocks.0.attn.lora_A.weight"], torch.ones(2, 4))


def test_collect_lora_params_normalizes_fsdp_wrapped_lora_keys(monkeypatch):
    """LTX-2.3: FSDP-wrapped LoRA keys stream to vLLM without FSDP/adapter segments.

    vLLM's ``parse_fine_tuned_lora_name`` only accepts checkpoint-style keys whose
    penultimate segment is ``lora_A``/``lora_B``; leaked ``_fsdp_wrapped_module.``
    segments make ``add_lora`` fail (``is unsupported LoRA weight``).
    """
    from collections import OrderedDict

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    lora_a_key = "transformer_blocks.0._fsdp_wrapped_module.to_q.lora_A.default._fsdp_wrapped_module.weight"
    lora_b_key = "transformer_blocks.0._fsdp_wrapped_module.to_q.lora_B.default._fsdp_wrapped_module.weight"
    tainted = OrderedDict({lora_a_key: torch.ones(2, 4), lora_b_key: torch.ones(4, 2)})
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_peft_lora_params_to_cpu", lambda *args, **kwargs: tainted)

    params = collect_lora_params(
        module,
        layered_summon=False,
        base_sync_done=True,
        is_diffusers=True,
    )

    assert list(params) == ["transformer_blocks.0.to_q.lora_A.weight", "transformer_blocks.0.to_q.lora_B.weight"]
    torch.testing.assert_close(params["transformer_blocks.0.to_q.lora_A.weight"], torch.ones(2, 4))
    torch.testing.assert_close(params["transformer_blocks.0.to_q.lora_B.weight"], torch.ones(4, 2))


def test_collect_lora_params_leaves_base_keys_untouched(monkeypatch):
    """Base-weight sync must not be mutated by the LoRA key normalizer."""
    from collections import OrderedDict

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    base_weight_key = "transformer_blocks.0._fsdp_wrapped_module.to_q.weight"
    lora_a_key = "transformer_blocks.0._fsdp_wrapped_module.to_q.lora_A.default._fsdp_wrapped_module.weight"
    base = OrderedDict({base_weight_key: torch.ones(4, 4), lora_a_key: torch.ones(2, 4)})
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_peft_lora_params_to_cpu", lambda *args, **kwargs: base)

    params = collect_lora_params(
        module,
        layered_summon=False,
        base_sync_done=True,
        is_diffusers=True,
    )

    # Non-LoRA keys pass through; only the LoRA key is normalized.
    assert base_weight_key in params
    assert "transformer_blocks.0.to_q.lora_A.weight" in params
    assert lora_a_key not in params
