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
"""Test-side evidence collection for the LTX-2.3 T2AV distillation smoke.

The smoke used to pass on process exit alone, which a skipped update also
satisfies: ``DiffusersFSDPEngine.optimizer_step`` logs a warning, zeroes the
gradients and returns when ``grad_norm`` is not finite
(``verl_omni/workers/engine/fsdp/diffusers_impl.py``), so the training loop still
exits 0. This module therefore watches the student update from inside the Ray
workers and prints one machine-readable line per optimizer step:

    [ltx2-opd-smoke] optimizer_step grad_norm=... lora_tensors=... \
        lora_tensors_changed=... lora_max_delta=... lora_nonfinite_deltas=...

``check_ltx2_opd_smoke.py`` combines those lines with the logged ``distill_kl``
metric and decides the smoke verdict, so this module only observes and never
changes training behaviour.

Install it through ``ray_kwargs.ray_init.runtime_env.worker_process_setup_hook``
and export ``LTX2_OPD_SMOKE_ASSERTIONS=1``; every worker process then calls
``install()`` before it runs any task, so no production code is touched.
"""

from __future__ import annotations

import math
import os

TAG = "[ltx2-opd-smoke]"
ENV_VAR = "LTX2_OPD_SMOKE_ASSERTIONS"
_WRAPPED_ATTR = "_ltx2_opd_smoke_wrapped"


def _enabled() -> bool:
    return os.environ.get(ENV_VAR, "0").lower() in {"1", "true", "yes"}


def _local(tensor):
    """Return the local shard of a ``DTensor``, or the tensor itself."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _trainable_snapshot(module) -> dict:
    """Clone the parameters the student can update, keyed by name."""
    if module is None:
        return {}
    try:
        named = list(module.named_parameters())
    except Exception:
        return {}
    trainable = [(name, param) for name, param in named if getattr(param, "requires_grad", False)]
    # LoRA weights are the only adapters this recipe updates; keep the snapshot
    # small when the pipeline is otherwise full precision.
    chosen = [(name, param) for name, param in trainable if "lora" in name.lower()] or trainable
    snapshot = {}
    for name, param in chosen:
        snapshot[name] = _local(param.detach()).float().cpu().clone()
    return snapshot


def _measure_update(before: dict, module) -> tuple[float, int, int]:
    """Return ``(max |delta|, changed tensors, non-finite deltas)`` for the snapshot."""
    if not before or module is None:
        return 0.0, 0, 0
    try:
        current = dict(module.named_parameters())
    except Exception:
        return 0.0, 0, 0
    max_delta = 0.0
    changed = 0
    nonfinite = 0
    for name, old in before.items():
        param = current.get(name)
        if param is None:
            continue
        new = _local(param.detach()).float().cpu()
        if new.shape != old.shape:
            continue
        delta = (new - old).abs().max().item()
        if not math.isfinite(delta):
            # A NaN/Inf update still proves the optimizer ran, but it is a
            # failure the checker must report, not a "no update" verdict.
            nonfinite += 1
            continue
        if delta > 0.0:
            changed += 1
            max_delta = max(max_delta, delta)
    return max_delta, changed, nonfinite


def _install_engine_hooks() -> None:
    from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

    original = DiffusersFSDPEngine.optimizer_step
    if getattr(original, _WRAPPED_ATTR, False):
        return

    def optimizer_step_with_smoke_evidence(self, *args, **kwargs):
        before = _trainable_snapshot(getattr(self, "module", None))
        grad_norm = original(self, *args, **kwargs)
        max_delta, changed, nonfinite = _measure_update(before, getattr(self, "module", None))
        print(
            f"{TAG} optimizer_step grad_norm={grad_norm!r} lora_tensors={len(before)} "
            f"lora_tensors_changed={changed} lora_max_delta={max_delta!r} "
            f"lora_nonfinite_deltas={nonfinite}",
            flush=True,
        )
        return grad_norm

    setattr(optimizer_step_with_smoke_evidence, _WRAPPED_ATTR, True)
    DiffusersFSDPEngine.optimizer_step = optimizer_step_with_smoke_evidence


def install() -> None:
    """Ray ``worker_process_setup_hook`` entry point; inert unless the smoke enables it."""
    if not _enabled():
        return
    try:
        _install_engine_hooks()
    except Exception as exc:  # pragma: no cover - the checker fails when no evidence appears
        print(f"{TAG} WARNING could not install the smoke hooks: {exc!r}", flush=True)
