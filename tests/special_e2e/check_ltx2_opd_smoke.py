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
"""Fail the LTX-2.3 T2AV distillation smoke unless it shows a real student update.

The smoke trains for one step and exits 0 even when nothing was learned, so this
checker reads the run's console log and requires evidence that distillation ran:

* the last logged step reports a finite, positive ``actor/distill_kl_loss`` --
  a NaN loss, or a KL of exactly zero (teacher and student means identical, i.e.
  the teacher path never contributed), fails;
* the same step reports a finite, positive ``actor/grad_norm`` and the engine
  never logged ``skipping this optimizer step`` (its non-finite-gradient path);
* ``ltx2_opd_smoke_hooks`` reported an optimizer step that moved at least one
  trainable parameter, with a finite ``lora_max_delta``.

Exit code 0 means all of the above held for every step found in the log.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STEP_RE = re.compile(r"\bstep:(\d+) - (.*)$")
_NUMBER_RE = re.compile(
    r"^\s*(?:np\.\w+\()?"
    r"(nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\)?\s*$"
)
_FIELD_RE = re.compile(r"([A-Za-z_]\w*)=([^\s]+)")
_HOOK_TAG = "[ltx2-opd-smoke]"
_SKIP_WARNING = "skipping this optimizer step"
_LOSS_KEY = "actor/distill_kl_loss"
_GRAD_NORM_KEY = "actor/grad_norm"
_STEP_KEY = "training/global_step"


def _parse_number(text: str) -> float | None:
    """Parse a console metric value, tolerating ``np.float64(...)`` wrappers."""
    match = _NUMBER_RE.match(text)
    if match is None:
        return None
    return float(match.group(1))


def _parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    match = _STEP_RE.search(line)
    if match is None:
        return None
    fields: dict[str, float] = {}
    for item in match.group(2).split(" - "):
        key, separator, value = item.partition(":")
        if not separator:
            continue
        number = _parse_number(value)
        if number is not None:
            fields[key.strip()] = number
    return int(match.group(1)), fields


def _hook_fields(line: str) -> dict[str, float]:
    fields: dict[str, float] = {}
    for key, value in _FIELD_RE.findall(line):
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields


def _check(condition: bool, failures: list[str], message: str, detail: str) -> None:
    if condition:
        print(f"[ok]   {detail}")
    else:
        print(f"[fail] {detail}")
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="console log of the training run")
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="minimum last global step the run must have reached (default: 1)",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"[fail] log file not found: {args.log}")
        return 1

    text = _ANSI_RE.sub("", args.log.read_text(encoding="utf-8", errors="replace"))
    lines = text.splitlines()

    steps: list[tuple[int, dict[str, float]]] = []
    hook_lines: list[dict[str, float]] = []
    for line in lines:
        parsed = _parse_step_line(line)
        if parsed is not None:
            steps.append(parsed)
        if _HOOK_TAG in line:
            fields = _hook_fields(line)
            if fields:
                hook_lines.append(fields)

    failures: list[str] = []
    print(f"log: {args.log}")

    last_step, fields = max(steps, key=lambda item: item[0]) if steps else (-1, {})
    _check(
        last_step >= args.min_steps,
        failures,
        f"the run never logged a training step >= {args.min_steps}",
        f"last logged step: {last_step} (need >= {args.min_steps})",
    )

    loss = fields.get(_LOSS_KEY)
    _check(
        loss is not None,
        failures,
        f"{_LOSS_KEY} is missing from the step line; the distillation objective may not be active",
        f"{_LOSS_KEY} logged: {loss}",
    )
    if loss is not None:
        _check(
            math.isfinite(loss),
            failures,
            f"{_LOSS_KEY} is not finite ({loss}); the distillation step produced no usable gradient",
            f"{_LOSS_KEY} finite: {loss}",
        )
        _check(
            loss > 0.0,
            failures,
            f"{_LOSS_KEY} is {loss}; the student and teacher means are identical, so the teacher "
            "checkpoint or the teacher replay is not contributing",
            f"{_LOSS_KEY} positive: {loss}",
        )

    grad_norm = fields.get(_GRAD_NORM_KEY)
    _check(
        grad_norm is not None,
        failures,
        f"{_GRAD_NORM_KEY} is missing from the step line; the actor update did not report a gradient norm",
        f"{_GRAD_NORM_KEY} logged: {grad_norm}",
    )
    if grad_norm is not None:
        _check(
            math.isfinite(grad_norm) and grad_norm > 0.0,
            failures,
            f"{_GRAD_NORM_KEY} is {grad_norm}; the distilled gradient never reached the student",
            f"{_GRAD_NORM_KEY} finite and positive: {grad_norm}",
        )

    _check(
        _SKIP_WARNING not in text,
        failures,
        f"the engine logged '{_SKIP_WARNING}', so the optimizer dropped the student update",
        "engine never skipped the optimizer step",
    )

    _check(
        bool(hook_lines),
        failures,
        f"no '{_HOOK_TAG}' line in the log; the worker hook never ran, so the student update is unverified",
        f"worker hook reported {len(hook_lines)} optimizer step(s)",
    )
    if hook_lines:
        max_delta = max(item.get("lora_max_delta", 0.0) for item in hook_lines)
        changed = max(item.get("lora_tensors_changed", 0.0) for item in hook_lines)
        tensors = max(item.get("lora_tensors", 0.0) for item in hook_lines)
        nonfinite = max(item.get("lora_nonfinite_deltas", 0.0) for item in hook_lines)
        _check(
            tensors > 0,
            failures,
            "the hook found no trainable parameter to compare; the LoRA adapter may not be attached",
            f"trainable tensors snapshotted: {int(tensors)}",
        )
        _check(
            nonfinite == 0,
            failures,
            f"{int(nonfinite)} trainable tensor(s) became non-finite during the update",
            f"non-finite updated tensors: {int(nonfinite)}",
        )
        _check(
            changed > 0 and max_delta > 0.0,
            failures,
            "every trainable student tensor kept its value across the optimizer step; distillation "
            "did not update the student",
            f"updated tensors: {int(changed)}, max |delta|: {max_delta:.6g}",
        )

    if failures:
        print()
        print("LTX-2.3 T2AV OPD smoke failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print()
    print("LTX-2.3 T2AV OPD smoke: finite distillation loss and a real student weight update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
