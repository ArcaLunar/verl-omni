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
"""The BPM recipe composes into a valid config: native vLLM VLM teacher, BPM loss, BPM agent loop."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.workers.config import DistillationTeacherModelConfig

from verl_omni.trainer.omni.bpm.agent_loop import BPMAgentLoopManagerTQ
from verl_omni.trainer.omni.omni_algos import build_omni_distillation_actor_loss

REPO_ROOT = Path(__file__).parents[4]
LAUNCHER = REPO_ROOT / "examples/bpm_trainer/qwen3_omni/run_qwen3_omni_thinker_bpm_lora_mmk12_muse_glimmer.sh"


def _launcher_argv(tmp_path, *extra, **env_overrides):
    """Run the launcher with a stub ``python3`` and return the arguments it passed."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python3"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARGV_OUT"\n')
    stub.chmod(0o755)
    argv_file = tmp_path / "argv.txt"
    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}", "ARGV_OUT": str(argv_file), **env_overrides}
    result = subprocess.run(["bash", str(LAUNCHER), *extra], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return result, argv_file.read_text().splitlines() if argv_file.exists() else []


@pytest.fixture(scope="module")
def recipe_config(tmp_path_factory):
    result, argv = _launcher_argv(
        tmp_path_factory.mktemp("bpm-recipe"), "trainer.total_epochs=1", TEACHER_PATH="/models/muse-glimmer"
    )
    assert result.returncode == 0, result.stderr
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="omni_trainer", overrides=argv[2:])
    OmegaConf.resolve(config)
    return config


def test_recipe_uses_a_native_vllm_vlm_teacher(recipe_config):
    distillation = omega_conf_to_dataclass(recipe_config.distillation)
    [teacher] = distillation.teacher_models.values()
    assert type(teacher) is DistillationTeacherModelConfig
    assert teacher.model_path == "/models/muse-glimmer"
    assert teacher.inference.name == "vllm"
    assert teacher.num_replicas == 2  # 4 teacher GPUs at TP=2
    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == 64
    assert not teacher.inference.enable_prefix_caching
    assert distillation.distillation_loss.loss_settings.use_topk


def test_recipe_builds_the_bpm_loss_and_agent_loop(recipe_config):
    loss_fn = build_omni_distillation_actor_loss(recipe_config)
    bpm = loss_fn.keywords["bpm_config"]
    assert (bpm.beta, bpm.whitespace_mask, bpm.teacher_modalities) == (0.0, "all", ["image"])
    assert bpm.teacher_generation_suffix == " to=user<|message|>"
    assert bpm.teacher_mm_processor_kwargs == {}
    manager_fqn = recipe_config.actor_rollout_ref.rollout.agent.agent_loop_manager_class
    assert load_class_from_fqn(manager_fqn, "AgentLoopManager") is BPMAgentLoopManagerTQ
    assert recipe_config.actor_rollout_ref.rollout.n == 1
    assert recipe_config.trainer.total_epochs == 1  # caller overrides come last


@pytest.mark.parametrize(
    ("env", "message"),
    [({"TEACHER_GPUS": "3"}, "multiple of TEACHER_TP"), ({"TOPK": "0"}, "TOPK must be a positive integer")],
)
def test_recipe_rejects_bad_topology(tmp_path, env, message):
    result, argv = _launcher_argv(tmp_path, **env)
    assert result.returncode != 0 and message in result.stderr
    assert argv == []
