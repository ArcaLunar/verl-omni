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
"""The Qwen3.5-0.8B-student BPM recipe composes into a valid config: native vLLM VLM teacher."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.workers.config import DistillationTeacherModelConfig

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.trainer.omni.bpm.agent_loop import BPMAgentLoopManagerTQ
from verl_omni.trainer.omni.omni_algos import build_omni_distillation_actor_loss

REPO_ROOT = Path(__file__).parents[4]
LAUNCHER = REPO_ROOT / "examples/bpm_trainer/qwen3_5/run_qwen3_5_0_8b_bpm_lora_mmk12_qwen3_vl.sh"
NPU_LAUNCHER = REPO_ROOT / "examples/bpm_trainer/qwen3_5/run_qwen3_5_0_8b_bpm_lora_mmk12_qwen3_vl_npu.sh"


def _launcher_argv(tmp_path, *extra, launcher=LAUNCHER, **env_overrides):
    """Run the launcher with a stub ``python3`` and return the arguments it passed."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python3"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARGV_OUT"\n')
    stub.chmod(0o755)
    argv_file = tmp_path / "argv.txt"
    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}", "ARGV_OUT": str(argv_file), **env_overrides}
    result = subprocess.run(["bash", str(launcher), *extra], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return result, argv_file.read_text().splitlines() if argv_file.exists() else []


def _compose(argv):
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="omni_trainer", overrides=argv[2:])
    OmegaConf.resolve(config)
    return config


@pytest.fixture(scope="module")
def recipe_config(tmp_path_factory):
    result, argv = _launcher_argv(
        tmp_path_factory.mktemp("bpm-qwen3-5-0-8b-recipe"),
        "trainer.total_epochs=1",
        STUDENT_PATH="/models/qwen3.5-0.8b",
        TEACHER_PATH="/models/qwen3-vl-4b",
    )
    assert result.returncode == 0, result.stderr
    return _compose(argv)


def test_recipe_uses_a_native_vllm_vlm_teacher(recipe_config):
    # Qwen3-VL has no verl-omni training adapter, so the teacher must stay on verl's vLLM replica:
    # OmniDistillationTeacherModelConfig would demand inference.name=vllm_omni for a top-k loss.
    assert OmniModelBase.peek_class("Qwen3VLForConditionalGeneration", "thinker") is None
    distillation = omega_conf_to_dataclass(recipe_config.distillation)
    [teacher] = distillation.teacher_models.values()
    assert type(teacher) is DistillationTeacherModelConfig
    assert teacher.model_path == "/models/qwen3-vl-4b"
    assert teacher.inference.name == "vllm"
    assert teacher.num_replicas == 4  # 4 teacher GPUs at TP=1
    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == 64
    assert not teacher.inference.enable_prefix_caching
    assert distillation.distillation_loss.loss_settings.use_topk


def test_recipe_keeps_the_student_on_native_vllm(recipe_config):
    # vLLM-Omni has no `qwen3_5` pipeline, so the student rollout must not request one.
    assert recipe_config.actor_rollout_ref.model.path == "/models/qwen3.5-0.8b"
    assert recipe_config.actor_rollout_ref.rollout.name == "vllm"
    assert "vllm_omni" not in recipe_config.actor_rollout_ref.rollout.engine_kwargs
    assert recipe_config.actor_rollout_ref.model.exclude_modules == ".*visual.*"


def test_recipe_adapts_the_hybrid_decoder_layers(recipe_config):
    # Qwen3.5-0.8B is 18 Gated-DeltaNet + 6 full-attention layers; LoRA must reach both.
    targets = list(recipe_config.actor_rollout_ref.model.target_modules)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(targets)


def test_recipe_defaults_to_hugging_face_model_ids(tmp_path):
    # Empty overrides clear any STUDENT_PATH / TEACHER_PATH exported by the caller.
    result, argv = _launcher_argv(tmp_path, "trainer.total_epochs=1", STUDENT_PATH="", TEACHER_PATH="")
    assert result.returncode == 0, result.stderr
    config = _compose(argv)
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3.5-0.8B"
    [teacher] = omega_conf_to_dataclass(config.distillation).teacher_models.values()
    assert teacher.model_path == "Qwen/Qwen3-VL-4B-Instruct"


def test_recipe_builds_the_bpm_loss_and_agent_loop(recipe_config):
    loss_fn = build_omni_distillation_actor_loss(recipe_config)
    bpm = loss_fn.keywords["bpm_config"]
    assert (bpm.beta, bpm.whitespace_mask, bpm.teacher_modalities) == (0.0, "all", ["image"])
    assert bpm.teacher_generation_suffix == ""
    assert bpm.teacher_mm_processor_kwargs == {}
    manager_fqn = recipe_config.actor_rollout_ref.rollout.agent.agent_loop_manager_class
    assert load_class_from_fqn(manager_fqn, "AgentLoopManager") is BPMAgentLoopManagerTQ
    assert recipe_config.actor_rollout_ref.rollout.n == 1
    assert recipe_config.trainer.total_epochs == 1  # caller overrides come last


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"TEACHER_GPUS": "3", "TEACHER_TP": "2"}, "multiple of TEACHER_TP"),
        ({"TOPK": "0"}, "TOPK must be a positive integer"),
    ],
)
def test_recipe_rejects_bad_topology(tmp_path, env, message):
    result, argv = _launcher_argv(tmp_path, **env)
    assert result.returncode != 0 and message in result.stderr
    assert argv == []


def test_npu_variant_adds_ascend_settings_on_top_of_the_recipe(tmp_path):
    result, argv = _launcher_argv(tmp_path, "trainer.total_epochs=1", launcher=NPU_LAUNCHER)
    assert result.returncode == 0, result.stderr
    config = _compose(argv)
    student_vllm = config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert student_vllm["additional_config"]["weight_nz_mode"] == 0
    assert student_vllm["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert config.actor_rollout_ref.rollout.logprobs_mode == "raw_logprobs"
    assert list(config.trainer.logger) == ["console", "tensorboard"]
    assert (config.trainer.n_gpus_per_node, config.distillation.n_gpus_per_node) == (8, 8)
    [teacher] = omega_conf_to_dataclass(config.distillation).teacher_models.values()
    teacher_vllm = teacher.inference.engine_kwargs["vllm"]
    assert (teacher.inference.name, teacher.num_replicas) == ("vllm", 8)
    assert teacher_vllm["additional_config"]["weight_nz_mode"] == 0
    assert teacher_vllm["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert teacher_vllm["max_logprobs"] == 64
