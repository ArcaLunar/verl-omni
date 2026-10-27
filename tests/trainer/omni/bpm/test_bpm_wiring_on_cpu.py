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
"""BPM wiring: config composition, actor-loss builder, agent-loop manager, pinned verl contracts."""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.trainer.distillation import distillation_ppo_loss
from verl.trainer.distillation.losses import get_distillation_loss_settings
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn

from verl_omni.trainer.omni.bpm import agent_loop
from verl_omni.trainer.omni.bpm.losses import bpm_distillation_ppo_loss
from verl_omni.trainer.omni.omni_algos import build_omni_distillation_actor_loss
from verl_omni.workers.config.omni import BPMConfig

CONFIG_DIR = Path(__file__).parents[4] / "verl_omni/trainer/config"


def _bpm_run_config():
    config = OmegaConf.load(CONFIG_DIR / "_generated_omni_trainer.yaml")
    config.distillation.enabled = True
    loss = config.distillation.distillation_loss
    loss.loss_mode, loss.loss_max_clamp, loss.use_policy_gradient = "bpm", None, False
    config.distillation.teacher_models.teacher_model.model_path = "teacher"
    config.actor_rollout_ref.actor.loss_agg_mode = "token-mean"
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    return config


@pytest.mark.parametrize("name", ["_generated_omni_trainer.yaml", "_generated_omni_megatron_trainer.yaml"])
def test_generated_configs_carry_the_bpm_group(name):
    bpm = omega_conf_to_dataclass(OmegaConf.load(CONFIG_DIR / name).bpm)
    assert isinstance(bpm, BPMConfig)
    assert (bpm.beta, bpm.stop_bridge, bpm.teacher_modalities) == (0.5, "bridge", ["image", "video"])


def test_bpm_is_a_logits_level_distillation_loss():
    assert get_distillation_loss_settings("bpm").use_topk


def test_actor_loss_builder_returns_the_bpm_loss():
    loss_fn = build_omni_distillation_actor_loss(_bpm_run_config())
    assert loss_fn.func is bpm_distillation_ppo_loss
    assert isinstance(loss_fn.keywords["bpm_config"], BPMConfig)
    assert loss_fn.keywords["config"].loss_agg_mode == "token-mean"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-mean", "token-mean"),
        ("distillation.distillation_loss.loss_max_clamp", 10.0, "clamped"),
        ("distillation.distillation_loss.use_policy_gradient", True, "supervised"),
    ],
)
def test_actor_loss_builder_rejects_settings_that_change_the_objective(path, value, message):
    config = _bpm_run_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=message):
        build_omni_distillation_actor_loss(config)


def test_manager_is_loadable_by_fqn_and_spawns_bpm_workers(monkeypatch):
    fqn = "verl_omni.trainer.omni.bpm.agent_loop.BPMAgentLoopManagerTQ"
    manager_cls = load_class_from_fqn(fqn, "AgentLoopManager")
    monkeypatch.setattr(AgentLoopManagerTQ, "__init__", lambda self, *args, **kwargs: None)
    assert manager_cls().agent_loop_workers_class is agent_loop.BPMAgentLoopWorkerTQ


def test_overridden_verl_contracts_are_unchanged():
    # BPM overrides or wraps these pinned verl entry points; a changed signature must fail here first.
    upstream = inspect.signature(AgentLoopWorker._compute_teacher_logprobs)
    ours = inspect.signature(agent_loop.BPMAgentLoopWorkerTQ.__ray_actor_class__._compute_teacher_logprobs)
    assert list(ours.parameters) == list(upstream.parameters)
    assert list(inspect.signature(distillation_ppo_loss).parameters) == [
        "config",
        "distillation_config",
        "model_output",
        "data",
        "dp_group",
        "student_logits",
        "data_format",
    ]
