# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Effect-v3 Stage-2 policy recipe for the fixed-base RoboCasa Atomic-5 set."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_atomic5_nano import (
    action_policy_robocasa365_atomic5_nano,
)

cs = ConfigStore.instance()

action_policy_robocasa365_atomic5_cosmos_behavior_stage2 = copy.deepcopy(action_policy_robocasa365_atomic5_nano)
action_policy_robocasa365_atomic5_cosmos_behavior_stage2["job"].update(
    project="cosmos_behavior",
    group="stage2_robocasa_atomic5_effect_v3",
    name="action_policy_robocasa365_atomic5_cosmos_behavior_stage2",
)
action_policy_robocasa365_atomic5_cosmos_behavior_stage2["model"]["config"]["behavior_stage2"] = dict(
    enabled=True,
    global_dim=256,
    phase_dim=128,
    effect_dim=128,
    effect_history_length=4,
    action_dim=7,
    horizon=32,
    num_anchors=8,
    hidden_dim=256,
    num_heads=4,
    prior_loss_weight=0.01,
    prior_dropout_rate=0.4,
    prior_inference_guidance_scale=0.5,
    global_prefix_tokens=1,
    leading_condition_steps=0,
)
action_policy_robocasa365_atomic5_cosmos_behavior_stage2["optimizer"]["keys_to_select"] += [
    "behavior_pbd", "behavior_adapter", "behavior_global_projector",
]
action_policy_robocasa365_atomic5_cosmos_behavior_stage2["optimizer"]["lr_multipliers"].update(
    behavior_pbd=5.0, behavior_adapter=5.0, behavior_global_projector=5.0,
)
action_policy_robocasa365_atomic5_cosmos_behavior_stage2["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]

_dataset = action_policy_robocasa365_atomic5_cosmos_behavior_stage2["dataloader_train"]["dataloader"]["datasets"]["robocasa365"]["dataset"]
_dataset["behavior_memory_bank"] = "${oc.env:BEHAVIOR_MEMORY_BANK}"
_dataset["behavior_phase_cache"] = "${oc.env:BEHAVIOR_PHASE_CACHE}"

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_atomic5_cosmos_behavior_stage2",
    node=action_policy_robocasa365_atomic5_cosmos_behavior_stage2,
)
