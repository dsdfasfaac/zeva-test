# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen-policy PIM adapter training for the verified Atomic-5 Stage-2 model."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_atomic5_zeva_stage2 import (
    action_policy_robocasa365_atomic5_zeva_stage2,
)

cs = ConfigStore.instance()

action_policy_robocasa365_atomic5_zeva_pim = copy.deepcopy(
    action_policy_robocasa365_atomic5_zeva_stage2
)
action_policy_robocasa365_atomic5_zeva_pim["job"].update(
    project="zeva",
    group="pim_adapter_robocasa_atomic5",
    name="action_policy_robocasa365_atomic5_zeva_pim",
)
behavior = action_policy_robocasa365_atomic5_zeva_pim["model"]["config"]["behavior_stage2"]
behavior.update(
    pim_memory_enabled=True,
    pim_persistent_length=4,
    pim_context_dim=256,
    # Exact released-model bypass at initialization.
    pim_gate_init=0.0,
)

# Freeze the complete verified GRU/Stage-2/Cosmos path.  Only the newly added
# prompt encoder, projection, and scalar gate receive gradients.
keys = action_policy_robocasa365_atomic5_zeva_pim["optimizer"]["keys_to_select"]
keys[:] = ["behavior_pim_encoder", "behavior_pim_projector", "behavior_pim_gate"]
action_policy_robocasa365_atomic5_zeva_pim["optimizer"]["lr_multipliers"].update(
    behavior_pim_encoder=5.0,
    behavior_pim_projector=5.0,
    behavior_pim_gate=1.0,
)
action_policy_robocasa365_atomic5_zeva_pim["checkpoint"]["keys_to_skip_loading"] += [
    "behavior_pim_encoder",
    "behavior_pim_projector",
    "behavior_pim_gate",
]

dataset = action_policy_robocasa365_atomic5_zeva_pim["dataloader_train"]["dataloader"]["datasets"][
    "robocasa365"
]["dataset"]
dataset["behavior_pim_training_bank"] = "${oc.env:BEHAVIOR_PIM_TRAINING_BANK}"
dataset["behavior_pim_history_length"] = 4
dataset["behavior_pim_context_dropout"] = 0.2
dataset["behavior_pim_support_dropout"] = 0.2

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_atomic5_zeva_pim",
    node=action_policy_robocasa365_atomic5_zeva_pim,
)
