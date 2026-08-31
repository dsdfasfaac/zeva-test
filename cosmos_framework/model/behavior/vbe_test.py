# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.behavior import (
    BehaviorEncoderConfig,
    CausalTransitionEncoder,
    VisualBehaviorEncoder,
    behavior_vbe_loss,
)


def test_causal_transition_encoder_is_checkpoint_compatible_public_name() -> None:
    assert CausalTransitionEncoder is VisualBehaviorEncoder


def test_vbe_forward_loss_and_ema() -> None:
    model = VisualBehaviorEncoder(BehaviorEncoderConfig(hidden_dim=32, retrieval_dim=16, phase_dim=16, num_layers=1, num_heads=4, use_mamba=False))
    frames = torch.rand(3, 5, 3, 48, 48)
    actions = torch.rand(3, 4, 4, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = model(frames, actions, valid)
    assert output["behavior"].shape == (3, 5, 32)
    assert output["retrieval"].shape == (3, 5, 16)
    losses = behavior_vbe_loss(output, actions, valid, torch.tensor([0, 0, 1]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    before = next(model.target_visual_encoder.parameters()).clone()
    with torch.no_grad():
        next(model.visual_encoder.parameters()).add_(0.01)
    model.update_ema_target()
    assert not torch.equal(before, next(model.target_visual_encoder.parameters()))
