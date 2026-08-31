from __future__ import annotations

import torch

from cosmos_framework.model.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    remap_legacy_cte_state_dict,
)


def test_legacy_cte_key_mapping_preserves_every_tensor() -> None:
    config = CausalTransitionEncoderConfig(
        hidden_dim=16,
        retrieval_dim=8,
        phase_dim=8,
        effect_dim=8,
        num_layers=1,
        num_heads=4,
        use_mamba=False,
    )
    source = CausalTransitionEncoder(config)
    legacy = {}
    for key, value in source.state_dict().items():
        old_key = key.replace("interaction_state_token", "behavior_token")
        old_key = old_key.replace(".interaction_state.", ".behavior.")
        legacy[old_key] = value.clone()

    restored = CausalTransitionEncoder(config)
    restored.load_state_dict(remap_legacy_cte_state_dict(legacy), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
