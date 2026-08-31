# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic adapters for released checkpoints that predate Zeva naming."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def remap_legacy_cte_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Map legacy CTE parameter keys without changing tensor values."""

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace("behavior_token", "interaction_state_token")
        new_key = new_key.replace(".behavior.", ".interaction_state.")
        if new_key in remapped:
            raise ValueError(f"Duplicate CTE checkpoint key after compatibility mapping: {new_key}")
        remapped[new_key] = value
    return remapped
