#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

# Short frozen-policy adaptation for the paper-aligned PIM residual.  The base
# checkpoint must be the verified Atomic-5 GRU Stage-2 iter-5k DCP.

set -u

TOML_FILE="examples/toml/sft_config/action_policy_robocasa365_atomic5_zeva_pim.toml"

: "${ROBOCASA365_ROOT:?Set ROBOCASA365_ROOT to the Atomic-5 RoboCasa365 dataset}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to the verified GRU Stage-2 iter-5k DCP}"
: "${BEHAVIOR_MEMORY_BANK:?Set BEHAVIOR_MEMORY_BANK to train_memory_effect_v3.pt}"
: "${BEHAVIOR_PHASE_CACHE:?Set BEHAVIOR_PHASE_CACHE to the Stage-1 effect-v3 feature-cache directory}"
: "${QWEN_VLM_PATH:?Set QWEN_VLM_PATH to Qwen3-VL-8B-Instruct}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to the Wan2.2 VAE checkpoint}"

# Accept the earlier artifact name as a compatibility alias.
BEHAVIOR_PIM_TRAINING_BANK="${BEHAVIOR_PIM_TRAINING_BANK:-${BEHAVIOR_PERSISTENT_EFFECT_BANK:-}}"
: "${BEHAVIOR_PIM_TRAINING_BANK:?Build the persistent effect bank first}"

export ROBOCASA365_ROOT BASE_CHECKPOINT_PATH BEHAVIOR_MEMORY_BANK BEHAVIOR_PHASE_CACHE
export BEHAVIOR_PIM_TRAINING_BANK QWEN_VLM_PATH WAN_VAE_PATH
DATASET_PATH="$ROBOCASA365_ROOT"
EXTRA_DATASET_CHECK='
[[ -f "$BEHAVIOR_MEMORY_BANK" ]] || { echo "ERROR: missing $BEHAVIOR_MEMORY_BANK" >&2; exit 1; }
[[ -d "$BEHAVIOR_PHASE_CACHE" ]] || { echo "ERROR: missing $BEHAVIOR_PHASE_CACHE" >&2; exit 1; }
[[ -f "$BEHAVIOR_PIM_TRAINING_BANK" ]] || { echo "ERROR: missing $BEHAVIOR_PIM_TRAINING_BANK" >&2; exit 1; }
[[ -d "$QWEN_VLM_PATH" ]] || { echo "ERROR: missing $QWEN_VLM_PATH" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
