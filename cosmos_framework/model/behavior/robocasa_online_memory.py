"""RoboCasa Atomic-5 online-memory contract.

This module keeps the RoboCasa contract separate from the ARX defaults in
``online_memory.py``.  In particular, RoboCasa uses the frozen Atomic-5 VBE,
7-dimensional arm actions and the left-agent/wrist 20-Hz temporal contract.
"""

from __future__ import annotations

from .online_memory import OnlineMemorySchema


ROBOCASA_ATOMIC5_TASKS = (
    "OpenStandMixerHead",
    "TurnOnElectricKettle",
    "CloseToasterOvenDoor",
    "TurnOnMicrowave",
    "CoffeeSetupMug",
)


def make_robocasa_atomic5_schema(
    *,
    vbe_hash: str = "",
    vae_temporal_hash: str = "",
    capacity: int = 64,
    top_k: int = 4,
) -> OnlineMemorySchema:
    """Return the single schema accepted by Atomic-5 online memory.

    ``vbe_hash`` and ``vae_temporal_hash`` are part of the schema hash.  A
    runtime/cache produced with a different Stage-1 or temporal contract is
    therefore rejected instead of silently mixing representations.
    """

    return OnlineMemorySchema(
        version="robocasa_atomic5_online_memory_v1",
        task_contract="robocasa365_atomic5_5task_left_wrist_arm7_20hz_cosmos32_exec16",
        phase_dim=128,
        visual_key_dim=128,
        effect_dim=128,
        action_dim=7,
        action_horizon=16,
        capacity=int(capacity),
        top_k=int(top_k),
        vbe_hash=str(vbe_hash),
        vae_temporal_hash=str(vae_temporal_hash),
    )
