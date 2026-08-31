# RoboCasa Atomic-5 PIM adapter

This path adds paper-aligned Persistent Interaction Memory (PIM) without
changing the released GRU Stage-2 policy path. PIM is episode-scoped, survives
attempt boundaries, accepts only completed 16-control effects, merges by
phase/effect similarity, and retrieves Top-K evidence using the current phase.

## Paper terminology and public API

New integrations should use the paper-facing names below. Historical names
remain available only for checkpoint and script compatibility; changing these
aliases does not rename any state-dict key.

| Paper term | Public code name | Compatible historical name |
| --- | --- | --- |
| Causal Transition Encoder (CTE) | `CausalTransitionEncoder` | `VisualBehaviorEncoder` |
| phase representation | `phase` / `phase_head` | unchanged |
| effect representation | `effect_pre`, `effect_post` | unchanged |
| Brief Interaction Trace (BIT) | `BriefInteractionTrace` | `behavior_effect` + `behavior_effect_valid` tensors |
| Persistent Interaction Memory (PIM) | `PersistentInteractionMemory` | external `_pim` runtime handle |
| PIM memory entry | `PIMMemoryEntry` | `PersistentInteractionEntry` |
| phase-conditioned retrieval | `PhaseConditionedPIMRetrieval` / `query_phase()` | direct `query_tensors()` calls |
| causal prompt / \(F_{mem}\) | `CausalPromptEncoder` / `PromptMemoryFusion` | `PersistentInteractionPromptEncoder` |
| policy injection | `inject_causal_prompt()` | `add_gated_pim_residual()` |
| frozen policy | `OmniMoTModel` / `Cosmos3VFMNetwork` | framework-native model names |

`CosmosBehaviorRetrievalHead` remains **Static Global Behavior Retrieval**. It
must not be described as PIM retrieval: it queries a frozen training bank from
the initial observation and instruction, whereas
`PhaseConditionedPIMRetrieval` queries same-episode interaction evidence using
the current phase.

The retrieved interaction evidence is projected into the existing global
behavior prefix through `tanh(gate)`. The gate starts at exactly zero, so the
untrained PIM checkpoint has the same prefix and token layout as the verified
Stage-2 model. Training freezes Cosmos, VBE, PBD, the behavior adapter, and the
existing global projector; only `behavior_pim_encoder`,
`behavior_pim_projector`, and `behavior_pim_gate` are optimized.

## 1. Build phase/effect support for short training

Use the memory bank and feature cache from the same GRU Stage-1 lineage:

```bash
export BEHAVIOR_MEMORY_BANK=/path/to/train_memory_effect_v3.pt
export BEHAVIOR_PHASE_CACHE=/path/to/stage2_effect_feature_cache_v3
export BEHAVIOR_PIM_TRAINING_BANK=/path/to/persistent_effect_bank_gru.pt

PYTHONPATH=. python -m \
  cosmos_framework.scripts.build_robocasa_persistent_effect_bank \
  --memory-bank "$BEHAVIOR_MEMORY_BANK" \
  --feature-cache "$BEHAVIOR_PHASE_CACHE" \
  --output "$BEHAVIOR_PIM_TRAINING_BANK"
```

Each training query excludes its own trajectory and retrieves same-task
support by current-phase cosine similarity. This matches runtime
phase-conditioned PIM retrieval and avoids leaking future effects from the
target trajectory.

## 2. Run the frozen-policy adapter training

Set `BASE_CHECKPOINT_PATH` to the verified GRU Stage-2 iter-5k DCP, then:

```bash
export ROBOCASA365_ROOT=/path/to/robocasa365/target
export BASE_CHECKPOINT_PATH=/path/to/stage2_iter_000005000
export QWEN_VLM_PATH=/path/to/Qwen3-VL-8B-Instruct
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export BEHAVIOR_MEMORY_BANK=/path/to/train_memory_effect_v3.pt
export BEHAVIOR_PHASE_CACHE=/path/to/stage2_effect_feature_cache_v3
export BEHAVIOR_PIM_TRAINING_BANK=/path/to/persistent_effect_bank_gru.pt

bash examples/launch_sft_action_policy_robocasa365_atomic5_cosmos_behavior_pim.sh
```

The default recipe is deliberately short: 500 updates with checkpoints every
100 updates. Select the checkpoint by paired closed-loop validation; do not
assume the last checkpoint is best.

## 3. Validate no-regression before conditioning

First run PIM in shadow mode on the original Stage-2 checkpoint. The server
builds and retrieves PIM, but sends an empty validity mask, so actions remain
on the original model path:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -u -m \
  cosmos_framework.scripts.action_policy_server_robocasa365_stage3_online_memory \
  --checkpoint-path "$BASE_CHECKPOINT_PATH" --allow-dcp-checkpoint \
  --experiment action_policy_robocasa365_atomic5_cosmos_behavior_stage2 \
  --experiment-overrides model.config.tokenizer.vae_path="$WAN_VAE_PATH" \
  --stage2-memory-bank "$BEHAVIOR_MEMORY_BANK" \
  --stage2-vbe-checkpoint /path/to/behavior_vbe_step_000500.pt \
  --stage3-retrieval-checkpoint /path/to/stage3_iter_000005000/best.pt \
  --pim-shadow --pim-top-k 4 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --action-dim 7 --action-chunk-size 32 --conditioning-fps 20 \
  --image-height 256 --image-width 512 --no-use-state --history-length 0 \
  --port 8300
```

For learned conditioning, change the checkpoint and experiment to the PIM
adapter checkpoint and
`action_policy_robocasa365_atomic5_cosmos_behavior_pim_inference`, replace
`--pim-shadow` with `--pim-enabled`, and keep all other inference settings
paired. Do not use the training experiment for serving: its warm-start skip
list intentionally omits PIM weights when loading the old Stage-2 checkpoint.

Before enabling PIM, also serve that trained adapter checkpoint with neither
PIM flag. Its PIM validity mask is empty, so paired requests must reproduce the
frozen Stage-2 action output. This checks both the checkpoint loader and the
zero-memory bypass, whereas shadow mode alone checks only the runtime memory
lifecycle.

Evaluate repeated attempts on one fixed environment seed:

```bash
PYTHONPATH=. python -m \
  cosmos_framework.scripts.eval_cosmos_behavior_robocasa365_online_memory \
  --task TurnOnMicrowave --host 127.0.0.1 --port 8300 \
  --episodes 10 --seed 195 --seed-mode fixed \
  --expect-memory-conditioning on \
  --output-dir /path/to/pim_eval/TurnOnMicrowave_seed195 --no-video
```

Use `--expect-memory-conditioning off` for shadow/no-memory runs. Report both
attempt-0 success (regression guard) and success versus attempt index (PIM
self-evolution evidence); the independent-seed 78% result is not evidence for
cross-attempt improvement.
