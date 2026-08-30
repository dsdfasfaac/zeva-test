---
license: other
license_name: openmdw-1.1
license_link: https://github.com/NVIDIA/cosmos-framework/blob/main/LICENSE
library_name: cosmos-framework
pipeline_tag: robotics
tags:
  - robotics
  - vision-language-action
  - cosmos
  - robocasa
  - behavior-retrieval
  - dcp
---

# Cosmos-Behavior RoboCasa Atomic-5 — Stage2-5k

This bundle contains the inference artifacts selected by a full Stage2 checkpoint sweep for Cosmos-Behavior on RoboCasa365 Atomic-5. The selected checkpoint is `iter_000005000`, which reached **195/250 = 78.0%** success with learned retrieval.

## Model lineage

- Backbone: Cosmos3-Nano action policy.
- Baseline: RoboCasa Atomic-5 `iter_000025000`.
- Stage 1: frozen effect-v3 VBE, behavior memory bank, and causal phase/effect features.
- Stage 2: `iter_000005000`, warm-started from the Atomic-5 baseline and trained with oracle global behavior.
- Stage 3: retrieval head trained specifically from the Stage2-5k readout; it is not reused from another checkpoint.
- Public source base: NVIDIA Cosmos Framework commit `ee58e41467f33c49ddde08b4d0ef4923876a95ac`, plus the Cosmos-Behavior inference overlay in the linked code release.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── NOTICE
├── ATTRIBUTIONS.md
├── RELEASE_INFO.json
├── weights/
│   ├── stage2_iter_000005000/model/
│   │   ├── .metadata
│   │   └── __0_0.distcp ... __7_0.distcp
│   ├── stage1/
│   │   ├── behavior_vbe_step_000500.pt
│   │   └── train_memory_effect_v3.pt
│   └── stage3_iter_000005000/
│       ├── best.pt
│       └── readouts/{train.pt,val.pt}
├── configs/
├── cosmos_framework/data/generator/action/normalizer_stats/
│   └── robocasa365_target_action_stats.json
└── evaluation/
    ├── sweep_summary.json
    ├── sweep_summary.csv
    └── verification_report.txt
```

The published model repository is inference-only. Optimizer, scheduler, and trainer DCP states are deliberately excluded. This repository contains the corresponding source code and runbooks.

## External dependencies not included

The following must be obtained separately under their own licenses:

- Qwen3-VL-8B-Instruct weights;
- Wan2.2-TI2V-5B VAE weights;
- RoboCasa365 simulator, assets, and the Atomic-5 dataset;
- Python/CUDA dependencies required by the linked Cosmos-Behavior code release.

Set paths before launching:

```bash
export QWEN_VLM_PATH=/path/to/Qwen3-VL-8B-Instruct
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export ROBOCASA365_ROOT=/path/to/robocasa365/target
```

## Start the learned-retrieval policy server

Clone [dsdfasfaac/zeva-test](https://github.com/dsdfasfaac/zeva-test), enter the framework root, and run:

```bash
export RELEASE_ROOT=/absolute/path/to/Cosmos-Behavior-RoboCasa-Atomic5-Stage2-5k
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python -u -m \
  cosmos_framework.scripts.action_policy_server_robocasa365_stage3 \
  --checkpoint-path "$RELEASE_ROOT/weights/stage2_iter_000005000" \
  --allow-dcp-checkpoint \
  --experiment action_policy_robocasa365_atomic5_cosmos_behavior_stage2 \
  --experiment-overrides \
    model.config.tokenizer.vae_path="$WAN_VAE_PATH" \
  --stage2-memory-bank \
    "$RELEASE_ROOT/weights/stage1/train_memory_effect_v3.pt" \
  --stage2-vbe-checkpoint \
    "$RELEASE_ROOT/weights/stage1/behavior_vbe_step_000500.pt" \
  --stage3-retrieval-checkpoint \
    "$RELEASE_ROOT/weights/stage3_iter_000005000/best.pt" \
  --stage3-retrieval-top-k 5 \
  --domain-name robocasa-panda-omron \
  --host 0.0.0.0 --port 8300 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --action-dim 7 --action-chunk-size 32 --conditioning-fps 20 \
  --image-height 256 --image-width 512 \
  --format-prompt-as-json \
  --no-use-state --history-length 0 \
  --output-dir /tmp/cosmos_behavior_stage2_5k_server
```

The `--no-use-state` flag disables the legacy action-as-state path. The fixed-base independent 9D proprioception path embedded in this model remains enabled.
`--format-prompt-as-json` is mandatory: Atomic-5 was trained with the structured
RoboCasa action prompt. The specialized Stage-3 server enables it by default,
but it is kept explicit above so the inference contract is auditable.

The included `robocasa365_target_action_stats.json` is the exact target-set
statistics used by the dataset adapter. For the selected arm7 channels `[5:12]`,
min-max normalization is numerically the identity (`min=-1`, `max=1`); the file
is still shipped to make dataset/config reconstruction complete.

## Evaluation contract

The published 78.0% result uses:

- RoboCasa365 `target` split;
- tasks: OpenStandMixerHead, TurnOnElectricKettle, CloseToasterOvenDoor, TurnOnMicrowave, CoffeeSetupMug;
- 50 episodes per task, seeds 195–244;
- left agent view plus wrist view, horizontally concatenated to 256×512;
- fixed-base arm9 proprioception and arm7 EEF/gripper actions;
- model domain name `robocasa-panda-omron`, which resolves to domain ID `22`
  (raw action width `12`, then Atomic-5 selects channels `[5:12]`);
- UniPC with 30 denoising steps, guidance 3.0, shift 5.0;
- a complete 32-action open-loop chunk at 20 Hz, maximum 300 control steps;
- one fresh simulator process per seed and no video recording.

Do not evaluate with `open-loop-steps=10`; it truncates the 1.6-second action target seen in training and invalidates comparison.

Use the bundled causal-effect evaluator rather than a baseline-only client:

```bash
python -m cosmos_framework.scripts.eval_cosmos_behavior_robocasa365 \
  --task OpenStandMixerHead --host 127.0.0.1 --port 8300 \
  --episodes 1 --seed 195 --open-loop-steps 32 \
  --split target --render-gpu-device-id 0 --no-video \
  --output-dir /tmp/cosmos_behavior_eval/OpenStandMixerHead_seed195
```

The evaluator sends the initial left-plus-wrist observation, arm9 proprioception,
and only already-observed four-control transition boundaries needed to construct
the causal 0→1→2→3→4 effect history. A baseline evaluator that omits these fields
does not reproduce Stage2/Stage3 inference.

## Result

| Task | Success |
| --- | ---: |
| OpenStandMixerHead | 48/50 (96%) |
| TurnOnElectricKettle | 45/50 (90%) |
| CloseToasterOvenDoor | 38/50 (76%) |
| TurnOnMicrowave | 47/50 (94%) |
| CoffeeSetupMug | 17/50 (34%) |
| **Total** | **195/250 (78.0%)** |

## License

The Cosmos model materials and bundled framework snapshot are distributed under the included OpenMDW-1.1 `LICENSE`. Preserve `LICENSE`, `NOTICE`, `ATTRIBUTIONS.md`, copyright notices, and notices of origin when redistributing. External dependencies and datasets may use different licenses and are not relicensed by this bundle.
