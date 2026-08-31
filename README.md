# zeva inference overlay

This repository contains the zeva code needed to serve the RoboCasa365 Atomic-5 policy with causal interaction extraction, dual-timescale memory, and in-context policy injection. It is an overlay for NVIDIA Cosmos Framework, not a standalone copy of the full framework.

The released policy reached **195/250 (78.0%)** closed-loop success across five RoboCasa365 Atomic tasks using 30 denoising steps.

## Paper-aligned CTE and PIM extension

The current `main` branch exposes the paper-facing Zeva API under `cosmos_framework.model.zeva`:

- `CausalTransitionEncoder` (CTE) and `CausalTransitionEncoderConfig`;
- `BriefInteractionTrace` (BIT);
- `PersistentInteractionMemory` (PIM) and
  `PhaseConditionedPIMRetrieval`;
- `CausalPromptEncoder` and gated causal-prompt policy injection;
- a PIM policy server and fixed-seed cross-attempt evaluator;
- a PIM-shadow formal path that keeps PIM write/retrieval lifecycle active
  while hard-gating its action residual off for frozen-policy regression tests.

The serialized policy parameter keys remain compatible, and legacy CTE keys
are deterministically remapped during strict loading; see
[checkpoint compatibility](docs/checkpoint_compatibility.md). In same-process lifecycle
off/on probes, PIM-shadow actions are bit-exact.

## What is included

- Stage-1 CTE definitions used to encode phase and causal-effect history;
- static task-context retrieval head and bank lookup;
- Causal Prompt construction and policy injection into the frozen policy;
- the RoboCasa365 static-task-context policy server;
- the matching causal-effect closed-loop evaluator (32-step execution by default);
- the Atomic-5 inference/model configuration;
- the RoboCasa365 target action statistics used by the dataset adapter;
- the exact RoboCasa domain mapping (`robocasa-panda-omron` → model domain ID 22);
- the exact evaluation contract and launch command.

PIM configuration, serving/evaluation code, and compact
verification utilities are included. Large checkpoints, datasets, evaluation
videos, and machine-specific launch files are intentionally excluded.

## Install the overlay

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git checkout ee58e41467f33c49ddde08b4d0ef4923876a95ac
cd ..

git clone https://github.com/dsdfasfaac/zeva-test.git zeva-inference
rsync -a \
  --exclude=.git \
  --exclude=README.md \
  zeva-inference/ cosmos-framework/
```

Install Cosmos Framework dependencies using the upstream instructions. Obtain the Stage2-5k model artifacts from:

```text
https://huggingface.co/chen123fu/zeva
```

`ee58e41467f33c49ddde08b4d0ef4923876a95ac` is a public NVIDIA commit. The model repository is inference-only and contains the frozen policy shards, CTE/PIM assets, static task-context retrieval head, configs, checksums, and evaluation summary. Qwen, Wan VAE, and RoboCasa assets remain external dependencies under their own licenses.

## Run inference

See [the Stage2-5k inference and evaluation runbook](docs/zeva_robocasa_atomic5_stage2_5k.md). The published result requires the full 32-action chunk, left-agent plus wrist cameras at 256×512, fixed-base arm9 proprioception, and 30 denoising steps.

## License

The overlay is distributed under the included OpenMDW-1.1 license. Preserve `LICENSE`, `NOTICE`, `ATTRIBUTIONS.md`, copyright notices, and notices of origin. External dependencies and model components may have additional terms.
