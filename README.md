# zeva inference overlay

This repository contains only the zeva code needed to serve the RoboCasa365 Atomic-5 Stage2-5k policy with learned Stage-3 behavior retrieval. It is an overlay for NVIDIA Cosmos Framework, not a standalone copy of the full framework.

The released policy reached **195/250 (78.0%)** closed-loop success across five RoboCasa365 Atomic tasks using 30 denoising steps.

## Paper-aligned CTE and PIM extension

The current `main` branch also includes the paper-facing behavior-memory API:

- `CausalTransitionEncoder` (CTE), with `VisualBehaviorEncoder` retained as a
  checkpoint-compatible alias;
- `BriefInteractionTrace` (BIT);
- `PersistentInteractionMemory` (PIM) and
  `PhaseConditionedPIMRetrieval`;
- `CausalPromptEncoder` and gated causal-prompt policy injection;
- an online-memory policy server and fixed-seed cross-attempt evaluator;
- a PIM-shadow formal path that keeps PIM write/retrieval lifecycle active
  while hard-gating its action residual off for frozen-policy regression tests.

The serialized Stage2/PIM parameter keys are intentionally unchanged. The
paper-aligned names are compatibility-preserving Python interfaces, so the
released checkpoints load without conversion. In same-process lifecycle
off/on probes, PIM-shadow actions are bit-exact.

## What is included

- Stage-1 VBE definitions used to encode causal phase/effect history;
- Stage-3 retrieval head and memory-bank lookup;
- Stage-2 behavior conditioning and residual injection in Cosmos3-Nano;
- the RoboCasa365 learned-retrieval policy server;
- the matching causal-effect closed-loop evaluator (32-step execution by default);
- the Atomic-5 inference/model configuration;
- the RoboCasa365 target action statistics used by the dataset adapter;
- the exact RoboCasa domain mapping (`robocasa-panda-omron` → model domain ID 22);
- the exact evaluation contract and launch command.

PIM adapter configuration, online-memory serving/evaluation code, and compact
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

`ee58e41467f33c49ddde08b4d0ef4923876a95ac` is a public NVIDIA commit. The model repository is inference-only and contains the Stage2 DCP model shards, Stage1 VBE/memory bank, Stage3 retrieval head, configs, checksums, and evaluation summary. Qwen, Wan VAE, and RoboCasa assets remain external dependencies under their own licenses.

## Run inference

See [the Stage2-5k inference and evaluation runbook](docs/zeva_robocasa_atomic5_stage2_5k.md). The published result requires the full 32-action chunk, left-agent plus wrist cameras at 256×512, fixed-base arm9 proprioception, and 30 denoising steps.

## License

The overlay is distributed under the included OpenMDW-1.1 license. Preserve `LICENSE`, `NOTICE`, `ATTRIBUTIONS.md`, copyright notices, and notices of origin. External dependencies and model components may have additional terms.
