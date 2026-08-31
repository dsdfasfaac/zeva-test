# zeva

Official implementation of **zeva**, a causal interaction memory framework for
robot manipulation.

Zeva extracts causal interaction states from observations and executed actions,
maintains short- and long-term interaction memory, and conditions a frozen
diffusion policy through a Causal Prompt.

## Architecture

- **Causal Transition Encoder (CTE)** produces the phase representation
  \(p_\tau\) and causal effect representation \(e_\tau\).
- **Brief Interaction Trace (BIT)** stores recent effects within an attempt.
- **Persistent Interaction Memory (PIM)** preserves interaction evidence across
  attempts.
- **Phase-Conditioned Retrieval** selects PIM entries relevant to the current
  phase.
- **Causal Prompt** fuses task context, phase, BIT, and PIM evidence.
- **Policy Injection** conditions the frozen diffusion policy.

The implementation is available under
[`cosmos_framework/model/zeva`](cosmos_framework/model/zeva).

## Results

The released RoboCasa365 Atomic-5 model achieves **195/250 (78.0%)** success:

| Task | Success |
| --- | ---: |
| OpenStandMixerHead | 48/50 |
| TurnOnElectricKettle | 45/50 |
| CloseToasterOvenDoor | 38/50 |
| TurnOnMicrowave | 47/50 |
| CoffeeSetupMug | 17/50 |

Evaluation uses seeds 195–244, a 32-action horizon, 30 UniPC steps, guidance
3.0, shift 5.0, and no retries.

## Installation

Zeva uses [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework)
at commit `ee58e41467f33c49ddde08b4d0ef4923876a95ac`.

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git checkout ee58e41467f33c49ddde08b4d0ef4923876a95ac
cd ..

git clone https://github.com/dsdfasfaac/zeva-test.git zeva
rsync -a --exclude=.git --exclude=README.md zeva/ cosmos-framework/
```

Install the Cosmos Framework dependencies following the upstream instructions.

## Models

Model weights and evaluation artifacts are hosted at
[chen123fu/zeva](https://huggingface.co/chen123fu/zeva).

## Reproduction

- [78% RoboCasa365 evaluation](docs/reproduce.md)
- [PIM and cross-attempt evaluation](docs/zeva_pim.md)

## License

This repository is released under the included OpenMDW-1.1 license. External
models, datasets, and simulators remain subject to their respective licenses.
