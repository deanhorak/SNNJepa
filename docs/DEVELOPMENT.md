# Development

## Build Loop

Configure:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build -j$(nproc) --target retina_classification

If the dependency cache is cold, the configure step will need network access once so CMake can fetch `spdlog`, `nlohmann/json`, `HighFive`, and `libsonata`.
```

## Dataset Expectations

The runner scripts use the same directory conventions as the original repo:
- `data/MNIST`
- `data/EMNIST`
- `data/cifar-10-batches-bin`
- or sibling `../data/...` directories when available

No dataset blobs are stored in the repo.

## Research Discipline

The repo should stay narrow.

When adding JEPA work:
- keep new experiments behind explicit config or CLI gates
- preserve the three baseline paths as regression checks
- document any new promoted path before adding more near-neighbor probes
- separate measurement scaffolding from promoted behavior

## Suggested First JEPA Milestones

1. Add explicit predictor and target latent states on top of the current sensory rollout.
2. Add masking or partial-view target generation over fixation sequences.
3. Add temporal latent-state consistency metrics before changing headline readout.
4. Keep baseline accuracy and runtime logs for MNIST, EMNIST, and CIFAR-10 after each major step.
