# SNNJepa

`SNNJepa` is a trimmed research workspace extracted from `SNNFrame` for one purpose: exploring JEPA-style learning on top of a spiking neural network substrate without carrying the full historical experiment surface.

The repo keeps:
- the core SNN runtime and declarative loading stack
- the shared retina classification benchmark used for the current MNIST, EMNIST, and CIFAR-10 baselines
- only the three current baseline configs and the scripts needed to run them

The repo deliberately leaves out:
- historical one-off experiments and demos
- old benchmark artifacts and audit outputs
- test data and local RocksDB stores
- closed exploratory configs outside the current MNIST, EMNIST, and CIFAR-10 baselines

## Current Scope

This workspace is the starting point for JEPA-oriented work where:
- the sensory front end is spiking and biologically structured
- future learning work can add predictor/target branches, masked prediction, temporal state alignment, and latent-state consistency
- the existing baselines remain runnable as regression checks

Current baseline surfaces:
- MNIST: `configs/mnist_retina_bilateral_experimental.sonata.json`
- EMNIST: `configs/emnist_retina_bilateral_experimental.sonata.json`
- CIFAR-10: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`

## Build

Prerequisites:
- CMake 3.16+
- C++17 compiler
- RocksDB development libraries
- system OpenMP support

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target retina_classification
```

The top-level CMake still fetches:
- `spdlog`
- `nlohmann/json`
- `HighFive`
- `libsonata`

If those dependencies are not already cached locally, the first configure step requires network access so `FetchContent` can download them.

Visualization dependencies remain optional. If `third_party/glfw`, `third_party/glad`, and `third_party/imgui` are present, the core library will build with visualization support. They are retained because parts of the runtime still compile against that surface.

## Run Baselines

MNIST:

```bash
scripts/run_mnist_retina_bilateral.sh
```

EMNIST:

```bash
scripts/run_emnist_retina_bilateral.sh
```

CIFAR-10:

```bash
scripts/run_cifar10_retina_bilateral_natural.sh
```

Each script accepts the same overrides used in `SNNFrame`, for example:

```bash
EXAMPLES_PER_CLASS=200 TEST_LIMIT=1000 scripts/run_cifar10_retina_bilateral_natural.sh
```

## JEPA Direction

This repo is not intended to chase surrogate-gradient or ANN-to-SNN conversion baselines. The research intent is to stay on a biologically grounded spiking substrate and introduce JEPA-like mechanisms through:
- latent target/predictor state construction
- temporal consistency across fixations
- masked or partial-view prediction over spiking sensory state
- object-centric recurrent state formation

## Current Status

The repo is now past the initial scaffolding phase.

Implemented:
- core framework relocation from `src/` into `include/snnfw/`
- JEPA module surface under `include/snnfw/jepa/`
- stage-1 latent tap extraction from the bilateral retina benchmark
- masking sampler with visible/hidden branch splits and no-leakage checks
- minimal JEPA trainer with explicit context encoder, predictor head, and EMA target encoder
- JEPA probe evaluation path that measures classifier accuracy from frozen JEPA embeddings
- temporal/fixation-aware JEPA targets via `jepa_target_mode=temporal_fixation`

Verified so far:
- retained baseline parity after the relocation:
  - MNIST full: `96.88% (9688/10000)`
  - EMNIST retained gate: `87.29% (4539/5200)`
  - CIFAR-10 retained gate: `33.70% (337/1000)`
- JEPA probe path is live and produces real JEPA-driven results rather than sidecar-only diagnostics
- temporal JEPA training is active when configs include real fixation diversity; the latest smoke run produced nonzero temporal pairs and zero fallback masked examples

Current interpretation:
- the JEPA plumbing is valid
- the frozen JEPA probe is not yet competitive with the protected baselines
- the next work is representation improvement, not downstream classifier tuning

See:
- [Architecture](docs/ARCHITECTURE.md)
- [Baselines](docs/BASELINES.md)
- [Development](docs/DEVELOPMENT.md)
- [JEPA Roadmap](docs/JEPA_ROADMAP.md)
- [JEPA Status](docs/JEPA_STATUS.md)
- [Repository Layout](docs/REPO_LAYOUT.md)
