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

See:
- [Architecture](docs/ARCHITECTURE.md)
- [Baselines](docs/BASELINES.md)
- [Development](docs/DEVELOPMENT.md)
- [JEPA Roadmap](docs/JEPA_ROADMAP.md)
- [Repository Layout](docs/REPO_LAYOUT.md)
