# Architecture

## Purpose

`SNNJepa` keeps the current retina-classification baseline intact while carving away unrelated experiment history. It is the base substrate for JEPA-style research in a spiking system.

## Active Runtime Surface

The main executable is:
- `experiments/retina_classification.cpp`

The executable uses:
- declarative configs under `configs/`
- dataset loaders in `src/MNISTLoader.cpp` and `src/EMNISTLoader.cpp`
- the retina front end in `src/adapters/RetinaAdapter.cpp`
- the classification and fusion surface in `src/classification/*`
- the core graph runtime in `src/Neuron.cpp`, `src/Synapse.cpp`, `src/NetworkPropagator.cpp`, `src/NeuralObjectFactory.cpp`, and related classes

## Baseline Processing Path

For the retained MNIST, EMNIST, and CIFAR-10 baselines:

1. Dataset samples are loaded from IDX or CIFAR-10 binary batches.
2. Domain-specific retinal adapters generate per-hemisphere feature streams.
3. Each hemisphere builds stage-1 activation patterns.
4. Corpus-callosum-style fusion combines hemisphere evidence.
5. The executable reports classification metrics and optional diagnostics.

## Why The Repo Still Contains Broad Core Code

The workspace intentionally removed extra experiments, scripts, and configs first. The core runtime under `src/` and `include/` is retained more broadly because:
- `retina_classification.cpp` still depends on a large shared surface
- aggressive pruning inside the runtime would create avoidable breakage before JEPA work starts
- the first goal is a stable, smaller research repo, not a risky deep refactor

## Intended JEPA Expansion Points

The most relevant extension points for JEPA work are:
- sensory-state construction in `src/adapters/RetinaAdapter.cpp`
- recurrent settling and population-state formation in `experiments/retina_classification.cpp`
- temporal rollout and replay logic in the same benchmark harness
- object-state, predictor-state, and target-state storage in the training path

The expected direction is to add predictor/target latent-state machinery without replacing the spiking substrate with ANN pretraining or surrogate-gradient pipelines.
