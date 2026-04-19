# Architecture

## Purpose

`SNNJepa` keeps the current retina-classification baseline intact while carving away unrelated experiment history. It is the base substrate for JEPA-style research in a spiking system.

## Active Runtime Surface

The main executable is:
- `experiments/retina_classification.cpp`

The executable uses:
- declarative configs under `configs/`
- dataset loaders in `include/snnfw/MNISTLoader.cpp` and `include/snnfw/EMNISTLoader.cpp`
- the retina front end in `include/snnfw/adapters/RetinaAdapter.cpp`
- the classification and fusion surface in `include/snnfw/classification/*`
- the core graph runtime in `include/snnfw/Neuron.cpp`, `include/snnfw/Synapse.cpp`, `include/snnfw/NetworkPropagator.cpp`, `include/snnfw/NeuralObjectFactory.cpp`, and related classes
- the JEPA research surface in `include/snnfw/jepa/*`

## Baseline Processing Path

For the retained MNIST, EMNIST, and CIFAR-10 baselines:

1. Dataset samples are loaded from IDX or CIFAR-10 binary batches.
2. Domain-specific retinal adapters generate per-hemisphere feature streams.
3. Each hemisphere builds stage-1 activation patterns.
4. Corpus-callosum-style fusion combines hemisphere evidence.
5. The executable reports classification metrics and optional diagnostics.

## Why The Repo Still Contains Broad Core Code

The workspace intentionally removed extra experiments, scripts, and configs first. The core runtime under `include/snnfw/` is retained more broadly because:
- `retina_classification.cpp` still depends on a large shared surface
- aggressive pruning inside the runtime would create avoidable breakage before JEPA work starts
- the first goal is a stable, smaller research repo, not a risky deep refactor

## Intended JEPA Expansion Points

The most relevant extension points for JEPA work are:
- sensory-state construction in `include/snnfw/adapters/RetinaAdapter.cpp`
- recurrent settling and population-state formation in `experiments/retina_classification.cpp`
- temporal rollout and replay logic in the same benchmark harness
- object-state, predictor-state, and target-state storage in the training path

The expected direction is to add predictor/target latent-state machinery without replacing the spiking substrate with ANN pretraining or surrogate-gradient pipelines.

## Current JEPA Runtime Shape

The current JEPA implementation is attached to the bilateral retina benchmark as a research path, not as the default classifier path.

Today it provides:
- stage-1 latent tap extraction from bilateral hemisphere training and evaluation samples
- branch-level masking with leakage checks
- a JEPA trainer with an explicit context encoder, predictor, and EMA target encoder
- probe evaluation over frozen JEPA embeddings
- an optional temporal target mode that pairs fixation `t` with fixation `t + 1` for the same source image, hemisphere, and surface

Important boundary:
- baseline benchmark accuracy still comes from the protected bilateral classifier path unless JEPA probe mode is explicitly enabled
- JEPA probe accuracy is a separate evaluation surface used to measure whether the learned representation is useful

This separation is intentional. It allows JEPA work to be validated without silently perturbing the retained baseline path.
