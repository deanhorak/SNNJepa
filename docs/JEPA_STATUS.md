# JEPA Status

## Current State

`SNNJepa` now has a working JEPA research surface layered onto the retained bilateral retina benchmark.

Implemented pieces:
- framework relocation into `include/snnfw/`
- JEPA config, sample, masking, extraction, loss, and trainer modules under `include/snnfw/jepa/`
- stage-1 latent export from `experiments/retina_classification.cpp`
- branch-level masking with explicit visible/hidden splits
- no-leakage validation for branch overlap and token reuse
- minimal JEPA trainer with:
  - context encoder
  - predictor head
  - EMA target encoder
  - latent norm and variance diagnostics
  - shuffled-target control reporting
- JEPA probe mode that evaluates classifier accuracy from frozen JEPA embeddings
- temporal/fixation-aware training targets through `jepa_target_mode=temporal_fixation`

## Verified Milestones

### Baseline Protection

The retained benchmark surface still matches the historical reference points after the core relocation:
- MNIST full: `96.88% (9688/10000)`
- EMNIST retained gate: `87.29% (4539/5200)`
- CIFAR-10 retained gate: `33.70% (337/1000)`

### JEPA Path Is Real

The JEPA path is no longer just a sidecar export. Probe evaluation now runs a classifier on JEPA-derived embeddings and reports an honest accuracy delta against the baseline path.

Early branch-mask JEPA probe runs were valid but substantially worse than baseline. That is a useful result because it confirms the evaluation path is measuring the new representation rather than replaying the old classifier outcome.

### Temporal Targets Are Active

Temporal JEPA training has been implemented and smoke-tested. When the config includes real fixation diversity, the trainer builds temporal examples from successive fixations of the same source image instead of falling back to single-sample branch masking.

The latest temporal smoke verification produced:
- nonzero temporal pairs
- nonzero temporal training examples
- zero fallback masked examples

## What Is Not Done Yet

- JEPA embeddings do not yet improve the protected MNIST, EMNIST, or CIFAR-10 benchmark gates
- JEPA is not yet promoted into the main bilateral decision path
- temporal JEPA probe reruns on larger CIFAR-10, EMNIST, and MNIST slices are still in progress
- the current JEPA trainer is intentionally minimal; it is suitable for representation experiments, not final model claims

## Working Interpretation

The current repo status supports three conclusions:
- the benchmark surface is stable enough for JEPA work
- the JEPA evaluation path is valid
- the limiting factor is representation quality, not missing plumbing

That makes the next phase straightforward: improve the temporal representation and rerun the protected comparisons before changing downstream classifier behavior.
