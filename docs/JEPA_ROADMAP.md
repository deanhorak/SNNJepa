# JEPA Roadmap

## Goal

Use the retained spiking retina classification substrate as the base for JEPA-style representation learning.

## Constraints

- stay on a spiking neural substrate
- do not switch to ANN-to-SNN conversion
- do not rely on surrogate gradients as the main training mechanism
- preserve baseline regression paths

## Near-Term Work

### 1. Latent State Split

Introduce separate:
- context / predictor state
- target state

These should be constructed from the existing spiking sensory rollout rather than from a detached ANN encoder.

### 2. Partial Observation Targets

Support masked-view or withheld-fixation targets so the predictor state must infer latent structure from incomplete sensory evidence.

### 3. Temporal Alignment

Add fixation-to-fixation latent consistency metrics:
- predictor-target cosine agreement
- within-image temporal stability
- cross-image separation

### 4. Sequence Replay As State Replay

Replay should reactivate sequences of latent states, not just static exemplar corrections.

## Medium-Term Work

- object-centric recurrent settling
- cross-hemisphere latent support before late fusion
- predictive sensory suppression and error-coded updates

## Exit Criteria For New JEPA Paths

A JEPA path should not be treated as successful unless it shows:
- better latent-state stability across fixations
- preserved or improved class separation
- no obvious representation collapse
- an interpretable gain on at least one retained baseline
