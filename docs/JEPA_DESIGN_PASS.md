# JEPA Design Pass

## Purpose

This document specifies a JEPA design for `SNNJepa` that is actually aligned with the assumptions behind the JEPA papers.

Implementation sequencing for this design lives in
[JEPA_IMPLEMENTATION_CHECKLIST.md](/home/dean/repos/SNNJepa/docs/JEPA_IMPLEMENTATION_CHECKLIST.md).

It is intentionally a design pass, not an implementation plan disguised as one.
The goal is to define:

- what counts as `x` and `y`
- what abstraction level JEPA should operate on
- what the predictor is actually allowed to predict
- what invariances and regularizers are required
- what must be true before new JEPA code should be written

This document supersedes the implicit assumption that the current CIFAR JEPA path is already "close enough" to I-JEPA or H-JEPA.
It is not.

## Research Assumptions To Match

The JEPA papers assume a prediction problem with the following properties:

- the target representation is abstract enough to discard irrelevant details
- the context is spatially or temporally distributed enough to support meaningful prediction
- the target is not a tiny local fragment of raw low-level state
- the predictor maps from context representation to target representation, not from a heavily compressed local hash to another local hash
- collapse prevention is explicit and strong enough to keep the representation informative

For image JEPA specifically, the core assumptions are:

- large target blocks
- sufficiently informative context
- prediction in representation space rather than reconstruction space

Those assumptions are not met by the current CIFAR JEPA path.

## Current Mismatch

The current CIFAR JEPA path differs from the papers in several structural ways:

- tokens are branch-level stage-1 slices, not spatial/semantic blocks
- raw stage-1 values are feature-hashed into a small projection before learning
- the trainer branch is extremely small relative to the compression imposed on the input
- the learned predictor is not the main thing the downstream probe embeds
- the JEPA target is still too close to raw low-level sensory state
- temporal context has been too local and too weak

The practical result has been consistent:

- direct surfaces outperform the learned JEPA embedding on CIFAR
- preserving direct signal helps
- "more JEPA-like" local modifications have not yet produced better bounded CIFAR probe accuracy than the current best line

## Design Decision

`SNNJepa` should stop treating raw stage-1 branch vectors as the right level for JEPA on CIFAR.

The redesigned JEPA should operate one level higher:

- predictor input: aggregated multiview sensory state
- target: abstract future summary state
- unit of prediction: structured view/state summaries, not raw branch fragments

This is closer to H-JEPA than to the current branch-mask setup.

## Stage 0 Lock

Stage 0 is now locked for CIFAR JEPA.

This section is the single active CIFAR JEPA problem definition.
Any future code work should use this section as the source of truth unless it is explicitly revised.

### Chosen `x`

The CIFAR JEPA context `x` is:

- multiple past fixations for the same source image
- both hemispheres
- structured sensory-summary tokens derived from each branch
- fixation-order and hemisphere identity metadata only as auxiliary features

Operationally:

- `x` is a distributed multiview context
- `x` is not a single visible branch
- `x` is not one whole raw branch token
- `x` is not a promoted-surface swap by itself

### Chosen `y`

The CIFAR JEPA target `y` is:

- future pooled hemisphere summary for the same source image

More specifically:

- `y` is built from future fixations
- `y` is pooled at the hemisphere-summary level
- `y` is intended to discard fixation-local unpredictable detail

Operationally:

- `y` is not a hidden branch
- `y` is not the raw next-fixation branch hash
- `y` is not the default promoted stage-1 surface

### Chosen Representation Level

The first implementation target is:

- Level 1: structured sensory summary

This means:

- do not begin with Level 0 raw sensory branch vectors as the main JEPA target
- do not skip directly to Level 2 recurrent/object summaries until Level 1 has been validated

### Chosen Tokenization Family

The first tokenization family is locked to:

- branch-by-subregion tokens

Interpretation:

- each branch must yield multiple tokens per fixation
- each token corresponds to a stable retinotopic or feature-group subregion inside the branch
- tokenization should preserve branch identity while creating a larger structured set than one token per branch

Current Stage 1 lock:

- split each branch slice into `4` equal contiguous subregions
- on the current CIFAR stage-1 surface this yields `12` tokens per hemisphere view

Why this family was chosen:

- it is the closest extension of the current substrate
- it creates large enough structured subsets for JEPA-style prediction
- it avoids jumping immediately to a more speculative object-centric token scheme

What is explicitly not the Stage 0 tokenization choice:

- one token per branch
- branch-by-name only tokenization
- promoted-surface tokenization as the default unit

### Chosen Validation Gate

Before temporal spike-code audit work or predictive-learning changes are allowed, the chosen tokenization and target must pass:

- direct nearest-neighbor probe on `y`
- direct centroid probe on `y`
- temporal consistency on `y`
- context-to-target agreement versus shuffled context

The minimum pre-trainer decision rule is:

- the chosen `y` must be at least competitive with current direct raw/recurrent surfaces
- the chosen `x` must carry more information about `y` than shuffled context

### Stage 0 Out-Of-Scope

The following are now explicitly out of scope for Stage 0:

- trainer redesign
- MLP predictor changes
- covariance regularization work
- probe readout retuning
- promoted-surface experiments

Those are blocked until tokenization and target-surface validation are complete.

## Target JEPA Problem

Define the JEPA prediction problem as:

- `x`: a distributed context built from multiple fixations, both hemispheres, and branch summaries for the same source image
- `y`: a future abstract summary state for the same source image

For CIFAR, the first meaningful `y` is:

- pooled future hemisphere summary

Not:

- one hidden branch
- one raw next-fixation branch hash
- one promoted-surface token

The redesigned JEPA should learn a representation that predicts what stable, future summary state the visual system will settle into, not the exact next low-level activation pattern.

## Representation Levels

Three representation levels should be distinguished explicitly.

### Level 0: Raw Sensory

- raw stage-1 branch vectors
- current promoted stage-1 variants
- fixation-specific low-level sensory state

Use:

- diagnostics
- direct probes
- baseline comparison

Do not use as the main JEPA target for CIFAR.

### Level 1: Structured Sensory Summary

This should become the first JEPA operating level.

A structured sensory summary should include:

- per-branch pooled statistics
- per-hemisphere pooled statistics
- cross-branch agreement/disagreement summaries
- cross-fixation stability summaries

This level is still tied to sensory evidence, but is abstract enough to be predictable.

### Level 2: Recurrent / Object / Scene Summary

This is the longer-term JEPA target surface.

Examples:

- recurrent sensory-state summary
- object-memory summary
- figure-ground ownership summary
- future hemisphere summary after recurrent settling

This is the preferred `y` level if it can be shown to retain class-relevant signal.

## Required Tokenization

JEPA should not continue to treat one whole branch as one token.

Required redesign:

- tokenize each fixation into multiple structured units
- tokens must correspond to meaningful subregions or feature groups
- tokens must support masking of large structured subsets, not tiny local gaps

Acceptable first tokenization options:

- branch-by-band or branch-by-subregion tokens
- retinotopic region groups within a branch
- pooled spatial sectors per hemisphere

Unacceptable option:

- single token per branch as the default JEPA unit

Reason:

- this does not provide large enough target blocks or sufficiently distributed context
- it makes JEPA closer to masked low-level regression than to abstract prediction

## Context Construction

The context `x` must be deliberately richer than the current local temporal setup.

For CIFAR temporal JEPA, the context should include:

- multiple past fixations for the same image
- both hemispheres
- all retained branch summaries for those fixations
- optional metadata for fixation order and hemisphere identity

But metadata must remain auxiliary, not the main predictive signal.

The context should be assembled as a structured set, then pooled or encoded.
It should not depend on one visible branch token.

## Target Construction

The target `y` must be an abstract future state.

Priority order:

1. Future pooled hemisphere summary
2. Future recurrent sensory summary
3. Future object-centric summary if available and separable

The target should explicitly remove unpredictable low-level details through pooling or encoder invariance.

The JEPA should not be asked to preserve every raw branch fluctuation across fixations.

## Encoder Design

The redesigned JEPA should have separate modules for:

- context encoder
- predictor
- target encoder

Minimum acceptable architecture:

- nonlinear MLP or small set encoder for context
- nonlinear predictor
- EMA target encoder

Not acceptable as the intended design:

- purely linear context encoder plus linear predictor

The target encoder should be allowed to enforce invariance by operating on the structured target summary, not on raw sensory vectors.

## Probe Design

The probe must evaluate what the JEPA branch actually learns.

That means:

- probe embeddings should include predictor-derived future-state information when the objective is future-state prediction
- the probe should not ignore the trained predictor and only summarize current-view context features

Separate probes should exist for:

- context-state embedding
- predicted-future embedding
- combined JEPA embedding

This prevents repeating the current failure mode where training optimizes one object while the probe evaluates another.

Stage 4 lock for CIFAR:

- the primary post-training CIFAR JEPA probe surface is the predicted future hemisphere-summary embedding
- context-state and combined embeddings remain diagnostic secondary probes
- a future trainer redesign is blocked from reintroducing a probe that ignores the trained predictor branch

## Loss Design

The loss should have four explicit parts.

### 1. Invariance / Prediction Loss

- prediction of target representation from context representation
- cosine or normalized distance is appropriate

### 2. Variance Floor

- keep per-dimension variance above threshold

### 3. Covariance Regularization

- decorrelate embedding dimensions

### 4. Optional Direct-Signal Preservation

- keep a bounded direct residual path only if it helps downstream probe quality

This residual is a practical bridge to the current substrate.
It should not become the whole representation.

## What Counts As Success

A JEPA path should not be considered viable unless it satisfies all of the following:

- target surface is more abstract than raw branch state
- context is distributed across multiple views or regions
- predictor output is part of the evaluated embedding
- direct probe on the target surface is at least competitive with current raw direct probes
- JEPA embedding beats the current bounded CIFAR JEPA line of `23.40%`

Anything short of that is research evidence, not a promotable design.

## Required Validation Ladder

Before any full CIFAR implementation is treated as serious, it must pass this ladder.

### Gate 1: Target-Surface Validity

Show that the chosen target surface has direct separability worth predicting.

Required:

- direct nearest-neighbor probe
- direct centroid probe
- temporal consistency

If the direct target surface is weak, do not train JEPA on it.

### Gate 2: Context Sufficiency

Show that context contains enough information to predict the target better than chance.

Required:

- context-only direct probe
- context-to-target agreement diagnostics

If context is too weak, change context construction before training.

### Gate 3: Learned JEPA Transfer

Show that the trained JEPA embedding beats the best direct baseline for the same target level.

If it does not, the representation is not helping.

### Gate 4: Bounded CIFAR Gate

Only then run the `100/500` bounded CIFAR comparison against:

- baseline `30.40% (152/500)`
- current best JEPA probe `23.40% (117/500)`

## Architecture To Build Next

The next milestone is not another JEPA trainer sweep.

The next milestone should be a temporal spike-code representation audit:

- measure temporal branch/subregion spike traces
- measure first-spike or rank-order evidence
- measure temporally discounted eligibility traces
- measure fixation-to-fixation stability and separation
- measure branch and hemisphere class separability from temporal spike dynamics

Only after that audit shows a better temporal code should JEPA-style prediction be plugged into the path as a learning signal.

When that gate is satisfied, the next JEPA architecture should be:

- target mode: future summary prediction
- tokenization: structured multi-token summaries within a fixation
- context: multiple past temporal spike traces from both hemispheres
- target: future temporal trace summary, pooled only after preserving useful timing structure
- model: nonlinear context encoder + nonlinear predictor + EMA target encoder
- regularization: variance + covariance
- learning signal: prediction error that gates local or mostly local temporal plasticity
- probe: includes predictor-derived embedding as a diagnostic, not as the only success surface

This is the first architecture in this repo that would plausibly satisfy the assumptions in the JEPA papers.

## Explicit Non-Goals

The following should not be reopened as default next steps:

- promoted-surface swaps without a stronger target surface argument
- more probe readout tuning as the main research direction
- more branch-mask variants on raw stage-1 CIFAR
- more tiny-dimensional linear JEPA trainer variants
- more JEPA trainer sweeps before a temporal spike-code audit
- more frozen summary concatenation tricks without changing the prediction problem itself

These are already documented as rabbit holes.

## Implementation Order

When code work resumes, it should happen in this order:

1. Define structured tokenization above raw branch level
2. Add direct probes for the new target surface
3. Add context builder for multi-fixation, bilateral context
4. Add the temporal spike-code representation audit
5. Use the audit to identify a temporal code that beats raw/recurrent direct surfaces
6. Add JEPA-style prediction error as a learning signal over that temporal code
7. Add predictor-aware diagnostic probe embedding
8. Run bounded CIFAR gate

If any stage fails its validation gate, stop and redesign that stage before proceeding.

## Final Decision Rule

Future JEPA code should only be written if it can be explained in terms of this document.

If a proposed change does not clearly answer:

- what is `x`
- what is `y`
- why `y` is abstract/predictable
- why `x` is sufficiently distributed/informative
- how the probe measures the trained object

then it is not ready for implementation.
