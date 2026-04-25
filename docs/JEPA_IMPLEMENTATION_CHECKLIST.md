# JEPA Implementation Checklist

This checklist translates [JEPA_DESIGN_PASS.md](/home/dean/repos/SNNJepa/docs/JEPA_DESIGN_PASS.md) into a staged implementation plan.

Important boundary:

- do not modify the JEPA trainer until Stage 0 through Stage 3 are complete and signed off
- this checklist is intentionally front-loaded toward tokenization, target-surface validation, and probe alignment
- any stage that fails its exit criteria should stop the rollout

## Scope

This checklist is specifically for the CIFAR JEPA redesign.

Protected comparison surface:

- baseline classifier: `30.40% (152/500)`
- current best bounded CIFAR JEPA probe: `23.40% (117/500)`

These are the numbers a new JEPA path must beat before it can be considered viable.

## Stage 0: Lock The Problem Definition

Goal:

- remove ambiguity about what JEPA is supposed to predict

Status:

- complete
- locked in [JEPA_DESIGN_PASS.md](/home/dean/repos/SNNJepa/docs/JEPA_DESIGN_PASS.md) under `Stage 0 Lock`

Required decisions:

- define `x` for CIFAR JEPA
- define `y` for CIFAR JEPA
- choose the first target representation level
- choose the first tokenization family

Required outputs:

- one short section added to `docs/JEPA_DESIGN_PASS.md` or a linked note that names:
  - chosen `x`
  - chosen `y`
  - chosen tokenization
  - chosen validation gate

Current recommended defaults:

- `x`: multiple past fixations, both hemispheres, structured branch/hemisphere summaries
- `y`: future pooled hemisphere summary
- tokenization family: structured subregion or branch-by-band tokens

Locked choices:

- `x`: multiple past fixations, both hemispheres, structured sensory-summary tokens, auxiliary fixation/hemisphere metadata
- `y`: future pooled hemisphere summary
- representation level: Level 1 structured sensory summary
- tokenization family: branch-by-subregion tokens
- pre-trainer validation gate:
  - direct nearest-neighbor probe on `y`
  - direct centroid probe on `y`
  - temporal consistency on `y`
  - context-to-target agreement versus shuffled context

Exit criteria:

- there is exactly one active CIFAR JEPA problem statement
- no competing “maybe raw branch”, “maybe promoted surface”, or “maybe direct next fixation” ambiguity remains

Do not proceed if:

- `x` and `y` are still being described in multiple incompatible ways

## Stage 1: Define Structured Tokenization

Goal:

- replace single-token-per-branch JEPA units with structured tokens

Status:

- complete
- locked by the Stage 1 audit in [build/cifar10_stage15_audit_1_10.json](/home/dean/repos/SNNJepa/build/cifar10_stage15_audit_1_10.json)

Locked implementation:

- token family: `branch-by-subregion`
- exact schema: equal contiguous subregions inside each branch slice
- configured split: `4` subregions per branch
- current CIFAR stage-1 result: `3 branches x 4 subregions = 12 tokens` per hemisphere view

Required work:

- specify the token schema for one fixation
- ensure each fixation produces multiple tokens per hemisphere
- keep token definitions deterministic and inspectable

Acceptable token families:

- branch-by-band tokens
- branch-by-subregion tokens
- retinotopic region-group tokens
- hemisphere sector summary tokens

Required instrumentation:

- per-token dimensionality
- token count per fixation
- token count per hemisphere
- token sparsity / energy summary

Required artifact:

- a small JSON or console audit for one CIFAR config showing:
  - token names
  - token counts
  - token sizes
  - nonzero/energy stats

Exit criteria:

- each fixation yields a structured multi-token set
- token boundaries are stable and explainable
- no default path remains where one whole branch is the only JEPA token

Stop conditions:

- tokenization is only a rename of existing whole-branch slices
- token counts stay effectively unchanged from the current branch-token path

## Stage 2: Build Candidate Target Surfaces

Goal:

- identify a target surface worth predicting before training JEPA

Status:

- complete
- selected target: `Future Hemisphere Summary Target`

Locked result from [build/cifar10_stage15_audit_1_10.json](/home/dean/repos/SNNJepa/build/cifar10_stage15_audit_1_10.json):

- raw direct `test_nn = 10.0%`
- future hemisphere summary target `test_nn = 20.0%`
- recurrent fixation direct `test_nn = 20.0%`
- recurrent future sensory summary target `test_nn = 10.0%`
- object-centric target: unavailable in the current CIFAR path

Candidate surfaces to validate in order:

1. pooled future hemisphere summary
2. recurrent future sensory summary
3. future object-centric summary if available

Required work:

- implement direct extraction of each candidate target surface
- expose each target surface without training JEPA

Required diagnostics for each candidate:

- direct nearest-neighbor probe
- direct centroid probe
- temporal consistency across fixations
- dimensionality and active-fraction summary

Required output:

- short comparison table in `docs/JEPA_STATUS.md`

Table fields:

- target surface name
- direct `test_nn`
- direct `test_centroid`
- temporal consistency
- dimension
- keep / reject decision

Exit criteria:

- one target surface is selected because its direct probe quality is at least competitive with current raw direct surfaces

Stop conditions:

- no target surface beats or matches the existing direct raw/recurrent probe quality
- target surface is abstract in theory but weak in direct probe practice

## Stage 3: Build The Context Surface

Goal:

- define the distributed context `x` that makes prediction of the chosen `y` plausible

Status:

- complete
- selected context: structured multifixation context from past-half fixations across both hemispheres

Locked result from [build/cifar10_stage15_audit_1_10.json](/home/dean/repos/SNNJepa/build/cifar10_stage15_audit_1_10.json):

- test same-image context/target cosine: `0.9688`
- test shuffled context/target cosine: `0.8561`
- test agreement margin: `+0.1127`

Required work:

- implement context assembly using:
  - multiple past fixations
  - both hemispheres
  - structured tokens or summaries from the chosen tokenization
- make context composition inspectable

Required diagnostics:

- context dimensionality or token count
- context coverage across fixations
- context-only direct probe on the chosen `y` task if feasible
- context-to-target similarity summary for same-image vs shuffled-image pairings

Required output:

- one audit artifact showing:
  - number of views in context
  - number of tokens in context
  - same-image context/target agreement
  - shuffled context/target agreement

Exit criteria:

- context carries more information about the chosen target than a shuffled baseline
- context is clearly distributed, not one local token or one branch fragment

Stop conditions:

- context does not beat shuffled agreement
- context remains effectively local

## Stage 4: Probe Alignment Spec

Goal:

- ensure the future JEPA probe will measure what the future JEPA model is actually trained to do

Status:

- complete
- locked in [JEPA_DESIGN_PASS.md](/home/dean/repos/SNNJepa/docs/JEPA_DESIGN_PASS.md) under `Probe Design`

Locked choice:

- primary future CIFAR JEPA probe surface: predicted future hemisphere-summary embedding

Required design decisions:

- define which embeddings will be exposed after training:
  - context-state embedding
  - predicted-future embedding
  - combined embedding
- define which one is the primary CIFAR JEPA probe surface

Required work:

- document the exact mapping from training objective to probe embedding
- forbid probe designs that ignore the trained predictor when prediction is the objective

Required output:

- a short “training object vs probe object” section in `docs/JEPA_DESIGN_PASS.md` or this file

Exit criteria:

- the future probe is explicitly tied to the future trained object
- no ambiguity remains about what post-training representation is being measured

Stop conditions:

- probe still mainly summarizes current-view context while training optimizes future-state prediction

## Stage 5: Pre-Audit Review Gate

Goal:

- formally decide whether the locked CIFAR problem definition is ready for temporal spike-code audit work

Status:

- complete
- decision: `go`

Locked review artifact:

- [build/cifar10_stage15_audit_1_10.json](/home/dean/repos/SNNJepa/build/cifar10_stage15_audit_1_10.json)
- Stage 5 decision string: `go: Stage1=pass, Stage2=pass, Stage3=pass, Stage4=pass`

Checklist:

- Stage 0 complete
- Stage 1 complete
- Stage 2 complete
- Stage 3 complete
- Stage 4 complete
- chosen target surface has direct signal worth predicting
- chosen context is distributed and informative
- probe is aligned to the future trained object

Required review note:

- add a dated note to `docs/JEPA_STATUS.md` saying:
  - chosen tokenization
  - chosen target surface
  - chosen context definition
  - chosen future probe embedding
  - approval to build the temporal spike-code representation audit

Exit criteria:

- there is an explicit recorded go/no-go decision for the temporal spike-code representation audit

Do not proceed if:

- any earlier stage is incomplete
- target-surface quality is still weak
- context sufficiency has not been demonstrated

## Stage 6: Temporal Spike-Code Representation Audit

This stage supersedes a direct trainer-redesign sweep as the next milestone.

Goal:

- prove that the temporal spiking code itself is becoming more separable before adding another JEPA trainer variant

Required audit surfaces:

- temporal branch/subregion spike traces
- first-spike or rank-order evidence
- temporally discounted eligibility traces
- fixation-to-fixation stability and separation
- branch and hemisphere class separability from temporal spike dynamics

Required diagnostics:

- temporal-code nearest-neighbor and centroid probes
- same-image versus shuffled-fixation temporal consistency
- per-branch/subregion active fraction and timing spread
- evidence accumulation over fixation order
- comparison against current raw/recurrent direct surfaces

Exit criteria:

- at least one temporal spike-code surface beats the current direct raw/recurrent audit surface at the same slice
- the improvement is visible before any JEPA-style prediction loss is added
- the code remains temporal-spiking based and does not collapse into rate-coded static vectors

Do not proceed if:

- the only improvement is a sidecar JEPA probe change
- the audit cannot separate spike timing effects from static activation magnitude
- branch or hemisphere separability does not improve

## Stage 7: JEPA-Style Predictive Learning Signal

This stage is blocked until Stage 6 shows a better temporal code.

When unlocked, JEPA-style prediction should be plugged into the temporal code as a learning signal, not treated as another standalone trainer sweep.

Required design:

- context: past temporal spike traces across fixations and hemispheres
- target: future temporal trace summary
- signal: prediction error over temporal summaries
- plasticity path: local or mostly local eligibility/STDP/BCM-style updates gated by prediction error
- readout: temporal-code audit and protected CIFAR path, not only a frozen JEPA probe

Allowed JEPA components:

- nonlinear context encoder
- nonlinear predictor
- EMA target encoder
- variance regularization
- covariance regularization
- predictor-aware probe embedding

These components are support machinery. They are not the main milestone.

## Stage 8: Validation Ladder After Predictive Learning Changes

Once predictive learning changes are approved and implemented:

1. rerun the temporal spike-code representation audit
2. run direct-vs-learned comparison on the same temporal surface
3. run bounded CIFAR `100/500` through the temporal spiking path
4. compare against:
   - baseline `30.40% (152/500)`
   - current best JEPA `23.40% (117/500)`

Promotion rule:

- the predictive path is only viable if it beats the current JEPA line of `23.40%`
- it is not promotable into the main path unless it also beats the bounded classifier baseline of `30.40%`

## Deliverables By Stage

Stage 0:

- locked `x` / `y` definition

Stage 1:

- structured tokenization audit

Stage 2:

- target-surface comparison table

Stage 3:

- context sufficiency audit

Stage 4:

- probe-alignment spec

Stage 5:

- explicit go/no-go note for temporal spike-code audit work

Stage 6:

- temporal spike-code representation audit

Stage 7:

- JEPA-style predictive learning signal over the audited temporal code

Stage 8:

- bounded CIFAR comparison vs `23.40%` and `30.40%`

## Explicit Reminders

- no more promoted-surface swaps as a default research path
- no more probe readout churn as the main direction
- no more small JEPA trainer edits before the temporal spike-code representation audit
- no more treating local trainer metrics as success if bounded CIFAR probe regresses
