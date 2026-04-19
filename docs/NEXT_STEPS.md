
# NEXT_STEPS

This document translates the lessons in `docs/RabbitHoles.md` into a practical working strategy for `SNNJepa`.
## Current Status

The repo now has the JEPA research plumbing needed for honest representation experiments:

- stage-1 latent extraction is wired into the bilateral benchmark
- masked visible/hidden branch sampling is implemented
- no-leakage checks are enforced at export and training time
- the JEPA trainer now uses a context encoder, predictor, and EMA target encoder
- probe evaluation on frozen JEPA embeddings is implemented
- temporal fixation targets are implemented and verified in smoke runs with real fixation diversity

What this means in practice:

- baseline parity is protected
- JEPA probe results are real JEPA-driven measurements
- current JEPA probe accuracy is still below the protected baselines
- the next gains must come from better representation learning, not downstream readout tuning

The goal is simple:

- protect the working benchmark surface

- avoid repeating closed dead ends

- focus new work on changes that can plausibly improve the mainline

---

## Protected Mainline

These are the current reference paths. They should be treated as the default comparison surface, not casually replaced.

- Unilateral Retina static

- Bilateral Retina static

- Bilateral Retina continuous

- CIFAR bilateral natural-features reference

Rule:

- every new vision experiment must beat an appropriate protected baseline before it is considered promotable

---

## What To Stop Doing

### 1. Stop Tuning Downstream Readout When Upstream Representation Is Weak

This is the single most repeated failure mode across the closed paths.

Do not spend time on:

- decision microcircuit tuning

- fusion tweaks

- replay tweaks

- classifier-space plasticity tweaks

- attractor/readout retuning

- matcher-vs-centroid toggles

unless upstream representation first shows better separability.

Working rule:

- no more downstream tuning unless branch, hemisphere, or stage-boundary separability measurably improves first

### 2. Stop Adding Biological Structure Without Proof Of Propagation

More biology-shaped structure has repeatedly increased complexity without improving performance.

Avoid defaulting to:

- extra cortical stages

- more branch splitting

- more hierarchy

- more laminar declarations

- more recurrent settling

- more figure-ground insertion

- more object-state logic

unless we first verify:

- synapses or effective pathways exist where expected

- spikes or features propagate

- nonzero downstream drive exists

- the new boundary improves class separation

### 3. Stop Treating Local Metric Improvement As Enough

Several failed experiments improved one or more local metrics such as:

- purity

- margins

- branch activity

- orientation energy

- initial accuracy

- replay corrections

but still failed final benchmark gates.

Working rule:

- no promotion from local metric movement alone

- promotion requires a real protected-gate win, preferably confirmed

### 4. Stop Retrying Same-Dimensional `g10` Reshaping Without A New Reason

This family has already been exercised heavily and mostly failed.

Examples include:

- contextual grouping

- eccentricity remapping

- temporal arbitration

- foveal/peripheral split

- transient/sustained split

- `ON/OFF + luminance` split

- fixation memory

- magno/parvo relay mixing

- edge/operator swaps

- complex-cell pooling

- contour-support banks

- guided saccades by current energy

- burst/tonic gating

- sensory gain plasticity

Default stance:

- do not reopen these with scalar retunes

- reopen only with a materially different representation hypothesis

---

## What To Keep

Not everything from failed branches was wasted. These reusable pieces should remain part of the toolbox.

- exact path-backed declarative group resolution

- wildcard and exact hierarchy path resolution

- incoming-drive-aware competition scoring

- bilateral Retina declarative hierarchy

- bilateral continuous-learning benchmark path

These improve the framework without forcing the project back into failed execution paths.

---

## Decision Rules For New Work

Before spending meaningful time on a new path, require all of the following.

### Gate 1: Mechanistic Validity

Show that the mechanism is actually live.

Examples:

- branch activity is nonzero

- intended channels are carrying energy

- expected stage output is populated

- there is no hidden runtime/config mismatch

### Gate 2: Stage-Wise Separability

Show improvement at the stage the idea is supposed to help.

Examples:

- branch centroid accuracy

- hemisphere purity

- post-normalization margin

- fusion proxy improvement

If the supposed target stage does not improve, stop.

### Gate 3: Protected Benchmark Win

The experiment must beat the protected benchmark at the required gate, not merely tie it and not merely improve smoke by one sample.

### Gate 4: Confirmation

If the win is small, rerun before promoting.

---

## What The Repo Is Really Telling Us

The current ceiling is probably **not** being caused primarily by:

- replay scheduling

- scalar reward tuning

- simple classifier-space plasticity

- fusion arbitration logic

- readout topology alone

The current ceiling is more likely caused by:

- weak or misaligned upstream representation

- insufficiently usable intermediate object structure

- mismatch between local feature improvements and the current fusion/readout surface

This means the next real gains are more likely to come from **representation-stage changes that create usable separability**, not from more downstream cleverness.

---

## Recommended Near-Term Research Posture

### Mainline Posture

Keep the existing Retina and CIFAR references stable.

### Research Posture

Only pursue experiments that satisfy this principle:

**A new mechanism should create a better code before it asks the classifier to be smarter.**

### Practical Screening Question

Before implementing any new idea, ask:

**What exact stage boundary should become more separable, and why?**

If there is no crisp answer, do not do the experiment.

---

## Good Candidate Directions

Based on the rabbit-hole history, the most defensible future work is likely in one of these forms.

### 1. A Materially Different Upstream Representation

Not another scalar retune of the current map, but a new code family that plausibly changes separability.

### 2. A Real Intermediate Representation Boundary

Not classifier-side concatenation or masking, but a distinct boundary that can preserve structure and support object-level organization.

### 3. Predictive Or Recurrent Mechanisms At The Right Level

Not decision-path centroid suppression, but mechanisms operating where actual-minus-expected structure can reshape representation.

### 4. Changes That Preserve Topology While Improving Object-Level Organization

Only if the new boundary can be instrumented and audited for separability before full benchmarking.

---

## Bad Candidate Directions

Do not spend more time on:

- small scalar sweeps around replay or plasticity

- more figure-ground append, mask, or vote variants on the same classifier surface

- more classifier-side fusion tricks

- more same-dimensional `g10` remaps

- more direct retunes of failed branch-split families

- more benchmark-harness mixing with immature graph execution

---

## Default Experimental Workflow

For every new path:

1. verify propagation and activity

2. measure stage-wise separability

3. compare against the protected baseline

4. confirm any narrow win

5. promote only after real benchmark evidence

If the path fails at stage-wise separability, stop early.

---

## Bottom-Line Directive

For current `SNNJepa` vision work, the default should be:

- protect the working benchmark paths

- stop revisiting known dead ends through minor retunes

- bias new work toward representation quality

- require stage-wise evidence before decision/readout tuning

- require real gate wins before promotion

Condensed to one line:

**Fix the code, not the chooser.**
