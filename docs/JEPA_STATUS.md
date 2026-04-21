# JEPA Status

## Purpose of This Document

This file is meant to be a high-context handoff for future work on the JEPA path in `SNNJepa`.
It captures:

- what the repo is trying to do
- what has already been implemented
- which results are valid and which were misleading
- what code was changed most recently
- where the current blockers are
- what the next conversation should do next

If a new conversation starts from this file, it should not need to reconstruct the recent JEPA work from scratch.

## Project Goal

`SNNJepa` is not trying to replace the retained retina benchmark with a new ANN stack. The intended direction is:

- keep the existing spiking / retina / bilateral classification pipeline runnable
- preserve baseline regressions on MNIST, EMNIST, and CIFAR-10
- add JEPA-style learning on top of stage-1 spiking representations
- use JEPA first as a representation-learning path and probe it honestly before trying to replace the main classifier

The local design reference for that direction is:

- [JEPA Implementation Strategy for SNNJepa.pdf](/home/dean/repos/SNNJepa/docs/JEPA%20Implementation%20Strategy%20for%20SNNJepa.pdf)

The most important external references used in the recent correction pass were:

- I-JEPA: https://arxiv.org/abs/2301.08243
- A-JEPA: https://arxiv.org/abs/2311.15830
- V-JEPA: https://openreview.net/forum?id=WFYbBOEOtv

## High-Level State

The repo now has a real JEPA research surface, not just placeholder scaffolding.

Implemented:

- framework relocation under `include/snnfw/`
- JEPA config, samples, masking, extraction, and trainer under `include/snnfw/jepa/`
- stage-1 tap export from `experiments/retina_classification.cpp`
- branch masking with visible / hidden splits
- no-leakage checks
- EMA target encoder, context encoder, predictor
- probe evaluation on frozen JEPA embeddings
- temporal target modes

What changed most recently:

- the temporal JEPA path was corrected so that temporal modes now build real temporal examples instead of silently degrading into masked single-view training
- the JEPA probe eval path was corrected so that temporal JEPA is evaluated on multi-fixation samples instead of single-view eval taps

Current reality:

- the JEPA plumbing is real
- the trainer is now capable of building honest temporal examples
- JEPA still does not beat the retained benchmark classifier
- but the current JEPA probe results are much more meaningful than the earlier fake-fallback results

## Files That Matter Most

Core JEPA code:

- [include/snnfw/jepa/JepaConfig.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaConfig.h)
- [include/snnfw/jepa/JepaSample.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaSample.h)
- [include/snnfw/jepa/JepaStateExtractor.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaStateExtractor.cpp)
- [include/snnfw/jepa/JepaTrainer.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.h)
- [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)

Benchmark integration:

- [experiments/retina_classification.cpp](/home/dean/repos/SNNJepa/experiments/retina_classification.cpp)

Useful configs:

- [build_nolto/jepa_configs/mnist_temporal_probe_200_1000.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/mnist_temporal_probe_200_1000.sonata.json)
- [build_nolto/jepa_configs/mnist_temporal_probe_hemisummary_200_1000.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/mnist_temporal_probe_hemisummary_200_1000.sonata.json)
- [build_nolto/jepa_configs/emnist_temporal_probe_500_1000.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/emnist_temporal_probe_500_1000.sonata.json)
- [build_nolto/jepa_configs/emnist_temporal_probe_hemisummary_500_1000.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/emnist_temporal_probe_hemisummary_500_1000.sonata.json)
- [build_nolto/jepa_configs/cifar10_temporal_probe_100_500.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/cifar10_temporal_probe_100_500.sonata.json)
- [build_nolto/jepa_configs/cifar10_temporal_probe_hemisummary_100_500.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/cifar10_temporal_probe_hemisummary_100_500.sonata.json)

## What Was Wrong Before

This is important because some earlier JEPA numbers looked better than they really were.

### 1. Temporal hemisphere-summary was not actually temporal

`temporal_hemisphere_summary` did not go through the multi-fixation extraction path in `buildJepaStage1TapInputs(...)`.
That meant it often produced:

- `temporal_examples = 0`
- `fallback_masked_examples > 0`

So earlier “temporal hemisphere-summary” wins were actually branch-mask fallback runs.

### 2. Temporal pairing was wrong

`buildTemporalExamples(...)` used a cyclic pairing:

- `0 -> 1`
- `1 -> 2`
- `2 -> 0`

That is not the intended JEPA temporal structure.
It now uses only forward adjacent pairs:

- `0 -> 1`
- `1 -> 2`

and does not wrap the last fixation back to the first.

### 3. Temporal probe evaluation was mismatched

Training JEPA on multi-fixation samples but evaluating the frozen JEPA probe on single-view eval taps was a bad mismatch.

That meant:

- the JEPA encoder learned from one sample structure
- the probe was judged on a different structure

The eval tap builder now mirrors the temporal train path for temporal target modes.

### 4. Silent fallback made bad runs look valid

Before the last correction, temporal target modes could silently fall back to masked examples.
Now temporal modes throw if they cannot produce temporal examples.

That change matters because it prevents future false-positive JEPA “improvements”.

## Most Recent Correctness Fixes

Recent code changes of consequence:

### In [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)

- temporal pair construction changed from cyclic wrap-around to strict `t -> t+1`
- temporal modes now fail if zero temporal examples are built
- fallback to masked examples is no longer allowed for temporal target modes

### In [experiments/retina_classification.cpp](/home/dean/repos/SNNJepa/experiments/retina_classification.cpp)

- temporal extraction now applies to both:
  - `temporal_fixation`
  - `temporal_hemisphere_summary`
- temporal JEPA probe eval now builds multi-fixation eval taps instead of single-view taps

These fixes are the reason the newest MNIST and EMNIST temporal results are more trustworthy than the earlier ones.

## Baseline Status

The retained benchmark surface remains intact.
Previously verified retained baselines:

- MNIST full retained reference: `96.88% (9688/10000)`
- EMNIST retained reference: `87.29% (4539/5200)`
- CIFAR-10 retained reference: `33.70% (337/1000)`

For the smaller JEPA probe slices used more recently:

- MNIST baseline on the JEPA slice: `93.30% (933/1000)`
- EMNIST baseline on the JEPA slice: `83.80% (838/1000)`
- CIFAR bounded JEPA slice baseline: `30.40% (152/500)`
- retained CIFAR baseline reference remains `33.70% (337/1000)` on the protected slice

## Valid JEPA Results

### Important rule for interpreting results

Only count a JEPA run as a valid temporal result if:

- `temporal_example_count > 0`
- `fallback_masked_example_count = 0`

Everything below is split into valid and invalid accordingly.

### Previously invalid or misleading results

These are not reliable temporal JEPA results and should not be used for decision making:

- earlier MNIST `temporal_hemisphere_summary` probe around `12.30%`
  - invalid because it was actually a fallback masked run
- earlier EMNIST `temporal_hemisphere_summary` probe around `9.60%`
  - invalid for the same reason
- earlier CIFAR hemisphere-summary attempts with `temporal_example_count = 0`
  - invalid temporal comparison

### Current valid MNIST results

MNIST is now the clearest signal that the JEPA path can improve materially with the right capacity.

Current best validated JEPA result:

- dataset: MNIST
- config: `mnist_temporal_probe_capacity_200_1000.sonata.json`
- target mode: `temporal_fixation`
- train budget: `200/class`
- test budget: `1000`
- baseline: `93.30% (933/1000)`
- JEPA probe: `84.90% (849/1000)`
- delta vs baseline: `-8.40 pts`
- log: [build/mnist_temporal_probe_capacity_200_1000_rerun.log](/home/dean/repos/SNNJepa/build/mnist_temporal_probe_capacity_200_1000_rerun.log)

Current trainer artifact on disk:

- [build/jepa_minimal_trainer_mnist_temporal_probe_200_1000.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_mnist_temporal_probe_200_1000.json)

Key metrics from that current artifact:

- `target_mode = temporal_fixation`
- `projection_dim = 32`
- `epoch_count = 20`
- `temporal_example_count = 8000`
- `fallback_masked_example_count = 0`
- `mean_loss = 0.16815613797660483`
- `mean_shuffled_loss = 0.24348509979979664`

Previously validated honest temporal hemisphere-summary reference:

- config: `temporal_hemisphere_summary`
- baseline: `93.30% (933/1000)`
- JEPA probe: `58.80% (588/1000)`
- delta vs baseline: `-34.50 pts`

Important artifact note:

- the newer MNIST capacity rerun reused `build/jepa_minimal_trainer_mnist_temporal_probe_200_1000.json`
- that means the current JSON on disk now reflects the capacity run, not the older `58.80%` hemisphere-summary run

Negative control worth remembering:

- `mnist_temporal_probe_mlp_200_1000.sonata.json` collapsed to `9.00% (90/1000)`
- that probe mode is currently much worse than the kNN-style probe and should not be treated as an improvement path without further work

Interpretation:

- the corrected JEPA path is no longer stuck in the `~58%` range on MNIST
- increased trainer capacity helps dramatically on the same bounded slice
- the main remaining gap on MNIST is now single digits instead of tens of points
- probe choice still matters a lot; the current MLP probe is unstable / poor

Capacity-transfer rerun on MNIST temporal hemisphere summary:

- config: `mnist_temporal_probe_hemisummary_200_1000.sonata.json`
- target mode: `temporal_hemisphere_summary`
- updated settings: `projection_dim = 32`, `epoch_count = 20`
- baseline: `93.30% (933/1000)`
- JEPA probe: `84.80% (848/1000)`
- delta vs baseline: `-8.50 pts`
- log: [build/mnist_temporal_probe_hemisummary_200_1000_capacity_rerun.log](/home/dean/repos/SNNJepa/build/mnist_temporal_probe_hemisummary_200_1000_capacity_rerun.log)

Interpretation:

- the same capacity change that helped MNIST temporal fixation also helps MNIST temporal hemisphere summary
- on the bounded MNIST slice, the two temporal target modes are now effectively tied

### Current valid EMNIST results

Both corrected EMNIST runs are valid temporal runs.

Baseline:

- `83.80% (838/1000)`

Original corrected temporal fixation reference:

- JEPA probe: `29.90% (299/1000)`
- delta vs baseline: `-53.90 pts`
- artifact: [build/jepa_minimal_trainer_emnist_temporal_probe_500_1000.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_emnist_temporal_probe_500_1000.json)

Key trainer values:

- `target_mode = temporal_fixation`
- `temporal_example_count = 8000`
- `fallback_masked_example_count = 0`
- `mean_loss = 0.13176290299522223`
- `mean_shuffled_loss = 0.1882622432218357`

Original corrected temporal hemisphere summary reference:

- JEPA probe: `29.50% (295/1000)`
- delta vs baseline: `-54.30 pts`
- artifact: [build/jepa_minimal_trainer_emnist_temporal_probe_hemisummary_500_1000.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_emnist_temporal_probe_hemisummary_500_1000.json)

Key trainer values:

- `target_mode = temporal_hemisphere_summary`
- `temporal_example_count = 8000`
- `fallback_masked_example_count = 0`
- `mean_loss` is slightly lower than the temporal-fixation run on the same slice
- `mean_shuffled_loss` is slightly below the fixation-mode counterpart but probe accuracy is also slightly worse

Interpretation:

- EMNIST currently favors `temporal_fixation` very slightly over `temporal_hemisphere_summary`
- the difference is small
- both are valid temporal JEPA runs
- both remain well below the baseline classifier

Capacity-transfer rerun results:

Temporal fixation with `projection_dim = 32`, `epoch_count = 20`:

- JEPA probe: `61.40% (614/1000)`
- delta vs baseline: `-22.40 pts`
- log: [build/emnist_temporal_probe_500_1000_capacity_rerun.log](/home/dean/repos/SNNJepa/build/emnist_temporal_probe_500_1000_capacity_rerun.log)

Temporal hemisphere summary with `projection_dim = 32`, `epoch_count = 20`:

- JEPA probe: `61.60% (616/1000)`
- delta vs baseline: `-22.20 pts`
- log: [build/emnist_temporal_probe_hemisummary_500_1000_capacity_rerun.log](/home/dean/repos/SNNJepa/build/emnist_temporal_probe_hemisummary_500_1000_capacity_rerun.log)

Updated interpretation:

- the capacity transfer improved EMNIST massively relative to the earlier `~30%` temporal runs
- EMNIST temporal fixation and temporal hemisphere summary are again effectively tied
- EMNIST is still below baseline, but the gap is now much smaller than before

## CIFAR Status

CIFAR now has a clean bounded end-to-end comparison for both corrected temporal modes.

### What is already validated on CIFAR

Fresh trainer artifacts and rerun logs now confirm that both CIFAR temporal modes can produce real temporal examples and complete probe evaluation on the smaller bounded slice:

- [build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json)
- [build/jepa_minimal_trainer_cifar10_temporal_probe_hemisummary_100_500.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_hemisummary_100_500.json)
- [build/cifar10_temporal_probe_100_500_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_rerun.log)
- [build/cifar10_temporal_probe_hemisummary_100_500_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_rerun.log)

Bounded baseline on this slice:

- `30.40% (152/500)`

Temporal fixation:

- JEPA probe: `16.20% (81/500)`
- delta vs baseline: `-14.20 pts`
- `temporal_example_count = 6000`
- `fallback_masked_example_count = 0`
- `mean_loss = 0.2016238821061088`
- `mean_shuffled_loss = 0.2407379494787585`

Temporal hemisphere summary:

- JEPA probe: `16.40% (82/500)`
- delta vs baseline: `-14.00 pts`
- `temporal_example_count = 6000`
- `fallback_masked_example_count = 0`
- `mean_loss = 0.20178586277049881`
- `mean_shuffled_loss = 0.2407854208979512`

This is important because it means the corrected temporal data path is working on CIFAR as well.

Current conclusion for CIFAR:

- the corrected temporal comparison is now captured cleanly
- `temporal_fixation` and `temporal_hemisphere_summary` are effectively tied on the bounded slice
- both remain far below the retained baseline classifier
- the CIFAR bounded configs were rerun with `projection_dim = 32`, `epoch_count = 20`

Capacity-transfer rerun results:

Temporal fixation with `projection_dim = 32`, `epoch_count = 20`:

- JEPA probe: `18.20% (91/500)`
- delta vs baseline: `-12.20 pts`
- log: [build/cifar10_temporal_probe_100_500_capacity_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_capacity_rerun.log)

Temporal hemisphere summary with `projection_dim = 32`, `epoch_count = 20`:

- JEPA probe: `18.00% (90/500)`
- delta vs baseline: `-12.40 pts`
- log: [build/cifar10_temporal_probe_hemisummary_100_500_capacity_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_capacity_rerun.log)

Updated interpretation:

- the higher-capacity settings improve CIFAR slightly relative to the earlier `~16%` JEPA runs
- the improvement is much smaller than on MNIST or EMNIST
- CIFAR temporal fixation remains slightly better than temporal hemisphere summary
- trainer loss and shuffled loss both worsened, so this does not look like a clean representation win yet

Structured embedding trial on CIFAR:

- experiment: richer probe embedding that added predictor-pooled features, per-group variances, and temporal deltas inside `encodeSample(...)`
- temporal fixation rerun: `18.00% (90/500)`
- temporal hemisphere summary rerun: `18.00% (90/500)`
- logs:
  - [build/cifar10_temporal_probe_100_500_structured_embedding_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_structured_embedding_rerun.log)
  - [build/cifar10_temporal_probe_hemisummary_100_500_structured_embedding_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_structured_embedding_rerun.log)

Interpretation:

- this did not improve CIFAR accuracy relative to the simpler pooled embedding
- it also made the JEPA probe pass substantially slower
- the trial was therefore reverted rather than kept as the new default

Promoted temporal tap semantics fix and smoke:

- correction: temporal JEPA taps now respect `jepa_tap_surface=promoted_stage1` instead of always feeding raw `extractPattern(...)` outputs during multi-fixation train/eval
- related correction: JEPA masking now leaves single-token promoted views unmasked instead of marking them as leakage failures, so temporal JEPA can consume promoted classifier-side vectors safely
- smoke run: CIFAR temporal fixation with a real promoted surface via `--figure-ground-mask-gain 0.35`
- budget: `20/class`, `200` test images
- baseline: `31.00% (62/200)`
- JEPA probe: `15.00% (30/200)`
- delta vs baseline: `-16.00 pts`
- log: [build/cifar10_temporal_probe_fgmask_promoted_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_fgmask_promoted_smoke_20_200.log)

Interpretation:

- this was a correctness-first smoke, not a new candidate winner
- it confirms the temporal JEPA path can now operate on a genuinely promoted classifier-side surface
- the first promoted-surface CIFAR smoke was materially worse than the current bounded raw-equivalent surface
- do not escalate this exact figure-ground-mask promoted surface to a larger CIFAR JEPA gate

Materially different promoted-boundary smoke:

- boundary: recurrent sensory-state settling with promoted temporal JEPA taps
- implementation note: temporal JEPA promoted taps now route through the same classifier-side promotion logic as the main decision path, and can also consume recurrent fixation patterns as the JEPA surface
- smoke run: CIFAR temporal fixation with `--recurrent-sensory-state-enabled`
- budget: `20/class`, `200` test images
- baseline: `29.00% (58/200)`
- JEPA probe: `15.00% (30/200)`
- delta vs baseline: `-14.00 pts`
- log: [build/cifar10_temporal_probe_recurrent_promoted_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_recurrent_promoted_smoke_20_200.log)

Interpretation:

- this was the right next promoted-boundary test because it is fixation-aware and materially different from the earlier figure-ground mask append
- the temporal JEPA trainer remained structurally valid on this boundary: `temporal_examples=1200`, `fallback_masked_examples=0`
- the promoted recurrent boundary did not improve CIFAR JEPA accuracy relative to the earlier promoted smoke; it landed at the same `15.00%` probe accuracy
- this is enough evidence to stop spending time on promoted-boundary smokes until there is a better CIFAR-side representation hypothesis

Representation audit implementation and first CIFAR smoke:

- code path: `retina_classification` now supports a JEPA representation audit that measures direct-surface separability and same-image fixation consistency on the exact train/test slice used by the run
- CLI/config knobs:
  - `--jepa-representation-audit-enabled`
  - `--jepa-representation-audit-sample-limit <n>`
  - `--jepa-representation-audit-output-path <path>`
  - config keys: `jepa_representation_audit_enabled`, `jepa_representation_audit_sample_limit`, `jepa_representation_audit_output_path`
- report output: JSON artifact plus console summary
- validation smoke:
  - config: `cifar10_temporal_probe_100_500.sonata.json`
  - budget: `20/class`, `200` test images
  - baseline classifier: `32.00% (64/200)`
  - JEPA probe: `13.50% (27/200)`
  - audit artifact: [build/cifar10_temporal_probe_representation_audit_smoke_20_200.json](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_smoke_20_200.json)
  - log: [build/cifar10_temporal_probe_representation_audit_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_smoke_20_200.log)

Audit readout:

- direct raw stage-1: `test_nn = 23.50%`, `train_loo = 21.89%`, `test_centroid = 28.00%`
- direct promoted stage-1: exactly the same as raw on this config
- direct recurrent fixation surface: `test_nn = 25.50%`, `train_loo = 19.40%`, `test_centroid = 28.50%`
- JEPA embedding: `test_nn = 14.50%`, `train_loo = 15.42%`, `test_centroid = 10.50%`
- fixation consistency is high on all direct surfaces (`~0.94` raw/promoted, `~0.96` recurrent), so CIFAR temporal views are internally stable

Interpretation:

- CIFAR direct surfaces are weak, but they are still materially better than the trained JEPA embedding on the same slice
- on this config, `promoted_stage1` is not changing the direct CIFAR representation at all; raw and promoted audits are numerically identical
- recurrent fixation patterns improve direct nearest-neighbor accuracy slightly, but JEPA is still worse than those direct recurrent surfaces
- the immediate CIFAR blocker is now more clearly JEPA representation degradation, not missing temporal stability

Direct-surface residual fix:

- code change: `encodeSample(...)` in `JepaTrainer.cpp` now preserves a direct-surface residual inside the frozen JEPA embedding instead of relying only on the learned `32d` context encoder output
- first attempt: a `32d` direct residual skip path only moved the bounded CIFAR JEPA probe from `13.50%` to `14.00%`
- second attempt: widen only the direct residual sketch to `256d` while keeping the JEPA trainer itself at `32d`
- validation smoke:
  - config: `cifar10_temporal_probe_100_500.sonata.json`
  - budget: `20/class`, `200` test images
  - baseline classifier: `32.00% (64/200)`
  - JEPA probe after fix: `17.00% (34/200)`
  - audit artifact: [build/cifar10_temporal_probe_representation_audit_widedirect_smoke_20_200.json](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_widedirect_smoke_20_200.json)
  - log: [build/cifar10_temporal_probe_representation_audit_widedirect_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_widedirect_smoke_20_200.log)

Updated audit readout:

- direct raw stage-1: `test_nn = 23.50%`
- direct recurrent fixation surface: `test_nn = 25.50%`
- JEPA embedding after wide direct residual: `test_nn = 16.50%`, up from `14.50%`
- JEPA embedding dimension after the fix: `2304`

Interpretation:

- this does not solve CIFAR, but it is the first representation-side fix that clearly moved the JEPA embedding toward the direct-surface signal instead of away from it
- the immediate JEPA failure mode was over-compressing away direct class information; widening the direct residual partially fixes that
- the larger bounded reruns confirm that the gain holds on both CIFAR temporal modes:
  - temporal fixation: baseline `30.00% (150/500)`, JEPA probe `22.80% (114/500)`, delta `-7.20 pts`
    - log: [build/cifar10_temporal_probe_100_500_widedirect_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_widedirect_rerun.log)
  - temporal hemisphere summary: baseline `30.00% (150/500)`, JEPA probe `22.80% (114/500)`, delta `-7.20 pts`
    - log: [build/cifar10_temporal_probe_hemisummary_100_500_widedirect_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_widedirect_rerun.log)
- the next useful step is no longer “verify the larger CIFAR gate”; it is to push the same representation-preserving idea further without regressing the now-improved bounded CIFAR line

Follow-up temporal-stat trial:

- experiment: extend the frozen JEPA embedding with direct per-fixation variances plus explicit direct consecutive-fixation deltas
- small smoke on CIFAR temporal fixation:
  - budget: `20/class`, `200` test images
  - baseline: `32.00% (64/200)`
  - JEPA probe: `20.00% (40/200)`
  - audit artifact: [build/cifar10_temporal_probe_representation_audit_temporalstats_smoke_20_200.json](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_temporalstats_smoke_20_200.json)
  - log: [build/cifar10_temporal_probe_representation_audit_temporalstats_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_representation_audit_temporalstats_smoke_20_200.log)
- full bounded fixation rerun:
  - baseline: `30.00% (150/500)`
  - JEPA probe: `22.60% (113/500)`
  - log: [build/cifar10_temporal_probe_100_500_temporalstats_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_temporalstats_rerun.log)

Interpretation:

- the added temporal statistics looked promising on the small smoke, but they did not beat the current best bounded CIFAR line
- the patch was therefore reverted, and the wide direct residual remains the best known CIFAR embedding path

## Current Working Hypothesis

The recent corrections support the following:

- earlier JEPA probe failures were partly due to invalid temporal plumbing, not just bad representations
- once temporal pairing and temporal eval are fixed, JEPA probe quality improves substantially on MNIST
- increasing JEPA trainer capacity on MNIST improves probe quality much more than the recent probe-only variants
- the same capacity transfer also improves EMNIST dramatically
- CIFAR temporal training and probe capture are now both structurally valid, but accuracy is still low

The current best interpretation is:

- the JEPA path is now worth iterating on
- MNIST is the best near-term surface for representation improvement work
- but it is still a representation-learning experiment, not a replacement for the baseline classifier

## Open Problems

### 1. Transfer the MNIST capacity gain

The best current JEPA improvement came from the higher-capacity MNIST temporal-fixation config:

- `projection_dim = 32`
- `epoch_count = 20`
- JEPA probe improved to `84.90%`

Need to test whether that gain transfers to:

- richer CIFAR JEPA settings than the simple `32d/20ep` transfer

### 2. JEPA still under baseline

Even after the temporal fixes and the MNIST capacity gain:

- MNIST is still `-8.40 pts` below baseline on the best bounded run
- MNIST temporal hemisphere summary is still `-8.50 pts` below baseline
- EMNIST is still about `-22 pts` below baseline after the capacity transfer
- CIFAR is still only about `18%` on the bounded slice after the capacity transfer

### 3. Probe path is unstable across probe choices

The new MLP probe path is implemented, but the current MNIST bounded run shows it is not working well yet:

- MNIST `mlp` probe only reached `9.00%`
- that is dramatically worse than the kNN-style probe on the same JEPA setup

So probe improvements are not just a matter of “use MLP instead”.

### 4. JEPA is not in the main decision path

The main classifier path still produces the benchmark accuracy.
JEPA is still a sidecar representation path plus probe, not a promoted inference path.

## Current Local Code Changes

At the time of writing, the working tree includes JEPA-related modifications in:

- [experiments/retina_classification.cpp](/home/dean/repos/SNNJepa/experiments/retina_classification.cpp)
- [include/snnfw/jepa/JepaConfig.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaConfig.h)
- [include/snnfw/jepa/JepaMaskSampler.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaMaskSampler.cpp)
- [include/snnfw/jepa/JepaTrainer.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.h)
- [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)

These changes include:

- probe MLP support
- additional loss / metadata config knobs
- corrected temporal tap construction
- corrected temporal eval taps
- temporal promoted-surface JEPA taps now honor the selected tap surface
- promoted single-token JEPA views no longer fail masking by construction
- JEPA representation audit support for direct raw/promoted/recurrent surfaces and trained JEPA embeddings
- direct-surface residual preservation in the frozen JEPA embedding, with a wider direct sketch for CIFAR-like high-dimensional taps
- strict temporal example enforcement

## Recommended Next Steps

Ordered by value:

1. Stop doing straight capacity-copy transfers for CIFAR.
   The `32d/20ep` change only improved CIFAR from about `16%` to about `18%`, which is too small to justify more blind config scaling.

2. Keep a short table of the valid temporal JEPA modes in one place.
   Capture a short table for:
   - MNIST `temporal_fixation`
   - MNIST `temporal_hemisphere_summary`
   - EMNIST `temporal_fixation`
   - EMNIST `temporal_hemisphere_summary`
   - CIFAR `temporal_fixation`
   - CIFAR `temporal_hemisphere_summary`

3. Stop treating the MLP probe as a likely win until it is debugged.
   The current result is a regression, not an improvement.

4. Representation-side improvements to consider next:
   - for CIFAR specifically, stop running new promoted-boundary smokes until the JEPA encoder stops underperforming the direct surfaces
   - investigate why `promoted_stage1` is identical to `raw_stage1` on the current CIFAR temporal config
   - the new wide direct residual improved bounded CIFAR from `13.50%` to `17.00%`, so confirm that gain on the `100/500` gate before changing the trainer objective again
   - improve the JEPA objective / encoder so the learned embedding preserves more of the direct-surface class signal already present in raw and recurrent fixations
   - only after that should richer predictor capacity or extra positional conditioning be revisited

5. Only after the probe becomes competitive should JEPA be tested inside the main bilateral decision path.

## Suggested Rerun Commands

These are the important corrected run forms to reuse.

MNIST higher-capacity temporal fixation probe:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/mnist_temporal_probe_capacity_200_1000.sonata.json \
  --train-images /home/dean/repos/data/MNIST/raw/train-images-idx3-ubyte \
  --train-labels /home/dean/repos/data/MNIST/raw/train-labels-idx1-ubyte \
  --test-images /home/dean/repos/data/MNIST/raw/t10k-images-idx3-ubyte \
  --test-labels /home/dean/repos/data/MNIST/raw/t10k-labels-idx1-ubyte \
  --examples-per-class 200 \
  --test-limit 1000 \
  --seed 42
```

MNIST valid hemisphere-summary probe:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/mnist_temporal_probe_hemisummary_200_1000.sonata.json \
  --train-images /home/dean/repos/data/MNIST/raw/train-images-idx3-ubyte \
  --train-labels /home/dean/repos/data/MNIST/raw/train-labels-idx1-ubyte \
  --test-images /home/dean/repos/data/MNIST/raw/t10k-images-idx3-ubyte \
  --test-labels /home/dean/repos/data/MNIST/raw/t10k-labels-idx1-ubyte \
  --examples-per-class 200 \
  --test-limit 1000 \
  --seed 42
```

EMNIST valid fixation probe:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/emnist_temporal_probe_500_1000.sonata.json \
  --train-images /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-train-images-idx3-ubyte \
  --train-labels /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-train-labels-idx1-ubyte \
  --test-images /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-test-images-idx3-ubyte \
  --test-labels /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-test-labels-idx1-ubyte \
  --examples-per-class 500 \
  --test-limit 1000 \
  --seed 42
```

EMNIST valid hemisphere-summary probe:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/emnist_temporal_probe_hemisummary_500_1000.sonata.json \
  --train-images /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-train-images-idx3-ubyte \
  --train-labels /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-train-labels-idx1-ubyte \
  --test-images /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-test-images-idx3-ubyte \
  --test-labels /home/dean/repos/SNNFrame/data/EMNIST/emnist-letters-test-labels-idx1-ubyte \
  --examples-per-class 500 \
  --test-limit 1000 \
  --seed 42
```

CIFAR bounded corrected runs:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/cifar10_temporal_probe_100_500.sonata.json \
  --train-images /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin \
  --train-labels /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin \
  --test-images /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin \
  --test-labels /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin \
  --examples-per-class 100 \
  --test-limit 500 \
  --seed 42
```

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
./build_nolto/experiments/retina_classification \
  --config build_nolto/jepa_configs/cifar10_temporal_probe_hemisummary_100_500.sonata.json \
  --train-images /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin \
  --train-labels /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin \
  --test-images /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin \
  --test-labels /home/dean/repos/SNNFrame/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin \
  --examples-per-class 100 \
  --test-limit 500 \
  --seed 42
```

## Short Summary

The project is past the “JEPA scaffolding only” stage.
The temporal JEPA path now produces honest probe results on MNIST, EMNIST, and CIFAR bounded slices.
The latest major result is that the higher-capacity MNIST temporal-fixation config reached `84.90%`, which is the best JEPA probe result so far and much closer to the `93.30%` baseline than earlier corrected runs.
The newest follow-up result is that the same capacity transfer also lifted MNIST temporal hemisphere summary to `84.80%` and EMNIST to about `61.5%` on both temporal modes.
The CIFAR follow-up is weaker: the same transfer only lifted CIFAR to about `18%`.

The next conversation should assume:

- JEPA plumbing is real
- temporal validity checks matter
- CIFAR bounded temporal comparison is now captured cleanly
- the current best JEPA result is the higher-capacity MNIST temporal-fixation run at `84.90%`
- MNIST temporal hemisphere summary is now effectively tied with that result at `84.80%`
- EMNIST now reaches about `61.5%` with the same higher-capacity settings
- CIFAR only improves modestly to about `18%` with the same higher-capacity settings
- a first richer CIFAR probe-embedding trial was neutral-to-negative and was reverted
- temporal JEPA now correctly supports real promoted-stage CIFAR surfaces, but both the figure-ground-mask and recurrent sensory-state boundary smokes were clearly worse
- the new representation audit shows CIFAR direct raw/promoted surfaces are better than the trained JEPA embedding, and raw vs promoted are currently identical on the default CIFAR temporal config
- preserving a wider direct residual inside the JEPA embedding improved the CIFAR audit smoke from `13.50%` to `17.00%`
- the same wider direct residual improved both bounded CIFAR temporal modes from about `18%` to `22.80%`
- the current MLP probe path is not competitive yet
- the next useful work is shifting from config scaling to CIFAR-specific representation improvements
