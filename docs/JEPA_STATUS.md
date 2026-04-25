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

Real promoted-surface audit on CIFAR:

- root cause follow-up: the bounded CIFAR configs were naming `jepa_tap_surface=promoted_stage1` without enabling any actual promotion stage, so `promoted_stage1` aliased `raw_stage1`
- code path now emits a runtime note when that alias happens, and the bounded CIFAR configs have been switched to explicit `raw_stage1` so the active path matches reality
- two actual promoted surfaces were then tested on the same bounded CIFAR temporal-fixation smoke gate:
  - budget: `10/class`, `100` test images
  - dataset: CIFAR-10 binary

Topographic promoted surface:

- run: existing bounded fixation config plus `--hemisphere-topographic-stage-enabled`
- baseline classifier: `23.00% (23/100)`
- JEPA probe: `21.00% (21/100)`
- audit artifact: [build/cifar10_representation_audit_topographic_smoke_10_100.json](/home/dean/repos/SNNJepa/build/cifar10_representation_audit_topographic_smoke_10_100.json)
- promoted direct audit:
  - raw direct: `test_nn = 16.00%`, `dim = 66024`
  - promoted direct: `test_nn = 16.00%`, `dim = 13100`
  - JEPA embedding: `test_nn = 15.00%`

Convergent promoted surface:

- run: existing bounded fixation config plus `--hemisphere-convergent-code-enabled`
- baseline classifier: `23.00% (23/100)`
- JEPA probe: `13.00% (13/100)`
- audit artifact: [build/cifar10_representation_audit_convergent_smoke_10_100.json](/home/dean/repos/SNNJepa/build/cifar10_representation_audit_convergent_smoke_10_100.json)
- promoted direct audit:
  - raw direct: `test_nn = 16.00%`, `dim = 66024`
  - promoted direct: `test_nn = 19.00%`, `dim = 540`
  - JEPA embedding: `test_nn = 17.00%`

Interpretation:

- both topographic and convergent stages are real promotions; they materially change the direct vector dimensionality and no longer alias raw
- topographic did not improve the direct audit over raw and did not improve end-to-end CIFAR accuracy
- convergent slightly improved direct nearest-neighbor separability on the smoke audit, but it hurt the actual CIFAR JEPA probe badly (`13%`), so it is not a better current path
- the correct current CIFAR baseline remains the improved raw-stage path with the wide direct residual, and future JEPA work should continue from explicit `raw_stage1` rather than pretending promotion is active

JEPA-specific probe readout correction:

- root cause: the JEPA probe had been inheriting the main classifier settings, so CIFAR JEPA evaluation was using the baseline retina classifier readout (`weighted_distance`, `k=9`) instead of the readout that matched the representation audit (`1-NN cosine`)
- code change: JEPA probe readout is now configured independently via:
  - `jepa_probe_classifier`
  - `jepa_probe_k`
  - `jepa_probe_exponent`
  - CLI overrides: `--jepa-probe-classifier`, `--jepa-probe-k`, `--jepa-probe-exponent`
- new JEPA defaults:
  - `probe_classifier = majority`
  - `probe_k = 1`
  - `probe_exponent = 1.0`
- the bounded CIFAR temporal configs now make that explicit in config rather than relying on inherited defaults

Readout validation:

- smoke gate: CIFAR temporal fixation, `10/class`, `100` test images
- inherited old readout (`weighted_distance`, `k=9`): `15.00%`
- corrected JEPA default (`majority`, `k=1`): `17.00%`
- optional `weighted_similarity`, `k=1`: also `17.00%`
- control rerun at `20/class`, `100` test images:
  - baseline classifier: `23.00%`
  - JEPA probe with corrected `k=1` readout: `23.00%`
  - this confirmed the earlier mismatch was real: the inherited `k=9` JEPA probe was washing out local embedding signal

Bounded CIFAR reruns with corrected JEPA probe:

- temporal fixation:
  - baseline: `32.80% (164/500)`
  - JEPA probe: `19.80% (99/500)`
  - delta: `-13.00 pts`
  - log: [build/cifar10_temporal_probe_100_500_probek1_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_probek1_rerun.log)
- temporal hemisphere summary:
  - baseline: `32.80% (164/500)`
  - JEPA probe: `20.00% (100/500)`
  - delta: `-12.80 pts`
  - log: [build/cifar10_temporal_probe_hemisummary_100_500_probek1_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_probek1_rerun.log)

Interpretation:

- the corrected JEPA readout is the right evaluation path and should be kept
- it improves CIFAR JEPA over the inherited `k=9` probe, but not enough to beat the earlier best bounded line (`22.80%`)
- fixation and hemisphere-summary remain nearly tied under the corrected readout
- the evaluation mismatch is now fixed; the remaining CIFAR problem is back to the embedding itself, not the probe classifier
- follow-up drift check: the earlier verbal comparison against a `30.00%` CIFAR baseline was stale; the archived `build/cifar10_temporal_probe_100_500_widedirect_rerun.log` baseline is actually `30.40% (152/500)`
- reproducibility check: rerunning the current raw-stage CIFAR smoke gate twice with the same explicit command line and `--seed 42` reproduced the same metrics both times: baseline `32.00% (32/100)`, JEPA probe `22.00% (22/100)`
- conclusion: the recent CIFAR baseline movement is not explained by obvious run-to-run randomness in the current seeded smoke path; treat it as a real setup/codepath difference between archived runs, not noise

Archived-semantics readout recheck:

- question: after all later fixes, is the CIFAR JEPA probe actually better with the older archived readout semantics than with the newer `majority, k=1` probe default?
- smoke rerun on the current explicit `raw_stage1` config:
  - command override: `--jepa-probe-classifier weighted_distance --jepa-probe-k 9 --jepa-probe-exponent 1.0`
  - baseline: `32.00% (32/100)`
  - JEPA probe: `24.00% (24/100)`
  - comparison: this beats the current `majority, k=1` smoke result of `22.00% (22/100)` on the same seed and slice
- bounded reruns on both CIFAR temporal modes with the same archived JEPA probe semantics:
  - temporal fixation: baseline `30.40% (152/500)`, JEPA probe `23.40% (117/500)`, delta `-7.00 pts`
    - log snapshot: [build/cifar10_temporal_probe_100_500_archived_probe_semantics_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_100_500_archived_probe_semantics_rerun.log)
  - temporal hemisphere summary: baseline `30.40% (152/500)`, JEPA probe `23.40% (117/500)`, delta `-7.00 pts`
    - log snapshot: [build/cifar10_temporal_probe_hemisummary_100_500_archived_probe_semantics_rerun.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_100_500_archived_probe_semantics_rerun.log)

Interpretation:

- the blanket assumption that `majority, k=1` is the best JEPA probe readout for CIFAR was wrong
- on the current wide-direct CIFAR embedding, the archived `weighted_distance, k=9` JEPA probe is measurably better
- the bounded CIFAR config files have therefore been switched back to explicit `weighted_distance, k=9` JEPA probe settings
- the next CIFAR work should continue from this restored probe path, not the weaker `k=1` probe baseline

Temporal-history summary MLP trial:

- goal: align the trainer more closely with JEPA-style temporal prediction by removing branch masking for temporal CIFAR, using full fixation-history context, predicting future pooled hemisphere summaries, and replacing the linear JEPA branch with a small normalized MLP plus covariance regularization
- implementation notes:
  - temporal training examples used accumulated fixation history as context and pooled future hemisphere summaries as targets
  - CIFAR configs were switched to `temporal_hemisphere_summary` with `jepa_mask_mode=none`
  - the trainer used a small nonlinear JEPA branch internally, but the trial was reverted after evaluation because it regressed
- smoke result (`20/class`, `200` test):
  - baseline: `32.00% (64/200)`
  - JEPA probe: `16.50% (33/200)`
  - log: [build/cifar10_temporal_probe_historysummary_mlp_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_historysummary_mlp_smoke_20_200.log)
- bounded reruns:
  - temporal fixation: baseline `30.40% (152/500)`, JEPA probe `19.00% (95/500)`, delta `-11.40 pts`
    - log: [build/cifar10_temporal_probe_historysummary_mlp_100_500.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_historysummary_mlp_100_500.log)
  - temporal hemisphere summary: baseline `30.40% (152/500)`, JEPA probe `19.00% (95/500)`, delta `-11.40 pts`
    - log: [build/cifar10_temporal_probe_hemisummary_historysummary_mlp_100_500.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_hemisummary_historysummary_mlp_100_500.log)

Interpretation:

- the trainer looked healthier numerically (`loss << shuffled_loss`, improved latent variance), but that did not transfer to probe accuracy
- this was not a viable improvement over the restored bounded CIFAR line of `23.40%`
- the code/config path was reverted after evaluation, so the repo remains on the best known CIFAR setup rather than this losing trial

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

1. Make the next milestone a temporal spike-code representation audit.
   The immediate goal is to measure whether the temporal spiking code itself is getting better before another JEPA trainer sweep. The audit should cover first-spike/rank-order evidence, temporally discounted traces, fixation-to-fixation stability, branch/subregion separability, and hemisphere separability from temporal spike dynamics.

2. Stop doing straight capacity-copy transfers for CIFAR.
   The `32d/20ep` change only improved CIFAR from about `16%` to about `18%`, which is too small to justify more blind config scaling.

3. Keep a short table of the valid temporal JEPA modes in one place.
   Capture a short table for:
   - MNIST `temporal_fixation`
   - MNIST `temporal_hemisphere_summary`
   - EMNIST `temporal_fixation`
   - EMNIST `temporal_hemisphere_summary`
   - CIFAR `temporal_fixation`
   - CIFAR `temporal_hemisphere_summary`

4. Stop treating the MLP probe as a likely win until it is debugged.
   The current result is a regression, not an improvement.

5. Representation-side improvements to consider next:
   - for CIFAR specifically, stop running new promoted-boundary smokes until the JEPA encoder stops underperforming the direct surfaces
   - investigate why `promoted_stage1` is identical to `raw_stage1` on the current CIFAR temporal config
   - the new wide direct residual improved bounded CIFAR from `13.50%` to `17.00%`, so confirm that gain on the `100/500` gate before changing the trainer objective again
   - improve the JEPA objective / encoder so the learned embedding preserves more of the direct-surface class signal already present in raw and recurrent fixations
   - only after that should richer predictor capacity or extra positional conditioning be revisited

6. Once the temporal spike-code audit shows a better code, plug JEPA-style prediction into that code as a learning signal.
   JEPA should provide prediction-error pressure over temporal spike traces and future temporal summaries. It should gate local plasticity or eligibility updates; it should not become the next standalone trainer sweep unless the audit shows the underlying temporal code is worth predicting.

7. Only after the temporal code and probe become competitive should JEPA be tested inside the main bilateral decision path.

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

## 2026-04-23 Stage 1-5 Pre-Trainer Gate

Stage 1 through Stage 5 are now complete for the CIFAR redesign path.

Artifact:

- [build/cifar10_stage15_audit_1_10.json](/home/dean/repos/SNNJepa/build/cifar10_stage15_audit_1_10.json)

Locked choices:

- tokenization: `branch-by-subregion`, equal contiguous subregions per branch
- concrete Stage 1 split: `4` subregions per branch, giving `12` tokens per hemisphere view on the current CIFAR stage-1 surface
- chosen target surface: `Future Hemisphere Summary Target`
- chosen context definition: past-half multifixation structured summaries from both hemispheres
- chosen future probe embedding: predicted future hemisphere-summary embedding

Smoke-gate evidence from the artifact:

- raw direct `test_nn = 10.0%`
- future hemisphere summary target `test_nn = 20.0%`
- recurrent fixation direct `test_nn = 20.0%`
- recurrent future sensory summary target `test_nn = 10.0%`
- test same-image context/target cosine `0.9688`
- test shuffled context/target cosine `0.8561`
- test context agreement margin `+0.1127`

Pre-trainer review decision:

- `go: Stage1=pass, Stage2=pass, Stage3=pass, Stage4=pass`

Interpretation:

- the new structured tokenization is real, inspectable, and no longer aliases whole-branch tokens
- the first abstract target worth predicting is the future pooled hemisphere summary, not the recurrent future summary
- the chosen multifixation context carries materially more information about the chosen target than shuffled context
- trainer redesign work is now justified, but it should target the locked Stage 0-4 problem definition rather than the old raw-branch temporal JEPA path

## 2026-04-23 Stage 6 Initial Trainer Redesign

Implemented:

- temporal JEPA examples are now built as one structured multifixation context per source image rather than per-hemisphere next-fixation pairs
- the temporal training target is now the locked future summary path rather than the old raw hidden-branch target
- the trainer path now uses a nonlinear `tanh` context encoder / predictor / target encoder stack
- the exposed JEPA embedding is now the predictor output for the future summary path, matching the Stage 4 probe lock

Files:

- [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)
- [include/snnfw/jepa/JepaTrainer.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.h)

Initial smoke result on CIFAR fixation config (`5/class`, `20` test):

- baseline classifier: `35.00% (7/20)`
- first Stage 6 smoke: JEPA probe `5.00% (1/20)` with over-normalized structured summaries
- second Stage 6 smoke after removing summary normalization: JEPA probe `15.00% (3/20)`

Current diagnosis:

- the Stage 6 semantics are now much closer to the locked design, but the trainer is still collapsing too hard
- trainer dump summary still shows near-zero context / target / prediction variance at this smoke scale
- this means the remaining Stage 6 problem is now concentrated in latent-scale and anti-collapse behavior, not in the old `x`/`y` definition

Interpretation:

- Stage 6 implementation is in place
- Stage 6 validation is not yet a win
- the next corrective work should focus on collapse control and latent scaling inside the new future-summary trainer, not on changing the locked target/context contract again

## 2026-04-23 Successor-Summary VICReg-JEPA First Pass

Implemented:

- temporal JEPA now builds a discounted successor target in the locked future-summary space instead of predicting only one pooled future summary
- the temporal trainer now applies variance pressure to both context and prediction latents
- the temporal trainer now applies explicit covariance penalties to both context and prediction latents
- trainer/config plumbing now exposes:
  - `jepa_covariance_penalty`
  - `jepa_successor_discount`

Files:

- [include/snnfw/jepa/JepaConfig.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaConfig.h)
- [include/snnfw/jepa/JepaTrainer.h](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.h)
- [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)
- [experiments/retina_classification.cpp](/home/dean/repos/SNNJepa/experiments/retina_classification.cpp)

Smoke result on CIFAR fixation config (`20/class`, `200` test):

- baseline classifier: `31.00% (62/200)`
- JEPA probe: `15.50% (31/200)`
- log: [build/cifar10_temporal_probe_successor_vicreg_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_successor_vicreg_smoke_20_200.log)
- trainer dump: [build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json)

Trainer diagnosis:

- `context_variance = 2.73e-06`
- `prediction_variance = 6.56e-07`
- `target_variance = 2.94e-06`
- `mean_covariance_penalty = 3.09e-13`
- `mean_loss = 0.00173`
- `mean_shuffled_loss = 0.00179`

Interpretation:

- the successor target and VICReg-style penalties are wired correctly and visible in the trainer dump
- but the default `32d`, `20 epoch` CIFAR setting still collapses almost completely
- this first pass is therefore not a performance win; it is a structural implementation milestone plus a sharper diagnosis
- the next work on this branch should focus on stronger latent scale / capacity changes within the successor-summary path, not on going back to raw next-fixation JEPA

## 2026-04-23 Batch-Normalized Alignment Fix

Implemented:

- temporal JEPA alignment loss now operates on batch-standardized prediction and target latents
- the target encoder is no longer updated from the context encoder during temporal JEPA training
- temporal JEPA now uses a reduced effective learning rate internally to stabilize the standardized objective

Files:

- [include/snnfw/jepa/JepaTrainer.cpp](/home/dean/repos/SNNJepa/include/snnfw/jepa/JepaTrainer.cpp)

Smoke result on CIFAR fixation config (`20/class`, `200` test):

- baseline classifier: `31.00% (62/200)`
- JEPA probe: `16.50% (33/200)`
- log: [build/cifar10_temporal_probe_successor_vicreg_normfix_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_successor_vicreg_normfix_smoke_20_200.log)
- trainer dump: [build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json)

Trainer diagnosis:

- aligned loss improved relative to shuffled loss:
  - `mean_loss = 0.4969`
  - `mean_shuffled_loss = 1.0237`
- latent variance still remains effectively collapsed at this smoke scale:
  - `context_var = 0.0000`
  - `pred_var = 0.0000`
  - `target_var = 0.0000`

Interpretation:

- this is the first corrected temporal JEPA run on CIFAR where shuffled targets are clearly worse than aligned targets
- so the training objective is now doing something meaningfully different from the earlier collapse regime
- but the representation is still too weak to improve probe accuracy yet
- this moves the branch from “fully degenerate collapse” to “weak but non-degenerate optimization,” which is progress, but not yet a usable CIFAR result

## 2026-04-23 Norm-Fix Capacity Follow-Up

Implemented:

- created a separate CIFAR fixation config with the corrected normalized-loss trainer and larger latent scale
- tested the corrected objective at `projection_dim=128`, `epochs=40`, `lr=0.005`, `covariance_penalty=0.1`

Files:

- [build_nolto/jepa_configs/cifar10_temporal_probe_100_500_normfix_capacity.sonata.json](/home/dean/repos/SNNJepa/build_nolto/jepa_configs/cifar10_temporal_probe_100_500_normfix_capacity.sonata.json)

Smoke result on CIFAR fixation config (`20/class`, `200` test):

- baseline classifier remained on the same `31.00%` smoke line
- JEPA probe: `17.00% (34/200)`
- log: [build/cifar10_temporal_probe_successor_vicreg_normfix_capacity_smoke_20_200.log](/home/dean/repos/SNNJepa/build/cifar10_temporal_probe_successor_vicreg_normfix_capacity_smoke_20_200.log)
- trainer dump: [build/jepa_minimal_trainer_cifar10_temporal_probe_100_500_normfix_capacity.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_100_500_normfix_capacity.json)

Trainer diagnosis:

- loss separation survived but weakened relative to the `32d` normalized-loss run:
  - `mean_loss = 0.8263`
  - `mean_shuffled_loss = 1.0001`
- printed latent variances still rounded to zero:
  - `context_var = 0.0000`
  - `pred_var = 0.0000`
  - `target_var = 0.0000`

Interpretation:

- increasing capacity after the normalization fix did not unlock a strong CIFAR improvement
- it slightly improved over the earlier degenerate successor/VICReg smoke line, but it did not beat the smaller normalized-loss run decisively
- the branch is now better behaved than before, but still not strong enough to compete with the best raw-residual CIFAR path

## 2026-04-23 Temporal LayerNorm Follow-Up

Implemented:

- added per-sample layer normalization on temporal future-summary pre-activations before `tanh`
- applied that normalization in:
  - temporal trainer forward passes
  - temporal trainer evaluation summaries
  - the aligned temporal `encodeSample()` path

Smoke results on CIFAR fixation config (`20/class`, `200` test):

- base normalized-loss config:
  - baseline: `32.00% (64/200)`
  - JEPA probe: `13.50% (27/200)`
- higher-capacity normalized-loss config:
  - baseline: `32.00% (64/200)`
  - JEPA probe: `14.00% (28/200)`

Trainer diagnosis:

- the small model finally showed materially higher latent spread:
  - `context_variance = 5.90e-05`
  - `prediction_variance = 3.73e-06`
  - `target_variance = 1.32e-04`
  - `mean_loss = 1.1445`
  - `mean_shuffled_loss = 1.3831`
- the larger model improved target variance but still kept prediction variance too small:
  - `context_variance = 1.88e-05`
  - `prediction_variance = 4.01e-07`
  - `target_variance = 7.20e-05`
  - `mean_loss = 1.2656`
  - `mean_shuffled_loss = 1.2940`

Interpretation:

- this change did fix the latent-scale problem more directly than the earlier variance-floor tweak
- but higher latent spread by itself did not improve CIFAR probe accuracy on this branch
- the successor-summary branch is now less collapsed numerically, yet still not producing a useful class geometry

## 2026-04-23 Temporal Probe Output Normalization Check

Implemented:

- unit-normalized the aligned future-summary prediction returned by temporal `encodeSample()`

Smoke result on the base normalized-loss CIFAR fixation config (`20/class`, `200` test):

- baseline: `32.00% (64/200)`
- JEPA probe: `13.50% (27/200)`

Interpretation:

- normalizing the probe output did not change the CIFAR result at all
- this suggests the current failure is not a simple probe-scale mismatch on the aligned temporal branch

## 2026-04-24 Temporal Spike-Code Representation Audit

Implemented:

- added a Stage 6 temporal spike-code audit section to the existing representation-audit JSON
- added CLI/config aliases:
  - `--temporal-spike-code-audit-enabled`
  - `--temporal-spike-code-audit-sample-limit <n>`
  - `--temporal-spike-code-audit-output-path <path>`
  - config keys: `temporal_spike_code_audit_enabled`, `temporal_spike_code_audit_sample_limit`, `temporal_spike_code_audit_output_path`
- temporal-audit-only runs now suppress JEPA trainer/probe execution unless the broader JEPA representation audit is also explicitly enabled
- the audit currently evaluates four temporal code surfaces:
  - discounted temporal trace
  - first-spike / rank-order code
  - consecutive-fixation delta code
  - projected onset + sustained composite code

Initial smoke result:

- command shape: CIFAR temporal config, `1/class`, `10` test, `--temporal-spike-code-audit-enabled`
- output: [build/temporal_spike_code_audit_smoke_1_10.json](/home/dean/repos/SNNJepa/build/temporal_spike_code_audit_smoke_1_10.json)
- trainer/probe: disabled in temporal-audit-only mode
- Stage 6 decision: `hold`
- best temporal code: first-spike / rank-order
- best temporal `test_nn`: `20.00%`
- direct reference `test_nn`: `20.00%`

Follow-up smoke after improving the temporal code:

- added a composite onset-sustained temporal spike code that combines normalized first-spike latency, discounted trace, consecutive delta, and peak-response channels per hemisphere
- added deterministic signed convergence into a compact downstream temporal population (`65,536` dimensions total) so the useful temporal geometry is not tied to a full `528,192`-dimension concatenation
- output: [build/temporal_spike_code_audit_smoke_1_10.json](/home/dean/repos/SNNJepa/build/temporal_spike_code_audit_smoke_1_10.json)
- trainer/probe: disabled in temporal-audit-only mode
- Stage 6 decision: `go`
- best temporal code: projected onset-sustained composite
- best temporal `test_nn`: `30.00%`
- direct reference `test_nn`: `20.00%`
- projected onset-sustained `test_centroid`: `30.00%`

Interpretation:

- the audit machinery now clears the direct-surface gate only after the temporal code adds useful event timing and sustained-response structure
- the next implementation step should scale this audit to the bounded CIFAR slice, keep the compact onset-sustained code as the candidate temporal representation, and only then add JEPA-style prediction error as a learning signal on that code

## 2026-04-24 JEPA Prediction On Temporal Spike Code

Implemented:

- added a JEPA target mode: `temporal_spike_code`
- added CLI/config plumbing:
  - `--jepa-target-mode temporal_spike_code`
  - `--jepa-temporal-spike-code-dim <n>`
  - config key: `jepa_temporal_spike_code_dim`
- the temporal JEPA trainer can now use:
  - context: past-half projected onset-sustained temporal spike code
  - target: future-half projected onset-sustained temporal spike code
- the JEPA probe now encodes test samples through the same temporal spike-code context path when the trained model is aligned to this mode

Smoke result:

- command shape: CIFAR temporal config, `1/class`, `10` test, `--jepa-target-mode temporal_spike_code`, `--jepa-temporal-spike-code-dim 32768`
- trainer dump: [build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json](/home/dean/repos/SNNJepa/build/jepa_minimal_trainer_cifar10_temporal_probe_100_500.json)
- trainer target mode: `temporal_spike_code`
- context input dim: `65,536`
- target input dim: `65,536`
- temporal examples: `10`
- mean loss: `1.7966`
- shuffled loss: `2.7906`
- context variance: `0.00125`
- prediction variance: `0.00092`
- target variance: `0.00643`
- baseline classifier: `20.00% (2/10)`
- JEPA probe: `20.00% (2/10)`

Interpretation:

- JEPA-style temporal prediction is now plugged into the audited spike code rather than the older future-summary path
- the smoke is not an accuracy win, but it is structurally healthier than the previous collapsed temporal-summary branch because aligned prediction is materially better than shuffled prediction
- the next implementation step should tune the temporal spike-code predictor on a moderate CIFAR slice, while keeping the representation audit gate in place
- the current CIFAR adapter path still reports `encoding=rate`, so the broader no-rate-coding objective still requires moving the upstream retina adapters to true temporal spike timing

## 2026-04-24 Temporal Spike-Code Probe Separation

Implemented:

- added a direct temporal spike-code probe path behind `--temporal-spike-code-probe-enabled`
- added CLI overrides for faster controlled JEPA experiments:
  - `--jepa-trainer-disabled`
  - `--jepa-probe-disabled`
  - `--jepa-projection-dim <n>`
  - `--jepa-trainer-epochs <n>`
  - `--jepa-trainer-learning-rate <v>`
  - `--jepa-variance-floor <v>`
  - `--jepa-variance-penalty <v>`
- the direct probe evaluates the projected onset-sustained temporal spike code as a representation surface, separate from the JEPA predicted embedding

Moderate CIFAR smoke:

- command shape: CIFAR temporal config, `5/class`, `50` test, `--temporal-spike-code-probe-enabled`, JEPA trainer/probe disabled
- baseline classifier: `32.00% (16/50)`
- direct temporal spike-code probe: `24.00% (12/50)`
- delta vs baseline: `-8.00 pts`

JEPA temporal-spike predictor capacity smoke:

- command shape: same slice, `--jepa-target-mode temporal_spike_code`, `--jepa-projection-dim 64`, `--jepa-trainer-epochs 30`, `--jepa-trainer-learning-rate 0.01`
- baseline classifier: `32.00% (16/50)`
- JEPA predicted temporal-spike probe: `16.00% (8/50)`
- trainer loss: `2.5092`
- shuffled loss: `2.6306`
- prediction variance: `0.0007`

Interpretation:

- direct temporal spike code is currently stronger than the JEPA-predicted embedding on the same slice
- neither is ready to replace the main bilateral classifier
- the useful direction is not more standalone JEPA probe sweeps; the next implementation should use temporal spike-code prediction error as a weighting/plasticity signal while keeping the direct temporal code available as a diagnostic and possible auxiliary vote

## 2026-04-24 Temporal Spike Prediction-Error Feedback

Implemented:

- added a gated auxiliary feedback path:
  - `--temporal-spike-prediction-error-feedback-enabled`
  - `--temporal-spike-prediction-error-feedback-gain <v>`
  - `--temporal-spike-prediction-error-feedback-threshold <v>`
  - `--temporal-spike-prediction-error-feedback-max-penalty <v>`
  - `--temporal-spike-prediction-error-plasticity-gain <v>`
- the JEPA model now exposes temporal spike-code prediction error for a sample
- when enabled, the main classifier still makes the decision; temporal spike prediction error only:
  - optionally down-weights the currently selected class confidence when prediction error is high
  - contributes to plasticity and replay priority through the existing decision context
- the JEPA probe can stay disabled while this auxiliary signal is active

Smoke results:

- slice: CIFAR temporal config, `5/class`, `50` test
- prior baseline on same slice without temporal feedback: `32.00% (16/50)`
- prior standalone JEPA predicted temporal-spike probe: `16.00% (8/50)`
- feedback mode, default confidence penalty:
  - baseline path with auxiliary feedback: `28.00% (14/50)`
  - trainer loss: `2.4895`
  - shuffled loss: `2.6218`
  - prediction variance: `0.0003`
- feedback mode, plasticity-only (`max_penalty=0`, `plasticity_gain=0.25`):
  - baseline path with auxiliary feedback: `28.00% (14/50)`

Interpretation:

- the requested architecture is now in place: temporal spike-code prediction error is an auxiliary weighting/plasticity signal, not a replacement classifier
- current temporal predictor quality is still too weak; on the moderate smoke it hurts online correction instead of improving it
- keep this path gated off by default until the temporal predictor has stronger loss separation and prediction variance
- calibration diagnostics are now implemented in the main evaluation summary:
  - they record temporal spike prediction error even when feedback is disabled
  - they report mean error for correct vs incorrect classifier decisions
  - they run a threshold sweep for "high temporal prediction error predicts classifier error"
  - when feedback is enabled, they also report decision changes, corrections, and regressions

Calibration smoke:

- slice: CIFAR temporal config, `5/class`, `50` test, temporal-spike JEPA model present, feedback disabled
- classifier accuracy remained `32.00% (16/50)`
- temporal prediction error:
  - samples: `50`
  - mean error: `0.3637`
  - correct-decision mean error: `0.3561`
  - incorrect-decision mean error: `0.3659`
  - mean cosine: `0.2725`
  - best diagnostic threshold: `0.3440`
  - balanced error-detection accuracy: `65.62%`
  - wrong precision: `84.09%`
  - wrong recall: `94.87%`
  - flagged: `44/50`
- same slice with feedback enabled:
  - classifier accuracy: `28.00% (14/50)`
  - feedback decision changes: `1`
  - corrections: `0`
  - regressions: `0`

Interpretation:

- the temporal prediction error has a small directionally useful separation, but it is badly calibrated: the best threshold flags almost the whole slice
- that explains why confidence penaltying and plasticity weighting currently hurt online correction
- next useful step is to improve the temporal spike-code predictor/error calibration itself, or restrict this signal to replay prioritization after it shows a larger correct-vs-incorrect gap

## 2026-04-24 Temporal Spike-Code Predictor Target Fix

Implemented:

- changed the temporal spike-code trainer target from an untrained random target-encoder projection of the future spike code to a deterministic norm-preserving projection of the future temporal spike code
- the context path now applies the same deterministic projection when encoding high-dimensional temporal spike code into the trainer input space
- `encodeSample()` and temporal prediction-error evaluation both use the same projection path, so training/evaluation no longer disagree on the temporal-spike-code input dimensionality

Smoke results:

- slice: CIFAR temporal config, `5/class`, `50` test, feedback disabled
- default `32d`, `20` epochs:
  - prior trainer: loss `2.4895`, shuffled loss `2.6218`, prediction variance `0.0003`
  - after target fix: loss `1.1747`, shuffled loss `2.7862`, prediction variance `0.0209`
  - classifier accuracy remained `32.00% (16/50)`
  - temporal prediction error calibration still weak:
    - mean error `0.3859`
    - correct error `0.3846`
    - incorrect error `0.3862`
    - best threshold `0.3527`, balanced accuracy `57.81%`, flagged `47/50`
- higher-capacity stable predictor (`projection_dim=128`, `epochs=40`, `lr=0.005`):
  - loss `1.1057`
  - shuffled loss `2.8299`
  - prediction variance `0.0214`
  - mean prediction error `0.2185`
  - mean cosine `0.5630`
  - best threshold `0.2264`, wrong precision `90.00%`, wrong recall `23.08%`, flagged `10/50`
  - classifier accuracy stayed `32.00% (16/50)`
  - standalone JEPA probe remained poor at `10.00% (5/50)`

Interpretation:

- this is a real predictor fix: the model now predicts a stable future temporal spike-code target, with much stronger loss separation and non-collapsed prediction variance
- the improved predictor still does not produce class geometry good enough to replace the classifier
- the error signal is now more selective at higher capacity, but it still does not separate correct and incorrect classifier decisions by mean error
- keep using temporal prediction error only as a gated auxiliary/plasticity candidate; the next useful improvement is to make the error signal conditional on classifier uncertainty or replay-only, rather than applying it broadly

## 2026-04-25 Temporal Spike-Code Predictor Rich-Target Pass

Implemented:

- added `--temporal-spike-code-probe-type <type>` so the direct temporal probe can compare temporal code families without recompiling
- added a projected rich onset-sustained temporal code:
  - first-spike latency/rank channel
  - discounted trace channel
  - consecutive delta channel
  - peak-response channel
  - temporal variance channel
  - first-to-last adaptation channel
- changed the JEPA temporal-spike-code predictor target/context path to use the same richer projected temporal code
- kept this path as predictor/error infrastructure only; it is still not promoted as a replacement classifier

Direct temporal-code checks on CIFAR `10/class`, `100` test, seed `42`, JEPA trainer/probe disabled:

- baseline classifier: `32.00% (32/100)`
- prior projected onset-sustained direct temporal probe: `17.00% (17/100)`
- projected rich onset-sustained direct temporal probe: `21.00% (21/100)`
- discounted trace direct temporal probe: `21.00% (21/100)`
- first-spike-rank direct temporal probe: `10.00% (10/100)`
- audit artifact: [build_nolto/cifar10_temporal_spike_code_audit_10_100.json](/home/dean/repos/SNNJepa/build_nolto/cifar10_temporal_spike_code_audit_10_100.json)
- temporal audit gate: `hold`, best temporal nearest-neighbor `21.00%`, reference direct nearest-neighbor `22.00%`
- audit note: discounted trace centroid reached `33.00%`, but the current gate is nearest-neighbor separability and still does not pass

Predictor wiring smoke:

- slice: CIFAR temporal config, `5/class`, `50` test, `--jepa-target-mode temporal_spike_code`, probe disabled, `projection_dim=64`, `epochs=5`
- trainer target examples now report `future_projected_rich_onset_sustained_spike_code`
- trainer artifact reports `temporal_spike_code_aligned: true`
- trainer loss: `1.1839`
- shuffled loss: `2.7987`
- prediction variance: `0.0183`
- classifier accuracy stayed `32.00% (16/50)`
- temporal prediction error was not useful yet: correct-error `0.3847`, incorrect-error `0.3818`

Interpretation:

- the richer temporal target is a measured improvement over the previous hard-coded projected code, but it still does not clear the direct-surface gate
- first-spike rank alone is currently too sparse/unstable for CIFAR on this pipeline
- the next representation work should improve the upstream temporal event code itself, especially replacing the current rate-coded retina adapter path with real latency/event timing before relying on JEPA prediction error

## 2026-04-24 Uncertainty-Gated Replay-Only Temporal Error

Implemented:

- added a replay-only temporal spike prediction-error path that does not change classifier confidence and does not scale immediate plasticity
- new CLI/config controls:
  - `--temporal-spike-prediction-error-replay-only-enabled`
  - `--temporal-spike-prediction-error-replay-gain <v>`
  - `--temporal-spike-prediction-error-replay-threshold <v>`
  - `--temporal-spike-prediction-error-replay-uncertainty-threshold <v>`
- the replay-only gate fires only when:
  - temporal spike prediction error is available
  - error is above the replay threshold
  - the classifier is uncertain or hemispheres disagree
- when it fires, it adds priority to replay only; it does not alter the current class choice
- diagnostics now report replay-gated samples, correct/incorrect split, and mean replay boost

Smoke results:

- slice: CIFAR temporal config, `5/class`, `50` test
- predictor: temporal spike code, `projection_dim=128`, `epochs=40`, `lr=0.005`
- replay-only threshold: `0.2264`
- uncertainty threshold: `0.35`
- gain `1.0`:
  - accuracy: `32.00% (16/50)`
  - replay-gated samples: `8`
  - gated correct: `0`
  - gated incorrect: `8`
  - mean replay boost: `0.2294`
- gain `3.0`:
  - accuracy: `32.00% (16/50)`
  - replay-gated samples: `8`
  - gated correct: `0`
  - gated incorrect: `8`
  - mean replay boost: `0.6882`
  - replays: `112`

Interpretation:

- this is the first temporal prediction-error use that cleanly targets mistakes on the smoke slice
- the signal is useful for selecting replay candidates, but current replay/consolidation mechanics do not convert that targeting into higher accuracy on this slice
- broad confidence penaltying and global plasticity remain rejected; replay-only targeting is the viable branch
- next useful work is to make replay of those gated mistakes stronger or more specific, for example by increasing correction repeats only for replay-gated mistakes or by pairing the temporal error gate with successful-consolidation replay

## 2026-04-24 CIFAR Training-Augmentation Smoke

Implemented:

- added a corpus disagreement arbiter diagnostic behind explicit CLI/config gates:
  - `--corpus-disagreement-arbiter-enabled`
  - `--corpus-disagreement-arbiter-min-margin <v>`
  - `--corpus-disagreement-arbiter-margin-delta <v>`
- the arbiter only acts when hemispheres disagree and one hemisphere has a stronger decision margin
- added correction/regression counters for that arbiter
- tested CIFAR training augmentation as a candidate accuracy path:
  - `training_augmentation_shift_px = 0.8`
  - `training_augmentation_rotation_deg = 4.0`
  - `training_augmentation_noise_std = 0.01`
  - variants swept at `1`, `3`, `5`, and `7`
- the checked-in CIFAR config is intentionally left without default augmentation because the gain did not survive the larger gate

Smoke results on CIFAR `5/class`, `50` test, seed `42`, JEPA trainer/probe disabled:

- prior baseline: `32.00% (16/50)`
- conservative arbiter (`min_margin=0.02`, `margin_delta=0.02`): no decision changes, `32.00% (16/50)`
- loose arbiter (`min_margin=0`, `margin_delta=0`): changed `7` decisions, found `1` correction, but final accuracy dropped to `26.00% (13/50)`
- 3-variant training augmentation preset: `34.00% (17/50)`
  - initial accuracy improved from `22.00%` to `30.00%`
  - post-correction accuracy improved from `32.00%` to `34.00%`
  - left/right stage-1 accuracy improved to `26.00%`/`28.00%`
- 5-variant training augmentation preset: `42.00% (21/50)`
  - initial accuracy improved to `32.00%`
  - post-correction accuracy improved to `42.00%`
  - left/right stage-1 accuracy landed at `24.00%`/`32.00%`
  - correction events dropped to `34`, corrected `5`, replays `97`
- 7-variant training augmentation preset: `30.00% (15/50)`
  - initial accuracy fell to `28.00%`
  - left stage-1 accuracy fell to `18.00%`
  - this suggests too many sampled variants start washing out the left-hemisphere code

Larger validation on CIFAR `10/class`, `100` test, seed `42`, JEPA trainer/probe disabled:

- no training augmentation: `32.00% (32/100)`
  - initial accuracy: `28.00%`
  - left/right stage-1 accuracy: `22.00%`/`21.00%`
- 1 variant: `28.00% (28/100)`
  - initial accuracy: `25.00%`
  - left/right stage-1 accuracy: `20.00%`/`16.00%`
- 3 variants: `29.00% (29/100)`
  - initial accuracy: `27.00%`
  - left/right stage-1 accuracy: `22.00%`/`18.00%`
- 5 variants: `29.00% (29/100)`
  - initial accuracy: `25.00%`
  - left/right stage-1 accuracy: `24.00%`/`17.00%`
- successful-consolidation replay without augmentation:
  - settings: `success_capacity=128`, `success_batch_size=2`, `success_repeats=2`
  - accuracy: `31.00% (31/100)`
  - initial accuracy: `27.00%`
  - replays increased from `211` to `276`

Interpretation:

- the disagreement-margin arbiter is useful as a diagnostic but should not be a default accuracy path
- the useful signal came from augmenting temporal/retinal training views on the tiny smoke only; it did not survive the larger `100`-sample gate
- leave training augmentation off by default for now
- successful-consolidation replay also stays off by default because it adds replay work without improving the larger gate
- the larger gate points back to improving the underlying temporal code and replay/plasticity mechanics rather than simply adding more augmented views
- this supports the next milestone direction: improve the spike-code representation itself, then let auxiliary prediction error shape plasticity around that stronger code

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
- Stage 1-5 of the CIFAR JEPA redesign are now locked around branch-by-subregion tokens, a future hemisphere-summary target, and a predictor-exposed future-summary probe
- the next useful work is not another JEPA trainer sweep
- the next milestone is a temporal spike-code representation audit
- once that audit shows a better temporal code, JEPA-style prediction should be plugged into that code as a prediction-error learning signal
