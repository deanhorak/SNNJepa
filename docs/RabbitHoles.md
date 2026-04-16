# RabbitHoles

This document records lines of work that consumed substantial time without producing a viable mainline result. The point is not to forbid future work. The point is to avoid repeating the same failed sequence without a new technical reason.

## Current Mainline To Protect

These are the benchmark paths worth keeping as the active reference surface:

- Unilateral Retina static: `86.17%`
- Bilateral Retina static: `87.29%`
- Bilateral Retina continuous: `87.50%` initial, `88.85%` post-correction
- CIFAR-10 bilateral natural-features experimental reference: `38.16%` on `1000/class, 5000` test

These live in:

- `configs/emnist_retina_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/emnist_retina_bilateral_continuous.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- `experiments/retina_classification.cpp`

Any new experimental vision path should be compared against these, not treated as a replacement by default.

## Rabbit Holes

### 1. Bolting Graph Runtime Into The Retina Benchmark Harness

What we tried:

- adding graph execution directly into `experiments/retina_classification.cpp`
- mixing mature Retina benchmarking with experimental instantiated-network execution

Why it was not productive:

- it coupled a stable benchmark harness to an immature research path
- it made it too easy to keep tuning graph behavior inside the wrong executable
- it obscured whether regressions came from the model or the harness

What to do instead:

- keep `emnist_retina_letters` as the stable bilateral Retina benchmark
- put graph-native experiments in separate executables or branches

Status:

- removed from the active branch surface

### 2. Treating The Multistage Graph Path As A Near-Term Replacement

What we tried:

- a separate multistage declarative cortical experiment
- stage1 -> association -> higher/fusion -> decision graph paths
- repeated config tuning to lift its accuracy

Why it was not productive:

- the path never became competitive with the working Retina baselines
- it stayed in the low single-digit / low double-digit range instead of approaching the high-80s baseline
- the representation was not class-separable enough upstream, so downstream tuning had very little value

What to do instead:

- keep this as a research track only
- require stage-wise separability evidence before spending more time on decision/readout tuning

### 3. Running A Laminar `L4 -> L2_3 -> L5` Config On The Wrong Runtime

What we tried:

- using a runtime path that effectively assumed `L4 -> L5`
- driving a more corticalized laminar config through it anyway

Why it was not productive:

- the runtime/config mismatch killed signal before it reached the intended downstream stage
- this looked like a model problem at first, but it was partly an execution-path mismatch

Lesson:

- before tuning a cortical config, verify that the runtime actually supports the declared laminar flow

### 4. Tuning The Decision Microcircuit Before Upstream Representations Were Separable

What we tried:

- decision nuclei
- excitatory/inhibitory settling
- winner fatigue
- homeostasis
- decision-stage recurrent tuning

Why it was not productive:

- the decision stage often received dead, weak, or poorly differentiated upstream drive
- the result was attractor collapse onto a few classes rather than improved recognition
- this was downstream tuning on top of an upstream representation problem

Lesson:

- do not tune decision dynamics until higher->readout and readout->decision separability has been measured and shown to be usable

### 5. Widening The Higher-Visual Bank Without Fixing Selectivity

What we tried:

- larger higher-visual banks
- richer higher-visual fan-in
- broader convergence into readout or decision

Why it was not productive:

- extra width increased complexity and search space without improving class separation
- broader fan-in often increased collapse, not discrimination
- stronger throughput into decision just amplified weak or wrong structure

Lesson:

- do not widen higher visual blindly
- fix selectivity before adding capacity

### 6. Stronger Feedforward / Stronger Readout Dynamics As A Substitute For Better Representation

What we tried:

- stronger `higher -> readout`
- stronger `readout -> decision`
- more recurrent readout cycles
- stronger readout inhibition
- tighter readout top-k and stricter readout targets

Why it was not productive:

- stronger drive changed which wrong basin won
- longer settling often made collapse worse on the next benchmark gate
- local readout tuning did not solve the missing class-separable upstream code

Lesson:

- do not keep increasing gain, recurrence, or inhibition when the real problem is upstream representation quality

### 7. Adding More Structural Hierarchy Before Verifying Propagation

What we tried:

- retinotopic tiled stage1
- tiled association stages
- distributed higher-visual banks
- direct class decision stages

Why it was not productive:

- structure became more biologically shaped, but useful differentiated drive still did not reliably reach the decision stage
- architecture complexity outran instrumentation

Lesson:

- before adding another area or stage, verify:
  - synapses exist where expected
  - spikes arrive in the right time window
  - post-synaptic drive is nonzero
  - class separation improves at that boundary

### 8. Treating CIFAR Fusion Tuning As The Main Bottleneck

What we tried:

- stronger disagreement arbitration
- margin-preference fusion tweaks
- corpus weighting variants on the CIFAR natural-image path

Why it was not productive:

- CIFAR errors were mostly cases where both hemispheres were already wrong
- fusion had limited headroom relative to the upstream representation problem
- the benchmark stayed flat while complexity increased

Lesson:

- do not spend time on fusion tuning when branch and hemisphere separability are already weak
- first measure how many errors are even recoverable by better arbitration

### 9. Tuning CIFAR Branch Aggregation When All Branches Are Already Collapsing The Same Way

What we tried:

- branch-level vote attribution
- checking whether one branch was dominating the wrong hemisphere label

Why it was not productive:

- all three branches in each hemisphere were usually biased toward the same wrong basin
- changing aggregation logic would not fix a branch representation that is already collapsed upstream

Lesson:

- if branch diagnostics show the same wrong target across all branches, stop tuning hemisphere aggregation
- fix the front-end representation instead

### 10. Repeated Natural-Image Front-End Tweaks Without Reviving The Edge Path

What we tried:

- chromaticity-only color opponency
- chromaticity plus gray-world color adaptation
- local contrast normalization on top of that path
- retinotopic and patchbank CIFAR variants
- luminance-only edge variants
- weaker thresholds and orientation reweighting
- small-kernel Gabor swaps
- overlapping edge receptive fields
- a separate coarser edge-analysis grid

Why it was not productive:

- some small-sample gates moved, but large-sample results did not hold
- several variants regressed sharply despite looking more biologically shaped
- the key diagnostic never changed: orientation slices stayed effectively dead (`raw_l2=0`, `raw_active=0%`)
- in that phase, the only CIFAR gains that generalized came from richer low-frequency appearance channels, not from the edge/orientation tweaks we were testing

Lesson:

- do not keep swapping edge operators or receptive-field geometry when orientation diagnostics remain zero
- before claiming a front-end improvement, verify that the orientation slice actually carries nonzero energy
- if the edge path is still zero, treat appearance-bank features as the real signal and stop tuning edge operators

### 11. Reviving The Entire CIFAR Edge Path At Once

What we tried:

- patch-local edge normalization on every branch
- larger edge-analysis patches on every branch
- reviving orientation signal globally once diagnostics showed the operator scale mismatch

Why it was not productive:

- it solved the wrong subproblem too aggressively
- orientation signal came alive, but overall accuracy regressed sharply
- the classifier collapsed toward a new wrong basin instead of improving (`ship` became dominant on the all-branch edge revival run)
- this showed that a globally revived coarse edge path can overpower the appearance path rather than complement it

Lesson:

- do not turn on coarse normalized edge analysis everywhere at once
- treat revived natural-image edge signal as a supplemental branch-level cue, not the main code path
- the only edge revival that held was the constrained hybrid on the `g10` branches, with reduced `orientation_feature_gain`

### 12. Temporal Coarse-To-Fine Inference Before The Coarse Cue Proved Useful

What we tried:

- a two-pass CIFAR inference path that masked deferred branch orientation slices on an initial coarse pass
- blending coarse and fine classifier confidence with explicit weights
- a narrower variant that deferred only the supplemental `g10` coarse-edge branches

Why it was not productive:

- even the narrower selective version regressed on the controlled CIFAR gate
- the coarse pass shifted confidence toward the wrong appearance-heavy basin instead of improving the final decision
- the branch representation was not yet stable enough for temporal staging to help; it just reweighted the same weak evidence

Lesson:

- do not add temporal coarse-to-fine arbitration until the supplemental coarse branch is already improving the representation by itself
- when testing temporal staging, start with a proven branch-level gain and verify that the coarse pass helps the same classes rather than just amplifying existing collapse

### 13. Broad Feature-Group Divisive Normalization On The CIFAR Hybrid Path

What we tried:

- per-region divisive normalization over orientation, chromatic, and appearance feature groups
- weak cross-group coupling so dominant appearance channels would be compressed before hemisphere classification

Why it was not productive:

- the controlled CIFAR gate regressed slightly instead of improving
- it changed class balance, but did not improve the underlying fusion or hemisphere separability enough to matter
- the current hybrid path appears to benefit more from preserving the existing appearance-plus-supplemental-edge balance than from globally re-scaling feature groups

Lesson:

- do not apply broad group-wise divisive normalization across all CIFAR branches without evidence that one group is truly saturating the classifier
- if divisive normalization is revisited, test it only on the active supplemental branch first, not on the whole natural-features path

### 14. Weak Local Contour Integration On The Active `g10` Branch

What we tried:

- a minimal branch-local contour facilitation step on the CIFAR hybrid path
- support applied only to neighboring `g10` subfields
- orientation-specific lateral support kept weak and branch-local

Why it was not productive:

- the controlled CIFAR gate still regressed slightly instead of improving
- it moved some class balances but did not improve the underlying representation enough to beat the current hybrid baseline
- the existing `g10` supplemental edge branch already appears close to the useful signal limit under the current classifier

Lesson:

- do not keep tuning branch-local contour support on the current CIFAR hybrid path without a larger representational change
- local contour integration is not the next lever if it cannot beat the gate even when constrained to the one branch that already carries live orientation signal

### 15. Stage-1 Branch Support Banks On The CIFAR Hybrid Path

What we tried:

- appended branch-local similarity features derived from multiple per-class support prototypes
- kept the bilateral scaffold and fusion path unchanged
- tested a minimal bank size (`2` support sets per class per branch) before any larger sweep

Why it was not productive:

- the small CIFAR gate improved only marginally
- the larger CIFAR run ended at exactly the same accuracy as the then-protected baseline (`36.60%`)
- that means the added representation complexity did not buy a real gain on the benchmark that matters

Lesson:

- do not add extra stage-1 branch-bank machinery unless it clears the larger natural-image benchmark, not just the small gate
- if a representation change ties the baseline at full scale, treat it as a dead end and keep the simpler path

### 16. Broad Mid-Scale Appearance Pyramids On Every CIFAR Branch

What we tried:

- replaced the pooled `appearance_bank` auxiliary path with a larger quadrant-aware appearance pyramid on every branch
- kept the bilateral scaffold and the hybrid `g10` edge branch unchanged

Why it was not productive:

- the controlled CIFAR gate collapsed sharply instead of improving
- the broader auxiliary vector distorted the working balance between appearance and the supplemental `g10` edge path
- adding mid-scale appearance detail everywhere at once overwhelmed the current classifier rather than preserving useful structure

Lesson:

- do not expand the auxiliary appearance code globally across all CIFAR branches in one step
- if mid-scale appearance is revisited, it needs to be branch-selective or classifier-aware from the start, not a blanket replacement of the current appearance bank

### 17. Stage-1 Shape/Surface Stream Splitting On The CIFAR Hybrid Path

What we tried:

- split the stage-1 hemisphere classifier input into a `shape` stream and a `surface` stream
- keep the bilateral scaffold unchanged while separately normalizing and reweighting the two streams
- validate the idea first on the small CIFAR gate, then on the protected `1000/class, 5000` benchmark

Why it was not productive:

- the small gate improved slightly (`30.70%` vs `30.00%`), but the full benchmark finished lower than the protected baseline
- after debugging the large-run failure mode, the completed result was `35.80%`, still below the then-protected `36.60%`
- the extra classifier-space split added complexity around stage-1 preparation and calibration without fixing the real upstream representation weakness
- the orientation-dead branches stayed dead; this mostly reweighted existing evidence instead of creating a better code

Lesson:

- do not promote classifier-space stream splitting based on a small-sample gate
- if the full CIFAR benchmark loses to the protected baseline, treat the stream split as a closed dead end even if the sampled gate moved up
- upstream Retina and pre-cortical representation remain the more plausible next levers than more stage-1 reweighting

Status:

- closed and documented
- removed from the active benchmark harness

### 18. Inference-Time Active Vision Before Multi-Fixation Evidence Helped

What we tried:

- added inference-time multi-fixation support on top of the promoted CIFAR baseline
- tested a wider `3`-fixation probe with remapping and a more conservative `2`-fixation probe without remapping
- replaced the original ring-offset targeting with a tile-saliency fixation policy

Why it was not productive:

- the protected baseline at `20/class, 200` was `29.50%`
- the wider active-inference probe dropped to `27.00%`
- the conservative active-inference probe still only reached `27.50%`
- the tile-saliency policy also stayed at `27.00%`
- this added sequential inference complexity without producing a sampled win over the already-promoted `training-time saccades + g10-only LGN` surface

Lesson:

- do not scale inference-time active vision without a clear sampled gain over the protected baseline
- changing fixation policy or remapping is not enough if the extra fixations only re-sample the same weak evidence
- the next justified work remains upstream of fusion and readout, where representation quality can improve before arbitration

Status:

- implemented and documented
- not promotable on the current CIFAR path

### 19. Promoting A Tiny Full-Benchmark Stream-Bank Gain Without Confirmation

What we tried:

- added `appearance_stream_bank` back only on `left_retina_g10` and `right_retina_g10`
- kept the winning `training-time saccades + g10-only LGN` surface unchanged otherwise
- advanced the path because the first full benchmark finished at `38.24%`, just above the protected `38.16%`

Why it was not productive:

- the first full-benchmark win was only `0.08` points
- the confirmatory rerun with a fresh seed finished at `37.68%`, below the protected `38.16%` reference
- that means the added stream-bank complexity did not demonstrate a stable mainline gain
- the mixed result is consistent with a marginal sampled fluctuation, not a robust new surface

Lesson:

- do not promote CIFAR changes that win by only a few test images without a confirmatory rerun
- a narrow single-run edge is not enough reason to replace a protected baseline
- on the current CIFAR path, `g10` stream-bank augmentation should stay secondary to stronger upstream ideas like explicit transient/sustained `ON/OFF` plus luminance channels

Status:

- documented and closed for promotion purposes
- keep `training-time saccades + g10-only LGN` as the protected reference

### 20. Another Static `g10` Auxiliary Bank Shuffle Without A Larger-Gate Win

What we tried:

- added a dedicated `luminance_stream_bank` mode for `left_retina_g10` and `right_retina_g10`
- kept the protected `training-time saccades + g10-only LGN` surface unchanged otherwise
- used only luminance, achromatic edge energy, and sustained/transient `ON/OFF` features instead of the broader appearance stream bundle

Why it was not productive:

- the first smoke exposed a real wiring bug where the new auxiliary slice stayed all zeros; that was fixed locally
- after the fix, the standard `50/class, 500 test` gate edged up only slightly to `28.80%` versus the protected `28.40%`
- the decisive `200/class, 1000 test` gate then finished at `32.10%`, below the protected `32.30%`
- that means a cleaner luminance plus `ON/OFF` code still did not produce a robust enough gain to justify scaling the benchmark

Lesson:

- do not keep reshuffling static `g10` auxiliary banks once the larger sampled gate stops improving
- a biologically cleaner auxiliary code is not enough by itself if it cannot beat the protected `200/1000` gate
- the next justified CIFAR work should move away from more static front-end bank variants and toward low-risk training protocol changes on the protected surface

Status:

- documented and closed for promotion purposes
- do not scale this path to `1000/class, 5000` without a materially new reason

### 21. Stacking More Training-Time Augmentation On Top Of The Already-Saccadic CIFAR Surface

What we tried:

- implemented training-only spike-domain augmentation on top of the protected `training-time saccades + g10-only LGN` baseline
- added deterministic extra training variants with small translations, slight rotations, and mild photon-like noise before the existing `4`-fixation training path
- tested both a default `2`-variant configuration and a milder `1`-variant override before spending the required `50/class, 500 test` gate

Why it was not productive:

- the protected baseline smoke at `20/class, 200 test` was already `29.50%`
- the default `2`-variant augmentation probe dropped to `27.00%`
- the milder `1`-variant probe regressed further to `24.50%`
- that means the current CIFAR surface is not simply under-augmented; extra training-time jitter beyond the promoted saccade path diluted useful evidence instead of improving invariance

Lesson:

- do not assume more training-time image perturbation helps once the baseline already includes multi-fixation saccadic sampling
- if a low-risk training protocol change cannot even clear the bounded smoke gate, do not advance it to `50/500`
- the next justified CIFAR work should move to a materially different low-risk protocol change, such as curriculum or review scheduling, rather than more image-space jitter on the same surface

Status:

- documented and closed for promotion purposes
- keep the augmentation code available for future controlled studies, but do not spend larger CIFAR budget on this policy as currently defined

### 22. Curriculum And Review Scheduling Without A Protected-Gate Win

What we tried:

- added declarative curriculum scheduling for stage-1 training order, using easier CIFAR groups first and harder classes later
- added an optional low-rate second review pass on top of the same protected `training-time saccades + g10-only LGN` baseline
- tested the default curriculum plus `10%` review pass, a milder `5%` review pass, and a curriculum-only ablation before spending the larger gate budget

Why it was not productive:

- the full curriculum plus `10%` review smoke dropped to `26.00%`
- reducing the review pass to `5%` still only reached `27.00%`
- removing the review pass entirely recovered to the protected smoke reference at `29.50%`, but the decisive curriculum-only `50/class, 500 test` gate only tied the protected `28.40%` instead of beating it
- that means the review pass is actively harmful on the current CIFAR surface, and curriculum ordering alone is not strong enough to justify a larger benchmark

Lesson:

- do not assume training-order tweaks will rescue the current CIFAR path once the upstream representation and fusion surface are already tightly constrained
- if the only surviving ablation can do no better than tie the protected small gate, stop before `200/1000`
- the next justified work should move away from more CIFAR schedule-order tweaks and toward a different class of bounded change, such as replay/eligibility behavior in the existing continuous-learning pipeline

Status:

- documented and closed for promotion purposes
- keep the curriculum schedule code available, but do not spend larger CIFAR budget on this track as currently defined

### 23. Delayed Replay On The CIFAR Continuous-Learning Path

What we tried:

- took the protected CIFAR `training-time saccades + g10-only LGN` surface and added the existing continuous-learning replay machinery
- swept replay delay on the bounded `20/class, 200 test` screen before spending larger gate budget
- compared immediate replay (`delay=0`) against delayed replay at `2`, `4`, and `8` steps

Why the delayed variants were not productive:

- immediate replay won the smoke at `32.50%`
- `delay=2` slipped to `32.00%`
- `delay=4` and `delay=8` both dropped to `31.50%`
- the replay summaries showed the expected mechanism: average eligibility decayed as delay increased, so the delayed variants were applying weaker corrections without any compensating accuracy gain

Lesson:

- on the current CIFAR surface, replay helps only when it stays close to the original uncertain error
- do not assume a more biologically delayed eligibility trace is automatically better if the current correction target is already sparse and recent
- treat replay delay as fixed at `0` unless there is a materially different replay-priority or reward-trace design to justify reopening it

Status:

- delayed replay variants are documented and closed on the current CIFAR path
- immediate replay is worth keeping and is now part of the promoted local baseline

### 24. First-Order Replay Scalar Sweeps Around The Promoted Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR immediate-replay surface fixed and only nudged first-order replay scalars
- ran a bounded queue-capacity sweep at `50/class, 500 test`
- then ran a bounded reward-gain sweep around the promoted `online_positive_reward_gain = 0.5` and `online_negative_reward_gain = 1.1`

Why it was not productive:

- queue capacity `128` regressed to `31.40%`
- queue capacities `256` and `512` both only tied the protected `31.60%` gate
- lowering positive reward gain to `0.35` and raising it to `0.65` both only tied `31.60%`
- lowering negative reward gain to `0.95` and raising it to `1.25` also only tied `31.60%`
- that means the promoted immediate-replay surface is robust to these simple scalar nudges, but there is no evidence that any of them improve the decisive small gate

Lesson:

- do not keep spending CIFAR budget on first-order queue-capacity or scalar reward-gain tuning once they show a flat protected-gate response
- the next justified work has to be a materially different plasticity rule, not more scalar replay parameter nudging
- if replay tuning is revisited later, it should be because a new interaction exists, not because these first-order sweeps were inconclusive

Status:

- documented and closed on the current CIFAR surface
- keep the promoted immediate-replay settings as the protected baseline and move on to a different plasticity mechanism

### 25. First-Order Reward-Gated STDP Prototype Evidence On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in reward-gated STDP-style prototype evidence bank in classifier space for hemisphere and fusion decisions
- used the existing online reward signal as the neuromodulatory gate
- screened the bounded follow-up at smoke, then advanced the best surviving setting through the protected `50/class, 500 test` and `200/class, 1000 test` gates

Why it was not productive enough:

- the default setting only reached `31.00%` at smoke
- the best tuned setting (`online_reward_stdp_gain = 0.06`, `online_reward_stdp_ltp = 0.12`, `online_reward_stdp_ltd = 0.06`) improved smoke to `32.00%`
- that tuned setting also beat the protected `50/500` gate at `32.00%` versus `31.60%`
- but it still missed the decisive `200/1000` gate at `33.60%`, below the protected `33.80%`
- the mechanism appears to help short-horizon online correction without scaling enough to move the more stable gate

Lesson:

- a first-order reward-gated prototype-evidence path is not enough by itself to beat the protected CIFAR baseline
- do not promote a new plasticity rule just because it wins smoke or the small gate if it still misses the decisive `200/1000` comparison
- the next justified move has to be a materially different local plasticity rule, such as triplet or voltage-dependent plasticity, not more mild tuning of this prototype-gated path

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the experimental code/config as reference, but do not spend more CIFAR budget on simple nudges of this reward-STDP prototype path

### 26. Classifier-Space Triplet-STDP Traces On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in triplet-STDP-style trace/prototype path in classifier space for hemisphere and fusion decisions
- screened two bounded settings at smoke:
  - `online_triplet_stdp_gain = 0.08`, `online_triplet_stdp_ltp = 0.12`, `online_triplet_stdp_ltd = 0.06`, `online_triplet_stdp_fast_decay = 0.88`, `online_triplet_stdp_slow_decay = 0.97`
  - `online_triplet_stdp_gain = 0.05`, `online_triplet_stdp_ltp = 0.10`, `online_triplet_stdp_ltd = 0.05`, `online_triplet_stdp_fast_decay = 0.90`, `online_triplet_stdp_slow_decay = 0.98`
- advanced both bounded settings to the protected `50/class, 500 test` gate because both smoke runs landed at the same score

Why it was not productive enough:

- both bounded triplet settings reached `32.00%` at smoke, so there was no clear winner even at the tiny gate
- at the protected `50/500` gate, both settings only reached `31.60%`, exactly tying the promoted immediate-replay baseline instead of beating it
- one stronger setting dipped to `30.60%` on the raw running test line before online correction recovered the final score back to `31.60%`, which reinforces that the extra triplet path was not adding stable net value
- because no triplet setting produced a protected-gate win, spending `200/1000` budget on this exact design would have been unjustified

Lesson:

- this classifier-space triplet-trace path is not adding enough usable signal on top of the promoted immediate-replay baseline; at best it reproduces the same corrected `50/500` result
- do not advance a new plasticity rule past smoke on parity alone; it needs an actual protected-gate win before taking more CIFAR budget
- the next justified move has to be a materially different local plasticity rule, such as a voltage-dependent path, not more gain/decay tuning of this triplet-trace design

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code path only as reference; do not spend more CIFAR budget on this exact triplet-STDP formulation

### 27. Classifier-Space Voltage-Gated Prototype Plasticity On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in voltage-dependent local-plasticity path in classifier space for hemisphere and fusion decisions
- screened three bounded settings at smoke:
  - `online_voltage_plasticity_gain = 0.06`, `online_voltage_plasticity_ltp = 0.14`, `online_voltage_plasticity_ltd = 0.08`, `online_voltage_plasticity_decay = 0.94`, `online_voltage_plasticity_threshold = 0.30`
  - `online_voltage_plasticity_gain = 0.04`, `online_voltage_plasticity_ltp = 0.10`, `online_voltage_plasticity_ltd = 0.05`, `online_voltage_plasticity_decay = 0.96`, `online_voltage_plasticity_threshold = 0.24`
  - `online_voltage_plasticity_gain = 0.08`, `online_voltage_plasticity_ltp = 0.12`, `online_voltage_plasticity_ltd = 0.03`, `online_voltage_plasticity_decay = 0.92`, `online_voltage_plasticity_threshold = 0.22`

Why it was not productive enough:

- the three bounded settings only reached `31.00%`, `31.50%`, and `31.00%` at smoke
- the protected immediate-replay smoke on the same CIFAR surface is `32.50%`, so this path never even earned a `50/500` gate
- the best surviving voltage setting was still `1.00` point below the protected smoke reference, which is worse than the earlier reward-STDP and triplet probes
- lowering the threshold and softening LTD did not recover the gap, which suggests the issue is the classifier-space prototype formulation itself, not just one bad voltage hyperparameter choice

Lesson:

- a depolarization-gated classifier-space prototype path is still not enough to improve the promoted immediate-replay baseline
- do not spend protected CIFAR gate budget on a new local-plasticity add-on that cannot at least clear the protected smoke reference
- the next justified move has to be materially different from these reward-, triplet-, and voltage-gated prototype/evidence-bank paths

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code path as reference if needed, but do not spend more CIFAR budget on this exact voltage-dependent formulation

### 28. BCM-Style Metaplastic Gating On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- avoided another inference-time evidence bank and instead added an opt-in BCM-style metaplastic gate on the existing reward-driven centroid/weight/exemplar updates
- screened three bounded settings at smoke:
  - `online_bcm_metaplasticity_ltp = 0.45`, `online_bcm_metaplasticity_ltd = 0.35`, `online_bcm_metaplasticity_decay = 0.96`, `online_bcm_metaplasticity_target = 0.28`
  - `online_bcm_metaplasticity_ltp = 0.65`, `online_bcm_metaplasticity_ltd = 0.20`, `online_bcm_metaplasticity_decay = 0.94`, `online_bcm_metaplasticity_target = 0.24`
  - `online_bcm_metaplasticity_ltp = 0.55`, `online_bcm_metaplasticity_ltd = 0.25`, `online_bcm_metaplasticity_decay = 0.97`, `online_bcm_metaplasticity_target = 0.26`

Why it was not productive enough:

- the best and third settings both only tied the protected immediate-replay smoke at `32.50%`
- the second setting slipped to `32.00%`
- the best settings did improve replay corrections versus the protected smoke run, but they still did not produce the one thing that matters for advancement: an actual smoke win
- by the repo's own rule, spending a protected `50/500` gate on parity alone would have repeated the same mistake already documented for earlier plasticity branches

Lesson:

- a BCM-style metaplastic gate on the current centroid/exemplar correction path is not enough by itself to justify more CIFAR budget on this surface
- tying the protected smoke is not a promotion signal and not enough to advance a new plasticity rule to the next gate
- the next justified move has to be materially different from the existing reward-, triplet-, voltage-, and BCM-gated classifier-space adaptations

Status:

- documented and closed for promotion on the current CIFAR surface
- the opt-in code path has been removed after evaluation; do not spend more CIFAR budget on this exact BCM-style metaplastic formulation

### 29. Same-Dimensional Contextual Grouping On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in, per-retina contextual grouping stage in `experiments/retina_classification.cpp`
- applied it only to the two `g10` branches so the protected `g9` and `dog_g9` paths stayed unchanged
- the grouping stage used bounded local contour support, internal border-ownership competition, surround suppression, divisive normalization, and a mild coarse-to-fine bias before the usual per-branch L2 normalization
- screened two bounded smoke settings:
  - `configs/cifar10_retina_bilateral_natural_features_contextual_grouping_experimental.sonata.json`
  - `configs/cifar10_retina_bilateral_natural_features_contextual_grouping_tune1_experimental.sonata.json`

Why it was not productive enough:

- the default bounded setting finished at `30.50%` (`61/200`)
- the milder retune finished at `30.00%` (`60/200`)
- both lost clearly to the protected immediate-replay smoke of `32.50%`
- the failure mode was not a crash or dead branch; the `g10` branches stayed active, but their class-separability collapsed
- on the default run, `left_retina_g10` centroid accuracy fell to `19.00%` and `right_retina_g10` to `17.00%`, which means the modulation degraded the branch representation instead of building a useful proto-object stage

Lesson:

- a same-dimensional modulation of the existing `g10` feature map is not enough to recover the figure-ground and contextual grouping that biological vision gets from V2-like circuitry
- if this area is revisited, it needs a materially different representation, not another small retune of contour gain, suppression, or ownership bias on the current map
- the next justified upstream move is more likely eccentricity-dependent retinal sampling / cortical magnification, or a truly explicit figure-ground representation, not another mild contextual rescaling pass

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact contextual-grouping formulation

### 30. Same-Dimensional Eccentricity Sampling / Cortical Magnification On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in nonuniform receptive-field sampler in `src/adapters/RetinaAdapter.cpp`
- enabled it only on the two `g10` branches so the protected `g9` and `dog_g9` paths stayed unchanged
- used a bounded symmetric foveal warp with a mixed strength knob and gamma exponent to compress central source patches and enlarge peripheral ones while keeping the branch dimensionality fixed
- screened two bounded smoke settings:
  - `configs/cifar10_retina_bilateral_natural_features_eccentricity_sampling_experimental.sonata.json` with `eccentricity_sampling_strength = 0.70`, `eccentricity_sampling_gamma = 1.80`
  - `configs/cifar10_retina_bilateral_natural_features_eccentricity_sampling_tune1_experimental.sonata.json` with `eccentricity_sampling_strength = 0.35`, `eccentricity_sampling_gamma = 1.40`

Why it was not productive enough:

- both bounded settings finished at `28.50%` (`57/200`)
- both lost clearly to the protected immediate-replay smoke of `32.50%`
- the failure mode was again branch-specific rather than a broken run: the untouched branches stayed near their prior behavior while the `g10` branches lost separability
- on the default run, `left_retina_g10` centroid accuracy was `19.00%` and `right_retina_g10` was `18.50%`
- the milder retune did not recover the gap; it only nudged `left_retina_g10` to `21.00%` while keeping the total smoke result flat at `28.50%`

Lesson:

- a same-dimensional cortical-magnification remap of the current `g10` branch is not enough to recover the foveal specialization that biological vision gets from a deeper retinal and cortical organization
- the current `g10` representation is fragile to geometry-only spatial remapping
- if this area is revisited, it needs a materially different representation, such as a temporal coarse-to-fine transient/sustained path or a more explicit multi-stream foveal/peripheral code, not another small retune of this remap

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact eccentricity-sampling formulation

### 31. Same-Dimensional Temporal Coarse-To-Fine Arbitration On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in temporal dual-pass path in `src/adapters/RetinaAdapter.cpp`
- enabled it only on the two `g10` branches so the protected `g9` and `dog_g9` paths stayed unchanged
- switched those `g10` branches to `appearance_stream_bank` and used the auxiliary transient/sustained channels to bias low-frequency coarse support first, then fine-detail support, while keeping the branch dimensionality fixed
- screened one bounded smoke setting:
  - `configs/cifar10_retina_bilateral_natural_features_temporal_coarse_fine_experimental.sonata.json`
  - `temporal_coarse_bias = 0.10`
  - `temporal_transient_gain = 0.55`
  - `temporal_sustained_gain = 0.40`
  - `temporal_cross_band_gain = 0.25`

Why it was not productive enough:

- the bounded smoke finished at `31.00%` (`62/200`)
- the corrected final smoke only tied the protected immediate-replay smoke at `32.50%`, which is not enough to justify a protected `50/500` gate
- the raw testing line still slipped to `31.00%`, so the added temporal arbitration was not producing a stronger pre-correction classifier state
- the failure mode was again branch-specific rather than a broken run: the `g10` branches stayed active, but their class-separability did not improve enough to matter
- `left_retina_g10` centroid accuracy fell to `21.50%` and `right_retina_g10` to `18.00%`
- the low/high split itself was not rescuing the branch: `left_retina_g10/low_band` fell to `17.50%`, `left_retina_g10/high_band` to `20.50%`, `right_retina_g10/low_band` to `17.00%`, and `right_retina_g10/high_band` to `19.00%`

Lesson:

- a same-dimensional temporal coarse-to-fine arbitration of the current `g10` map is not enough to recover the transient/sustained stream advantages that biological vision gets from genuinely distinct pathways
- the current `g10` representation can stay active under this gate while still failing to beat the protected smoke, so activity alone is not a promotion signal
- if this area is revisited, it needs a materially different representation, such as explicit foveal/peripheral sub-branches or truly separate transient/sustained populations with their own late fusion path, not another bounded rescaling pass inside the same map

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact same-dimensional temporal-arbitration formulation

### 32. Explicit `g10` Foveal/Peripheral Branch Split On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added opt-in region-masking support in `src/adapters/RetinaAdapter.cpp`
- replaced each baseline `g10` adapter with explicit `g10_foveal` and `g10_peripheral` adapters that attach to separate hierarchy paths and fuse late through the existing bilateral classifier stack
- screened two bounded smoke settings:
  - `configs/cifar10_retina_bilateral_natural_features_foveal_peripheral_experimental.sonata.json`
    - foveal weight `0.65`, peripheral weight `0.45`
    - mask radius `0.26`, softness `0.08`
  - `configs/cifar10_retina_bilateral_natural_features_foveal_peripheral_tune1_experimental.sonata.json`
    - foveal weight `0.80`, peripheral weight `0.25`
    - mask radius `0.36`, softness `0.14`

Why it was not productive enough:

- the default split finished at `32.00%` (`64/200`)
- the milder retune finished at `31.00%` (`62/200`)
- both stayed below the protected immediate-replay corrected smoke of `32.50%`
- the default run showed the split was alive but weak: `left_retina_g10_foveal` fell to `13.50%` centroid accuracy and `left_retina_g10_peripheral` to `19.00%`; the right side was similarly weak at `15.50%` and `16.00%`
- the milder retune did not rescue it enough: `left_retina_g10_foveal` only reached `15.50%`, `left_retina_g10_peripheral` `20.00%`, `right_retina_g10_foveal` `17.00%`, and `right_retina_g10_peripheral` `17.00%`

Lesson:

- an explicit central/peripheral split of the current `g10` branch is still not enough to recover the kind of foveal/peripheral specialization biological vision gets from deeper retinal and cortical organization
- even with true late fusion and independent branch paths, the current `g10` code loses too much separability when split by geography
- if this area is revisited, it needs a materially different branch axis, such as explicit transient/sustained or `ON/OFF` populations with their own late fusion path, not another retune of this geography-only split

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact foveal/peripheral split formulation

### 33. Explicit Transient/Sustained `g10` Branch Split On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added opt-in temporal-stream branch gating in `src/adapters/RetinaAdapter.cpp`
- replaced each baseline `g10` adapter with explicit `g10_transient` and `g10_sustained` adapters that attach to separate hierarchy paths and fuse late through the existing bilateral classifier stack
- switched those new `g10` branches to `appearance_stream_bank` so the split used the real transient/sustained auxiliary channels already produced by the retina code
- screened one bounded smoke setting:
  - `configs/cifar10_retina_bilateral_natural_features_transient_sustained_experimental.sonata.json`
  - transient branch weight `0.55`, sustained branch weight `0.65`
  - transient floor `0.22`, sustained floor `0.28`
  - preferred-stream gain `0.90`
  - opposing-stream suppression `0.15` transient / `0.12` sustained

Why it was not productive enough:

- the bounded smoke finished at `29.50%` (`59/200`)
- it lost clearly to the protected immediate-replay corrected smoke of `32.50%`
- both temporal branches stayed active, but neither became strong enough to justify the split:
  - left: `left_retina_g10_transient` reached `20.50%`, `left_retina_g10_sustained` `20.00%`
  - right: `right_retina_g10_transient` reached `18.00%`, `right_retina_g10_sustained` `18.00%`
- the hemispheres were not the main issue by themselves: `Left Hemisphere/combined` still reached `34.50%` centroid accuracy and `Right Hemisphere/combined` `31.50%`
- the actual collapse was at fusion: `fusion` centroid accuracy fell to `22.50%`, so the late-fused representation was worse than the protected surface even though the new branches were alive

Lesson:

- an explicit transient/sustained split of the current `g10` branch is still not enough to recover the temporal-stream advantages biological vision gets from deeper retinal and thalamocortical specialization
- simply splitting the branch and late-fusing it is not sufficient when the resulting stream-specific evidence remains weak and poorly aligned for fusion
- if this area is revisited, it needs a materially different parallel code, such as explicit `ON/OFF` plus achromatic luminance branches with their own late fusion path, not another retune of this transient/sustained split

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact transient/sustained branch-split formulation

### 34. Explicit Late-Fused `ON/OFF + Luminance` `g10` Branch Split On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added opt-in luminance-branch gating in `src/adapters/RetinaAdapter.cpp`
- replaced each baseline `g10` adapter with explicit `g10_on`, `g10_off`, and `g10_luminance` adapters that attach to separate hierarchy paths and fuse late through the existing bilateral classifier stack
- switched those new `g10` branches to `luminance_stream_bank` so the split used the real luminance and transient/sustained `ON/OFF` auxiliary channels already produced by the retina code
- screened one bounded smoke setting:
  - `configs/cifar10_retina_bilateral_natural_features_onoff_luminance_experimental.sonata.json`
  - `ON` / `OFF` branch weight `0.35`
  - luminance branch weight `0.50`
  - preferred-drive gain around `0.90-0.95`
  - opponent suppression around `0.08-0.18`

Why it was not productive enough:

- the clean smoke rerun finished at only `27.50%` (`55/200`)
- it lost clearly to the protected immediate-replay corrected smoke of `32.50%`
- all six explicit `g10` sub-branches stayed alive, but none were strong enough to justify the split:
  - left: `left_retina_g10_on` reached `22.50%`, `left_retina_g10_off` `21.50%`, `left_retina_g10_luminance` `21.00%`
  - right: `right_retina_g10_on` reached `18.50%`, `right_retina_g10_off` `19.00%`, `right_retina_g10_luminance` `18.00%`
- the hemispheres were not completely dead by themselves: `Left Hemisphere/combined` still reached `35.00%` centroid accuracy and `Right Hemisphere/combined` `32.00%`
- the actual collapse was again at fusion: `fusion` centroid accuracy fell to `22.50%`, so the late-fused representation was materially worse than the protected surface even though the new sub-branches were active
- replay correction did almost nothing for it: only `2` of `147` correction events succeeded

Lesson:

- an explicit late-fused `ON/OFF + luminance` split of the current `g10` branch is still not enough to recover the parallel-stream advantages biological vision gets from deeper retinal and thalamic specialization
- simply adding more biologically named late-fused sub-branches is not sufficient when the resulting stream-specific evidence remains weak and poorly aligned for fusion
- if this area is revisited, it needs materially different pre-cortical dynamics, such as stream-specific LGN relay timing or gain structure feeding the existing cortical branch, not another split of the current `g10` map into more late-fused sub-branches

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the code/config path as reference if needed, but do not spend more CIFAR budget on this exact `ON/OFF + luminance` branch-split formulation

### 35. `g10`-Only Fixation-Aware Stage-1 Memory On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- reran the protected `200/1000` gate with `flow_audit` enabled to establish a fresh control on the current binary:
  - `build/flow_audit_baseline_gate_200_1000_rerun.log`
  - `build/flow_audit_baseline_gate_200_1000_rerun_testing_summary.json`
- added opt-in stage-1 fixation-memory aggregation in `experiments/retina_classification.cpp`
- enabled it only on `left_retina_g10` and `right_retina_g10` through `stage1_fixation_memory_mode = mean_max_summary`
- left `g9`, `dog_g9`, replay, and corpus-callosum fusion unchanged
- screened one bounded `200/1000` audited gate:
  - `configs/cifar10_retina_bilateral_natural_features_g10_fixation_memory_experimental.sonata.json`
  - `build/flow_audit_g10_fixation_memory_gate_200_1000.log`
  - `build/flow_audit_g10_fixation_memory_gate_200_1000_testing_summary.json`

Why it was not productive enough:

- the fresh protected control finished at `33.70%` (`337/1000`); the fixation-aware probe only nudged that to `33.80%` (`338/1000`)
- the probe did improve hemisphere neighborhood purity:
  - left `mean_topk_purity`: `0.237889 -> 0.275667`
  - right `mean_topk_purity`: `0.239556 -> 0.274000`
- but it failed the actual gating requirement because both `g10` post-normalization margins got slightly worse:
  - left `mean_post_margin`: `-0.00387905 -> -0.00389373`
  - right `mean_post_margin`: `-0.00376781 -> -0.00378284`
- the offline fusion proxies also moved the wrong way:
  - `interaction` centroid accuracy: `35.9 -> 33.3`
  - `confidence-concat` centroid accuracy: `35.7 -> 33.4`
  - `hemisphere-concat` stayed flat at `29.8`
- the representation change shifted stage-1 view balance instead of cleanly strengthening both hemispheres: left view accuracy improved to `31.50%`, right view accuracy fell to `29.70%`
- because it missed the margin gate even with the purity gain, it did not justify a `1000/5000` run

Lesson:

- changing how the current `g10` fixation set is summarized can improve local neighborhood purity without improving the underlying `g10` discriminative margin
- on this CIFAR surface, the remaining ceiling no longer looks like a pure fixation-aggregation problem
- if this area is revisited, it should be through materially different raw `g10` feature code or pre-cortical relay dynamics, not another `mean` vs `summary` reshuffle of the same fixations

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code/config as reference if needed, but do not spend more CIFAR budget on more fixation-memory retunes of the current `g10` map

### 36. Band-Mixed Magnocellular/Parvocellular-Like LGN Relay On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added opt-in dual-relay band-image construction in `src/adapters/RetinaAdapter.cpp`
- enabled it only on `left_retina_g10` and `right_retina_g10` through `lgn_parallel_relay_enabled`
- used a coarse achromatic magno-like relay, a finer parvo-like relay, and blended them back into the existing `g10` branch instead of creating more late-fused cortical branches
- screened a fresh same-build protected smoke plus two bounded probes:
  - protected smoke: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_magno_parvo.log`
  - default relay split: `configs/cifar10_retina_bilateral_natural_features_magno_parvo_lgn_experimental.sonata.json`
  - conservative retune: `configs/cifar10_retina_bilateral_natural_features_magno_parvo_lgn_tune1_experimental.sonata.json`

Why it was not productive enough:

- the protected smoke on the same build finished at `32.00%` (`64/200`)
- the default relay split collapsed to `23.50%` (`47/200`)
- the conservative retune only recovered to `27.00%` (`54/200`)
- both probes lost clearly enough that there was no justified `50/500` gate
- the failure was branch-local and upstream:
  - default split stage-1 view accuracy fell to `19.00%` left and `21.50%` right
  - conservative retune only recovered to `21.00%` on both views
  - `g10` stayed active, but the default split pushed `raw_active` down into the mid-`50%` range and kept `g10` centroid accuracy around `20-20.5%`
  - even the conservative retune only moved `left_retina_g10` to `23.00%` and `right_retina_g10` to `21.00%`
- replay correction did not rescue it: only `2` corrections on the default run and `3` on the conservative retune succeeded

Lesson:

- a band-mixed coarse/fine LGN relay on the current `g10` map is not enough to recover the benefits of true magnocellular/parvocellular specialization
- the coarse/fine blend weakened the usable `g10` representation instead of strengthening it
- if this area is revisited, it needs materially different pre-cortical computation or a different raw `g10` feature code, not another retune of this relay-mixing formulation

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code/config as reference if needed, but do not spend more CIFAR budget on more band-mixed magno/parvo relay retunes of the current `g10` map

### 37. Raw `g10` Orientation-Energy Feature Code On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in `orientation_energy` edge operator in `src/features/OrientationEnergyOperator.cpp`
- enabled it only on `left_retina_g10` and `right_retina_g10`
- used a gradient-histogram orientation code with structure-tensor sharpening instead of the existing Sobel-style directional differences
- screened a fresh same-build protected smoke plus two bounded probes:
  - protected smoke: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_orientation_energy.log`
  - default raw-feature probe: `configs/cifar10_retina_bilateral_natural_features_orientation_energy_experimental.sonata.json`
  - sharper retune: `configs/cifar10_retina_bilateral_natural_features_orientation_energy_tune1_experimental.sonata.json`

Why it was not productive enough:

- the protected smoke on the same build finished at `32.00%` (`64/200`)
- both bounded raw-feature probes finished at only `30.00%` (`60/200`)
- that was not close enough to justify a `50/500` gate
- the result was mixed rather than uniformly bad:
  - the default probe materially improved raw `g10` branch separability
  - `left_retina_g10` centroid accuracy moved from `21.00%` to `26.00%`
  - `right_retina_g10` centroid accuracy moved from `18.00%` to `27.00%`
  - the hemisphere combined centroid proxies rose from `30.00%` and `27.00%` to `37.50%` and `37.00%`
- but that upstream gain did not survive the current end-to-end surface:
  - stage-1 view accuracy only rose from `26.50%`/`25.00%` to `27.50%`/`27.50%`
  - fusion centroid accuracy only reached `30.00%`
  - replay still corrected only `6` cases on the default probe
  - final post-correction accuracy stayed below the protected smoke
- the sharper retune pushed `g10` centroid accuracy a little further to `27.50%`/`27.50%`, but task accuracy still stayed flat at `30.00%`

Lesson:

- a materially different raw `g10` code can improve branch-local separability without improving the protected CIFAR objective
- on this surface, a better `g10` branch alone is not enough if the new code is not aligned with the current fusion/readout surface
- if this area is revisited, it should be through a materially different raw feature family, such as a phase-sensitive or quadrature simple-cell code, or through a raw `g10` auxiliary representation designed to match the new edge code, not another small retune of the current orientation-energy/tensor-histogram formulation

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in operator/configs as reference if needed, but do not spend more CIFAR budget on more small orientation-energy/tensor retunes of the current `g10` map

### 38. Raw `g10` Quadrature-Gabor Feature Code On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in `quadrature_gabor` edge operator in `src/features/QuadratureGaborOperator.cpp`
- enabled it only on `left_retina_g10` and `right_retina_g10`
- used phase-invariant even/odd Gabor energy with patch mean-centering instead of the current Sobel-style or tensor-histogram raw code
- screened a fresh same-build protected smoke plus two bounded probes:
  - protected smoke: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_quadrature_gabor.log`
  - default raw-feature probe: `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_experimental.sonata.json`
  - permissive retune: `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_tune1_experimental.sonata.json`

Why it was not productive enough:

- the protected smoke on the same build finished at `32.00%` (`64/200`)
- both bounded quadrature-Gabor probes collapsed to `22.50%` (`45/200`)
- that was far below the bar for a `50/500` gate
- this was not the same kind of failure as orientation-energy:
  - `g10` orientation support effectively vanished
  - `left_retina_g10/orientation` and `right_retina_g10/orientation` both reported `raw_active=0.00%`
  - the surviving `g10` branch was almost entirely the existing auxiliary appearance bank
  - stage-1 view accuracy fell to `18.50%` left and `21.00%` right
  - fusion centroid accuracy collapsed to `20.00%`
- the permissive retune did not change the outcome at all on this surface, which is a strong sign that the problem is geometric rather than scalar

Lesson:

- a direct quadrature-Gabor/simple-cell swap is too weak on the current `5x5` `g10` edge-analysis patches
- the current patch support and thresholded feature path do not sustain usable phase-sensitive orientation energy here
- if this family is revisited, it needs materially different spatial support or a different stage boundary, not another scalar retune of the same direct replacement
- given the orientation-energy result immediately before it, the more defensible next move is now an operator-matched raw `g10` auxiliary representation on top of the stronger orientation-energy base, not another direct Gabor-family swap

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in operator/configs as reference if needed, but do not spend more CIFAR budget on more direct quadrature-Gabor retunes of the current `g10` patch geometry

### 39. `g10` Complex-Cell Pooling Plus Divisive Normalization On The Immediate-Replay Baseline

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in same-dimensional `complex_cell_enabled` stage in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`
- pooled `g10` orientation responses across the pooled-region and subfield blocks, then applied orientation-wise divisive normalization before the existing band-selection/output path
- enabled the new stage only on `left_retina_g10` and `right_retina_g10`
- paired it with the stronger `orientation_energy` raw `g10` carrier, because the earlier direct orientation-energy follow-up improved branch-local separability without enough task lift
- screened a fresh same-build protected smoke, two bounded probes, and one audited rerun of the best probe:
  - protected smoke: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_complex_cell.log`
  - default probe: `configs/cifar10_retina_bilateral_natural_features_complex_cell_experimental.sonata.json`
  - milder retune: `configs/cifar10_retina_bilateral_natural_features_complex_cell_tune1_experimental.sonata.json`
  - audit rerun: `build/cifar10_retina_complex_cell_flow_audit_smoke_20_200.log`

Why it was not productive enough:

- the protected smoke on the same build finished at `32.00%` (`64/200`)
- the default complex-cell probe finished at `31.50%` (`63/200`)
- the milder retune fell to `29.50%` (`59/200`)
- the best probe therefore did not justify a `50/500` gate
- this was again not a pure upstream collapse:
  - on the best probe, `left_retina_g10/orientation` rose to `25.50%` centroid accuracy and `right_retina_g10/orientation` reached `22.00%`
  - the hemisphere combined centroid proxies improved to `36.00%` left and `33.50%` right
  - the audited `g10` margins improved relative to the protected audited smoke:
    - left `mean_post_margin`: `-0.00469` vs `-0.00592`
    - right `mean_post_margin`: `-0.00492` vs `-0.00594`
  - `g10` stayed strongly active rather than collapsing: audited post active fraction stayed around `75-76%`
- but the current end-to-end surface still got worse where it mattered:
  - stage-1 view accuracy dropped to `23.50%` left and `23.00%` right
  - the audited fusion alternatives degraded instead of improving:
    - `interaction_centroid_accuracy`: `27.5%` vs protected `29.0%`
    - `confidence_concat_centroid_accuracy`: `27.0%` vs protected `30.0%`
    - `hemisphere_concat_centroid_accuracy`: `19.5%` vs protected `18.5%`
  - replay correction remained small, with only `6` successful corrections on the audited best probe
- the milder retune pushed branch-local `g10` centroid accuracy a bit further on some slices, but task accuracy still moved in the wrong direction

Lesson:

- a biologically motivated complex-cell style pooling and divisive-normalization stage can improve local `g10` separability on this surface
- that still is not enough by itself, because the current benchmark has no object-part or border-assignment stage that can exploit the improved local code
- the missing function is no longer best described as “more edge energy”; it is contour/border organization on top of the improved local orientation support
- if this family is revisited, it should be through an operator-matched contour continuation, end-stopping, or border-ownership representation on top of the stronger local `g10` code, not another same-dimensional scalar retune of pooling or divisive normalization

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code/configs as reference if needed, but do not spend more CIFAR budget on more same-dimensional complex-cell pooling or divisive-normalization retunes of the current `g10` map

### 40. `g10` Contour-Support Auxiliary Bank On Top Of Orientation-Energy Plus Complex-Cell

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in `contour_support_bank` auxiliary mode in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`
- derived six same-dimensional `g10` auxiliary channels from the post-complex-cell orientation map:
  - collinear continuation
  - cocircular / curvature support
  - end-stopping
  - junctionness
  - border-owner-left
  - border-owner-right
- enabled it only on `left_retina_g10` and `right_retina_g10` in `configs/cifar10_retina_bilateral_natural_features_contour_support_experimental.sonata.json`
- evaluated it under the strict audited smoke gate against a same-build protected control:
  - protected audit: `build/cifar10_retina_baseline_flow_audit_for_contour_support_20_200.log`
  - protected summary: `build/flow_audit_baseline_contour_support_smoke_testing_summary.json`
  - valid strict probe: `build/cifar10_retina_contour_support_flow_audit_fix2_20_200.log`
  - valid strict summary: `build/flow_audit_contour_support_smoke_fix2_testing_summary.json`

Why it was not productive enough:

- the protected same-build audit finished at `32.00%` (`64/200`)
- the valid strict probe finished at only `30.00%` (`60/200`)
- so it failed the headline gate immediately and did not justify a `50/500` follow-up
- importantly, this was not a pure local-feature collapse:
  - both hemisphere `mean_topk_purity` scores improved:
    - left: `0.186111` vs protected `0.184444`
    - right: `0.187778` vs protected `0.185556`
  - both audited `g10` post-margins moved materially toward zero:
    - left: `-0.00287558` vs protected `-0.00592188`
    - right: `-0.00301946` vs protected `-0.00593531`
  - `confidence_concat_centroid_accuracy` held the protected line at `30.0%`
- but the experiment still failed where the strict criteria actually demanded a win:
  - `interaction_centroid_accuracy` slipped to `28.5%` vs the protected `29.0%`
  - final smoke stayed well below the required `>32.0%`
  - stage-1 accuracy by view remained weak at only `23.00%` left and `24.00%` right
  - replay stayed small, correcting only `5` of `145` correction events

Lesson:

- this surface can respond to contour/border-style local evidence in a measurable way; the `g10` local code and hemisphere neighborhood purity both improved
- that still is not enough if the added contour evidence stays same-dimensional and gets pushed through the current hemisphere-level matcher/fusion surface unchanged
- the problem is no longer “there is no contour signal at all”; it is that the current pipeline cannot cash out that better local contour signal into a better hemisphere-level object representation
- if this area is revisited, it must be through a materially different intermediate stage boundary or grouping/readout representation, not more same-dimensional `contour_support_bank` retunes on the current `g10` map

Status:

- documented and closed for promotion on the current CIFAR surface
- do not spend more CIFAR budget on more same-dimensional `contour_support_bank` retunes of the current `g10` surface
- keep the opt-in code/config as reference only

### 41. Guided Training-Time Saccades Scored By Existing `g10` Energy

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in guided fixation selector in `experiments/retina_classification.cpp`
- left the protected baseline config unchanged and enabled the probe only through CLI/JSON flags
- generated candidate fixation offsets around the original image, scored each candidate with the existing `g10` orientation/auxiliary energy, and selected `4` fixations from `8` candidates with a small inhibition-of-return penalty
- evaluated it under the strict audited smoke gate against a same-build protected control:
  - protected audit: `build/cifar10_guided_saccades_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_guided_saccades_baseline_testing_summary.json`
  - guided probe: `build/cifar10_guided_saccades_flow_audit_20_200.log`
  - guided summary: `build/flow_audit_guided_saccades_testing_summary.json`

Why it was not productive enough:

- the protected same-build audit finished at `32.00%` (`64/200`)
- the guided-saccade probe finished at only `29.00%` (`58/200`)
- so it failed the headline gate immediately and did not justify a `50/500` follow-up
- this was not a total local-feature collapse:
  - left hemisphere `mean_topk_purity` improved to `0.191111` from protected `0.184444`
  - right hemisphere `mean_topk_purity` only tied at `0.185556`
  - audited `g10` post-margins moved slightly toward zero:
    - left: `-0.00580663` vs protected `-0.00592188`
    - right: `-0.00589308` vs protected `-0.00593531`
  - audited `g10` orientation energy was unchanged:
    - left: `0.913151` vs protected `0.913151`
    - right: `0.912764` vs protected `0.912764`
- but the experiment failed where the strict criteria actually demanded a win:
  - stage-1 view accuracy dropped to `25.50%` left and `23.50%` right, from protected `26.50%` left and `25.00%` right
  - `interaction_centroid_accuracy` dropped to `27.0%` from protected `29.0%`
  - `confidence_concat_centroid_accuracy` dropped to `27.0%` from protected `30.0%`
  - final smoke stayed below the protected `32.00%` result

Lesson:

- replacing fixed training saccade offsets with a selector that simply chases current `g10` feature energy does not improve the current CIFAR surface
- the likely failure mode is that the selector prefers high-energy clutter or already-dominant texture, not class-discriminative object evidence
- if guided saccades are revisited, the policy needs a materially different objective, such as uncertainty reduction, object-boundary novelty, or prediction error, not just local `g10` energy
- do not scale this specific policy to `8` or `48` fixations; the strict smoke gate already failed

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- continue path work with a materially different candidate, such as a thalamic relay dynamic, rather than another high-energy fixation-selector retune

### 42. `g10` Burst/Tonic LGN Relay Mode Switch

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in local-salience burst/tonic mode switch on top of the existing `lgn_relay_enabled` center-surround path in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`
- left `g9`, `dog_g9`, replay, and corpus-callosum fusion unchanged
- left the protected baseline config unchanged and enabled the probe only through g10-only CLI overrides in `experiments/retina_classification.cpp`
- evaluated it under the strict audited smoke gate against a same-build protected control:
  - protected audit: `build/cifar10_burst_tonic_lgn_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_burst_tonic_lgn_baseline_testing_summary.json`
  - burst/tonic probe: `build/cifar10_burst_tonic_lgn_flow_audit_20_200.log`
  - burst/tonic summary: `build/flow_audit_burst_tonic_lgn_testing_summary.json`
- probe flags: `--g10-lgn-burst-tonic-enabled --g10-lgn-burst-threshold 0.04 --g10-lgn-burst-extra-strength 0.12 --g10-lgn-burst-slope 8.0 --g10-lgn-burst-neuromodulator 1.0`

Why it was not productive enough:

- the protected same-build audit finished at `32.00%` (`64/200`)
- the burst/tonic probe finished at `31.50%` (`63/200`)
- so it failed the headline gate and did not justify a `50/500` follow-up
- this was not a pure drive collapse:
  - initial accuracy improved to `30.00%` from protected `28.50%`
  - left hemisphere `mean_topk_purity` improved to `0.186111` from protected `0.184444`
  - left audited `g10` post-margin moved toward zero: `-0.00579146` vs protected `-0.00592188`
  - audited `g10` orientation energy stayed effectively unchanged: left `0.913284` vs `0.913151`, right `0.912758` vs `0.912764`
- but the strict scale criteria moved the wrong way:
  - final smoke fell to `31.50%`, below the protected `32.00%`
  - stage-1 view accuracy dropped to `24.50%` left and `24.50%` right, from protected `26.50%` left and `25.00%` right
  - `interaction_centroid_accuracy` dropped to `24.5%` from protected `29.0%`
  - `confidence_concat_centroid_accuracy` dropped to `26.0%` from protected `30.0%`
  - right hemisphere `mean_topk_purity` fell to `0.181667` from protected `0.185556`
  - replay successes fell to `3` from protected `7`

Lesson:

- a local contrast/salience-gated burst mode can perturb the current `g10` relay without killing raw orientation energy
- the perturbation does not produce a better end-to-end object code on this surface; it improved the pre-correction decision but made the fusion and replay surface worse
- do not spend CIFAR budget retuning this same local-salience relay gain policy
- if thalamic gating is revisited, it needs a materially different gating signal, such as uncertainty, reward context, or prediction error, not only center-surround salience amplitude

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- continue path work with a different mechanism, not another local-salience center-surround relay retune

### 43. `g10` Sensory Triplet/BCM Feature-Gain Plasticity

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in sensory-layer-only plasticity rule in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`
- implemented local per-feature gain updates with fast/slow triplet traces, a BCM-style sliding activity threshold, and training-only learning that freezes for evaluation
- left classifier-space triplet/voltage/BCM, replay, corpus-callosum fusion, `g9`, and `dog_g9` unchanged
- left the protected baseline config unchanged and enabled the probe only through g10-only CLI overrides in `experiments/retina_classification.cpp`
- evaluated it under the strict audited smoke gate against a same-build protected control:
  - protected audit: `build/cifar10_sensory_triplet_bcm_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_sensory_triplet_bcm_baseline_testing_summary.json`
  - sensory triplet/BCM probe: `build/cifar10_sensory_triplet_bcm_flow_audit_20_200.log`
  - sensory triplet/BCM summary: `build/flow_audit_sensory_triplet_bcm_testing_summary.json`
- probe flags: `--g10-sensory-triplet-bcm-enabled --g10-sensory-triplet-bcm-learning-rate 0.015 --g10-sensory-triplet-bcm-ltp 0.12 --g10-sensory-triplet-bcm-ltd 0.04 --g10-sensory-triplet-fast-decay 0.80 --g10-sensory-triplet-slow-decay 0.97 --g10-sensory-bcm-threshold-decay 0.985 --g10-sensory-bcm-target-activation 0.08`

Why it was not productive enough:

- the protected same-build audit finished at `32.00%` (`64/200`)
- the sensory triplet/BCM probe finished at `32.50%` (`65/200`)
- this is a real one-sample headline lift, but the audit does not support spending a `50/500` gate on this exact setting
- the positive signs:
  - initial accuracy improved to `29.50%` from protected `28.50%`
  - final smoke improved to `32.50%` from protected `32.00%`
  - left hemisphere `mean_topk_purity` improved to `0.186667` from protected `0.184444`
  - right hemisphere `mean_topk_purity` improved to `0.187778` from protected `0.185556`
  - audited `g10` orientation energy increased:
    - left `mean_post_orientation_l2`: `0.920339` vs protected `0.913151`
    - right `mean_post_orientation_l2`: `0.920196` vs protected `0.912764`
- the failure signs:
  - both audited `g10` post-margins worsened:
    - left: `-0.00647548` vs protected `-0.00592188`
    - right: `-0.00665063` vs protected `-0.00593531`
  - `interaction_centroid_accuracy` dropped to `25.5%` from protected `29.0%`
  - `confidence_concat_centroid_accuracy` dropped to `26.5%` from protected `30.0%`
  - replay successes fell to `6` from protected `7`
  - stage-1 accuracy only rearranged by view: left fell to `25.00%` from `26.50%`, while right rose to `26.00%` from `25.00%`

Lesson:

- a genuinely sensory-layer-local triplet/BCM-style gain rule can move the current `g10` representation without collapsing raw orientation energy
- the movement is not yet aligned with the downstream object code: raw orientation energy and hemisphere purity improved while branch margins and fusion separability worsened
- do not scale this exact sensory-gain rule; a one-sample smoke lift is not enough when the flow audit says the fusion surface got worse
- if this family is revisited, the gain update needs a stricter gate, such as prediction-error, class-agnostic novelty that preserves fusion separability, or an explicit constraint on post-normalization margins

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this exact bounded setting unless a repeat also preserves the fusion audit metrics

### 44. Prediction-Error-Gated Corpus-Callosum Feedback Proxy

What we tried:

- kept the promoted CIFAR `training-time saccades + g10-only LGN + immediate replay` surface fixed
- added an opt-in corpus-callosum decision-path proxy for prediction-error-gated feedback in `experiments/retina_classification.cpp`
- computed top-down expectation from the predicted class centroid for each hemisphere, then computed a local prediction error as `1 - cosine(actual_stage1_pattern, predicted_class_centroid)`
- exposed the mechanism without mutating the protected config:
  - `--prediction-error-feedback-enabled`
  - `--prediction-error-feedback-gain`
  - `--prediction-error-feedback-threshold`
  - `--prediction-error-feedback-max-penalty`
  - `--prediction-error-feedback-min-confidence`
- tested two bounded forms:
  - whole-hemisphere evidence scaling
  - targeted predicted-label expectation suppression, where only the predicted hypothesis and vote are downweighted
- evaluated under the strict audited smoke gate against a same-build protected control:
  - protected audit: `build/cifar10_prediction_error_feedback_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_prediction_error_feedback_baseline_testing_summary.json`
  - whole-hemisphere proxy: `build/cifar10_prediction_error_feedback_flow_audit_20_200.log`
  - whole-hemisphere summary: `build/flow_audit_prediction_error_feedback_testing_summary.json`
  - targeted predicted-label proxy: `build/cifar10_prediction_error_feedback_labelgate_flow_audit_20_200.log`
  - targeted predicted-label summary: `build/flow_audit_prediction_error_feedback_labelgate_testing_summary.json`
- probe flags for both forms: `--prediction-error-feedback-enabled --prediction-error-feedback-gain 1.0 --prediction-error-feedback-threshold 0.06 --prediction-error-feedback-max-penalty 0.18 --prediction-error-feedback-min-confidence 0.05`

Why it was not productive enough:

- the protected same-build audit finished at `32.00%` (`64/200`)
- the whole-hemisphere feedback proxy also finished at `32.00%` (`64/200`)
- the targeted predicted-label suppression proxy also finished at `32.00%` (`64/200`)
- initial accuracy stayed fixed at `28.50%`
- stage-1 view accuracy stayed fixed at `26.50%` left and `25.00%` right
- replay corrections stayed fixed at `7` successes and `419` replay events
- flow-audit fusion alternatives stayed unchanged:
  - `interaction_centroid_accuracy`: `29.0%`
  - `confidence_concat_centroid_accuracy`: `30.0%`
  - `hemisphere_concat_centroid_accuracy`: `18.5%`
- hemisphere audit aggregates stayed unchanged:
  - left `mean_topk_purity`: `0.184444`
  - right `mean_topk_purity`: `0.185556`
  - left `mean_centroid_margin`: `-0.00337642`
  - right `mean_centroid_margin`: `-0.00324997`
- audited `g10` branch metrics also stayed unchanged:
  - left `g10` post-margin: `-0.00592188`
  - right `g10` post-margin: `-0.00593531`
  - left `g10` post-orientation L2: `0.913151`
  - right `g10` post-orientation L2: `0.912764`

Lesson:

- centroid-based prediction-error gating at the existing corpus-callosum decision path is not a useful lever on the current CIFAR surface
- this proxy is not equivalent to graph-native L5/6-to-L4 predictive coding; it happens after the weak `g10` representation has already been formed and after class prototypes are already highly overlapping
- the no-lift result argues against spending CIFAR budget on more scalar centroid-error suppression in the decision path
- if predictive feedback is revisited, it needs to be implemented at a true intermediate representation boundary with explicit actual-minus-expectation features and a concrete fusion metric, not as another class-centroid confidence reweighting rule

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this decision-path proxy

### 45. Neuromodulator-Gated Eligibility With Variable-Delay Replay Consolidation

What we tried:

- kept the protected CIFAR baseline and upstream representation fixed
- added a default-off replay-path probe in `experiments/retina_classification.cpp` for:
  - uncertainty- and reward-scaled eligibility traces
  - variable replay delay derived from the scaled eligibility
  - a small successful-sequence consolidation queue replayed between ordinary replay steps
- exposed the mechanism without mutating the protected config:
  - `--online-neuromodulator-eligibility-enabled`
  - `--online-neuromodulator-uncertainty-gain`
  - `--online-neuromodulator-positive-reward-gain`
  - `--online-neuromodulator-eligibility-max`
  - `--online-replay-variable-delay-enabled`
  - `--online-replay-success-consolidation-enabled`
  - `--online-replay-success-consolidation-capacity`
  - `--online-replay-success-consolidation-batch-size`
  - `--online-replay-success-consolidation-repeats`
- first checked the same-build audited smoke gate:
  - protected audit: `build/cifar10_neuromodulator_replay_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_neuromodulator_replay_baseline_testing_summary.json`
  - probe audit: `build/cifar10_neuromodulator_replay_flow_audit_20_200.log`
  - probe summary: `build/flow_audit_neuromodulator_replay_testing_summary.json`
- then spent a direct same-build `50/500` gate:
  - protected gate: `build/cifar10_neuromodulator_replay_baseline_gate_50_500.log`
  - probe gate: `build/cifar10_neuromodulator_replay_gate_50_500.log`
- bounded probe flags:
  - `--online-neuromodulator-eligibility-enabled --online-neuromodulator-uncertainty-gain 1.0 --online-neuromodulator-positive-reward-gain 0.5 --online-neuromodulator-eligibility-max 1.75 --online-replay-variable-delay-enabled --online-replay-delay-steps 4 --online-replay-success-consolidation-enabled --online-replay-success-consolidation-capacity 64 --online-replay-success-consolidation-batch-size 1 --online-replay-success-consolidation-repeats 1`

Why it was not productive enough:

- the same-build audited smoke improved slightly from `32.00%` (`64/200`) to `33.50%` (`67/200`)
- that smoke lift did not come from a better upstream code:
  - both audited `g10` post-margins were unchanged
  - both audited `g10` post-orientation L2 values were unchanged
  - `confidence_concat_centroid_accuracy` stayed fixed at `30.0%`
  - `hemisphere_concat_centroid_accuracy` stayed fixed at `18.5%`
  - hemisphere purity only moved trivially:
    - left `mean_topk_purity`: `0.184444` -> `0.185000`
    - right `mean_topk_purity`: `0.185556` -> `0.186111`
  - the only fusion proxy movement was a tiny `interaction_centroid_accuracy` lift from `29.0%` to `29.5%`
- the larger same-build gate reversed the apparent gain:
  - protected baseline finished at `30.80%` on `500` test samples
  - the neuromodulator replay probe finished at `30.40%` on `500` test samples
- the `50/500` result means the smoke bump was replay churn, not a stable classification improvement

Lesson:

- extending eligibility and replaying rewarded sequences can increase replay activity without improving the actual object code
- when the audited `g10` and hemisphere margins stay flat, a small smoke lift is not enough evidence to scale a replay-policy variant
- this family is now closed in its current scalar form: better replay scheduling cannot compensate for the current upstream representation ceiling
- if continuous-learning work is revisited, it needs a materially different source of information at replay time, not another uncertainty/reward/delay reshuffle of the current prototypes and exemplars

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend more CIFAR budget on this neuromodulator-eligibility plus success-consolidation replay variant

### 46. V2-Like Hemisphere Convergent Stage From Pooled Branch Conjunctions

What we tried:

- kept the protected CIFAR baseline fixed and added a new opt-in hemisphere-internal stage boundary in `experiments/retina_classification.cpp`
- instead of feeding the classifier the raw concatenated branch pattern, built a convergent hemisphere code from:
  - pooled summaries of each live branch
  - pooled orientation/auxiliary/band projections where available
  - pairwise branch conjunction and disagreement terms
- exposed the mechanism without mutating the protected config:
  - `--hemisphere-convergent-code-enabled`
  - `--hemisphere-convergent-summary-bins`
  - `--hemisphere-convergent-residual-gain`
  - `--hemisphere-convergent-interaction-gain`
- ran a same-build strict audited smoke gate:
  - protected control: `build/cifar10_convergent_stage_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_convergent_stage_baseline_testing_summary.json`
  - convergent-stage probe: `build/cifar10_convergent_stage_flow_audit_20_200.log`
  - convergent-stage summary: `build/flow_audit_convergent_stage_testing_summary.json`
- bounded probe flags:
  - `--hemisphere-convergent-code-enabled --hemisphere-convergent-summary-bins 12 --hemisphere-convergent-residual-gain 0.35 --hemisphere-convergent-interaction-gain 1.0`

Why it was not productive enough:

- the same-build protected control finished at `32.00%` (`64/200`)
- the convergent-stage probe collapsed to `24.50%` (`49/200`)
- stage-1 view accuracy collapsed:
  - protected: left `26.50%`, right `25.00%`
  - probe: left `15.00%`, right `16.50%`
- initial accuracy also collapsed:
  - protected: `28.50%`
  - probe: `19.00%`
- fusion audit alternatives degraded sharply:
  - `interaction_centroid_accuracy`: `29.0%` -> `21.5%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `20.0%`
  - `hemisphere_concat_centroid_accuracy`: `18.5%` -> `11.0%`
- hemisphere purity also degraded:
  - left `mean_topk_purity`: `0.184444` -> `0.136667`
  - right `mean_topk_purity`: `0.185556` -> `0.154444`
- nearest-neighbor similarity jumped almost to identity:
  - left `mean_best_neighbor_similarity`: `0.887603` -> `0.988428`
  - right `mean_best_neighbor_similarity`: `0.887609` -> `0.988703`
- importantly, the raw branch code did not improve:
  - both audited `g10` post-margins were unchanged
  - both audited `g10` post-orientation L2 values were unchanged
  - the new stage boundary changed only the classifier input, and it made that input more collapsed, not more separable

Lesson:

- a materially different stage boundary is necessary, but this particular one is wrong: pooled branch conjunctions over the current branch outputs destroy useful hemisphere discrimination instead of building object-level structure
- the collapse signature matters more than the new code’s novelty: when `topk_purity` falls and best-neighbor similarity rises toward `1.0`, the representation is becoming less useful even if centroid margins do not look catastrophic
- do not spend more CIFAR budget on retuning pooled branch-summary conjunctions; this family is closed in its current form
- if a higher-order intermediate stage is revisited, it must preserve more explicit spatial/topographic structure than this pooled summary code

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this convergent pooled-summary stage

### 47. Topographic Hemisphere Stage With Local Continuity/Junction Features

What we tried:

- kept the protected CIFAR baseline fixed and added a new default-off hemisphere-internal stage boundary in `experiments/retina_classification.cpp`
- unlike Rabbit Hole #46, this stage preserved region order and local neighborhood structure instead of pooling branch summaries
- for each branch with valid retinotopic layout, built a topographic classifier-side map from:
  - per-region orientation energy residuals
  - local continuity terms along the preferred orientation
  - local junction terms from adjacent orientations at the same region
  - per-region auxiliary summaries
- exposed the mechanism without mutating the protected config:
  - `--hemisphere-topographic-stage-enabled`
  - `--hemisphere-topographic-residual-gain`
  - `--hemisphere-topographic-continuity-gain`
  - `--hemisphere-topographic-junction-gain`
  - `--hemisphere-topographic-auxiliary-gain`
- evaluated under the same-build strict audited smoke gate:
  - protected control: `build/cifar10_convergent_stage_baseline_flow_audit_20_200.log`
  - protected summary: `build/flow_audit_convergent_stage_baseline_testing_summary.json`
  - topographic-stage probe: `build/cifar10_topographic_stage_flow_audit_20_200.log`
  - topographic-stage summary: `build/flow_audit_topographic_stage_testing_summary.json`
- bounded probe flags:
  - `--hemisphere-topographic-stage-enabled --hemisphere-topographic-residual-gain 0.35 --hemisphere-topographic-continuity-gain 0.60 --hemisphere-topographic-junction-gain 0.35 --hemisphere-topographic-auxiliary-gain 0.25`

Why it was not productive enough:

- the same-build protected control finished at `32.00%` (`64/200`)
- the topographic-stage probe still regressed badly to `25.00%` (`50/200`)
- stage-1 view accuracy stayed above the pooled-summary collapse, but still regressed materially:
  - protected: left `26.50%`, right `25.00%`
  - probe: left `22.50%`, right `17.00%`
- initial accuracy also regressed:
  - protected: `28.50%`
  - probe: `22.00%`
- fusion proxies remained far below the protected surface:
  - `interaction_centroid_accuracy`: `29.0%` -> `23.0%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `23.0%`
  - `hemisphere_concat_centroid_accuracy`: `18.5%` -> `17.5%`
- hemisphere purity still regressed:
  - left `mean_topk_purity`: `0.184444` -> `0.167778`
  - right `mean_topk_purity`: `0.185556` -> `0.159444`
- nearest-neighbor similarity was less collapsed than Rabbit Hole #46 but still too high:
  - left `mean_best_neighbor_similarity`: `0.887603` -> `0.974387`
  - right `mean_best_neighbor_similarity`: `0.887609` -> `0.974631`
- as with Rabbit Hole #46, the raw branch code itself did not improve:
  - both audited `g10` post-margins were unchanged
  - both audited `g10` post-orientation L2 values were unchanged

Lesson:

- preserving topography is necessary, but this specific local continuity/junction transform still compresses the hemisphere code into an over-similar classifier surface
- the fact that the raw branch audits stayed unchanged while the hemisphere/fusion metrics regressed means the failure is in the new stage construction itself, not in upstream branch activity
- do not spend more CIFAR budget on retuning this exact local-continuity/junction stage family
- if a higher-order stage is revisited again, it likely needs explicit multi-scale topographic structure or learned sparse selection rather than another fixed local transform over the current branch outputs

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this topographic local-continuity/junction stage

### 48. Classifier-Side Figure-Ground Residual Append

What we tried:

- kept the protected CIFAR baseline fixed and used the new explicit figure-ground scaffold only as a default-off classifier-side residual append in `experiments/retina_classification.cpp`
- left the raw hemisphere pattern intact and appended:
  - the normalized explicit figure-ground state pattern
  - a small normalized figure-ground summary vector
- exposed the mechanism without mutating the protected config:
  - `--figure-ground-stage-enabled`
  - `--figure-ground-classifier-enabled`
  - `--figure-ground-classifier-gain 0.35`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_figure_ground_classifier_control_flow_audit_20_200.log`
  - control summary: `flow_audit_figure_ground_classifier_control_testing_summary.json`
  - probe: `build/cifar10_figure_ground_classifier_probe_flow_audit_20_200.log`
  - probe summary: `flow_audit_figure_ground_classifier_probe_testing_summary.json`

Why it was not productive enough:

- the same-build control stayed on the protected surface at `32.00%` (`64/200`)
- the classifier-side figure-ground probe collapsed immediately to `10.00%` (`20/200`)
- both hemispheres collapsed onto the same single class:
  - hemisphere agreement: `62.50%` -> `100.00%`
  - stage-1 view accuracy: left `26.50%` -> `10.00%`, right `25.00%` -> `10.00%`
  - initial accuracy: `28.50%` -> `10.00%`
  - post-correction accuracy: `32.00%` -> `10.00%`
- the fusion audit alternatives also collapsed:
  - `interaction_centroid_accuracy`: `29.0%` -> `10.0%`
- hemisphere purity roughly halved:
  - left `mean_topk_purity`: `0.184444` -> `0.0933333`
  - right `mean_topk_purity`: `0.185556` -> `0.0933333`
- replay could not rescue the collapse:
  - corrected samples: `7` -> `0`
  - the probe predicted `bird` for every test sample

Lesson:

- the explicit figure-ground state is stable across fixations, but naive classifier-side concatenation is not enough; it overwhelms the current distance space and destroys class structure
- the failure is not upstream branch silence: the raw branch audits stayed on the same surface while the classifier space collapsed
- do not spend more CIFAR budget on gain-retuning this exact raw-plus-figure-ground append
- if figure-ground is promoted later, it likely needs gated integration, sparse selection, or a dedicated object-level memory boundary rather than direct concatenation into the current hemisphere classifier vector

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this direct figure-ground residual append

### 49. Same-Dimensional Figure-Ground Mask On The Raw Hemisphere Code

What we tried:

- kept the protected CIFAR baseline fixed and used the explicit figure-ground scaffold only as a same-dimensional multiplicative mask on the existing hemisphere pattern
- for each branch with valid retinotopic layout, derived a per-region weight from the explicit figure-ground state and reweighted the original branch values in place
- exposed the mechanism without mutating the protected config:
  - `--figure-ground-mask-enabled`
  - `--figure-ground-mask-gain 0.35`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_figure_ground_mask_control_flow_audit_20_200.log`
  - control summary: `flow_audit_figure_ground_mask_control_testing_summary.json`
  - probe: `build/cifar10_figure_ground_mask_probe_flow_audit_20_200.log`
  - probe summary: `flow_audit_figure_ground_mask_probe_testing_summary.json`

Why it was not productive enough:

- the same-build control stayed on the protected surface at `32.00%` (`64/200`)
- the figure-ground mask probe stayed sane, but still regressed to `31.00%` (`62/200`)
- the best metrics moved in opposite directions:
  - initial accuracy improved slightly: `28.50%` -> `29.00%`
  - left `mean_topk_purity` improved slightly: `0.184444` -> `0.190000`
  - both `mean_best_neighbor_similarity` values improved slightly:
    - left `0.887603` -> `0.881953`
    - right `0.887609` -> `0.881736`
- but the gate-critical metrics got worse:
  - final accuracy: `32.00%` -> `31.00%`
  - corrected samples: `7` -> `4`
  - right stage-1 view accuracy: `25.00%` -> `19.50%`
  - `interaction_centroid_accuracy`: `29.0%` -> `28.0%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `27.0%`
- the result is therefore mixed but non-promotable: it preserves the class space far better than Rabbit Hole #48, but it still does not clear the protected smoke gate or improve the fusion surface

Lesson:

- same-dimensional figure-ground gating is much safer than direct concatenation, but this simple multiplicative mask is still too blunt
- the slight gain in left hemisphere purity and the lower neighbor similarity suggest the figure-ground signal is not useless
- however, hurting right-view accuracy and fusion confidence means this mask does not yet isolate object structure cleanly enough to justify a larger gate
- do not spend a `50/500` gate on gain-retuning this exact raw-pattern mask
- if figure-ground is revisited again, it should likely be through a more selective object-memory or gated-association boundary rather than direct per-region scaling of the whole hemisphere vector

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this direct same-dimensional figure-ground mask

### 50. Separate Figure-Ground Object-Memory Vote

What we tried:

- kept the protected CIFAR baseline fixed and used the explicit figure-ground scaffold only to build a separate object-memory vote, leaving the main hemisphere classifier pattern untouched
- for each hemisphere training pattern, built a sparse figure-dominant pattern by keeping only the top `30%` of figure-ground-signaled regions per retinotopic branch and stored that as a separate exemplar bank
- at inference time, classified that sparse figure-ground object pattern against the separate bank and blended the resulting confidence back into the protected hemisphere confidence with a bounded gain
- exposed the mechanism without mutating the protected config:
  - `--figure-ground-object-memory-enabled`
  - `--figure-ground-object-memory-gain 0.25`
  - `--figure-ground-object-memory-keep-fraction 0.30`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_figure_ground_object_memory_control_flow_audit_20_200.log`
  - control summary: `flow_audit_figure_ground_object_memory_control_testing_summary.json`
  - probe: `build/cifar10_figure_ground_object_memory_probe_flow_audit_20_200.log`
  - probe summary: `flow_audit_figure_ground_object_memory_probe_testing_summary.json`

Why it was not productive enough:

- the same-build control stayed on the protected surface at `32.00%` (`64/200`)
- the figure-ground object-memory probe was live, but still regressed to `30.50%` (`61/200`)
- the new bank was not a dead code path:
  - each hemisphere stored `200` figure-ground object-memory patterns
  - right stage-1 view accuracy improved slightly: `25.00%` -> `26.50%`
  - left stage-1 view accuracy held flat at `26.50%`
- but the gate-critical metrics still got worse:
  - initial accuracy: `28.50%` -> `28.00%`
  - corrected samples: `7` -> `5`
  - final accuracy: `32.00%` -> `30.50%`
  - `interaction_centroid_accuracy`: `29.0%` -> `28.5%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `28.0%`
- the most important audit signal was that the separate vote did not actually change the hemisphere neighbor structure:
  - left `mean_topk_purity`: unchanged at `0.184444`
  - right `mean_topk_purity`: unchanged at `0.185556`
  - left `mean_best_neighbor_similarity`: unchanged at `0.887603`
  - right `mean_best_neighbor_similarity`: unchanged at `0.887609`
- this means the object-memory bank was active but did not create new usable separability beyond the protected hemisphere code

Lesson:

- a separate figure-ground object-memory vote is much safer than direct concatenation and less blunt than whole-pattern masking, but this exact sparse-region exemplar bank is still too redundant or noisy to improve the protected classifier surface
- the unchanged purity and neighbor metrics matter more than the slight right-view lift: the new bank did not actually reshape the local class geometry
- do not spend a `50/500` gate on gain-retuning or keep-fraction retuning this exact object-memory vote
- if figure-ground is revisited again, it likely needs a materially different object-level representation or confusion-targeted selective association step, not another parallel vote over sparsified raw regions

Status:

- documented and closed for promotion on the current CIFAR surface
- keep the opt-in code path as reference only
- do not spend a `50/500` gate on this separate figure-ground object-memory vote

## Rabbit Hole #51: Bounded Recurrent Sensory-State Settling

What we tried:

- implemented a default-off recurrent sensory-state path in `experiments/retina_classification.cpp`
- when enabled, stage-1 storage no longer keeps the raw extracted feature vector; it stores a settled state built from:
  - multi-fixation extraction even when the protected baseline uses `saccade_training_only=on`
  - feedforward retinal drive
  - recurrent self-state persistence across update cycles
  - figure-ground-derived same-dimensional drive
  - fixation-to-fixation continuity from the previous settled state
  - early callosal support mixed before the settled state is stored/classified
- routed inference through the same settled-state path, so replay and testing both operate on the new state representation instead of only patching readout-time behavior
- exposed the bounded probe with:
  - `--recurrent-sensory-state-enabled`
  - `--recurrent-sensory-cycles 3`
  - `--recurrent-sensory-feedforward-gain 0.55`
  - `--recurrent-sensory-state-gain 0.35`
  - `--recurrent-sensory-figure-ground-gain 0.25`
  - `--recurrent-sensory-continuity-gain 0.20`
  - `--recurrent-sensory-callosal-gain 0.15`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_recurrent_sensory_control_samebuild_flow_audit_20_200.log`
  - control summary: `flow_audit_recurrent_sensory_control_samebuild_testing_summary.json`
  - probe: `build/cifar10_recurrent_sensory_probe_samebuild_flow_audit_20_200.log`
  - probe summary: `flow_audit_recurrent_sensory_probe_samebuild_testing_summary.json`

Why it was not productive enough:

- the protected same-build control stayed at `32.00%` (`64/200`)
- the recurrent sensory-state probe finished at `31.00%` (`62/200`)
- the new path was live and materially different:
  - elapsed time rose from `81.56s` to `315.66s`
  - recurrent settled-memory construction alone took about `81s` for the `200` stored patterns
  - both training and inference ran through the settled-state path
- but the gate-critical metrics still regressed:
  - initial accuracy: `28.50%` -> `27.50%`
  - final accuracy: `32.00%` -> `31.00%`
  - left stage-1 view accuracy: `26.50%` -> `24.50%`
  - right stage-1 view accuracy: `25.00%` -> `20.50%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `28.0%`
- there were a few encouraging but insufficient signs:
  - `interaction_centroid_accuracy`: `29.0%` -> `29.5%`
  - left `mean_topk_purity`: `0.184444` -> `0.190556`
  - right `mean_topk_purity`: `0.185556` -> `0.186667`
- the more important failure signal is that nearest-neighbor collapse pressure increased:
  - left `mean_best_neighbor_similarity`: `0.887603` -> `0.908979`
  - right `mean_best_neighbor_similarity`: `0.887609` -> `0.908935`
- that means the bounded recurrent path slightly tightened local class clusters while simultaneously making the overall hemisphere representation more self-similar and less useful for the existing readout surface

Lesson:

- this is not the same failure mode as the direct figure-ground append/mask/vote probes; it did not collapse the class space and it did improve a few local purity and interaction metrics
- but it is still not promotable because the current readout remains exemplar/classifier matching over settled vectors, and this exact gain setting weakens both stage-1 view accuracy and final CIFAR accuracy while increasing neighbor similarity
- the main alignment point remains valid: if this family is revisited, the next step should be a more explicit recurrent object-state readout or object-level settling boundary, not another same-surface gain retune over the current classifier space

Status:

- documented as the first bounded recurrent state-settling probe
- not promotable on the current CIFAR surface
- do not spend a `50/500` gate on this exact recurrent gain setting

## Rabbit Hole #52: Recurrent State + Population-Centroid Readout

What we tried:

- kept the bounded recurrent sensory-state path from Rabbit Hole #51
- replaced the recurrent path's hemisphere exemplar-matcher readout with a class-state centroid readout
- for recurrent-path inference and corpus-callosum calibration, classification no longer called the stage-1 matcher against `trainingPatterns`; it read out cosine evidence against class centroids built from the settled recurrent states
- exposed the change with:
  - `--recurrent-population-readout-enabled`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_recurrent_population_readout_control_20_200.log`
  - control summary: `flow_audit_recurrent_population_readout_control_testing_summary.json`
  - probe: `build/cifar10_recurrent_population_readout_probe_20_200.log`
  - probe summary: `flow_audit_recurrent_population_readout_probe_testing_summary.json`

Why it was not productive enough:

- the protected same-build control stayed at `32.00%` (`64/200`)
- the recurrent + population-readout probe regressed harder, to `29.50%` (`59/200`)
- the centroid-style readout changed the error shape substantially, but not in the right direction:
  - stage-1 view accuracy shifted from `26.50%/25.00%` to `29.00%/28.00%`
  - yet fusion degraded badly:
    - `interaction_centroid_accuracy`: `29.0%` -> `23.0%`
    - `confidence_concat_centroid_accuracy`: `30.0%` -> `23.0%`
- hemisphere-local purity did not improve enough to matter:
  - left `mean_topk_purity`: `0.184444` -> `0.188333`
  - right `mean_topk_purity`: `0.185556` -> `0.183333`
- nearest-neighbor collapse pressure stayed high, just like the previous recurrent probe:
  - left `mean_best_neighbor_similarity`: `0.887603` -> `0.908711`
  - right `mean_best_neighbor_similarity`: `0.887609` -> `0.908599`
- centroid margins also stayed worse than control in both hemispheres

Lesson:

- simply swapping the recurrent path from exemplar matching to class-centroid population readout does not solve the current mismatch
- it improves neither the class geometry enough nor the fusion surface enough; instead it amplifies a flatter, more self-similar hemisphere code and makes the final bilateral decision surface worse
- this means the next biologically aligned step is not another matcher-vs-centroid toggle on the same settled state
- if this family is revisited, it needs a genuinely different readout boundary, likely an explicit object-state or attractor-style readout rather than class centroids over the current settled vector

Status:

- documented and closed for promotion on the current CIFAR surface
- do not spend a `50/500` gate on this exact recurrent-population-readout formulation

## Rabbit Hole #53: Recurrent State + Object-State Attractor Readout

What we tried:

- kept the bounded recurrent sensory-state path from Rabbit Hole #51
- replaced the recurrent-path readout with an explicit object-state attractor layer instead of exemplar matching or class centroids
- built a per-hemisphere object-state bank from the settled recurrent training states:
  - `3` attractor units per class
  - `30` attractor units per hemisphere on the strict `20/class` gate
- each test state was read out by:
  - cosine drive into the object-state units
  - recurrent self-persistence
  - same-class support amplification
  - cross-class competition
  - final class vote from the settled object-state units
- exposed the path with:
  - `--recurrent-object-state-readout-enabled`
  - `--recurrent-object-state-units-per-class 3`
  - `--recurrent-object-state-cycles 4`
  - `--recurrent-object-state-input-gain 1.0`
  - `--recurrent-object-state-self-gain 0.35`
  - `--recurrent-object-state-class-support-gain 0.40`
  - `--recurrent-object-state-competition-gain 0.55`
- evaluated under the same-build strict audited smoke gate:
  - control: `build/cifar10_recurrent_object_state_control_20_200.log`
  - control summary: `flow_audit_recurrent_object_state_control_testing_summary.json`
  - probe: `build/cifar10_recurrent_object_state_probe_20_200.log`
  - probe summary: `flow_audit_recurrent_object_state_probe_testing_summary.json`

Why it was not productive enough:

- the protected same-build control stayed at `32.00%` (`64/200`)
- the recurrent + object-state-attractor probe regressed badly to `27.50%` (`55/200`)
- the readout was live:
  - recurrent settled-state storage completed normally
  - each hemisphere built `30` object-state attractors
- the failure mode was broad, not just a mild regression:
  - initial accuracy: `28.50%` -> `20.50%`
  - final accuracy: `32.00%` -> `27.50%`
  - left/right stage-1 accuracy: `26.50%/25.00%` -> `23.00%/19.50%`
  - `interaction_centroid_accuracy`: `29.0%` -> `18.0%`
  - `confidence_concat_centroid_accuracy`: `30.0%` -> `18.5%`
- nearest-neighbor collapse pressure again stayed high:
  - left `mean_best_neighbor_similarity`: `0.887603` -> `0.908604`
  - right `mean_best_neighbor_similarity`: `0.887609` -> `0.908502`
- the resulting confusion structure showed broad truck-dominant collapse on multiple classes, meaning the attractor competition did not discover a cleaner object-level decision surface

Lesson:

- adding an explicit attractor-style readout boundary is directionally closer to the biological goal, but this first formulation still operates on a class-labeled prototype bank over the same settled state, and it made both hemisphere and fusion behavior worse
- the broad degradation means the current attractor dynamics are too coarse or too label-tied to serve as a useful object-state layer
- if this area is revisited, the next step should not be a mild retune of gains or unit count; it needs a materially different object-state representation or a different upstream state geometry before attractor competition

Status:

- documented and closed for promotion on the current CIFAR surface
- do not spend a `50/500` gate on this exact recurrent-object-state-attractor readout

## Reusable Work Worth Keeping

Not everything from the failed path was wasted. These pieces are reusable and should be retained:

- exact path-backed declarative group resolution in `NetworkConstructor`
- wildcard/exact hierarchy path support for layer and population resolution
- incoming-drive-aware competition scoring in `CompetitionManager`
- bilateral Retina declarative hierarchy with `attach_path` and `fusion_path`
- bilateral continuous-learning benchmark path

These improve the framework or the benchmark surface without committing us to the failed graph path.

## Rules For Revisiting This Area

Do not reopen the multistage graph path unless at least one of these is true:

- there is a new runtime with native support for the declared laminar computation
- there is new instrumentation showing meaningful class separation at `higher -> readout`
- there is a new biologically motivated representation stage with a concrete reason to expect separation gains

If we do revisit it, the order should be:

1. verify propagation and timing
2. measure stage-wise separability
3. improve representation
4. only then tune decision/readout dynamics

## Practical Default

For current vision work:

- use the bilateral Retina static and continuous paths as the mainline benchmark
- use the CIFAR natural-features path as the current natural-image reference, not as proof that the edge path is solved
- treat graph-native cortical vision as experimental research, not the default execution path
