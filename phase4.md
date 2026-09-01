# PHASE 4 FINAL MASTER TRAINING DIRECTIVE
## Fresh Full-Scale Training, Finalist Bake-Off, Calibration, FP/FN Optimization & Frozen Evaluation

CONTROLLING DOCUMENTS
---------------------

Read and obey:

1. `AUTH_PHASE1.md`
2. `docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md`
3. All Phase 4 reports under `reports/`
4. The completed Phase 3 reports and frozen Phase 2 baseline provenance

This document is the controlling execution directive for the final
Phase-4 training process.

Do NOT rely on narrative summaries when numerical data are available.
Read the machine-readable JSON.

Do NOT silently alter the scientific protocol.

Do NOT invent results.

Do NOT reuse stale predictions or stale feature arrays.

Do NOT silently reuse old fusion weights.

Frozen pretrained foundation weights are allowed.

Fresh derived features/predictions for this experiment must be generated
from raw images.

============================================================
0. PRIMARY OBJECTIVE
============================================================

The objective is NOT simply:

    maximize AUROC

The objective is:

    EXTREMELY LOW FP
    +
    EXTREMELY LOW FN
    +
    HIGH TPR
    +
    HIGH AUROC/AUPRC
    +
    STRONG GENERATOR/OOD GENERALIZATION
    +
    ROBUSTNESS UNDER REALISTIC TRANSFORMATIONS
    +
    GOOD CALIBRATION
    +
    PRACTICAL RTX 3050 PERFORMANCE

Definitions:

    REAL -> REAL = TN
    REAL -> AIGC = FP
    AIGC -> REAL = FN
    AIGC -> AIGC = TP

Priority:

1. Keep FP/FPR extremely low.
2. Preserve as much TPR as possible.
3. Reduce FN/FNR without causing an unacceptable FP increase.
4. Preserve robustness and OOD generalization.
5. Preserve practical latency/VRAM.

Do NOT claim "zero false positives" from zero observed FP in a finite
sample.

Always report:

    FP / N_real

and, where practical, an uncertainty interval.

============================================================
1. IMPORTANT PHASE-3 / PHASE-4 RECONCILIATION
============================================================

Before training, read:

    reports/phase4_phase3_reconciliation.json
    reports/phase4_phase3_reconciliation.md

The historical contradiction involving:

    186 Phase-2 errors
    249 Phase-3 Candidate-B errors
    "negative error reduction"

has already been explained.

Treat the Phase-2 82.5K model as the frozen historical baseline.

Do NOT interpret the old Phase-3 20K experiments as evidence that the
smaller-data candidates beat the full Phase-2 model.

The correct conclusion is:

    data scale materially affected the result.

============================================================
2. IMPORTANT: DO NOT BLINDLY ACCEPT THE PHASE-4 AUTHORIZATION ARCHITECTURE
============================================================

The current reports contain a methodological inconsistency.

The Phase-4 micro-challenge ranks:

    Cand_C_CLIP_SigLIP_Edge

as the strongest tested candidate at the stated operating point.

Its reported results are approximately:

    AUROC = 0.9971
    FPR @ 0.80 = 1.51%
    TPR @ 0.80 = 97.10%
    FP = 64
    FN = 176
    total errors = 240

Yet the authorization report recommends:

    CLIP + SigLIP + DINO + SRM + Edge

without presenting that exact architecture as the empirical winner of
the same micro-challenge.

Therefore:

DO NOT blindly train the recommended Forensic Quad architecture.

Instead conduct a controlled full-scale finalist bake-off.

Let fresh data decide.

============================================================
3. VALIDATION-GOVERNANCE CORRECTION
============================================================

The previously used:

    PHASE2_VAL

has been repeatedly used for:

- architecture comparison
- threshold analysis
- calibration experiments
- error analysis
- complementarity analysis
- Phase-3 model selection
- Phase-4 micro-challenge

Therefore treat PHASE2_VAL as:

    HISTORICAL DEVELOPMENT EVIDENCE

NOT as a pristine final development evaluation.

Do not discard it.
Do not pretend it has never been observed.

Most importantly:

CREATE A NEW PRISTINE FINAL-DEVELOPMENT SPLIT.

============================================================
4. CREATE A PRISTINE FINAL DEVELOPMENT SET
============================================================

Using the complete approved corpus currently available:

    103,137 approved samples

with the currently locked:

    10,316 internal test samples

construct a new:

    FINAL_DEV
    FINAL_CALIBRATION
    FINAL_TRAIN

partition.

The new FINAL_DEV and FINAL_CALIBRATION images must come from images
that were NOT previously used as:

- Phase-3 validation data
- Phase-4 micro-challenge validation data
- previous threshold-selection data
- previous architecture-selection data
- previous calibration data

The safest implementation is:

    identify all previously exposed experiment image IDs
    exclude them from pristine-dev selection
    cryptographically prove the resulting split

Do not merely rename the old validation split.

Use:

    SHA-256
    +
    path/identity verification
    +
    perceptual duplicate analysis where practical

Target minimum scale:

    FINAL_DEV         >= 5,000 images
    FINAL_CALIBRATION >= 2,500 images

Prefer larger sets if the clean-data inventory allows it.

Maintain meaningful diversity in both.

Report:

- total count
- real count
- AIGC count
- generator families
- real domains
- resolutions
- source datasets

============================================================
5. INTERNAL TEST REMAINS LOCKED
============================================================

The current internal test:

    PHASE2_INTERNAL_TEST
    ~10K images

must remain untouched until the final candidate has been frozen.

Do NOT use it for:

- architecture selection
- fusion selection
- loss selection
- threshold selection
- calibration
- hyperparameter selection
- hard-negative mining
- early stopping

It is a FINAL HOLDOUT.

============================================================
6. EXTERNAL OOD REMAINS LOCKED
============================================================

Do NOT access for tuning:

- Synthbuster
- AIGIBench
- Chameleon
- VCT²
- WildRF
- SynthWildX
- official locked hackathon validation

They may only be evaluated after the final candidate is completely
frozen.

Run each approved OOD benchmark once for the final evaluation.

============================================================
7. CURRENT PHASE-4 DATA INVENTORY
============================================================

The currently audited approved corpus contains approximately:

    103,137 images

with:

    42,369 REAL
    60,768 AIGC

and source families including:

    WikiArt
    authentic photography
    COCO
    synthetic general corpus
    Quality Paradox modern diffusion
    SID diffusion
    HFCF
    FLUX/SD3-modern material where legitimately represented

Do NOT blindly enforce 50/50 physical storage composition.

The physical corpus and sampled training distribution may differ.

Record both.

============================================================
8. TRAINING DISTRIBUTION
============================================================

Use generator-aware and domain-aware sampling.

The sampler must prevent:

- one generator dominating
- one real domain dominating
- HFCF shortcut learning
- easy-generator dominance
- excessive repetition of a narrow subset

Retain increased exposure to:

AIGC:

- SID low-artifact diffusion
- Quality Paradox modern diffusion
- subtle photorealistic AIGC
- modern generators
- low-step diffusion
- post-processed AIGC
- compressed/downsampled AIGC

REAL:

- COCO
- natural photography
- studio photography
- difficult bokeh
- macro photography
- strong legitimate textures
- compression
- optical blur
- unusual lighting
- high-frequency camera imagery
- artwork

The current hard-example findings must be explicitly represented in the
sampling strategy.

Do not fabricate weights.

Test the proposed Strategy-E weighting on fresh validation.

============================================================
9. PHASE-4 FINALIST ARCHITECTURE BAKE-OFF
============================================================

Do NOT immediately commit to one architecture.

Freshly train small fusion heads using the PRISTINE_FINAL_DEV protocol.

Test at minimum:

A:
    CLIP + SigLIP

B:
    CLIP + SigLIP + SRM

C:
    CLIP + SigLIP + Edge

D:
    CLIP + SigLIP + DINO

E:
    CLIP + SigLIP + SRM + Edge

F:
    CLIP + SigLIP + SRM + DINO

G:
    CLIP + SigLIP + DINO + SRM + Edge

H:
    All-9 experts

The purpose of H is scientific verification.

Do NOT assume it will win.

============================================================
10. FUSION METHODS
============================================================

For the serious finalists evaluate:

1. Linear/logistic fusion
2. 2-layer MLP
3. 2-layer MLP + LayerNorm
4. 2-layer MLP + GELU
5. Structured branch dropout
6. Lightweight gated residual fusion

Do NOT create unnecessarily large fusion heads.

Avoid massive parameter counts in the head.

The pretrained foundation models remain frozen initially.

============================================================
11. SPECIALIST ROLE
============================================================

Treat:

CLIP + SigLIP

as the semantic core candidate.

Treat:

SRM
Edge
DINO
ConvNeXt
EVA
FFT
Patch-MIL

as auxiliary evidence candidates.

The central scientific question is:

    Can auxiliary evidence reduce FP/FN without destabilizing
    the strong semantic detector?

Do NOT allow a noisy specialist to dominate the decision.

============================================================
12. FFT
============================================================

The current evidence indicates:

    2D-FFT is approximately non-discriminative as a standalone feature.

Do NOT automatically include it.

However, retain it in the explicitly required all-expert experiment.

If it produces no measurable benefit:

    mark FFT as DROPPED

and explain why.

============================================================
13. PATCH-MIL
============================================================

The current evidence suggests Patch-MIL is mostly redundant.

It may participate in the all-expert challenge.

Do not include it in the final architecture merely because it exists.

============================================================
14. DINO / EVA / CONVNEXT
============================================================

Do not interpret their standalone performance alone.

Evaluate their marginal contribution.

For each, calculate:

    ΔAUROC
    ΔAUPRC
    ΔFPR
    ΔFNR
    ΔTPR
    ΔWorst
    Δrobustness
    Δlatency
    ΔVRAM

relative to the selected semantic core.

============================================================
15. TRAINING DATA SCALE EXPERIMENT
============================================================

The Phase-3 reconciliation established that reducing the fitting set
from ~82.5K to 20K materially degraded performance.

Therefore the final architecture must receive sufficient data.

Before the definitive full-data run, test:

    20K
    40K
    ~80K

or the largest practical equivalent.

Do not repeat the entire experiment unnecessarily if the larger-scale
results clearly establish the trend.

The purpose is to establish whether the architecture benefits from
additional diversity and whether the specialist branches remain useful
when trained at scale.

============================================================
16. FRESH FEATURE EXTRACTION
============================================================

IMPORTANT:

Do NOT reuse stale Phase-2 or Phase-3 feature caches for the final
experiment.

Generate fresh features from raw images.

Each fresh feature cache must be keyed by:

    manifest SHA-256
    backbone checkpoint SHA-256
    preprocessing SHA/version
    feature dimension
    dtype
    extraction code version

Save provenance metadata.

Reuse features ONLY among experiments within THIS SAME PHASE-4 RUN when
the provenance key is identical.

That is legitimate within-run reuse.

Historical feature caches are not legitimate inputs.

============================================================
17. RTX 3050 EXECUTION STRATEGY
============================================================

Hardware:

    RTX 3050 6GB
    ~31GB RAM
    ~24GB swap
    ~400GB NVMe available

Use:

    SOURCE
      ↓
    NVMe STAGING
      ↓
    ASYNCHRONOUS PREFETCH
      ↓
    PINNED HOST RAM
      ↓
    NON-BLOCKING GPU TRANSFER
      ↓
    GPU

Do NOT attempt to hold the entire 400+ GB corpus in RAM.

Do NOT intentionally use swap as a cache.

Do NOT increase swap merely to make the system appear to have more
memory.

Keep:

    sustained swap activity ≈ 0

Use RAM for:

- hot working sets
- page cache
- pinned prefetch buffers
- bounded staging

Use NVMe for:

- feature cache
- active dataset staging
- intermediate data

============================================================
18. MAXIMIZE GPU UTILIZATION
============================================================

Benchmark:

- DataLoader workers
- prefetch factor
- persistent workers
- pinned memory
- asynchronous loading
- batch size
- mixed precision
- NVMe staging layout

Select the configuration based on measured:

    images/sec

not theoretical throughput.

A healthy run must show:

- GPU utilization
- progress in processed images
- feature cache growth
- loss/metric changes
- stable VRAM
- stable RAM
- zero sustained swap I/O

============================================================
19. TRAINING EFFICIENCY
============================================================

Foundation backbones:

    FROZEN

initially.

Train only:

    fusion head
    calibration parameters
    gating parameters
    any explicitly approved lightweight adapters

Do not fine-tune all foundation backbones initially.

If the final evidence suggests frozen features have reached a ceiling,
perform a separate small controlled adapter experiment.

Do NOT immediately fine-tune the billion-parameter backbones.

============================================================
20. LOSS FUNCTION
============================================================

Use:

    asymmetric binary cross entropy

with:

    lambda_FP = 2.0

as the baseline.

Do NOT blindly assume it is globally optimal.

Run a small controlled comparison:

    lambda_FP = 1.5
    lambda_FP = 2.0
    lambda_FP = 2.5

Use PRISTINE_FINAL_DEV only.

Do not perform a giant hyperparameter grid.

The selected lambda must balance:

    FPR
    TPR
    FNR
    calibration

rather than merely minimizing FPR.

============================================================
21. HARD-NEGATIVE / HARD-POSITIVE CURRICULUM
============================================================

After the first full-scale training cycle:

identify:

REAL with highest P(AIGC)

and:

AIGC with lowest P(AIGC)

Analyze:

- source
- generator
- domain
- resolution
- compression
- expert disagreement

Do NOT mine from internal test or OOD.

Create an optional hard-example weighting stage.

Give special attention to:

REAL:
    COCO macro photography
    strong bokeh
    studio flash
    high-frequency legitimate textures

AIGC:
    SID low-step diffusion
    low-artifact diffusion
    Quality Paradox photorealism
    modern subtle generators

============================================================
22. CALIBRATION
============================================================

Do NOT simply reuse:

    T = 1.2526

from the previous experiment.

Generate fresh calibration predictions.

Compare:

1. Temperature scaling
2. Platt scaling
3. Isotonic regression only if sample size supports it

Measure:

    ECE
    Brier
    calibration curves

The selected calibration must be fitted ONLY on FINAL_CALIBRATION.

============================================================
23. THRESHOLD OPTIMIZATION
============================================================

Do NOT hard-code:

    tau = 0.80

as the final threshold.

Generate a dense threshold curve.

At minimum:

    0.50
    0.55
    0.60
    0.65
    0.70
    0.75
    0.80
    0.85
    0.90
    0.92
    0.94
    0.95
    0.96
    0.97
    0.98
    0.99

Also search the exact empirical thresholds necessary to obtain:

    FPR <= 5%
    FPR <= 2%
    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%

Report:

    threshold
    TP
    TN
    FP
    FN
    FPR
    FNR
    TPR
    TNR
    precision
    recall
    F1

============================================================
24. DEPLOYMENT POLICY
============================================================

Test whether a three-way decision is superior to forced binary output:

    HIGH-CONFIDENCE REAL
    REVIEW / ABSTAIN
    HIGH-CONFIDENCE AIGC

The review band must be learned from validation/calibration evidence.

Do NOT invent a review band just because one was used before.

If abstention reduces catastrophic FP/FN, quantify:

    coverage
    FP reduction
    FN reduction
    retained TPR

============================================================
25. ROBUSTNESS
============================================================

Evaluate serious finalists under:

    Clean
    JPEG Q90
    JPEG Q70
    JPEG Q50
    JPEG Q30
    Gaussian Blur sigma=1
    Gaussian Blur sigma=2
    resize 0.75x
    resize 0.50x
    noise std=0.05
    noise std=0.10
    crop
    color brightness
    color contrast
    sharpening
    social-media recompression

For every condition report:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    TNR

Also report:

    mean RI
    worst-case AUROC
    clean-to-worst degradation

============================================================
26. GENERATOR GENERALIZATION
============================================================

Break down AIGC detection by generator family.

Where sufficient samples exist report separately:

    FLUX
    SD3
    SDXL
    Midjourney
    PixArt
    SID
    HFCF
    Quality Paradox
    other approved families

Also report REAL FPR by domain:

    COCO
    general photography
    WikiArt
    other authentic sources

The final report must identify potential shortcut learning.

============================================================
27. OOD
============================================================

Only after the final candidate is frozen:

evaluate:

    Synthbuster
    AIGIBench
    Chameleon
    VCT²
    WildRF
    SynthWildX

where legitimately available and approved.

Run these evaluations ONCE.

Do NOT retrain after seeing OOD results.

OOD performance is evidence, not a tuning signal.

============================================================
28. INTERNAL TEST
============================================================

Only after:

    architecture frozen
    fusion frozen
    loss frozen
    calibration frozen
    threshold frozen
    review policy frozen

evaluate the locked internal test ONCE.

Do not make any post-test changes.

Report exactly:

    N_real
    N_AIGC
    TP
    TN
    FP
    FN
    FPR
    FNR
    TPR
    TNR
    AUROC
    AUPRC
    ECE
    Brier

Do not optimize after seeing it.

============================================================
29. FULL-SCALE TRAINING DATA STRATEGY
============================================================

For the CURRENT 103,137-sample approved corpus:

Use all eligible non-test data efficiently.

Do NOT discard large quantities simply to force an arbitrary 50/50
physical manifest.

Instead use:

    generator-aware sampler
    domain-aware sampler
    FP/FN-aware sampling

The final training manifest should contain the largest scientifically
useful eligible corpus.

The locked internal test remains excluded.

The fresh FINAL_DEV and FINAL_CALIBRATION subsets remain excluded from
training.

============================================================
30. LONG-TERM 400–600+ GB CORPUS EXPANSION
============================================================

This Phase-4 model should establish the final scalable training pipeline.

Do NOT assume 103K samples are the ultimate ceiling.

After the 103K controlled full-scale training succeeds, the SAME
pipeline should be capable of expanding to the larger approved
400–600+ GB corpus.

For the expanded corpus:

    inventory
    deduplicate
    verify labels
    stratify generators
    stratify real domains
    isolate validation
    isolate calibration
    isolate test
    stage on NVMe
    train with generator-aware sampling

Do not simply dump every image into one uncontrolled pool.

The purpose of larger data is:

    more generator diversity
    more difficult real images
    more subtle AIGC
    more post-processing diversity
    better generalization

NOT merely a larger number of near-duplicates.

============================================================
31. EXPERIMENT CHECKPOINTS
============================================================

Long jobs must checkpoint:

    model
    optimizer
    scheduler
    epoch
    step
    sampler state
    RNG state
    manifest hash
    architecture
    preprocessing
    loss
    calibration configuration

Use atomic writes.

Never overwrite the only known-good checkpoint.

============================================================
32. LIVE EXECUTION AUDIT
============================================================

A process existing is not evidence that training is healthy.

Periodically verify:

    process state
    CPU time
    GPU utilization
    VRAM
    RAM
    swap
    NVMe throughput
    processed samples
    cache size
    loss
    validation metrics
    ETA

If GPU utilization collapses:

    diagnose I/O first.

If RAM begins swapping:

    reduce prefetch/cache buffers.

If VRAM OOM occurs:

    reduce batch size or representation footprint.

Do not silently change architecture.

============================================================
33. REQUIRED MACHINE-READABLE REPORTS
============================================================

Produce:

    reports/phase4_final_dev_manifest.json
    reports/phase4_final_dev_integrity.json
    reports/phase4_final_calibration_manifest.json
    reports/phase4_fresh_data_provenance.json

    reports/phase4_fullscale_architecture_bakeoff.json
    reports/phase4_fullscale_fusion_comparison.json
    reports/phase4_fullscale_loss_comparison.json

    reports/phase4_final_feature_cache_integrity.json
    reports/phase4_final_training_telemetry.json

    reports/phase4_final_fp_fn_forensics.json
    reports/phase4_final_calibration.json
    reports/phase4_final_threshold_analysis.json
    reports/phase4_final_robustness.json
    reports/phase4_final_generator_breakdown.json
    reports/phase4_final_domain_breakdown.json

    reports/phase4_final_internal_test.json
    reports/phase4_final_ood_results.json

    reports/phase4_final_training_report.json
    reports/phase4_final_training_report.md

Also update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

with the actual measured results and provenance.

============================================================
34. FINAL ARCHITECTURE DECISION RULE
============================================================

The final architecture is NOT predetermined.

However, the current empirical hypotheses are:

    CLIP + SigLIP
        = semantic core candidate

    SRM
        = forensic residual candidate

    Edge
        = useful high-pass/local structural candidate

    DINO
        = structural/geometry candidate

    ConvNeXt
        = possible pixel-grid complement

    EVA
        = potentially useful but expensive

    FFT
        = currently weak/redundant

    Patch-MIL
        = currently weak/redundant

These are hypotheses, not commands.

The final architecture must be selected from fresh full-scale evidence.

============================================================
35. ALL-9 EXPERIMENT
============================================================

The all-9 architecture MUST remain an experimental candidate.

Do NOT suppress its evaluation.

But do NOT force it into deployment.

If all-9 wins:

    keep it.

If all-9 loses:

    document that clearly.

The scientific objective is:

    find the smallest set of experts that captures the useful evidence.

Not:

    use every model because every model exists.

============================================================
36. MULTI-OBJECTIVE RANKING
============================================================

Rank finalists using:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    worst-case robustness
    OOD performance
    ECE
    Brier
    latency
    VRAM
    parameter count

However:

DO NOT allow a single composite score to hide catastrophic FPR/FNR.

Always display the underlying metrics.

A candidate with slightly better AUROC but substantially worse FPR
should not automatically win.

============================================================
37. STATISTICAL HONESTY
============================================================

For every result distinguish:

    training
    development
    calibration
    internal-test
    OOD

Never mix these.

Never say:

    "test accuracy"

when referring to validation.

Never call an empirical threshold universally optimal.

Never claim:

    "near-zero probability of FP"

from finite observations.

Never treat model disagreement as proof of complementarity.

Never treat correlation alone as proof of redundancy.

============================================================
38. FINAL DECISION REPORT
============================================================

At the end produce a single authoritative report containing:

    FINAL ARCHITECTURE
    FINAL EXPERT BRANCHES
    FINAL FEATURE DIMENSIONS
    FINAL FUSION MECHANISM
    FINAL TRAINABLE PARAMETERS

    FINAL LOSS
    FINAL LAMBDA_FP
    FINAL REGULARIZATION

    FINAL CALIBRATION METHOD
    FINAL TEMPERATURE / PARAMETERS

    FINAL THRESHOLD
    FINAL ABSTENTION / REVIEW POLICY

    FINAL TRAINING DATA SIZE
    FINAL REAL/AIGC DISTRIBUTION
    FINAL GENERATOR DISTRIBUTION

    FINAL VALIDATION DESIGN
    FINAL CALIBRATION DESIGN
    FINAL TEST DESIGN

    FINAL LATENCY
    FINAL VRAM
    FINAL RAM
    FINAL THROUGHPUT

    FINAL INTERNAL TEST METRICS
    FINAL OOD METRICS

    REMAINING FAILURE MODES
    REMAINING RISKS
    RECOMMENDED NEXT EXPERIMENT

============================================================
39. REQUIRED FINAL COMPARISON
============================================================

The final report MUST include a table comparing:

    Phase 2 frozen baseline
    best Phase-4 candidate
    all-9 candidate
    best compact candidate

At minimum:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    worst AUROC
    RI
    ECE
    Brier
    latency
    VRAM
    trainable params

Also show:

    FP delta
    FN delta
    net error delta

============================================================
40. HARD STOP CONDITIONS
============================================================

STOP immediately if:

- validation leakage appears
- internal-test contamination appears
- OOD contamination appears
- stale cache reuse appears
- manifest SHA changes unexpectedly
- labels conflict
- duplicate contamination appears
- numerical instability appears
- sustained swap thrashing appears
- repeated GPU OOM occurs
- checkpoint integrity fails
- metrics cannot be reproduced
- reported numbers contradict raw predictions
- the final architecture violates compute limits

Do NOT silently repair a failed gate.

Explain the failure.

============================================================
41. FULL-SCALE TRAINING AUTHORIZATION LOGIC
============================================================

The current instruction authorizes:

    data reconstruction
    fresh development split
    finalist bake-off
    fresh feature extraction
    full-scale head training
    calibration
    threshold analysis
    robustness analysis

It DOES NOT authorize silently changing the scientific protocol.

Once the finalist bake-off and fresh-data integrity gates pass, proceed
automatically with the full-scale current-corpus training.

Do NOT stop merely because a previous report said "human authorization"
if this directive explicitly authorizes the next controlled stage.

However:

Do NOT proceed to the 400–600+ GB expanded-corpus training until the
current-corpus final report demonstrates that the pipeline is stable and
the selected architecture genuinely earns its place.

============================================================
42. FINAL SCIENTIFIC PRINCIPLE
============================================================

Do not optimize for impressive numbers.

Do not optimize for the easiest generator.

Do not optimize for the easiest validation set.

Do not optimize for zero FP at the cost of enormous FN.

Do not optimize for zero FN at the cost of accusing REAL images.

Do not use more models merely because they are available.

Do not use fewer models merely for elegance.

USE EVIDENCE.

The detector should learn:

    semantic evidence
    structural evidence
    forensic evidence
    generator diversity
    authentic-domain diversity
    transformation robustness

while avoiding:

    dataset shortcuts
    generator shortcuts
    compression shortcuts
    overfitting
    validation leakage
    stale derived data
    unnecessary model complexity

The ultimate target is:

    VERY LOW FPR
    +
    VERY LOW FNR
    +
    HIGH TPR
    +
    HIGH OOD GENERALIZATION
    +
    HIGH ROBUSTNESS
    +
    GOOD CALIBRATION
    +
    PRACTICAL RTX 3050 INFERENCE

============================================================
43. BEGIN
============================================================

BEGIN WITH:

    STEP 0:
    Read all Phase-4 reports and perform the final numerical /
    methodological reconciliation.

THEN:

    STEP 1:
    Construct and cryptographically verify the pristine
    FINAL_DEV and FINAL_CALIBRATION sets.

THEN:

    STEP 2:
    Generate fresh features from RAW images.

THEN:

    STEP 3:
    Run the full-scale finalist architecture bake-off.

THEN:

    STEP 4:
    Select the empirically strongest finalist.

THEN:

    STEP 5:
    Train that finalist at full current-corpus scale.

THEN:

    STEP 6:
    Perform fresh calibration and threshold optimization.

THEN:

    STEP 7:
    Perform robustness and generator/domain analysis.

THEN:

    STEP 8:
    Freeze the candidate.

THEN:

    STEP 9:
    Evaluate the locked internal test ONCE.

THEN:

    STEP 10:
    Evaluate locked OOD benchmarks ONCE.

THEN:

    STEP 11:
    Produce the final machine-verifiable report.

THEN:

    STOP and preserve all artifacts.

============================================================
NO HALLUCINATION / NO CHEATING / NO STALE DATA
============================================================

EVERY important reported number must be traceable to:

    actual prediction arrays
    actual labels
    actual manifest
    actual checkpoint
    actual experiment configuration

If a statement cannot be demonstrated:

    WRITE "NOT ESTABLISHED."

Never fill a missing result with an expectation.

Never convert an expectation into a measured result.

Never hide a worse result.

Never overwrite an old result merely because it is unfavorable.

Never silently change the protocol.

BEGIN.