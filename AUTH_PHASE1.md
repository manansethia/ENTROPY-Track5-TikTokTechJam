============================================================
MASTER AUTHORIZATION PROMPT
PHASE 1 → LARGE-SCALE TRAINING
ACCURACY + ROBUSTNESS + LOW FP/FN + COMPUTATIONAL EFFICIENCY
============================================================

READ THIS ENTIRE PROMPT BEFORE EXECUTING ANY TRAINING.

This is the authoritative execution instruction for the next phase of
the AIGC detection research project.

The objective is NOT merely to obtain a high validation AUROC.

The objective is to build the most accurate, robust and efficient
practical AIGC detector possible under the available hardware:

GPU:
    NVIDIA RTX 3050 6 GB VRAM

HOST:
    ~31–32 GB physical RAM
    ~24 GB existing swap
    ~397 GB available NVMe staging/storage

DATA:
    50K Phase-1 dataset immediately available
    ~400+ GB approved corpus available for subsequent expansion

PRIMARY CLASSIFICATION SEMANTICS:

    y = 0 → AUTHENTIC / REAL
    y = 1 → AIGC / FAKE

Therefore:

    TN = Real correctly classified as Real
    FP = Real incorrectly classified as Fake
    FN = AIGC incorrectly classified as Real
    TP = AIGC correctly classified as AIGC

Primary operational concern:

    VERY LOW FP
    VERY LOW FN
    HIGH TNR / specificity
    HIGH TPR / sensitivity
    HIGH AUROC / AUPRC
    GOOD calibration
    STRONG robustness under transformations
    STRONG unseen-generator generalization
    REASONABLE inference latency and VRAM

Do NOT optimize only for accuracy.

============================================================
0. ABSOLUTE DATA / RESULT INTEGRITY RULE
============================================================

The Master Prompt and the current Knowledge Base are authoritative.

However:

DO NOT blindly trust historical metrics.

DO NOT reuse historical prediction arrays.

DO NOT reuse historical classifier weights.

DO NOT reuse historical feature caches.

DO NOT use old train/validation predictions as training data.

DO NOT use previous fusion weights.

DO NOT use previous thresholds as if they were ground truth.

DO NOT silently reproduce previous numerical results.

Pretrained FOUNDATION MODEL WEIGHTS are allowed and expected.

What is forbidden is leakage from previous experiments through
derived predictions, fitted heads, normalization parameters,
feature caches, thresholds or validation-derived artifacts.

Every newly trained head must be demonstrably fitted from the current
approved training data.

If an existing artifact is used for convenience, first prove that it
is a fixed pretrained model component rather than an experiment-derived
artifact.

When uncertain:

    RECOMPUTE FROM RAW DATA.

Scientific correctness is more important than saving computation.

============================================================
1. FIRST: READ THE AUTHORITATIVE PROJECT DOCUMENTS
============================================================

Before launching Phase-1 training:

Read:

    MASTER_PROMPT.MD

and:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Also inspect the current reports relevant to:

    - fresh decision gate
    - data governance
    - distribution analysis
    - sampling strategy
    - infrastructure/I/O benchmark
    - pilot training
    - model/fusion evaluation

Do not overwrite historical reports.

Do not silently reinterpret old results.

Clearly distinguish:

    HISTORICAL RESULT
    CURRENT FRESH RESULT
    NEW TRAINING RESULT

Update the Knowledge Base after every major completed experimental
phase with exact provenance.

============================================================
2. CURRENT EMPIRICAL BASELINE
============================================================

The latest fresh decision gate established approximately:

CLIP-ViT-L:
    Clean AUROC ≈ 0.9783
    RI ≈ 0.9061
    Worst ≈ 0.8244
    FPR ≈ 8.0%

SigLIP:
    Clean AUROC ≈ 0.9737
    RI ≈ 0.9054
    Worst ≈ 0.8193
    FPR ≈ 6.0%

DINOv2:
    Clean AUROC ≈ 0.8711
    RI ≈ 0.8456
    Worst ≈ 0.7993

EVA-02:
    Clean AUROC ≈ 0.9154
    RI ≈ 0.8574
    Worst ≈ 0.7854

ConvNeXt-V2:
    Clean AUROC ≈ 0.8793
    RI ≈ 0.8282
    Worst ≈ 0.7615

Forensic specialists were weaker standalone.

Fresh controlled fusion showed:

CLIP + SigLIP:
    Clean ≈ 0.9857
    RI ≈ 0.9258
    Worst ≈ 0.8420
    FPR ≈ 3.3%

CLIP + SigLIP + DINO:
    Clean ≈ 0.9845
    RI ≈ 0.9346
    Worst ≈ 0.8664
    FPR ≈ 4.0%

These are DEVELOPMENT-GATE results only.

They are NOT deployment guarantees.

Do not assume that these exact weights, thresholds or architecture
must be used during final training.

Use them as evidence.

============================================================
3. ARCHITECTURE PRINCIPLE
============================================================

Do NOT blindly deploy every available expert.

Also do NOT prematurely discard experts merely because their standalone
AUROC is weaker.

The final architecture must be evidence-driven.

The candidate expert pool is:

    CLIP-ViT-L/14
    SigLIP-SO400M
    DINOv2-Registers-L
    EVA-02-Large
    ConvNeXt-V2-Tiny
    2D-FFT
    SRM/DWT
    Edge-Specialist
    Patch-MIL

The research question is:

    Which combination provides the best combination of:

        low FP
        low FN
        high robustness
        high OOD generalization
        calibration
        complementary error coverage
        latency
        VRAM
        parameter efficiency

?

Do not equate:

    "more models"

with:

    "better detector".

But also explicitly test whether a broad all-expert fusion provides
meaningful additional error coverage.

============================================================
4. PHASE-1 DATA
============================================================

Use the existing 50K Phase-1 corpus as the immediate training corpus.

Current known composition:

    17,373 authentic
    32,627 synthetic

Do NOT delete the remaining approved data.

Do NOT artificially discard large quantities of valid data merely to
force a 50/50 manifest.

Instead use:

    complete approved data pool
    +
    generator-aware sampling
    +
    controlled loss weighting

where scientifically justified.

The current generator-aware strategy is:

    50% Real
    50% AIGC

with synthetic allocation approximately:

    SID Diffusion       45%
    Diffusion General   20%
    HFCF                35%

This is a CURRENT HYPOTHESIS, not an immutable law.

Before full training, run a lightweight sampling sanity check.

Compare:

    natural distribution
    50/50 class sampling
    generator-aware hybrid sampling

using a small training pilot.

Do not use external OOD datasets to choose the sampling strategy.

If a different strategy clearly improves held-out development
performance without increasing FP/FN or destroying generator coverage,
document and use it.

============================================================
5. FULL CORPUS STRATEGY
============================================================

Phase 1:

    50K curated/deduplicated training corpus

Phase 2:

    progressively incorporate the complete approved 400+ GB corpus.

The long-term objective is to learn from essentially ALL useful approved
data, not permanently restrict training to 50K images.

The larger corpus contains:

    WikiArt hard negatives
    Quality Paradox
    Defactify
    multi-generator Parquet data
    SID
    FLUX/SD3/SDXL material
    additional authentic photography
    other approved generator families

The model must therefore learn across:

    photography
    artwork
    screenshots/web imagery
    compression-heavy imagery
    high-resolution imagery
    subtle photorealistic AIGC
    multiple diffusion families
    GAN-generated imagery
    different generator pipelines
    different resolutions
    different image domains

Avoid generator memorization.

The goal is:

    DETECT AIGC ARTIFACTS

not:

    IDENTIFY THE TRAINING DATASET.

============================================================
6. DATA SPLIT GOVERNANCE
============================================================

Create explicit:

    TRAIN
    VALIDATION
    INTERNAL TEST

partitions.

No image may cross partitions.

Perform:

    SHA-256 deduplication

and, where computationally practical:

    perceptual near-duplicate detection.

Near-duplicate detection is important because cryptographic hashes alone
do not detect resized/recompressed copies.

Do not split near-duplicate families across train and validation.

Record:

    exact counts
    class counts
    generator counts
    source dataset counts
    domain counts
    hash overlap
    near-duplicate overlap

in:

    reports/phase1_dataset_integrity.json

============================================================
7. VALIDATION MUST BE LARGE ENOUGH
============================================================

Do NOT rely on 50-image or 100-image validation sets.

For Phase 1, use thousands of images where computationally feasible.

Prefer:

    5,000 validation images

and:

    5,000 internal test images

when the approved pool permits.

The internal test must remain untouched during training and
hyperparameter selection.

Do not repeatedly inspect the same test results and then modify the
model.

That converts the test set into validation data.

============================================================
8. LOCKED EXTERNAL OOD DATA
============================================================

The following remain strictly external:

    Synthbuster
    AIGIBench
    Chameleon
    VCT²
    WildRF
    SynthWildX
    official locked hackathon validation data

Do NOT:

    train on them
    fit normalization on them
    tune thresholds on them
    choose architecture using them
    tune fusion weights using them
    repeatedly inspect them during development

Only after the model is frozen should external OOD evaluation occur.

============================================================
9. HARD NEGATIVES
============================================================

Hard negatives are essential.

The model must see difficult REAL images including:

    heavy JPEG compression
    blur
    sharpening
    unusual textures
    paintings
    drawings
    screenshots
    web imagery
    high-frequency textures
    unusual lighting
    photographic noise
    old/vintage imagery
    difficult artistic styles

These are particularly important because the operational objective
includes extremely low false-positive rate.

A real image that looks "AI-like" to one specialist must not automatically
be classified as fake.

============================================================
10. GPU/RAM/NVME PIPELINE
============================================================

DO NOT repeatedly stream individual images directly from HDD.

Use the available NVMe as the primary dataset staging/cache layer.

Preferred pipeline:

    HDD / source dataset
            ↓
    NVMe staged dataset
            ↓
    asynchronous DataLoader workers
            ↓
    RAM prefetch / hot buffer
            ↓
    pinned host memory
            ↓
    non-blocking GPU transfer
            ↓
    RTX 3050 inference/training

The previous I/O benchmark showed:

    Direct HDD:
        ~183 img/s

    Direct NVMe:
        ~187 img/s

    NVMe + Async Pinned RAM:
        ~625 img/s

Therefore prioritize:

    NVMe + asynchronous pinned RAM

over direct HDD access.

============================================================
11. RAM POLICY
============================================================

Prefer keeping frequently accessed training data in RAM where practical.

Do NOT attempt to force the entire 400+ GB corpus into 31 GB RAM.

That would cause memory pressure and potentially destructive swapping.

Instead implement a hierarchical cache:

    HOT:
        frequently accessed samples → RAM

    WARM:
        staged training corpus → NVMe

    COLD:
        original source data → HDD

Use:

    persistent workers
    prefetch_factor
    pinned memory
    non_blocking CUDA transfers
    asynchronous prefetching
    sequential/locality-aware staging where possible

Monitor:

    RAM usage
    page faults
    swap activity
    NVMe throughput
    HDD throughput
    GPU utilization
    DataLoader wait time

The objective is:

    GPU continuously supplied with data.

============================================================
12. SWAP POLICY
============================================================

Do NOT use swap as normal dataset memory.

Do NOT intentionally increase swap to compensate for insufficient RAM.

Existing ~24 GB swap is sufficient as a safety buffer if the system is
stable.

The preferred hierarchy is:

    RAM > NVMe > HDD > swap

Swap should remain emergency capacity only.

If sustained swap activity appears:

    reduce RAM cache
    reduce DataLoader workers
    reduce prefetch depth
    reduce batch size

Do NOT allow the system to thrash.

============================================================
13. NVME CACHE DESIGN
============================================================

Build a deterministic NVMe staging system.

Requirements:

    - cache raw training images or efficiently decoded representations
    - cryptographic manifest
    - source provenance
    - cache version
    - dataset version
    - no silent stale-cache reuse
    - integrity verification

If a feature cache is created, it must have:

    model checkpoint hash
    preprocessing version
    image manifest hash
    feature dimensionality
    dtype
    creation timestamp
    experiment ID

A feature cache may only be reused when ALL of these match.

Otherwise recompute.

============================================================
14. MIXED PRECISION / MEMORY EFFICIENCY
============================================================

Use mixed precision where numerically safe:

    FP16 or BF16 where supported and stable

Keep sensitive calculations in FP32 when required:

    loss accumulation
    probability calibration
    metric computation
    normalization statistics where necessary
    threshold calculations

Use:

    inference_mode()
    autocast
    gradient accumulation where useful
    activation checkpointing only if it improves feasible batch size
    gradient clipping if instability occurs

Do not sacrifice numerical correctness merely for speed.

============================================================
15. DO NOT FINE-TUNE EVERY HUGE BACKBONE IMMEDIATELY
============================================================

RTX 3050 6 GB is constrained.

Initially:

    freeze foundation backbones

and train:

    lightweight representation heads
    fusion head
    calibration head

Only fine-tune a backbone if evidence shows that it materially improves
held-out validation performance.

If fine-tuning is justified:

    use parameter-efficient fine-tuning where appropriate
    freeze most layers
    use small learning rates
    monitor catastrophic overfitting

Do not spend 12 hours fine-tuning a backbone unless a pilot proves it
is worthwhile.

============================================================
16. TRAINING OBJECTIVE
============================================================

The primary loss must reflect the real operational objective.

Start with:

    asymmetric BCE

with stronger penalty on false positives.

Current candidate:

    lambda_FP = 2.0

But DO NOT assume 2.0 is universally optimal.

Run a small controlled comparison:

    λFP ∈ {1.0, 1.5, 2.0, 2.5, 3.0}

using TRAIN only for fitting and VALIDATION for selection.

Do not tune using internal test or external OOD.

Measure:

    FPR
    FNR
    TPR
    TNR
    AUROC
    AUPRC
    ECE
    Brier
    calibration curve

Choose the smallest FP penalty that produces the required reduction
without unnecessarily increasing FN.

============================================================
17. DO NOT USE THRESHOLD 0.50 BY DEFAULT
============================================================

A probability threshold of 0.50 is not sacred.

After the model is trained:

evaluate threshold curves on validation.

Explicitly report thresholds producing approximately:

    FPR ≤ 5%
    FPR ≤ 2%
    FPR ≤ 1%
    FPR ≤ 0.5%
    FPR ≤ 0.1%

For each report:

    threshold
    FPR
    TPR
    FNR
    TNR
    precision
    recall
    specificity
    accuracy

Select the operational threshold based on the actual project objective.

Do NOT tune this threshold on external OOD data.

============================================================
18. CALIBRATION
============================================================

Raw sigmoid output is not automatically a trustworthy probability.

Evaluate:

    temperature scaling
    Platt scaling
    isotonic regression

Fit calibration ONLY on the designated validation/calibration data
according to the predefined protocol.

Report:

    ECE
    Brier score
    reliability curve
    calibration error by subgroup

Do not use test/OOD data to fit calibration.

The final detector must distinguish:

    raw score
    calibrated P(AIGC)

============================================================
19. TRANSFORMATION ROBUSTNESS
============================================================

Every serious model must be evaluated under the core perturbations:

    Clean
    JPEG30
    Blur2
    Resize0.25
    Noise0.10
    Crop80
    ColorJitter

Additionally evaluate the broader robustness matrix when computationally
possible:

    JPEG 30–90
    Blur σ 0.5–2.0
    Downscale 0.25–0.5
    Noise σ 0.02–0.10
    Color Jitter
    Center Crop

Use the SAME trained model across transformations.

Never fit a separate model for each transformation.

Report:

    each AUROC
    mean RI
    worst AUROC
    degradation from clean
    FPR
    FNR

============================================================
20. GENERATOR-SPECIFIC EVALUATION
============================================================

Report AIGC detection separately for each known generator family.

At minimum distinguish:

    FLUX
    SD3
    SDXL
    Midjourney
    DALL-E
    StyleGAN
    other available families

The model must not appear strong merely because one generator dominates.

Report:

    per-generator TPR
    per-generator FNR
    AUROC
    AUPRC

Also report:

    worst-generator performance.

============================================================
21. AUTHENTIC-DOMAIN EVALUATION
============================================================

For REAL images, report FPR/TNR separately for:

    COCO
    general photography
    high-resolution photography
    WikiArt / artwork
    web imagery
    compression-heavy images

This is essential.

A detector with 0% aggregate FPR can still have unacceptable FPR in one
specific authentic domain.

============================================================
22. ALL-EXPERT EXPERIMENT
============================================================

Before final architecture selection, perform ONE controlled
ALL-EXPERT fusion experiment.

This is explicitly required because the scientific question is whether
the combined knowledge of all representations produces useful
complementary error coverage.

Test:

    CLIP
    + SigLIP
    + DINO
    + EVA
    + ConvNeXt
    + FFT
    + SRM/DWT
    + Edge
    + Patch-MIL

Use lightweight fusion only.

Do NOT concatenate enormous intermediate feature maps if unnecessary.

Prefer:

    normalized expert logits/probabilities

or:

    compact learned projections

followed by:

    small MLP / logistic fusion head

Compare:

    individual experts
    CLIP baseline
    CLIP + SigLIP
    best compact fusion
    ALL-EXPERT fusion

The all-expert experiment is a RESEARCH EXPERIMENT.

It does NOT automatically become the final architecture.

============================================================
23. FUSION FORMULAS
============================================================

Test multiple controlled fusion mechanisms:

    probability average
    weighted probability average
    logit average
    learned logistic regression
    small MLP

For learned fusion:

    train only fusion parameters initially.

Do not let the fusion head memorize validation data.

Use:

    train split → fit
    validation split → model selection
    internal test → final internal assessment

Record exact fusion weights.

If learned weights collapse toward one expert, report that.

If weak experts receive near-zero weights, report that.

Do not force every expert to contribute equally.

============================================================
24. ERROR COMPLEMENTARITY
============================================================

For each expert and fusion combination calculate:

    Pearson correlation
    Spearman correlation
    prediction disagreement
    FN overlap
    FP overlap
    A→B rescue
    B→A rescue
    net rescue
    new FP introduced
    new FN introduced

Also calculate:

    oracle best-of-two

BUT clearly label oracle metrics as:

    ANALYTICAL UPPER-BOUND

not achievable deployed performance.

Do not claim:

    "oracle AUROC = model AUROC".

============================================================
25. FP/FN OPTIMIZATION
============================================================

The final model must explicitly optimize both error types.

Do not chase:

    FPR → 0%

if doing so causes:

    FNR → extremely high.

Likewise do not chase:

    FNR → 0%

while causing:

    FPR → unacceptable.

Construct an operational tradeoff table.

At minimum:

    τ = 0.50
    τ selected for FPR ≤ 5%
    τ selected for FPR ≤ 2%
    τ selected for FPR ≤ 1%
    τ selected for FPR ≤ 0.5%

The final recommended operating point must be justified using
validation evidence.

============================================================
26. TRAINING SCHEDULE
============================================================

Because RTX 3050 is constrained, use staged training.

STAGE A:
    Data pipeline benchmark

STAGE B:
    Small pilot

STAGE C:
    Phase-1 50K training

STAGE D:
    Large-corpus expansion

STAGE E:
    final calibration

STAGE F:
    frozen internal test

STAGE G:
    locked external OOD evaluation

Do not begin Stage D until Stage C demonstrates that the architecture
is learning useful signal.

============================================================
27. PILOT BEFORE EXPENSIVE TRAINING
============================================================

Before committing to the full 50K run:

Train a small pilot using approximately:

    5K–10K training images

and a sufficiently large held-out validation set.

The pilot must answer:

    - does loss converge?
    - does validation AUROC improve?
    - does FPR decrease?
    - does FNR remain acceptable?
    - does calibration improve?
    - does GPU utilization remain high?
    - is I/O the bottleneck?
    - is RAM pressure acceptable?
    - is swap stable?

If the pilot is broken:

    STOP.

Fix it before full training.

Do not waste the full dataset on a broken pipeline.

============================================================
28. TRAINING CHECKPOINTS
============================================================

Save checkpoints only with complete metadata:

    dataset manifest hash
    code version/hash
    model configuration
    optimizer
    scheduler
    epoch
    training seed
    preprocessing version
    loss parameters
    sampler configuration

Never overwrite the best checkpoint without recording why.

Keep:

    latest
    best validation AUROC
    best RI
    best low-FPR operating point

as distinct artifacts.

============================================================
29. EARLY STOPPING
============================================================

Do not select the model solely from training loss.

Monitor:

    validation AUROC
    validation RI
    worst-condition AUROC
    FPR
    FNR
    ECE

Prefer a checkpoint that generalizes robustly.

If clean AUROC rises while worst-condition performance collapses,
investigate before continuing.

============================================================
30. SPEED OPTIMIZATION
============================================================

Maximize GPU utilization without destabilizing the system.

Tune:

    batch size
    workers
    prefetch_factor
    persistent_workers
    pinned memory

using measured telemetry.

Do NOT simply maximize worker count.

CPU saturation is acceptable only if it increases end-to-end throughput
without causing memory pressure.

The target is:

    minimal GPU idle time
    maximal images/sec
    stable RAM
    zero sustained swap

Benchmark:

    images/sec
    GPU utilization
    GPU memory
    CPU utilization
    RAM
    swap
    NVMe throughput
    batch preparation time
    GPU compute time
    end-to-end time

============================================================
31. FEATURE EXTRACTION STRATEGY
============================================================

If frozen backbones are used:

    extract features ONCE

and store them on NVMe.

Do NOT recompute identical backbone features for every fusion-head
experiment.

But every feature cache must be cryptographically tied to:

    exact image manifest
    exact pretrained checkpoint
    exact preprocessing
    exact feature extractor version

If any dependency changes:

    invalidate cache.

This is the ONLY acceptable form of caching.

Do not use historical feature caches whose provenance cannot be proven.

============================================================
32. FULL 400+ GB EXPANSION
============================================================

After Phase-1 architecture validation:

expand training to the complete approved corpus.

Do not simply concatenate everything.

Create a generator/domain-aware manifest.

Ensure representation of:

    multiple generators
    authentic domains
    hard negatives
    subtle AIGC
    compression
    artwork
    photography
    different resolutions

Avoid any one dataset becoming the dominant shortcut.

The sampler should ensure useful exposure to minority generator families.

============================================================
33. DATA CURRICULUM
============================================================

If beneficial, use curriculum training:

Early:

    broad/easy examples

Middle:

    diverse generators + hard negatives

Late:

    subtle AIGC
    difficult real images
    adversarial transformations
    rare generator families

Do NOT use curriculum merely because it sounds sophisticated.

Benchmark it against ordinary generator-aware sampling.

Keep the simpler method if performance is equivalent.

============================================================
34. NO UNNECESSARY HYPERPARAMETER SWEEP
============================================================

Do not run massive sweeps on RTX 3050.

Use targeted experiments only.

Priority:

    loss FP weight
    learning rate
    batch/effective batch size
    sampler distribution
    fusion method
    calibration method
    threshold

Do not optimize dozens of irrelevant parameters.

Every experiment must answer a scientific question.

============================================================
35. FINAL MODEL SELECTION
============================================================

The final model must maximize:

    OOD robustness
    clean AUROC
    AUPRC
    low FPR
    low FNR
    high worst-case AUROC
    calibration
    complementary error coverage

subject to:

    <2B total instantiated parameters

and practical:

    RTX 3050 6GB

deployment.

Latency matters.

A 1.6B architecture that gives negligible improvement over a 1.0B
architecture should not automatically win.

Likewise, a 30M specialist should be retained if it provides measurable
independent error rescue at negligible cost.

============================================================
36. REQUIRED FINAL REPORTS
============================================================

Produce:

    reports/phase1_training_run.json
    reports/phase1_training_metrics.json
    reports/phase1_validation_report.json
    reports/phase1_generator_breakdown.json
    reports/phase1_authentic_domain_breakdown.json
    reports/phase1_transformation_robustness.json
    reports/phase1_calibration_report.json
    reports/phase1_threshold_analysis.json
    reports/phase1_error_analysis.json
    reports/phase1_fusion_analysis.json
    reports/phase1_io_performance.json
    reports/phase1_memory_performance.json

For Phase 2:

    reports/full_corpus_training_manifest_audit.json
    reports/full_corpus_training_report.json
    reports/full_corpus_validation_report.json
    reports/full_corpus_ood_report.json

============================================================
37. REQUIRED CONFUSION-MATRIX REPORTING
============================================================

For every major checkpoint report:

    TP
    TN
    FP
    FN

and:

    FPR
    TPR
    FNR
    TNR
    Precision
    Recall
    Accuracy

Do not report only accuracy.

Especially report FP and FN by:

    generator
    authentic domain
    transformation
    threshold

============================================================
38. FINAL OOD EVALUATION
============================================================

Only after architecture, training recipe, calibration and threshold are
FROZEN:

evaluate on locked external datasets.

Run each exactly according to the predefined protocol.

Do not modify the model afterward based on those results.

If external performance is poor:

    document failure

rather than silently tuning until the test improves.

If another iteration is desired, that becomes a NEW development cycle
with a new model/version and fresh governance.

============================================================
39. KNOWLEDGE BASE SYNCHRONIZATION
============================================================

After every major stage:

update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Record:

    what was tested
    why it was tested
    exact dataset
    exact split
    exact model
    exact loss
    exact sampler
    exact hyperparameters
    exact result
    interpretation
    limitations
    next decision

Never replace historical results.

Append new evidence with timestamps/version identifiers.

============================================================
40. STOP CONDITIONS
============================================================

STOP immediately if:

    - train/validation leakage is detected
    - near-duplicate leakage is detected
    - stale derived data is accidentally reused
    - labels are inconsistent
    - model outputs become constant
    - loss becomes NaN
    - sustained swap thrashing occurs
    - GPU OOM repeatedly occurs
    - validation metrics become suspiciously perfect
    - preprocessing differs unexpectedly
    - a checkpoint fails integrity validation
    - external OOD data enters training/selection
    - a metric cannot be reproduced from saved predictions

Do not hide failures.

Report them.

============================================================
41. EXECUTION ORDER
============================================================

Execute in exactly this order:

    1. Read Master Prompt + Knowledge Base
    2. Verify current data governance
    3. Verify 50K manifest integrity
    4. Verify train/val/test separation
    5. Verify near-duplicate isolation
    6. Benchmark NVMe/RAM pipeline
    7. Run small sampling pilot
    8. Run small model-training pilot
    9. Validate FP/FN loss weighting
   10. Validate fusion strategy
   11. Train Phase-1 model
   12. Evaluate all transformations
   13. Evaluate generator/domain subgroups
   14. Calibrate probabilities
   15. Determine operating threshold
   16. Evaluate internal test ONCE
   17. Freeze Phase-1 checkpoint
   18. Expand to approved full corpus
   19. Train/continue with generator-aware sampling
   20. Repeat validation
   21. Freeze final model
   22. Evaluate locked external OOD
   23. Update Knowledge Base
   24. Produce final scientific report

Do not skip stages merely because earlier experiments looked good.

============================================================
42. CRITICAL: DO NOT WAIT FOR HUMAN APPROVAL BETWEEN EVERY STEP
============================================================

The purpose of this prompt is to allow autonomous execution of the
validated pipeline.

You MAY proceed automatically through routine engineering steps.

However, you MUST STOP for human review if:

    - a scientific assumption changes
    - a data-governance decision changes
    - a new dataset is proposed
    - an external benchmark would influence model selection
    - a new architecture outside the approved expert pool is proposed
    - a result is suspicious or irreproducible
    - the final architecture would exceed 2B parameters
    - the model exhibits severe FP/FN tradeoff
    - the expected training time materially exceeds the available window

Do not silently make major scientific decisions.

============================================================
43. FINAL OBJECTIVE
============================================================

The final detector should aim for:

    VERY HIGH AIGC DETECTION ACCURACY

while simultaneously achieving:

    EXTREMELY LOW FALSE POSITIVES
    LOW FALSE NEGATIVES
    HIGH TRUE-REAL / TRUE-NEGATIVE RATE
    HIGH TRUE-AIGC DETECTION RATE
    STRONG CALIBRATION
    STRONG TRANSFORMATION ROBUSTNESS
    STRONG UNSEEN-GENERATOR GENERALIZATION
    LOW LATENCY
    LOW VRAM
    <2B PARAMETERS

Do not promise "zero FP" or "zero FN".

Those are empirical outcomes, not design assumptions.

The scientific goal is to push both toward zero without sacrificing
generalization.

============================================================
44. MOST IMPORTANT ENGINEERING PRINCIPLE
============================================================

USE COMPUTATION INTELLIGENTLY.

Do not waste RTX 3050 compute on:

    repeated feature extraction
    redundant experiments
    huge hyperparameter sweeps
    stale-cache recomputation
    unnecessary backbone fine-tuning

Use:

    NVMe staging
    RAM hot buffering
    pinned memory
    asynchronous loading
    persistent workers
    mixed precision
    feature caching WITH strict provenance
    lightweight heads
    targeted pilots

The desired pipeline is:

    HDD
      ↓
    NVMe
      ↓
    RAM hot cache
      ↓
    pinned memory
      ↓
    GPU
      ↓
    frozen representation
      ↓
    compact learned head/fusion
      ↓
    calibration
      ↓
    threshold
      ↓
    robust evaluation

The GPU should spend its time computing, not waiting for HDD I/O.

============================================================
45. FINAL EXECUTION RULE
============================================================

Do not tell me that a model is "excellent" based on a single AUROC.

Do not tell me FP is "near zero" based on a tiny validation set.

Do not claim OOD generalization before locked OOD evaluation.

Do not claim zero-shot performance if the generator was development-exposed.

Do not claim a model is better merely because it is larger.

Do not claim a model is worse merely because its standalone score is
lower if its representation provides useful complementary information.

Every major claim must be supported by:

    exact dataset
    exact sample count
    exact split
    exact prediction source
    exact metric
    exact checkpoint
    exact experiment identifier

The purpose of this project is not to manufacture a good number.

The purpose is to discover and train a detector that genuinely
generalizes.

START WITH THE DATA/PIPELINE/PILOT GATES.

THEN TRAIN.

DO NOT SKIP THE SCIENTIFIC GATES.

============================================================
END OF MASTER AUTHORIZATION PROMPT
============================================================