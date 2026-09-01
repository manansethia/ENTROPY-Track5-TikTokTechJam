# MASTER CONTROL PROMPT — AUTONOMOUS 20-HOUR AIGC DETECTOR SPRINT
# BUILDBOT + KAGGLE GPU/TPU + MAC AUXILIARY
# PURPOSE: DO NOT GET LOST — ALWAYS KNOW WHAT TO DO NEXT AND WHEN

We have approximately 20 hours of wall-clock time remaining.

The project must operate autonomously through:
DATA → VALIDATION → BENCHMARKS → TRAINING → EVALUATION → SELECTION → CALIBRATION → FREEZE → DEPLOYMENT TEST.

Do not repeatedly ask the user what to do next for routine work.
Follow this protocol until completion.

===============================================================================
0. NON-NEGOTIABLE GOAL
===============================================================================

Fix the current major real-world failure modes:

    REAL high-resolution → AIGC
    REAL high-quality/color-corrected → AIGC
    REAL portraits/selfies/headshots → AIGC
    REAL cropped/edited → AIGC
    REAL brightness/contrast edited → AIGC

Observed counterexample:

    shrinking REAL can make it become REAL
    shrinking AIGC can also make it become REAL

Therefore:

    DO NOT solve the problem by simply shrinking every input.

The goal is:

    high-res REAL ≠ AIGC merely because of resolution/quality
    ordinary edits ≠ AIGC
    genuine synthetic evidence remains detectable

===============================================================================
1. IMMUTABLE PRODUCTION CONTROL
===============================================================================

NEVER modify, overwrite, or delete:

    checkpoints/production/final_champion_frozen_model.pt

Expected SHA-256:

    91a6a3814c30f0b62f7b63e63fa81fe595c83b9edf91733ef9a8d3358e133438

Use a COPY for every experiment.

Every candidate must record:

    source checkpoint
    source SHA-256
    dataset manifest hash
    code revision
    configuration
    random seed

===============================================================================
2. GLOBAL AUTHORITY
===============================================================================

BUILDBOT IS THE AUTHORITATIVE MACHINE.

Buildbot owns:

    master dataset manifest
    training manifest
    canonical evaluation
    final comparison
    calibration
    final model selection
    final frozen checkpoint
    deployment artifacts

KAGGLE is:

    independent remote GPU/TPU worker infrastructure

MAC M1 is:

    lightweight auxiliary CPU infrastructure

Do NOT implement distributed training across machines.

Do NOT synchronize gradients over the network.

===============================================================================
3. CENTRAL JOB ORCHESTRATOR
===============================================================================

Create and maintain:

    reports/master_execution_state.json
    reports/master_execution_state.md

The state file MUST contain:

    current_phase
    current_task
    task_status
    task_start_time
    elapsed_time
    ETA
    blocker
    next_task
    dependency_status

Also maintain:

    reports/accelerator_budget.json

Hard external budgets:

    GPU <= 30 cumulative GPU-hours
    TPU <= 20 cumulative TPU-hours

Never exceed either budget.

===============================================================================
4. REQUIRED STATUS FORMAT
===============================================================================

Every automatic status update must contain:

    CURRENT TIME
    CURRENT PHASE
    WHAT IS RUNNING
    WHY IT IS RUNNING
    HARDWARE/WORKER
    DATASET
    COMPLETED
    REMAINING
    CURRENT RATE
    ETA
    GPU HOURS USED / REMAINING
    TPU HOURS USED / REMAINING
    DISK FREE
    RAM
    SWAP
    NEXT ACTION

Never report only:

    "running"
    "in progress"
    "pipeline proceeding"

The user must always know:

    WHAT
    WHY
    HOW MUCH
    HOW LONG
    WHAT NEXT

===============================================================================
5. MASTER EXECUTION ORDER
===============================================================================

Follow this exact high-level sequence:

    PHASE A
    Storage + process inventory

    ↓

    PHASE B
    Dataset download / streaming / persistent storage

    ↓

    PHASE C
    Dataset classification + deduplication + contamination audit

    ↓

    PHASE D
    External-model benchmark

    ↓

    PHASE E
    High-resolution / multi-crop / transformation diagnostics

    ↓

    PHASE F
    Build clean remediation manifest

    ↓

    PHASE G
    PORTRAIT-REM-1 training

    ↓

    PHASE H
    Epoch-by-epoch evaluation

    ↓

    PHASE I
    Optional second candidate ONLY if justified

    ↓

    PHASE J
    Final comparison

    ↓

    PHASE K
    Calibration

    ↓

    PHASE L
    FREEZE

    ↓

    PHASE M
    Deployment verification

Never jump randomly between phases.

Routine completed phases do not need user confirmation.

===============================================================================
6. DATASET DOWNLOAD PLAN
===============================================================================

Continue all currently approved downloads.

However, do NOT wait for every optional evaluation dataset before
training.

Separate downloads into:

A. TRAINING-CRITICAL
B. EVALUATION-CRITICAL
C. OPTIONAL

TRAINING-CRITICAL data gets highest priority.

EVALUATION-CRITICAL downloads continue in parallel.

OPTIONAL downloads are paused if they compete with training I/O or
risk disk capacity.

-------------------------------------------------------------------------------
6A. REQUIRED DOWNLOAD TELEMETRY
-------------------------------------------------------------------------------

For EVERY download task record:

    dataset
    shard/archive
    total size
    downloaded size
    percentage
    current MB/s
    rolling average MB/s
    ETA
    number of completed shards
    remaining shards
    disk consumed
    disk remaining

Example:

    NTIRE TRAIN
    3/5 shards
    17.2 / 25.0 GB
    68.8%
    185 MB/s
    ETA 7m 03s

NEVER report a generic ETA for all downloads.

Give one ETA per download.

-------------------------------------------------------------------------------
6B. DOWNLOAD PRIORITY
-------------------------------------------------------------------------------

Priority order:

    1. TRAIN-ELIGIBLE high-res REAL/AIGC data
    2. data required to create PORTRAIT-REM-1
    3. robustness training data
    4. external evaluation datasets
    5. optional benchmarks

Do NOT download an evaluation-only dataset merely because it is large
and available.

===============================================================================
7. DATASET ROLE RULE
===============================================================================

Before any dataset enters training:

    classify:
        TRAIN-ELIGIBLE
        EVALUATION-ONLY

Keep evaluation-only data physically/logically outside the training
manifest.

HiRes-50K remains evaluation-only if its current dataset designation
still says so.

Any benchmark/test dataset designated evaluation-only stays evaluation-only.

===============================================================================
8. STORAGE PLAN
===============================================================================

Buildbot currently has substantial free disk.

Before each large extraction:

    df -h
    df -i

Maintain safe free space.

Delete ONLY:

    verified caches
    incomplete downloads
    exact duplicate archives
    recreatable temporary extraction
    obsolete staging

Do NOT delete:

    frozen production model
    locked evaluation data
    provenance
    canonical manifests
    required source code

If disk usage exceeds 85%:

    pause optional downloads

If disk usage exceeds 90%:

    stop noncritical ingestion and clean verified temporary data.

===============================================================================
9. STREAMING + SHARDING
===============================================================================

For very large datasets:

    prefer streaming or sharded processing

Do NOT load an entire multi-GB dataset into RAM.

Use:

    bounded prefetch
    persistent workers
    local cache for repeated epochs
    shard-level processing

Choose streaming vs local caching based on measured throughput.

Record which strategy is used.

===============================================================================
10. KAGGLE REMOTE WORKER ARCHITECTURE
===============================================================================

Use the existing Kaggle API headlessly.

Do NOT rely on the browser once a worker is configured.

Use independent workers.

For each worker:

    push
    launch
    verify accelerator
    run task
    collect logs
    collect output
    record runtime
    record resource usage
    stop when done

-------------------------------------------------------------------------------
10A. KAGGLE GPU SELF-TEST
-------------------------------------------------------------------------------

Before any useful computation:

    torch.cuda.is_available()
    torch.cuda.device_count()
    actual GPU name
    actual VRAM
    real CUDA tensor operation

If GPU is unavailable:

    terminate the worker immediately.

If the requested GPU is not the actual GPU:

    record actual GPU
    continue only if suitable.

Do not burn quota on CPU-only workers.

-------------------------------------------------------------------------------
10B. KAGGLE WORKER PRIORITY
-------------------------------------------------------------------------------

Worker GPU-1:
    Main independent high-res remediation candidate

Worker GPU-2:
    SPAI / any-resolution benchmark

Worker GPU-3:
    CommunityForensics ViT-Small benchmark

Worker GPU-4:
    High-res robustness benchmark

Worker GPU-5:
    Multi-crop/native-resolution benchmark

Worker GPU-6:
    Unseen-generator benchmark

Worker GPU-7:
    Additional candidate only if first candidate needs comparison

Do NOT duplicate jobs without a scientific reason.

-------------------------------------------------------------------------------
10C. KAGGLE STORAGE
-------------------------------------------------------------------------------

Use persistent Kaggle Dataset storage for large reusable datasets.

Do NOT use /kaggle/working as the long-term dataset store.

Where account limits permit, organize large data into meaningful datasets/shards.

Examples:

    aigc-highres-real-v1
    aigc-highres-synthetic-v1
    aigc-robustness-v1
    aigc-diagnostic-v1

Store:

    manifest
    source/license metadata
    SHA metadata
    README

Avoid duplicating identical files across datasets.

-------------------------------------------------------------------------------
10D. MODEL STORAGE
-------------------------------------------------------------------------------

Use Kaggle persistent model/artifact storage for large temporary
experimental checkpoints when appropriate.

Do NOT repeatedly transfer every optimizer-state checkpoint to Buildbot.

Return only:

    best candidate
    final candidate
    metadata
    reports

===============================================================================
11. TPU STRATEGY
===============================================================================

TPU quota is separate from GPU quota.

Do NOT port the complete 735M PyTorch/CUDA production pipeline to TPU
unless it is already compatible.

TPU may be used for:

    small independent compatible model
    compatible distillation experiment
    other low-engineering-cost tensor workloads

Before consuming real TPU budget:

    initialize TPU
    run tensor operation
    run one training step
    benchmark throughput

If setup/debugging exceeds 15 minutes:

    terminate TPU work.

Never sacrifice the 20-hour deadline to port the main model to XLA.

===============================================================================
12. MAC M1 ROLE
===============================================================================

Use Mac M1 only for lightweight parallel CPU work:

    hashing
    metadata extraction
    manifest generation
    lightweight preprocessing
    decompression/download assistance

Do NOT use significant Mac RAM.

Do NOT build distributed training across Mac + Buildbot.

===============================================================================
13. EXTERNAL MODEL PRIORITY
===============================================================================

Benchmark in this order:

    1. SPAI / TFG-model
    2. CommunityForensics ViT-Small
    3. our current frozen model
    4. any additional model only if setup is cheap

Use the SAME diagnostic images wherever possible.

Priority diagnostic cases:

    high-res REAL
    portraits
    selfies
    high-grade camera photos
    color-corrected
    cropped REAL
    brightness/contrast edited REAL
    high-res AIGC

Record:

    probability
    class
    latency
    VRAM/RAM

===============================================================================
14. HIGH-RESOLUTION DIAGNOSTIC
===============================================================================

For selected REAL and AIGC images test:

    original
    75%
    50%
    33%
    25%
    16%
    10%

Also compare:

    bicubic
    bilinear
    Lanczos

Record:

    P(AIGC)
    classification
    probability shift

Do not conclude from one image.

Use a statistically meaningful diagnostic subset.

===============================================================================
15. MULTI-CROP DIAGNOSTIC
===============================================================================

Compare:

A:
    full image → 224

B:
    native high-resolution local crops → 224 → shared encoder

C:
    global view + native local crops → shared encoder + lightweight fusion

Measure:

    clean AUC
    robust AUC
    high-res REAL FPR
    high-res AIGC TPR
    latency
    VRAM

If multi-crop materially improves the target failure, prioritize it.

===============================================================================
16. REAL POST-PROCESSING ROBUSTNESS
===============================================================================

For the same REAL image generate:

    mild crop 5–15%
    brightness ±5–10%
    contrast ±5–10%
    mild color adjustment
    JPEG Q90–95
    moderate resize
    mild sharpening
    mild denoising

Measure classification stability.

Also apply equivalent transformations to AIGC.

Do NOT demand invariance to transformations that destroy genuinely
discriminative evidence.

===============================================================================
17. REMEDIATION DATA POOL
===============================================================================

Build:

    REAL:
        high-resolution
        portraits
        selfies
        headshots
        smartphone
        DSLR/mirrorless
        studio
        HDR
        bokeh
        high-ISO
        color grading
        color correction
        retouching
        sharpening
        denoising
        JPEG/web processing
        varied aspect ratios

    AIGC:
        multiple generators
        high-resolution
        photorealistic
        portraits/people
        professional-looking
        varied aspect ratios
        varied quality/compression

Match distributions across:

    resolution
    aspect ratio
    compression
    visual quality
    processing

===============================================================================
18. HARD-NEGATIVE MINING
===============================================================================

Mine REAL false positives from approved TRAIN only.

Prioritize:

    high-res REAL
    portrait REAL
    selfie REAL
    edited REAL
    cropped REAL
    brightness/contrast REAL

Also retain difficult AIGC false negatives.

Never fabricate hard cases merely to hit a target count.

===============================================================================
19. MAIN TRAINING — PORTRAIT-REM-1
===============================================================================

Start from a COPY of frozen production.

First remediation should primarily correct:

    training data distribution
    real hard-negative balance
    geometry matching
    mild realistic post-processing
    high-resolution coverage

Use the multi-crop architecture ONLY if its ablation is empirically
positive.

Do not simultaneously introduce:

    VLM feedback
    multiple new losses
    architecture search
    aggressive blur/JPEG invariance

unless a controlled result justifies it.

===============================================================================
20. TRAINING RESOURCE OPTIMIZATION
===============================================================================

Use the largest stable batch size.

Benchmark a few batch sizes automatically.

Use:

    AMP/mixed precision
    pinned memory
    persistent workers
    efficient prefetch
    efficient image decode
    gradient accumulation only if needed

Keep:

    GPU highly utilized
    CPU feeding it efficiently
    RAM used as bounded cache

Do NOT intentionally:

    OOM
    swap thrash
    fill disk

===============================================================================
21. EPOCH CONTROL
===============================================================================

After every epoch:

    save checkpoint
    evaluate
    record result
    decide continue/stop

Do NOT automatically run all planned epochs.

If performance is clearly worsening, stop early.

Use the best epoch, not necessarily the final epoch.

===============================================================================
22. REQUIRED EPOCH METRICS
===============================================================================

Competition:

    Clean ROC-AUC
    Robust ROC-AUC
    Final 50/50 score

Standard:

    Accuracy
    FP
    FN
    AUPRC

Production:

    TPR @ 0.10% FPR
    TPR @ 0.01% FPR

Real robustness:

    high-res REAL FPR
    portrait FPR
    selfie FPR
    edited-real FPR
    crop-real FPR
    brightness-real FPR
    contrast-real FPR

Generalization:

    SID/LDM
    unseen-generator AUC
    pseudo-OOD AUC
    edge-case accuracy

===============================================================================
23. MODEL SELECTION
===============================================================================

Primary:

    FINAL_SCORE =
        0.50 * AUC_clean
        +
        0.50 * AUC_robust

Secondary safeguards:

    high-res REAL FPR
    portrait FPR
    unseen-generator AUC
    SID/LDM
    low-FPR TPR
    edge cases

A candidate must not be promoted merely because its training loss fell.

If a candidate improves one metric by severely damaging another,
reject it.

===============================================================================
24. DYNAMIC COMPUTE REALLOCATION
===============================================================================

At every candidate checkpoint ask:

    Is this candidate improving?

If YES:

    continue or allocate more compute.

If NO:

    stop early.

Unused GPU budget may be reassigned to:

    strongest candidate
    highest-value validation
    unseen-generator benchmark

Do not spend remaining compute on a clearly inferior candidate.

===============================================================================
25. REQUIRED DOWNLOAD ETA TABLE
===============================================================================

Maintain a live table like:

    DATASET                  TOTAL     DONE       RATE       ETA
    ----------------------------------------------------------------
    HiRes-50K                XX GB     XX GB      XXX MB/s   HH:MM
    NTIRE TRAIN              XX GB     XX GB      XXX MB/s   HH:MM
    Quality Paradox          XX GB     XX GB      XXX MB/s   HH:MM
    other training source    XX GB     XX GB      XXX MB/s   HH:MM

Update this whenever download state changes materially.

===============================================================================
26. REQUIRED MODEL PRIORITY TABLE
===============================================================================

Maintain:

    MODEL/EXPERIMENT
    PURPOSE
    HARDWARE
    STATUS
    TIME USED
    ETA
    PRIORITY
    DECISION

Priority levels:

    P0 = required for final model
    P1 = high-value parallel experiment
    P2 = useful if spare time
    P3 = optional / stop if time constrained

P0 always wins over P1/P2/P3.

===============================================================================
27. TIME BUDGET
===============================================================================

### T+0 to T+2h
- storage audit
- finish essential downloads
- verify Kaggle GPU workers
- establish benchmark baselines
- finalize training manifest as soon as possible

### T+2h to T+10h
- PORTRAIT-REM-1 training
- high-value Kaggle experiments
- robustness diagnostics
- unseen-generator tests

### T+10h to T+14h
- compare candidates
- stop weak experiments
- concentrate compute on best candidate

### T+14h to T+16h
- final external/high-res evaluation
- final clean/robust benchmark
- final error analysis

### T+16h to T+17h
- final candidate selection
- calibration

### T+17h
HARD FREEZE RESEARCH.

No new:
    datasets
    architectures
    losses
    research experiments

### T+17h to T+19h
- final validation
- checkpoint integrity
- calibration verification
- deployment integration

### T+19h to T+20h
- CPU inference test
- CUDA inference test
- Mac MPS test
- packaging
- final documentation

===============================================================================
28. DEADLINE ESCALATION RULE
===============================================================================

If the project falls behind schedule:

At T+4h:
    prioritize training readiness over optional benchmarks.

At T+8h:
    stop low-value downloads.

At T+12h:
    stop experimental branches that are not clearly promising.

At T+15h:
    choose top candidate.

At T+17h:
    freeze research.

At T+19h:
    package only.

===============================================================================
29. FAILURE HANDLING
===============================================================================

For any worker/environment:

If blocked >15 minutes by:

    dependency
    network
    accelerator
    CUDA/XLA
    dataset access

then:

    stop
    record cause
    reassign if valuable
    continue without blocking the project

Never spend 30–60 minutes fighting one worker.

===============================================================================
30. FINAL CALIBRATION
===============================================================================

After final model selection:

    fit temperature on CAL split
    compute exact operational thresholds
    verify no test leakage

Record:

    calibration temperature
    threshold at FPR 1%
    threshold at FPR 0.5%
    threshold at FPR 0.1%
    threshold at FPR 0.05%
    threshold at FPR 0.01%

===============================================================================
31. FINAL FREEZE
===============================================================================

Before declaring production:

    save final checkpoint
    SHA-256
    parameter hash
    parameter count
    fresh-process reload
    inference sanity check

Then:

    FREEZE FINAL MODEL

No training afterwards.

===============================================================================
32. FINAL DEPLOYMENT
===============================================================================

Only AFTER freeze:

    CUDA benchmark
    CPU benchmark
    MPS benchmark
    preprocessing parity
    output schema validation

Then package:

    model
    metadata
    preprocessing
    inference
    API
    benchmark
    README
    reports

===============================================================================
33. FINAL ARTIFACTS
===============================================================================

Required:

    reports/master_execution_state.json
    reports/master_execution_state.md
    reports/accelerator_budget.json
    reports/dataset_inventory.json
    reports/download_status.json
    reports/model_priority.json
    reports/final_model_comparison.json
    reports/final_model_comparison.md
    reports/robustness_evaluation.json
    reports/unseen_generator_evaluation.json

Final checkpoint:

    checkpoints/production/final_champion_frozen_model.pt

===============================================================================
34. AUTONOMOUS DECISION RULE
===============================================================================

For every completed task:

    1. mark COMPLETE
    2. record result
    3. update ETA table
    4. update budget
    5. identify dependencies unlocked
    6. immediately start the highest-priority unblocked task

Never stop simply because a task completed.

Never start work merely because hardware is available.

Always choose:

    highest-priority
    unblocked
    useful
    time-efficient task

===============================================================================
35. FINAL RULE

The agent must NEVER lose the execution plan.

It must always know:

    WHAT is happening now
    WHY it is happening
    WHAT depends on it
    WHAT happens next
    HOW LONG it should take
    WHAT can run in parallel
    WHAT should be stopped

The mission is:

    correct the real-photo shortcut
    improve high-resolution robustness
    preserve AIGC detection
    validate on unseen generators
    use available compute efficiently
    finish within 20 hours
    freeze the best verified model
    move immediately to deployment
