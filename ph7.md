# PHASE 7 — FINAL PRE-FULL-CORPUS VALIDATION
# VERIFY CONDITIONAL SYSTEM + THRESHOLDS + DATA SCALE
# THEN AUTHORIZE DEFINITIVE 400–600+ GB TRAINING

============================================================
AUTHORITY
============================================================

Read:

    AUTH_PHASE1.md
    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Treat Phase 4, Phase 5 and Phase 6 as historical experimental evidence.

Do NOT modify their frozen checkpoints.

The verified Phase-4 / Phase-6 primary model is:

    CLIP-ViT-L/14
    +
    SigLIP-SO400M-224
    +
    SRM-DWT

with:

    Structured Branch Dropout MLP
    hidden_dim = 256
    LayerNorm
    GELU
    dropout = 0.15

The proposed conditional verifier is:

    DINOv2-Registers
    +
    Edge-Specialist

============================================================
1. DO NOT START THE 400–600 GB FINAL TRAINING YET
============================================================

Before the definitive full-corpus run, perform ONE final validation
experiment.

Do NOT run another broad architecture search.

Do NOT add arbitrary new experts.

Do NOT perform a giant hyperparameter sweep.

The only remaining questions are:

1. Does the conditional DINO/Edge verifier genuinely improve the
   final system?
2. What is the correct ultra-low-FPR operating curve?
3. What is the correct threshold/review policy?
4. What exact architecture/loss/calibration configuration should be
   frozen for full-corpus training?
5. Is the evidence strong enough to commit the entire approved corpus?

============================================================
2. CRITICAL: RECONCILE THE CONDITIONAL-VERIFIER CLAIM
============================================================

The Phase-6 report states:

    18 FP rescued
    112 FN rescued
    6.8% Stage-2 invocation

but also explicitly states that the LOCKED INTERNAL TEST was evaluated
with Stage-1 alone.

Therefore determine exactly:

A. Which development images produced the 18 FP rescues?
B. Which development images produced the 112 FN rescues?
C. Did Stage-2 actually change each prediction?
D. What were the Stage-1 and Stage-2 scores?
E. Was the final decision improved or merely changed?
F. What new FP did Stage-2 introduce?
G. What new FN did Stage-2 introduce?
H. What is the net change in errors?
I. What is the change in AUROC/AUPRC?
J. What is the change in the FPR/TPR frontier?

Create:

    reports/phase7_conditional_verifier_audit.json
    reports/phase7_conditional_verifier_audit.md

============================================================
3. FRESH DEVELOPMENT VALIDATION OF STAGE-2
============================================================

Use a fresh development population not previously used to optimize
the Stage-2 routing policy if available.

Do NOT use:

    locked internal test
    external OOD

for tuning.

Evaluate:

A. Stage-1 only

B. Stage-1 + Stage-2 DINO/Edge

Compare:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    TNR
    FP
    FN

and especially:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

============================================================
4. TEST MULTIPLE ROUTING WINDOWS
============================================================

The existing recommendation is:

    [0.35, 0.85]

Do NOT assume this is optimal.

Evaluate a small controlled set:

    [0.30, 0.70]
    [0.35, 0.75]
    [0.35, 0.85]
    [0.40, 0.90]

For each report:

    Stage-2 invocation %
    FP rescued
    FN rescued
    new FP
    new FN
    net error change
    TPR
    FPR
    FNR
    average latency
    P95 latency
    P99 latency
    worst-case latency

Select the routing window based on validation evidence.

============================================================
5. CRITICAL THRESHOLD RECONCILIATION
============================================================

The current Phase-6 report states:

    TPR = 90.41% at tau = 0.9993
    FPR = 0.09%

but the supplied threshold table only goes to:

    tau = 0.99
    TPR = 87.08%
    FPR = 0.46%

This must be independently reconciled.

Do NOT trust the narrative.

Recompute the threshold curve directly from raw prediction arrays.

Include at least:

    tau = 0.90
    0.92
    0.94
    0.95
    0.96
    0.97
    0.98
    0.99
    0.995
    0.997
    0.998
    0.999
    0.9993
    0.9995
    0.9997
    0.9999

For every point report:

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

If a threshold/result cannot be reproduced:

    MARK IT INVALID.

============================================================
6. LOW-FPR OPERATING FRONTIER
============================================================

Calculate the best achievable TPR for:

    FPR <= 1.0%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

For each constraint report:

    selected threshold
    TP
    TN
    FP
    FN
    TPR
    FNR
    precision

Do NOT claim an FPR that is below the resolution supported by the number
of real samples.

If the real-image sample count is insufficient to support a 0.01% claim:

    explicitly write:
    "INSUFFICIENT SAMPLE SIZE FOR RELIABLE EMPIRICAL RESOLUTION."

============================================================
7. THREE-WAY OPERATING POLICY
============================================================

Test:

    HIGH-CONFIDENCE REAL
    REVIEW / VERIFY
    HIGH-CONFIDENCE AIGC

The review zone should route into the DINO/Edge verifier.

Do NOT simply invent the bands.

Search a small controlled set around the validated operating frontier.

Report:

    coverage
    Stage-2 invocation
    FP
    FN
    FPR
    FNR
    TPR
    average latency
    P95 latency

Determine whether the three-way system is genuinely superior to a
single-threshold Stage-1 system.

============================================================
8. CALIBRATION RECONCILIATION
============================================================

Do NOT blindly reuse:

    T = 1.208419

Re-fit temperature scaling on the designated calibration set.

Verify:

    ECE
    Brier
    high-confidence calibration

especially near:

    p > 0.95

and:

    p > 0.99

Compare calibrated vs uncalibrated predictions.

============================================================
9. HARD-EXAMPLE VALIDATION
============================================================

Verify the previously discovered hard pools:

HARD REAL:
    macro/bokeh photography
    studio flash
    high-frequency legitimate imagery
    fine-art textures

HARD AIGC:
    subtle SID diffusion
    low-artifact diffusion
    modern photorealistic AIGC
    Quality Paradox

Measure:

    baseline FP/FN
    post-training FP/FN
    verifier corrections

Do not assume that the hard pools remain hard after additional training.

============================================================
10. FINAL ARCHITECTURE COMPARISON
============================================================

Compare only:

A.
    Tri-Stream Stage-1

B.
    Tri-Stream + conditional DINO/Edge

C.
    Tri-Stream + conditional DINO/Edge + optional expert gating

Do NOT repeat the discarded all-9 architecture search.

The all-9 model remains a historical control and does not need retraining
unless a specific inconsistency is discovered.

============================================================
11. FULL-CORPUS DATA READINESS
============================================================

After the architectural validation, audit the COMPLETE approved corpus.

Do not stop at the current ~100K scale.

The final training objective is to exploit the maximum scientifically
useful approved data from the available 400–600+ GB corpus.

Use all eligible unique data that passes:

    provenance checks
    exact deduplication
    near-duplicate screening where feasible
    label validation
    train/dev/calibration/test isolation

Do NOT artificially cap the final corpus at 100K merely because earlier
phases used smaller sets.

============================================================
12. FINAL CORPUS COMPOSITION
============================================================

Preserve diversity across:

AIGC:

    SID
    Quality Paradox
    FLUX
    SDXL
    SD3
    Midjourney
    PixArt
    HFCF
    other approved generator families

REAL:

    COCO
    photography
    WikiArt
    archival
    high-frequency real imagery
    difficult camera imagery
    legitimate digital art
    hard-negative photography

Do NOT allow one generator to dominate.

Do NOT pretend repeated samples are new information.

Report:

    unique images
    training exposures
    repeat factor
    per-generator exposure
    per-domain exposure

============================================================
13. HARD-EXAMPLE MIX
============================================================

Include:

    NORMAL DATA
    +
    HARD REAL
    +
    HARD AIGC

Use hard-example weighting.

Do NOT train exclusively on hard examples.

Keep enough ordinary examples to prevent over-specialization.

============================================================
14. LOSS
============================================================

Use:

    lambda_FP = 2.5

as the default baseline.

Test only:

    lambda_FP = 2.0
    lambda_FP = 2.5
    lambda_FP = 3.0

on the final development population.

Select based primarily on:

    TPR subject to FPR <= 0.1%

and secondarily:

    FNR
    AUROC
    AUPRC
    robustness
    calibration

============================================================
15. FROZEN BACKBONES FIRST
============================================================

Keep:

    CLIP
    SigLIP
    DINO
    other specialists

frozen initially.

Train:

    fusion head
    routing head
    calibration parameters

Do NOT fine-tune all 1B+ parameters automatically.

Only test lightweight adapter tuning if fresh evidence shows that frozen
features have reached a meaningful performance ceiling.

============================================================
16. FEATURE EXTRACTION
============================================================

For the final corpus:

Generate fresh features from RAW images.

Do NOT use historical Phase-2/3/4/5/6 feature caches.

Within the final run, a cache may be reused if its provenance exactly
matches:

    manifest SHA
    checkpoint SHA
    preprocessing hash
    feature dimension
    dtype
    extraction-code version

Store this provenance beside the cache.

============================================================
17. STORAGE / RAM / NVME
============================================================

Hardware:

    RTX 3050 6 GB
    ~31 GB RAM
    ~24 GB swap
    ~400 GB+ NVMe

Use:

    source storage
        ->
    NVMe staging
        ->
    asynchronous prefetch
        ->
    pinned RAM
        ->
    non-blocking GPU transfer

Do NOT load the full 400–600 GB corpus into RAM.

Do NOT use swap as a dataset cache.

Use RAM as bounded hot cache.

Use NVMe as persistent large cache.

Target:

    sustained swap I/O ≈ 0

============================================================
18. TRAINING CHECKPOINTS
============================================================

Checkpoint:

    model
    optimizer
    scheduler
    sampler state
    RNG
    epoch
    step
    manifest hash
    architecture
    preprocessing
    loss
    routing policy

Never overwrite the only good checkpoint.

============================================================
19. FINAL INTERNAL TEST
============================================================

Only after:

    architecture frozen
    fusion frozen
    loss frozen
    routing frozen
    calibration frozen
    threshold/review policy frozen

run the locked internal test ONCE.

If Stage 2 is part of the proposed final system:

the locked internal test must be run through the ACTUAL FINAL END-TO-END
PIPELINE, including Stage 2 routing.

Do NOT report Stage-1-only internal-test numbers as though they represent
the complete final system.

============================================================
20. EXTERNAL OOD
============================================================

Only after final model freezing:

run the approved external evaluation on:

    Synthbuster
    AIGIBench
    Chameleon
    VCT²
    WildRF
    SynthWildX

where legitimately available.

Do not tune afterward.

============================================================
21. EXPLANATION / FORENSIC REASONING EXTENSION
============================================================

Do NOT make free-form explanation generation part of the initial
classification loss.

First establish the final classifier.

Then, as an AUXILIARY forensic reasoning layer, create:

    classification
        +
    evidence tags
        +
    localization / evidence map
        +
    explanation

For difficult or uncertain examples, generate candidate evidence such as:

    hand/finger anomaly
    facial geometry
    text/glyph inconsistency
    perspective inconsistency
    reflection/shadow inconsistency
    texture inconsistency
    brushstroke inconsistency
    edge anomaly
    frequency/residual anomaly
    compression/resampling artifact

BUT:

These categories are hypotheses.

Do not claim an artifact exists unless it is supported by measurable
evidence.

The explanation system must NOT be allowed to create arbitrary textual
justifications that become self-reinforcing labels.

============================================================
22. EXPLANATION VERIFICATION
============================================================

For difficult cases:

1. Generate candidate explanation.
2. Extract the claimed evidence region.
3. Check the evidence against independent specialists.
4. Perform counterfactual/occlusion testing where computationally
   reasonable.
5. Score explanation consistency.

Example:

    Original P(AIGC) = 0.97

    Mask claimed suspicious hand
    P(AIGC) = 0.60

This provides evidence that the hand was relevant.

But if:

    Original = 0.97
    Mask hand = 0.96

then the explanation may not be causally supported.

Do NOT use unrestricted LLM self-judgment as the sole truth source.

============================================================
23. EXPLANATION REWARD / PENALTY
============================================================

Only after the evidence verification pipeline exists, test an auxiliary
reward system such as:

    correct classification + evidence supported     -> positive reward
    correct classification + unsupported evidence    -> neutral/small penalty
    wrong classification                              -> strong penalty
    confidently fabricated evidence                  -> stronger penalty

Do NOT use arbitrary reward magnitudes without first testing stability.

The classification objective remains primary.

The evidence objective is auxiliary.

Never allow an explanation reward to override the actual image label.

============================================================
24. HARD FP/FN FORENSIC LOOP
============================================================

For every hard-example mining round:

REAL:
    mine highest P(AIGC)
    identify why the model is suspicious

AIGC:
    mine lowest P(AIGC)
    identify what synthetic evidence is being missed

Then:

    independent specialist analysis
        ->
    candidate explanation
        ->
    evidence verification
        ->
    targeted reweighting
        ->
    retraining

Maximum:

    2 hard-example mining rounds

unless validation evidence explicitly justifies another.

Keep validation/test untouched.

============================================================
25. REQUIRED FINAL REPORTS
============================================================

Produce:

    reports/phase7_conditional_verifier_audit.json
    reports/phase7_threshold_reconciliation.json
    reports/phase7_operating_policy.json
    reports/phase7_calibration_reconciliation.json
    reports/phase7_hard_example_validation.json
    reports/phase7_full_corpus_inventory.json
    reports/phase7_final_architecture_validation.json

Then, once the final system is authorized:

    reports/final_training_manifest.json
    reports/final_training_provenance.json
    reports/final_training_telemetry.json
    reports/final_training_results.json
    reports/final_calibration.json
    reports/final_threshold_analysis.json
    reports/final_fp_fn_analysis.json
    reports/final_generator_breakdown.json
    reports/final_domain_breakdown.json
    reports/final_robustness.json
    reports/final_internal_test.json
    reports/final_ood_results.json

Also update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

============================================================
26. ANTI-HALLUCINATION RULE
============================================================

Every important number must be recoverable from:

    raw predictions
    labels
    manifest
    checkpoint
    configuration
    telemetry

If two reports disagree:

    RECOMPUTE.

If the result cannot be reproduced:

    MARK "NOT ESTABLISHED."

Never choose the more favorable number.

============================================================
27. FINAL AUTHORIZATION GATE
============================================================

After Phase 7 completes, produce:

    reports/final_full_corpus_training_authorization.json
    reports/final_full_corpus_training_authorization.md

The authorization report must state:

    FINAL_ARCHITECTURE
    FINAL_ROUTING
    FINAL_LOSS
    FINAL_LAMBDA_FP
    FINAL_CALIBRATION
    FINAL_THRESHOLD
    FINAL_REVIEW_BAND

    TRAINING_CORPUS_SIZE
    UNIQUE_IMAGES
    REAL_COUNT
    AIGC_COUNT
    GENERATOR_DISTRIBUTION
    REAL_DOMAIN_DISTRIBUTION

    EXPECTED_THROUGHPUT
    EXPECTED_TRAINING_TIME
    EXPECTED_VRAM
    EXPECTED_RAM

    REMAINING_RISKS

If all gates pass:

    FULL_CORPUS_TRAINING = AUTHORIZED

Otherwise:

    FULL_CORPUS_TRAINING = NOT_AUTHORIZED

============================================================
FINAL PRINCIPLE
============================================================

We are now optimizing the extreme end of the ROC curve.

Do not chase a prettier AUROC number.

The real target is:

    FPR << 1%
    +
    maximum possible TPR
    +
    low FNR
    +
    strong generator generalization
    +
    strong real-domain specificity
    +
    robustness
    +
    calibration
    +
    efficient inference

Use the large corpus to increase diversity.

Use hard REAL images to reduce false accusations.

Use subtle AIGC to reduce missed fakes.

Use DINO/Edge conditionally if they demonstrably help.

Use evidence-based explanations as an auxiliary forensic layer.

Do not let the explanation system become a source of label contamination.

Do not use all experts merely because they exist.

Do not discard useful experts merely because they are weaker alone.

Let the fresh evidence decide.

BEGIN WITH:

    STEP 1 — RECONCILE STAGE-2 VERIFIER
    STEP 2 — RECOMPUTE ULTRA-LOW-FPR THRESHOLD CURVE
    STEP 3 — VALIDATE FINAL END-TO-END ROUTING
    STEP 4 — AUTHORIZE FULL-CORPUS TRAINING