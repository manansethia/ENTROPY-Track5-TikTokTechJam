# PHASE 6 — FINAL ARCHITECTURE VALIDATION BEFORE FULL-CORPUS TRAINING

AUTHORITY
---------
Read:

    AUTH_PHASE1.md
    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Treat the verified Phase-4 checkpoint and Phase-5 reports as historical
evidence.

DO NOT start the 400–600+ GB final training yet.

The goal of Phase 6 is to resolve the remaining Phase-5 ambiguities and
produce ONE trustworthy final training specification.

============================================================
1. FREEZE PHASE-5
============================================================

Verify and hash:

    checkpoints/phase5/phase5_champion_model.pt

Record:

- checkpoint SHA256
- architecture
- preprocessing
- feature dimensions
- loss
- lambda_FP
- calibration
- threshold
- dataset hash
- code version

Do not modify the Phase-5 checkpoint.

============================================================
2. RECONCILE PHASE-5 PERFORMANCE
============================================================

The reported Phase-5 locked test is:

    AUROC = 0.9986
    AUPRC = 0.9990
    FP = 40
    FN = 146
    FPR = 0.94%
    TPR = 97.60%

Verify these directly from raw prediction arrays.

Compare against Phase 4:

    AUROC = 0.9986
    AUPRC = 0.9991
    FP = 42
    FN = 129
    FPR = 0.99%
    TPR = 97.88%

Do NOT describe Phase 5 as universally better.

Determine exactly where it is better and where it is worse.

============================================================
3. CRITICAL CONDITIONAL-VERIFIER AUDIT
============================================================

The Phase-5 report claims:

    18 FP rescued
    112 FN rescued
    net error delta = -28

while the final locked test contains:

    40 FP
    146 FN

Reconcile this.

Determine:

A. Are the 18/112 numbers from the locked test?
B. Are they from validation?
C. Are they from a separate verifier profiling set?
D. Is the verifier actually connected to the final checkpoint?
E. Does the locked-test prediction pipeline actually invoke Stage 2?
F. What are the exact Stage-1 and Stage-2 predictions for each affected
   image?

Do NOT accept a narrative explanation.

Trace the actual image IDs and prediction arrays.

Create:

    reports/phase6_conditional_verifier_provenance.json

If the verifier was NOT included in the final locked-test result,
state this explicitly.

============================================================
4. RECOMPUTE THE TRUE END-TO-END PIPELINE
============================================================

Measure separately:

A. RAW IMAGE -> CLIP
B. RAW IMAGE -> SigLIP
C. RAW IMAGE -> SRM
D. RAW IMAGE -> Stage-1 fusion
E. RAW IMAGE -> Stage-2 DINO/Edge
F. FULL RAW IMAGE -> FINAL DECISION

Do NOT report cached-feature throughput as image inference throughput.

Report:

    preprocessing time
    backbone time
    fusion time
    verifier time
    end-to-end latency
    average latency
    P95 latency
    P99 latency
    worst-case latency
    VRAM
    RAM
    throughput

Also report:

    percentage of images triggering Stage 2

The previous "845,000 images/sec" number MUST be clearly classified as
cached-head throughput if that is what it represents.

============================================================
5. TEST WHETHER THE CONDITIONAL VERIFIER ACTUALLY HELPS
============================================================

Using a fresh development split:

Compare:

A.
    Phase-5 Stage 1 only

B.
    Stage 1 + DINO/Edge verifier

C.
    Stage 1 + DINO + Edge + ConvNeXt

D.
    Stage 1 + DINO + Edge + EVA

E.
    Stage 1 + all useful specialists

Do NOT automatically include FFT/Patch-MIL.

Measure:

    AUROC
    AUPRC
    FP
    FN
    FPR
    FNR
    TPR
    TNR

and especially:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

============================================================
6. TEST THE VERIFIER AS A TRUE CONDITIONAL SYSTEM
============================================================

Do not simply run Stage 2 on every image.

Measure the routing policy.

For multiple candidate uncertainty windows, report:

    coverage
    Stage-2 invocation rate
    FP reduction
    FN reduction
    TPR
    FPR
    average latency
    P95 latency

Candidate windows may include:

    [0.30, 0.70]
    [0.35, 0.75]
    [0.35, 0.85]
    [0.40, 0.90]

but do NOT assume any one is correct.

Select using development data.

============================================================
7. HARD-EXAMPLE DATA
============================================================

The Phase-5 hard mining findings are important.

Hard REAL:

- extreme optical bokeh
- macro photography
- intense studio flash
- fine-art texture
- high-frequency sensor characteristics

Hard AIGC:

- subtle SID latent diffusion
- low-artifact diffusion
- photorealistic Quality Paradox
- modern subtle generators

Do NOT discard the remainder of the corpus.

Create:

    NORMAL DATA
    +
    HARD REAL
    +
    HARD AIGC

The hard pools should be weighted, not allowed to replace the full
distribution.

Use diversity constraints so the model does not simply memorize the
hard pools.

============================================================
8. EXPAND TRAINING DATA SUBSTANTIALLY
============================================================

The eventual objective is to exploit the complete approved corpus.

Do NOT stop at 68K training images.

Inventory the full approved corpus again.

Use the largest scientifically useful unique subset.

Ultimately target:

    hundreds of thousands of unique images

and, where the approved corpus supports it:

    millions of unique images.

Do NOT count repeated sampling as new independent data.

Report:

    unique images
    effective samples/epoch
    repeat factor
    generator exposure
    real-domain exposure

============================================================
9. GENERATOR DIVERSITY
============================================================

Prevent any single generator from dominating.

Explicitly balance exposure across available:

    FLUX
    SDXL
    SD3
    Midjourney
    PixArt
    SID
    HFCF
    Quality Paradox
    other approved generators

The goal is:

    generator-invariant AIGC detection

rather than:

    generator-specific classification.

============================================================
10. REAL-DOMAIN DIVERSITY
============================================================

Increase difficult REAL exposure.

Prioritize actual available:

    COCO
    high-frequency photography
    macro
    bokeh
    studio photography
    archival photography
    WikiArt
    painting
    sketches
    legitimate digital art
    compression-heavy photographs
    resized real images

Especially preserve categories that caused Phase-5 FP.

============================================================
11. DATASET STRUCTURE
============================================================

Create independent:

    FINAL6_TRAIN
    FINAL6_DEV
    FINAL6_CALIBRATION
    FINAL6_INTERNAL_TEST

Use exact SHA256 deduplication.

Use perceptual duplicate detection where feasible.

Do not allow previous development data to contaminate the new final
development set unless explicitly documented.

The internal test remains locked.

============================================================
12. TRAINING ARCHITECTURE
============================================================

Test only the architectures justified by the evidence.

At minimum:

A.
    CLIP + SigLIP + SRM
    Structured Dropout

B.
    CLIP + SigLIP + SRM
    Structured Dropout
    +
    conditional DINO + Edge verifier

C.
    CLIP + SigLIP + SRM
    Structured Dropout
    +
    conditional DINO + Edge + ConvNeXt

D.
    full useful specialist conditional system

E.
    all-9 control

Keep large backbones frozen initially.

Do NOT fine-tune 1B+ parameters automatically.

============================================================
13. ADAPTATION
============================================================

Phase 5 found:

    Frozen head AUROC ≈ 0.9991
    LoRA AUROC ≈ 0.9992

The gain is tiny relative to the additional trainable parameters and
VRAM.

Therefore:

Do NOT automatically select LoRA.

Only retain adaptation if fresh large-data training demonstrates a
meaningful improvement in:

    low-FPR TPR
    FN
    OOD
    robustness

============================================================
14. LOSS
============================================================

Use:

    lambda_FP = 2.5

as the baseline.

Test only a small controlled set around it:

    2.0
    2.5
    3.0

Choose based on:

    TPR subject to FPR constraint

not ordinary accuracy.

============================================================
15. ULTRA-LOW-FPR OBJECTIVE
============================================================

The primary deployment targets are:

    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

For each architecture provide:

    best achievable TPR under each FPR constraint

Do NOT claim an operating point if the evaluation population cannot
resolve it.

============================================================
16. CALIBRATION
============================================================

Fit fresh calibration.

Test:

    temperature scaling
    Platt scaling
    isotonic where justified

Do NOT automatically reuse:

    T = 1.213654

from Phase 5.

Measure:

    ECE
    Brier
    high-confidence tail calibration

============================================================
17. THRESHOLD
============================================================

Generate the full threshold curve.

Do NOT assume:

    tau = 0.80

is final.

Find operating points for:

    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

Report:

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

============================================================
18. ROBUSTNESS
============================================================

Evaluate:

    clean
    JPEG90
    JPEG70
    JPEG50
    JPEG30
    blur
    resize
    noise
    crop
    color jitter
    sharpening
    recompression

Calculate:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    RI
    worst-case AUROC

============================================================
19. GENERATOR OOD
============================================================

Report generator-specific results.

Do not allow a high aggregate score to hide a generator collapse.

Especially inspect:

    SID
    Quality Paradox
    FLUX
    SDXL
    SD3
    Midjourney
    PixArt

============================================================
20. EXTERNAL OOD
============================================================

Keep locked:

    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

Do not tune against them.

Evaluate only after the candidate is frozen.

============================================================
21. INTERNAL TEST
============================================================

After architecture/fusion/loss/calibration/threshold are all frozen:

evaluate FINAL6_INTERNAL_TEST ONCE.

Do not tune after seeing the result.

============================================================
22. FINAL LARGE-CORPUS TRAINING
============================================================

ONLY AFTER the conditional-verifier audit establishes which system
actually improves the low-FPR/FN tradeoff should the full approved corpus
be trained.

Use the largest scientifically useful corpus practical on the machine.

Do not artificially cap at 100K merely because earlier phases used
smaller datasets.

Use:

    NVMe staging
    RAM hot cache
    pinned memory
    asynchronous prefetch
    non-blocking GPU transfer

Do NOT put hundreds of GB into RAM.

Do NOT use swap as active cache.

============================================================
23. HARD-NEGATIVE ITERATION
============================================================

Perform at most two hard-example mining rounds.

Each round:

    train
    mine
    analyze
    reweight
    retrain

Preserve a completely independent development set.

Never mine from internal test or OOD.

============================================================
24. REQUIRED FINAL OUTPUT
============================================================

Produce:

    reports/phase6_conditional_verifier_provenance.json
    reports/phase6_end_to_end_latency.json
    reports/phase6_large_cocktail_comparison.json
    reports/phase6_routing_comparison.json
    reports/phase6_hard_example_effect.json
    reports/phase6_scale_comparison.json
    reports/phase6_loss_comparison.json
    reports/phase6_calibration.json
    reports/phase6_threshold_analysis.json
    reports/phase6_robustness.json
    reports/phase6_generator_breakdown.json
    reports/phase6_domain_breakdown.json
    reports/phase6_final_architecture_decision.json
    reports/phase6_final_training_plan.md

Update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

============================================================
25. FINAL DECISION
============================================================

The final recommendation must answer:

1. Is Stage 2 genuinely part of the final system?
2. Does DINO actually help?
3. Does Edge actually help?
4. Does ConvNeXt help?
5. Does EVA justify its cost?
6. Does all-9 help?
7. Which experts should be dropped?
8. What architecture achieves the best TPR at FPR <= 0.1%?
9. What architecture achieves the best TPR at FPR <= 0.01%?
10. What architecture has the best overall FP/FN tradeoff?
11. What is the actual end-to-end inference latency?
12. What is the recommended loss?
13. What is the recommended calibration?
14. What is the recommended threshold?
15. What is the recommended final training corpus size?
16. Should the full 400–600+ GB corpus now be used?
17. Should LoRA/adapters be used?
18. What is the final training configuration?

============================================================
26. ABSOLUTE SCIENTIFIC RULE
============================================================

If the conditional verifier does not actually improve the locked-test
or genuinely fresh-development FP/FN frontier:

DROP IT.

If the larger cocktail does not improve:

DO NOT USE IT.

If all-9 is worse:

DOCUMENT THAT AND DROP IT.

If a small 3–5 expert architecture wins:

USE THE SMALLER ARCHITECTURE.

Do not add complexity for prestige.

============================================================
27. FINAL STOP
============================================================

Do NOT automatically launch the final 400–600+ GB training merely because
Phase 5 finished.

Complete the Phase-6 validation first.

Then produce the final training specification and STOP.

The next full-scale run should consume the maximum scientifically useful
approved data and should be the last major architecture-selection cycle
before final model production.

BEGIN WITH:

    STEP 0 — VERIFY PHASE-5 CHECKPOINT AND RECONCILE CONDITIONAL VERIFIER