# PHASE 5 MASTER DIRECTIVE
# ULTRA-LOW-FPR + HARD-EXAMPLE MINING + CONDITIONAL MULTI-EXPERT VERIFIER

============================================================
AUTHORITY
============================================================

Read:

    AUTH_PHASE1.md
    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Read and verify the complete frozen Phase-4 reconciliation:

    reports/phase4_final_reconciliation.json
    reports/phase4_final_reconciliation.md
    reports/phase4_final_report.json

The authoritative Phase-4 baseline is:

    Cand_C_Structured_Dropout

Architecture:

    CLIP-ViT-L/14
    +
    SigLIP-SO400M-224
    +
    SRM-DWT

Representation:

    2212 dimensions

Fusion:

    Structured Branch Dropout MLP
    p = 0.15
    hidden = 256
    LayerNorm
    GELU

Trainable parameters:

    567,297

Frozen checkpoint:

    checkpoints/phase4/phase4_champion_model.pt

Checkpoint SHA-256:

    b53479d0aa7c4eb1f4af9e8f4d6a39fc53ac260fdea7b58b42bc68253de37b59

This checkpoint is FROZEN and must never be modified.

============================================================
PRIMARY OBJECTIVE
============================================================

The present Phase-4 detector is already very strong:

    INTERNAL TEST AUROC = 0.9986
    INTERNAL TEST AUPRC = 0.9991
    FPR @ tau=0.80 = 0.99%
    TPR @ tau=0.80 = 97.88%

However:

    42 FP / 4238 REAL
    129 FN / 6078 AIGC

still remain.

At deployment scale, 1% FPR is too large.

Therefore the primary objective of Phase 5 is:

    DRAMATICALLY REDUCE FPR

while preserving as much TPR as possible.

Secondary objective:

    DRAMATICALLY REDUCE FN

especially subtle modern AIGC.

The optimization target is therefore:

    HIGH TPR
    subject to
    VERY LOW FPR

and simultaneously:

    LOW FNR

Do NOT optimize ordinary accuracy alone.

============================================================
CONFUSION MATRIX DEFINITIONS
============================================================

Use one immutable definition:

    REAL -> REAL = TN
    REAL -> FAKE = FP
    FAKE -> REAL = FN
    FAKE -> FAKE = TP

Positive class:

    AIGC / FAKE = 1

Negative class:

    AUTHENTIC / REAL = 0

Always use these definitions.

============================================================
PHASE 5 CORE HYPOTHESIS
============================================================

The current model has already demonstrated strong general discrimination.

Therefore the next gain is expected to come primarily from:

1. hard-example diversity
2. difficult authentic images
3. subtle modern AIGC
4. better conditional specialist evidence
5. stronger calibration
6. low-FPR operating-point optimization
7. targeted adaptation rather than blind full-backbone fine-tuning

Do NOT assume that simply increasing parameter count will improve the
extreme-low-FPR regime.

Test the hypothesis.

============================================================
STEP 0 — FREEZE PHASE-4
============================================================

Before doing anything:

1. SHA-256 the Phase-4 checkpoint.
2. SHA-256 the Phase-4 manifests.
3. Record all baseline artifacts.
4. Create a separate Phase-5 experiment namespace.

Never overwrite Phase-4 files.

Historical Phase-4 data can be used for context and comparison.

Do NOT use its internal-test predictions for training.

============================================================
STEP 1 — COMPLETE APPROVED DATASET INVENTORY
============================================================

Scan the complete APPROVED training corpus currently available.

Do not rely on directory names alone.

Inspect actual dataset content and metadata.

Inventory:

- dataset
- source
- total images
- REAL/AIGC label
- generator family
- generator/model
- image resolution
- image format
- image domain
- provenance
- duplicate status
- SHA-256
- perceptual duplicate where feasible

Do NOT access locked external benchmarks.

The locked evaluation datasets remain completely isolated.

============================================================
STEP 2 — HARD-EXAMPLE DISCOVERY FROM THE ACTUAL DATA
============================================================

We specifically want the agent to FIND difficult examples inside the
approved datasets.

Do not invent hard-example categories.

Use actual dataset evidence.

Create two pools:

    HARD_REAL_POOL
    HARD_AIGC_POOL

------------------------------------------------------------
2A. HARD REAL POOL
------------------------------------------------------------

Search the approved REAL datasets for images likely to challenge the
detector.

Sources may include, where actually present:

- COCO
- WikiArt
- photography datasets
- archival photography
- high-resolution photography
- difficult authentic imagery
- digital art
- CGI/3D renders
- compressed imagery
- web imagery
- other approved REAL sources

Potential hard-real properties to investigate:

- strong bokeh
- shallow depth of field
- macro photography
- foliage
- hair/fur
- fine textures
- high-frequency structures
- HDR
- high local contrast
- unusual lighting
- heavy sharpening
- JPEG artifacts
- resizing
- motion blur
- optical blur
- sensor noise
- unusual color processing
- digitally processed but authentic imagery

IMPORTANT:

These are CANDIDATE categories only.

Do not assume they are hard.

Use actual model scoring and data analysis to establish which categories
produce elevated false-AIGC scores.

------------------------------------------------------------
2B. HARD AIGC POOL
------------------------------------------------------------

Search the approved AIGC corpus for difficult synthetic examples.

Prioritize actual available examples from:

- SID diffusion
- Quality Paradox
- FLUX
- SDXL
- SD3
- Midjourney
- PixArt
- other approved modern generators

Look especially for:

- photorealistic AIGC
- low-artifact AIGC
- subtle diffusion
- modern generators
- difficult photographic scenes
- post-processed AIGC
- JPEG-compressed AIGC
- downsampled AIGC
- blurred AIGC
- visually realistic AIGC

Again:

Do NOT assume a category is hard.

Measure actual model confidence.

============================================================
STEP 3 — MODEL-BASED HARD MINING
============================================================

Use the FROZEN Phase-4 model to score a large candidate subset of the
approved TRAINING DATA ONLY.

Never score the internal test or locked OOD data for mining.

For REAL images:

    rank by P(AIGC) descending.

The highest-scoring REAL images are candidate hard negatives.

For AIGC images:

    rank by P(AIGC) ascending.

The lowest-scoring AIGC images are candidate hard positives.

Store:

- image ID
- source
- generator/domain
- score
- resolution
- metadata
- reason for inclusion
- original training split

Create:

    reports/phase5_hard_negative_mining.json
    reports/phase5_hard_positive_mining.json

Create manifests:

    manifests/phase5_hard_real.jsonl
    manifests/phase5_hard_aigc.jsonl

Do NOT move validation/test images into these pools.

============================================================
STEP 4 — STRATIFIED HARD-EXAMPLE COMPOSITION
============================================================

Do NOT simply select the top 10,000 highest-scoring images.

That could create another shortcut.

Construct hard-example pools with diversity constraints.

For HARD_REAL:

balance across available real domains.

For HARD_AIGC:

balance across available generator families.

Also retain ordinary non-hard samples.

The model needs:

    NORMAL DATA
    +
    HARD REAL
    +
    HARD AIGC

Do not train exclusively on mistakes.

============================================================
STEP 5 — QUANTIFY HARD-EXAMPLE VALUE
============================================================

Measure:

    baseline FP rate on hard-real pool
    baseline FN rate on hard-AIGC pool

Break down by:

- dataset
- generator
- domain
- resolution
- compression
- image type

Determine whether the difficult examples are concentrated.

Report:

    fraction of all errors represented by hard pool
    generator concentration
    domain concentration

Do not claim improvement until it is measured.

============================================================
STEP 6 — CONSTRUCT A FRESH PHASE-5 DEVELOPMENT SET
============================================================

The Phase-4 development set has already been heavily examined.

Create a new FINAL_PHASE5_DEV.

Minimum target:

    >= 10,000 images

Prefer larger if the approved corpus supports it.

Create:

    PHASE5_TRAIN
    PHASE5_DEV
    PHASE5_CALIBRATION
    PHASE5_INTERNAL_TEST

The Phase-5 DEV and CALIBRATION pools must be genuinely independent from
the historical Phase-2/3/4 development populations whenever the corpus
permits.

Use:

    exact SHA-256 deduplication
    +
    perceptual similarity checks where feasible

for split separation.

============================================================
STEP 7 — HARD-EXAMPLE CURRICULUM
============================================================

Train the model in stages.

STAGE A:

Train on ordinary diverse data.

STAGE B:

Introduce hard-real and hard-AIGC examples.

STAGE C:

Evaluate whether hard-example training improves the actual low-FPR
operating region.

Do NOT continuously mine the same errors indefinitely.

Use a bounded number of mining rounds.

At each round preserve an independent validation set.

============================================================
STEP 8 — ARCHITECTURE EXPERIMENT
============================================================

Do NOT assume the 3-stream architecture is optimal.

Evaluate the following candidates:

A.
    CLIP + SigLIP + SRM

B.
    CLIP + SigLIP + SRM + Edge

C.
    CLIP + SigLIP + SRM + DINO

D.
    CLIP + SigLIP + SRM + Edge + DINO

E.
    CLIP + SigLIP + SRM + Edge + DINO + ConvNeXt

F.
    CLIP + SigLIP + SRM + Edge + DINO + ConvNeXt + EVA

G.
    ALL 9 EXPERTS

Do not omit G.

It is a required scientific control.

Do NOT automatically deploy G.

============================================================
STEP 9 — CONDITIONAL SPECIALIST ARCHITECTURE
============================================================

The most important new architectural experiment is a TWO-STAGE detector.

STAGE 1:

    CLIP + SigLIP + SRM

fast primary detector.

STAGE 2:

For uncertain / suspicious images, invoke additional specialists:

    DINO
    Edge
    ConvNeXt
    EVA
    optional Patch-MIL
    optional FFT only as a control

The Stage-2 specialists should not necessarily receive equal weight.

Implement a small reliability-aware gated verifier.

Conceptually:

    Image
      |
      v
    Stage 1
      |
      +---- very confident REAL ------> REAL
      |
      +---- very confident AIGC ------> AIGC
      |
      +---- uncertain ----------------> Stage 2
                                      |
                                      v
                               specialist verifier
                                      |
                                      v
                                calibrated score

Do NOT hard-code the Stage-1 confidence boundaries.

Learn/test them using PHASE5_DEV.

============================================================
STEP 10 — ULTRA-LOW-FPR OBJECTIVE
============================================================

The key operating constraints are now:

    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

For every model report:

    TPR at each FPR constraint

This is more important than a tiny AUROC difference.

Example required table:

    FPR <= 1.00%  -> TPR = ?
    FPR <= 0.50%  -> TPR = ?
    FPR <= 0.10%  -> TPR = ?
    FPR <= 0.05%  -> TPR = ?
    FPR <= 0.01%  -> TPR = ?

Do NOT report unattainable operating points.

If the empirical validation set cannot resolve a target, explicitly say:

    INSUFFICIENT SAMPLE SIZE

rather than fabricating precision.

============================================================
STEP 11 — THRESHOLD SEARCH
============================================================

Perform a dense threshold search.

Do not assume:

    tau = 0.80

is optimal.

Search the full score range.

For every serious candidate report:

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

Identify:

1. ultra-safe operating point
2. balanced operating point
3. high-recall operating point

============================================================
STEP 12 — STATISTICAL REQUIREMENT FOR VERY LOW FPR
============================================================

At extremely low FPR, sample size matters.

If:

    FP = 0

report:

    FP = 0 / N_REAL

and an appropriate uncertainty bound.

Do NOT say:

    "true FPR = 0"

from a finite sample.

For very small target rates, increase the number of independent REAL
images in PHASE5_DEV whenever possible.

A very low-FPR claim requires a correspondingly large authentic
evaluation population.

============================================================
STEP 13 — CALIBRATION
============================================================

Do not blindly reuse:

    T = 1.208419

from Phase 4.

Fit fresh calibration on:

    PHASE5_CALIBRATION

Compare:

- temperature scaling
- Platt scaling
- isotonic where statistically justified

Evaluate:

- ECE
- Brier
- reliability
- high-confidence tail calibration

The high-confidence tail is particularly important for ultra-low-FPR
deployment.

============================================================
STEP 14 — LOSS
============================================================

Use λ_FP = 2.0 as the current baseline.

Compare a SMALL number of candidates:

    1.5
    2.0
    2.5
    3.0
    4.0

Do NOT run a giant hyperparameter sweep.

The selection criterion should emphasize:

    TPR subject to FPR <= target

rather than ordinary validation accuracy.

============================================================
STEP 15 — CONTROLLED ADAPTATION
============================================================

Initially keep large foundation backbones frozen.

If the hard-example experiments demonstrate a clear representation
ceiling, test lightweight adaptation.

Candidates:

- LoRA
- adapters
- last-block fine-tuning
- final transformer block fine-tuning
- projection-layer tuning

Do NOT fine-tune the entire 1B+ backbone immediately.

For each adaptation test report:

- trainable parameters
- VRAM
- latency
- AUROC
- AUPRC
- FPR
- FNR
- TPR

Only retain adaptation if it provides meaningful held-out improvement.

============================================================
STEP 16 — ROBUSTNESS
============================================================

Evaluate serious candidates under:

    Clean
    JPEG 90
    JPEG 70
    JPEG 50
    JPEG 30
    Blur 0.5
    Blur 1.0
    Blur 2.0
    Resize 0.5
    Resize 0.25
    Noise 0.02
    Noise 0.05
    Noise 0.10
    Center Crop 80
    Color Jitter
    mild sharpening
    recompression

Report:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    TNR

Calculate:

    RI
    worst AUROC
    clean-to-worst degradation

============================================================
STEP 17 — GENERATOR BREAKDOWN
============================================================

Report by:

    FLUX
    SDXL
    SD3
    Midjourney
    PixArt
    SID
    HFCF
    other available families

Also report:

REAL FPR by:

    COCO
    WikiArt
    photography
    archival
    other real sources

============================================================
STEP 18 — ERROR FORENSICS
============================================================

For the final candidate, identify:

TOP FALSE POSITIVES

and:

TOP FALSE NEGATIVES

Record:

- image ID
- score
- threshold
- source
- generator/domain
- resolution
- transformation
- expert scores
- expert disagreement

Determine whether the new architecture genuinely fixes Phase-4 errors.

Do not merely count total errors.

============================================================
STEP 19 — SPECIALIST RESCUE
============================================================

For each specialist calculate:

    FP rescue
    FN rescue
    new FP introduced
    new FN introduced

Compare:

    Stage 1 alone

vs

    Stage 1 + Stage 2

Report:

    ΔFP
    ΔFN
    ΔTPR
    ΔFPR

The specialist must earn its runtime cost.

============================================================
STEP 20 — EFFICIENCY
============================================================

Use:

    NVMe
      ->
    RAM hot cache
      ->
    pinned prefetch
      ->
    non-blocking GPU transfer
      ->
    RTX 3050

Do NOT load the entire 400–600 GB corpus into RAM.

Do NOT use swap intentionally as active memory.

Use approximately 31 GB RAM intelligently.

Keep sustained swap activity approximately zero.

Record:

- GPU utilization
- VRAM
- RAM
- swap
- NVMe throughput
- CPU utilization
- images/sec
- Stage-1 latency
- Stage-2 latency
- worst-case end-to-end latency

A conditional detector should report BOTH:

    average latency
    worst-case latency

============================================================
STEP 21 — PARAMETER BUDGET
============================================================

The final instantiated architecture MUST remain:

    < 2,000,000,000 parameters

However, increased parameter count is acceptable when it provides
measurable improvement.

Do not reject a model simply because it is larger.

Do not accept a model simply because it is larger.

Report:

    total parameters
    trainable parameters
    peak VRAM

============================================================
STEP 22 — REQUIRED CANDIDATE COMPARISON
============================================================

Compare at minimum:

    Phase-4 frozen baseline

    CLIP + SigLIP + SRM

    CLIP + SigLIP + SRM + Edge

    CLIP + SigLIP + SRM + DINO

    CLIP + SigLIP + SRM + Edge + DINO

    CLIP + SigLIP + SRM + Edge + DINO + ConvNeXt

    large 7-expert cocktail

    all-9

    best conditional two-stage detector

For every candidate:

    AUROC
    AUPRC
    TPR
    FPR
    FNR
    RI
    Worst
    ECE
    Brier
    latency
    VRAM
    params

PLUS:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

============================================================
STEP 23 — MODEL SELECTION
============================================================

The winning model is NOT:

"highest AUROC"

and NOT:

"lowest FPR"

and NOT:

"most experts."

The preferred candidate is the one that gives the best:

    LOW-FPR TPR
    +
    LOW FNR
    +
    ROBUSTNESS
    +
    GENERALIZATION
    +
    CALIBRATION
    +
    EFFICIENCY

If a larger model provides only negligible improvement over a smaller
one, prefer the smaller model.

If a larger model provides a major improvement in the low-FPR regime,
keep the larger model.

============================================================
STEP 24 — FINAL INTERNAL TEST
============================================================

The internal test remains locked.

Only after:

- architecture selection
- fusion selection
- loss selection
- calibration selection
- threshold selection
- hard-example strategy
- adaptation strategy

are frozen:

run the internal test ONCE.

Then never tune again.

============================================================
STEP 25 — EXTERNAL OOD
============================================================

Only after the model is completely frozen:

evaluate:

    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

if approved and available.

Do not tune after seeing OOD results.

If OOD performance is poor:

    REPORT IT.

Do not silently adapt the model to the OOD benchmark.

============================================================
STEP 26 — HARD-NEGATIVE ITERATION LIMIT
============================================================

Do not create an infinite feedback loop.

Maximum:

    2 hard-example mining rounds

unless the evidence demonstrates that a third round is necessary.

Each round must maintain independent development data.

============================================================
STEP 27 — REQUIRED ARTIFACTS
============================================================

Produce:

    reports/phase5_dataset_inventory.json
    reports/phase5_dataset_distribution.json
    reports/phase5_duplicate_audit.json

    reports/phase5_hard_negative_mining.json
    reports/phase5_hard_positive_mining.json
    reports/phase5_hard_example_analysis.json

    reports/phase5_architecture_bakeoff.json
    reports/phase5_conditional_verifier.json
    reports/phase5_specialist_rescue.json

    reports/phase5_loss_comparison.json
    reports/phase5_adaptation_comparison.json

    reports/phase5_calibration.json
    reports/phase5_threshold_analysis.json

    reports/phase5_robustness.json
    reports/phase5_generator_breakdown.json
    reports/phase5_domain_breakdown.json

    reports/phase5_fp_fn_forensics.json
    reports/phase5_efficiency.json

    reports/phase5_internal_test.json
    reports/phase5_ood_results.json

    reports/phase5_final_architecture_decision.json
    reports/phase5_final_report.md

Update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

with all measured results.

============================================================
STEP 28 — PROVENANCE REQUIREMENTS
============================================================

Every derived artifact must record:

    manifest SHA
    checkpoint SHA
    architecture
    preprocessing version
    seed
    software versions
    date/time

Every feature cache must be cryptographically tied to:

    manifest
    checkpoint
    preprocessing
    code version

No stale cache reuse.

============================================================
STEP 29 — NO HALLUCINATION / NO CHEATING
============================================================

NEVER:

- invent dataset counts
- invent generator composition
- invent hard-example categories
- invent performance
- invent confidence intervals
- call 0 observed FP "zero true FPR"
- tune the internal test
- tune the OOD benchmarks
- reuse old predictions silently
- reuse old fusion weights silently
- silently alter labels
- silently change train/validation/test boundaries
- silently change the loss
- silently change threshold-selection rules
- hide unfavorable results
- declare a specialist useful solely because it disagrees
- call correlation proof of redundancy
- call disagreement proof of complementarity

If the evidence is insufficient:

    WRITE "NOT ESTABLISHED."

============================================================
STEP 30 — FAILURE CONDITIONS
============================================================

STOP if:

- leakage occurs
- train/test overlap occurs
- locked OOD data enters development
- stale derived data is detected
- hard-example mining touches test/OOD
- labels are inconsistent
- checkpoint provenance is broken
- predictions become constant
- NaN/Inf appears
- repeated OOM occurs
- sustained swap thrashing occurs
- metrics cannot be reproduced
- report values contradict raw predictions

Do not silently fix a scientific failure.

============================================================
STEP 31 — FINAL DECISION GATE
============================================================

At the end, answer:

1. What architecture produces the lowest FPR?
2. What architecture produces the lowest FNR?
3. What architecture produces the best TPR at FPR <= 1%?
4. What architecture produces the best TPR at FPR <= 0.5%?
5. What architecture produces the best TPR at FPR <= 0.1%?
6. Can the model approach FPR <= 0.01% with useful TPR?
7. Which hard REAL categories dominate remaining FP?
8. Which AIGC generators dominate remaining FN?
9. Which expert provides unique FP rescue?
10. Which expert provides unique FN rescue?
11. Does the conditional verifier beat single-stage fusion?
12. Does the all-9 cocktail actually help?
13. What is the smallest model achieving near-maximum low-FPR TPR?
14. Does lightweight adaptation materially improve it?
15. What loss weighting is best?
16. What calibration is best?
17. What threshold is best?
18. Should an abstention/review band be deployed?
19. What is the average and worst-case inference latency?
20. Should the full 400–600 GB corpus now be used for definitive training?

============================================================
STEP 32 — FINAL TRAINING DECISION
============================================================

If a candidate clearly wins the Phase-5 fresh-data bake-off:

freeze that architecture.

Then recommend the exact full-scale training configuration.

If no candidate clearly wins:

do NOT force a choice.

Report:

    NO CLEAR WINNER

and explain what evidence is missing.

============================================================
STEP 33 — IMPORTANT: FULL 400–600 GB TRAINING
============================================================

The ultimate approved corpus remains approximately 400–600+ GB.

DO NOT blindly train every file.

The purpose of full-corpus training is to add:

- generator diversity
- difficult REAL diversity
- subtle AIGC
- post-processing diversity
- domain diversity

not merely duplicates.

After Phase-5 architecture selection, design the final large-corpus
training manifest using the same:

    generator-aware
    domain-aware
    hard-example-aware

sampling principles.

Use the maximum useful approved diversity that the RTX 3050 can train
within the project time budget.

============================================================
FINAL PRINCIPLE
============================================================

The current Phase-4 model is already extremely strong.

Therefore, Phase 5 is NOT about making a larger model for its own sake.

It is about attacking the remaining:

    FALSE POSITIVES
    +
    FALSE NEGATIVES

with evidence.

Use the ACTUAL APPROVED DATASETS to find the difficult images.

Mine difficult REAL images from the real corpus.

Mine difficult AIGC images from the synthetic corpus.

Test whether additional structural/forensic experts can rescue those
errors.

Use conditional inference where that gives better accuracy without
paying maximum compute on every image.

Allow more parameters and more processing time when the measured
accuracy gain justifies the cost.

But never sacrifice scientific validity.

The final target is:

    FPR << 1%
    +
    HIGH TPR
    +
    LOW FNR
    +
    STRONG OOD GENERALIZATION
    +
    ROBUSTNESS
    +
    CALIBRATION
    +
    PRACTICAL PERFORMANCE

LET THE FRESH DATA DECIDE.

BEGIN WITH STEP 0.


============================================================
IMPORTANT — MAXIMIZE USEFUL TRAINING DATA
============================================================

Do NOT interpret the current ~100K-scale corpus as the final data ceiling.

The long-term goal is to exploit the full approved 400–600+ GB corpus,
subject to deduplication, provenance validation, label quality, and strict
train/dev/calibration/test separation.

For FINAL TRAINING, prefer:

    MAXIMUM UNIQUE, HIGH-QUALITY, APPROVED DATA

rather than an arbitrarily small fixed image count.

Do NOT discard valid data merely to force a convenient manifest size.

Instead:

1. Inventory all approved data.
2. Deduplicate it.
3. Remove invalid/ambiguous samples.
4. Remove train/dev/test contamination.
5. Preserve generator diversity.
6. Preserve authentic-domain diversity.
7. Use a generator-aware sampler to prevent dominant generators from
   overwhelming smaller families.

The effective training distribution may therefore differ from the
physical corpus distribution.

============================================================
AIGC DATA EXPANSION
============================================================

Use as much eligible AIGC data as practical, especially from generators
that are underrepresented in previous phases.

Prioritize diversity across:

- generator architectures
- generator versions
- diffusion families
- flow-matching generators
- T2I
- I2I
- generative editing
- photorealistic AIGC
- subtle/low-artifact AIGC
- post-processed AIGC

Do NOT allow a single synthetic family to dominate gradient updates.

============================================================
REAL DATA EXPANSION
============================================================

Likewise use as much eligible REAL data as practical.

Strongly prioritize difficult REAL imagery that can produce false
positives:

- photography
- high-frequency natural textures
- foliage
- macro
- bokeh
- HDR
- studio photography
- low-light/high-ISO images
- JPEG-compressed photographs
- resized photographs
- archival imagery
- artwork
- legitimate digital art
- CGI/3D renders
- unusual lighting/color processing

Do not train only on "easy real photographs."

============================================================
FULL-CORPUS TRAINING POLICY
============================================================

For the final training stage:

Use the LARGEST SCIENTIFICALLY USEFUL approved corpus that the hardware
and project time budget can process.

If the entire approved corpus is computationally feasible:

    USE IT.

If the entire corpus is too large for the available training window:

    DO NOT arbitrarily cut to a small number.

Instead select the largest subset that maximizes:

    generator diversity
    authentic-domain diversity
    hard-example diversity
    resolution diversity
    post-processing diversity

while preserving reproducible sampling.

Report exactly what fraction of the approved corpus is used.

============================================================
IMPORTANT DISTINCTION
============================================================

DO NOT confuse:

    physical dataset size

with:

    effective training exposure.

A generator with 500,000 images does not need 100x the gradient exposure
of a generator with 5,000 images.

Use generator-aware sampling to balance exposure while retaining access
to the full corpus.

Record:

    unique images available
    unique images used
    effective samples/epoch
    sampling weights
    expected repeat factor

Never describe repeated samples as independent observations.