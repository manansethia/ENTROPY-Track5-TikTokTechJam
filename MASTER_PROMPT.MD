<USER_REQUEST>
MASTER EXECUTION PROTOCOL
AIGC ROBUST DETECTION PROJECT
============================================================

This is the authoritative execution plan for the project.

Treat this prompt as the master protocol and reconcile it with the
existing repository, reports, configs, scripts, checkpoints, dataset
manifests, and previous experiment results.

IMPORTANT:
Do not blindly redo completed work.
Inspect the current state first.
Reuse valid completed artifacts.
Only rerun experiments when a diagnostic demonstrates that the existing
result is invalid, incomplete, or methodologically unsound.

The ultimate objective is NOT to maximize one development-set AUROC.

The objective is to build the strongest practical image-level AIGC
detector possible under the competition constraint:

    COMPLETE FINAL MODEL < 2 BILLION PARAMETERS

while maximizing:

    clean discrimination
    robustness to realistic post-processing
    generator OOD generalization
    low false-positive rate
    low false-negative rate
    calibration
    complementary evidence
    reasonable inference latency
    reasonable VRAM usage

The RTX 3050 has 6 GB VRAM.
The server has 32 GB RAM and substantial SSD/HDD storage.

Use CPU RAM, disk feature caching, mixed precision, sequential model
execution and offloading intelligently.

============================================================
0. CURRENT PROJECT STATE
============================================================

The 400-image Stage-1 dataset is ONLY a small architecture-discovery /
expert-profiling dataset.

It is NOT the complete training corpus.

Large approved training/development resources include approximately:

    GenImage++                         ~193 GB
    SID Parquet streams                ~129 GB
    Defactify                          ~14 GB
    WikiArt authentic hard negatives   ~25 GB
    AIGI Quality Paradox                ~7.5 GB
    ArtBench                            ~6 GB
    Vintage/Archival                    ~8 GB
    DRCT-2M                             selective acquisition

Other external/OOD resources include:

    Synthbuster
    AIGIBench
    Chameleon
    VCT²
    WildRF
    SynthWildX
    official locked hackathon validation data

The large datasets are intended for actual supervised learning,
hard-example mining, robustness training and development.

The 400-image dataset is intended for fast expert comparison and
architecture discovery.

Do NOT mistake Stage 1 for the final training benchmark.

============================================================
1. DATA GOVERNANCE IS ABSOLUTE
============================================================

Read and enforce:

    configs/dataset_policy.yaml

Every dataset must have an explicit role:

    TRAIN
    DEVELOPMENT
    INTERNAL_OOD
    EXTERNAL_EVAL
    UNAVAILABLE

Never silently change a dataset's role.

Never use EXTERNAL_EVAL data for:

    training
    feature learning
    hard-example mining
    architecture selection
    fusion-weight selection
    threshold selection
    hyperparameter tuning
    calibration

This includes:

    AIGIBench
    Synthbuster
    Chameleon
    VCT²
    WildRF
    SynthWildX
    locked hackathon validation data

unless a dataset explicitly contains a documented TRAIN split that is
independently designated as training data.

The test/evaluation portion remains locked.

============================================================
2. GENERATOR CONTAMINATION
============================================================

Maintain explicit generator exposure metadata.

A generator is STRICT ZERO-SHOT only when it has had no exposure through:

    training images
    development images
    feature caches
    hard-example mining
    architecture selection
    threshold tuning
    hyperparameter selection
    manual iterative optimization

IMPORTANT:

FLUX.1 was used in the Stage-1 development subset.

Therefore:

    FLUX = DEVELOPMENT-EXPOSED
    FLUX = NOT STRICT ZERO-SHOT

SDXL and Imagen are training-seen according to the existing audit.

Do not later claim FLUX is unseen OOD.

Maintain:

    reports/generator_contamination_audit.json

and update it whenever exposure changes.

For strict OOD generators, record:

    training_images = 0
    development_images = 0
    feature_exposure = 0
    hard_mining_exposure = 0
    architecture_selection_exposure = 0
    threshold_selection_exposure = 0

Report OOD performance per generator, not only as one aggregate number.

============================================================
3. CURRENT STAGE-1 EXPERT POOL
============================================================

Required experts:

    1. SigLIP-SO400M
    2. DINOv2-Large + Registers
    3. EVA-02-Large
    4. CLIP ViT-L/14
    5. ConvNeXt-V2-Tiny
    6. 2D FFT Spectral
    7. SRM/DWT forensic branch
    8. E²GenF Edge Specialist
    9. Patch/MIL Expert
    10. AIDE
    11. DDA

Optional:

    Swin-L

Do NOT assume all experts belong in the final architecture.

The goal is to find the smallest genuinely complementary set.

============================================================
4. STAGE-1 RESULTS ALREADY ESTABLISHED
============================================================

The current Stage-1 results establish approximately:

    CLIP:
        Clean AUROC ≈ 0.9072
        RI ≈ 0.8235
        Worst ≈ 0.7450

    SigLIP:
        Clean AUROC ≈ 0.7811
        RI ≈ 0.7461
        Worst ≈ 0.7039

Generic representation heuristics:

    DINO ≈ 0.4838
    EVA ≈ 0.4876
    ConvNeXt ≈ 0.4810

Forensic heuristic baselines:

    FFT ≈ 0.2203
    SRM/DWT ≈ 0.4951
    Edge ≈ 0.4621
    Patch-MIL ≈ 0.4798

AIDE and DDA previously returned exactly:

    0.5000

across conditions.

These must NOT be treated as meaningful detector-performance results
until their inference paths have been validated.

IMPORTANT:

DINO/EVA/ConvNeXt were evaluated using untrained representation
heuristics.

Therefore their low Stage-1 AUROC does NOT prove that their learned
representations are useless.

Likewise, weak forensic standalone AUROC does not prove that the
features are useless for fusion.

============================================================
5. DO NOT RESTART VALID STAGE-1
============================================================

The previous Stage-1 experiment has already established:

    400 images
    × 7 core conditions
    × 11 experts

    = 30,800 image-level evaluations

and:

    13 batches/condition
    × 7 conditions
    × 11 experts

    = 1,001 batch forwards

Do not rerun all 1,001 forwards unless an audit demonstrates that the
original experiment was invalid.

============================================================
6. AIDE / DDA DIAGNOSTIC
============================================================

FIRST PRIORITY.

For BOTH AIDE and DDA:

    verify checkpoint integrity
    verify architecture
    verify parameter count
    verify trainable parameters
    verify state_dict loading
    report missing keys
    report unexpected keys
    verify classifier head exists
    verify classifier weights are non-zero
    inspect raw logits
    inspect probability statistics
    verify output shape
    verify preprocessing
    verify RGB ordering
    verify normalization
    verify resize/crop requirements
    verify real/fake polarity
    verify P(AIGC) semantics
    run 10 obvious real examples
    run 10 obvious synthetic examples
    verify predictions are not constant

If broken:

    ADAPTER_FAILED
    or
    CHECKPOINT_FAILED

If repairable:

    fix adapter
    rerun ONLY AIDE/DDA Stage-1

Never report constant 0.5 output as a legitimate detector result.

Write:

    reports/aide_dda_diagnostic.json

============================================================
7. FOUR-TIER EXPERT TAXONOMY
============================================================

Maintain this distinction.

TIER 1 — ZERO-SHOT VLM

    CLIP
    SigLIP

These are zero-shot semantic baselines.

TIER 2 — PRETRAINED AIGC DETECTORS

    AIDE
    DDA

These are specialized trained detectors.

TIER 3 — GENERIC VISION REPRESENTATIONS

    DINO
    EVA
    ConvNeXt

These are generic visual representations and require supervised heads
to properly test their AIGC usefulness.

TIER 4 — FORENSIC SPECIALISTS

    FFT
    SRM/DWT
    Edge
    Patch/MIL

These provide frequency, residual, boundary and local evidence.

Never compare these tiers as though their Stage-1 outputs were produced
by equivalent methodologies.

============================================================
8. SUPERVISED REPRESENTATION PROBES
============================================================

After AIDE/DDA diagnostics, perform controlled supervised probes.

For:

    DINO
    EVA
    ConvNeXt
    FFT
    SRM/DWT
    Edge
    Patch/MIL

Initially:

    freeze feature extractor
    extract features
    train a small classifier

Use:

    0 = REAL
    1 = AIGC

Output:

    pred = P(AIGC)

Start with:

    logistic regression / linear probe

Then, ONLY if justified:

    small 1–2 layer MLP

Do not immediately fine-tune entire backbones.

Use identical development train/validation partitions.

Evaluate:

    Clean AUROC
    JPEG30 AUROC
    Blur2 AUROC
    Resize0.25 AUROC
    Noise0.10 AUROC
    Crop80 AUROC
    ColorJitter AUROC

Then calculate:

    RI
    Worst AUROC
    Robustness degradation
    AUPRC
    FPR
    FNR
    ECE
    Brier score
    latency
    peak VRAM
    parameter count

Write:

    reports/supervised_representation_benchmark.json

============================================================
9. ZERO-SHOT CLIP / SIGLIP BASELINES
============================================================

Preserve the original zero-shot CLIP and SigLIP measurements.

Do not retrain them during this stage.

Their baseline results must remain available for comparison.

Their predictions must still follow:

    pred = P(AIGC)

Do not optimize their thresholds against external evaluation data.

============================================================
10. ROBUSTNESS DEFINITIONS
============================================================

Use the official seven-condition Stage-1 core matrix:

    Clean
    JPEG30
    Blur sigma=2.0
    Resize 0.25x then upscale
    Gaussian Noise sigma=0.10
    Center Crop 80%
    Color Jitter

Define:

    RI =
    (AUROC_clean
     + AUROC_JPEG30
     + AUROC_Blur2
     + AUROC_Resize0.25
     + AUROC_Noise0.10
     + AUROC_Crop80
     + AUROC_ColorJitter) / 7

Define:

    AUROC_worst = minimum AUROC across seven conditions

Define:

    Delta_robust =
    AUROC_clean - AUROC_worst

Later, the definitive robustness matrix should include:

    JPEG 90/70/50/30
    Blur 0.5/1.0/2.0
    Resize 0.5/0.25
    Noise 0.02/0.05/0.10
    Crop80
    ColorJitter

Do not optimize specifically to the fixed seven Stage-1 conditions.

============================================================
11. ERROR COMPLEMENTARITY
============================================================

For every promising representation against CLIP/SigLIP calculate:

    Pearson correlation
    Spearman correlation
    FN overlap
    FP overlap
    disagreement rate
    A → B rescue
    B → A rescue
    new errors introduced

Use:

    Rescue(A → B)
    =
    |Errors_A ∩ Correct_B| / |Errors_A|

Do NOT automatically preserve an expert simply because rescue >50%.

A weak model can have high disagreement without providing useful
information.

The relevant question is whether its information improves a learned
fusion model on held-out development data.

Write:

    reports/error_complementarity_matrix.json

============================================================
12. LARGE-SCALE TRAINING DATASET
============================================================

Once promising representations are identified, construct the real
training corpus from approved TRAIN/DEVELOPMENT data.

DO NOT blindly concatenate hundreds of GB.

Create explicit manifests containing:

    image path
    dataset source
    generator family
    generator version
    real/AIGC label
    split
    image dimensions
    provenance
    cryptographic hash
    perceptual hash where practical

Detect:

    exact duplicates
    near duplicates
    train/development contamination

Use generator-aware splitting.

Random image-level splitting alone is insufficient.

Construct:

    TRAIN
    DEVELOPMENT
    INTERNAL OOD

with generator exposure explicitly recorded.

============================================================
13. LARGE-SCALE SUPERVISED TRAINING
============================================================

Use the large datasets for actual learning.

Do NOT train the final detector on only the 400-image Stage-1 set.

Initially:

    freeze large backbones
    cache features to /mnt/ai-storage
    train lightweight AIGC heads

Only fine-tune a backbone when frozen-feature experiments show a
measurable benefit.

Prefer efficient training techniques appropriate to the RTX 3050:

    mixed precision
    gradient accumulation
    feature caching
    CPU/RAM offloading
    sequential backbone processing

Use the substantial HDD/SSD storage intelligently.

============================================================
14. ROBUSTNESS TRAINING
============================================================

Training should contain a probabilistic mixture of transformations.

JPEG:

    quality 90 / 70 / 50 / 30

Blur:

    sigma 0.5 / 1.0 / 2.0

Resize:

    0.5x / 0.25x then upscale

Noise:

    sigma 0.02 / 0.05 / 0.10

Color:

    brightness / contrast / saturation ±20%

Crop:

    center crop 80%

Do NOT apply the same transformation to every image.

Do NOT overfit to the exact benchmark transformation values.

Maintain an untouched validation subset.

Measure whether robustness training improves transformed performance
without destroying clean performance.

============================================================
15. HARD-EXAMPLE MINING
============================================================

After a first supervised model exists, mine difficult examples ONLY
from approved TRAIN/DEVELOPMENT data.

Prioritize:

    false positives
    false negatives
    high-confidence mistakes
    expert disagreement
    heavy compression
    blur
    resizing
    crop
    color manipulation
    noise
    stylized authentic artwork
    photographic AI
    difficult diffusion outputs
    reconstruction/editing artifacts

Never mine from:

    AIGIBench
    Synthbuster
    Chameleon
    VCT²
    WildRF
    SynthWildX
    locked hackathon validation

Create an explicit hard-example manifest.

Use iterative hard-example mining carefully so it does not become
a disguised form of OOD benchmark optimization.

============================================================
16. CONTROLLED FUSION ABLATIONS
============================================================

After individual supervised probes:

Test:

    A. CLIP
    B. best learned structural expert
    C. best learned forensic expert
    D. CLIP + SigLIP
    E. CLIP + structural
    F. CLIP + forensic
    G. CLIP + SigLIP + forensic
    H. CLIP + structural + forensic
    I. best compact combination

Use small learned fusion heads.

Do not concatenate every representation blindly.

Every added expert must justify itself through held-out development
performance and/or meaningful complementary evidence.

For every fusion report:

    Clean AUROC
    RI
    Worst AUROC
    degradation
    AUPRC
    FPR
    TPR
    FNR
    ECE
    Brier score
    latency
    peak VRAM
    parameter count

Write:

    reports/pairwise_fusion_benchmark.json

============================================================
17. DUAL-EVIDENCE RELIABILITY ROUTER
============================================================

Only after ordinary fusion has been benchmarked should the
Dual-Evidence Reliability Router be tested.

The router should learn to distinguish:

    semantic evidence
    structural evidence
    forensic evidence
    AI evidence
    real evidence
    uncertainty/disagreement

The router should output:

    P(AIGC)

and optionally:

    confidence
    uncertainty

Keep the router lightweight.

Do not make it large merely to increase parameter count.

Compare it against simpler fusion baselines.

The router earns inclusion only if it improves held-out development
performance without unacceptable increases in:

    false positives
    latency
    VRAM
    complexity

============================================================
18. SWIN-L
============================================================

Swin-L is OPTIONAL.

Do not let it delay critical experiments.

If resources permit, evaluate it as a frozen representation using the
same supervised probe methodology.

Compare it against:

    CLIP
    SigLIP
    DINO
    EVA
    ConvNeXt
    forensic branches
    AIDE
    DDA

Include Swin-L only if it provides meaningful complementary value.

Do not include it merely because it is large.

============================================================
19. PARETO ARCHITECTURE SELECTION
============================================================

Do not select architecture based solely on clean AUROC.

Evaluate:

    Clean AUROC
    Robustness Index
    Worst-case AUROC
    OOD AUROC
    AUPRC
    FPR
    FNR
    ECE
    Brier score
    error rescue
    new errors
    latency
    VRAM
    parameter count

Candidate A Pareto-dominates candidate B iff:

    A is no worse than B on every selected objective
    AND
    A is strictly better on at least one objective.

Non-dominated candidates remain eligible for further ablation.

Do not use an arbitrary additive score as the sole selection criterion.

Final inclusion requires demonstrated benefit on held-out development
data.

============================================================
20. FPR / OPERATING POINT
============================================================

FPR <1% is a DESIRABLE SAFETY TARGET.

It is NOT a competition disqualification criterion unless explicitly
required by the competition.

Report threshold curves at:

    0.50
    0.60
    0.70
    0.80
    0.85
    0.90
    0.95

Report:

    FPR
    TPR/Recall
    Precision
    FNR

Choose the final operating threshold on HELD-OUT DEVELOPMENT data.

Freeze the threshold before external evaluation.

============================================================
21. CALIBRATION
============================================================

After architecture selection:

evaluate:

    AUROC
    AUPRC
    FPR
    TPR
    FNR
    ECE
    Brier score
    calibration curves
    confidence distributions

Calibrate using development data only.

Do not calibrate against external OOD benchmarks.

============================================================
22. EXTERNAL OOD EVALUATION
============================================================

ONLY AFTER:

    architecture frozen
    training complete
    hard mining complete
    robustness training complete
    calibration complete
    threshold frozen

evaluate untouched external data.

Priority:

    1. Synthbuster
    2. AIGIBench
    3. Chameleon
    4. VCT²
    5. WildRF
    6. SynthWildX
    7. official locked hackathon validation

Report EACH DATASET separately.

For generator evaluation report:

    generator
    samples
    AUROC
    AUPRC
    FPR
    FNR
    ECE
    confidence distribution

Also report aggregate statistics where useful, but never hide weak
generators behind an aggregate score.

DO NOT modify the model after seeing external OOD results unless the
dataset has an explicitly permitted development/training split.

Otherwise that evaluation is final evidence.

============================================================
23. FINAL 2-BILLION-PARAMETER AUDIT
============================================================

The competition requires:

    COMPLETE FINAL MODEL < 2,000,000,000 PARAMETERS

This means the actual final instantiated architecture.

Do NOT simply add individual backbone parameter counts from separate
experiments.

Before submission:

    instantiate the complete final model
    count every parameter
    count trainable parameters
    count non-trainable parameters
    record checkpoint size
    record inference memory
    save authoritative audit

The final model MUST remain below 2B.

Write/update:

    reports/final_model_parameter_audit.json

============================================================
24. VRAM / MEMORY ENGINEERING
============================================================

The target GPU:

    RTX 3050 6GB

Use:

    FP16/BF16 where supported
    feature caching
    gradient accumulation
    sequential expert execution
    CPU/RAM offloading
    disk-backed feature stores
    subprocess isolation when useful

Between expert runs:

    delete model
    delete tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() where appropriate

Record:

    torch.cuda.memory_allocated()
    torch.cuda.memory_reserved()
    peak memory
    before/after values

Verify that VRAM returns near baseline.

Do not assume empty_cache() alone proves complete memory release.

============================================================
25. EXPERIMENT REPRODUCIBILITY
============================================================

Every experiment must record:

    git commit
    dataset manifest version
    dataset hashes/version
    generator exposure
    model/checkpoint
    parameter count
    preprocessing
    augmentation configuration
    random seed
    optimizer
    learning rate
    batch size
    precision
    training steps/epochs
    checkpoint path
    validation metrics
    peak VRAM
    inference latency

No important result should exist without provenance.

============================================================
26. RESULTS MUST BE HONEST
============================================================

Never claim:

    "99% accuracy"
    "unbreakable"
    "zero-shot"

unless the exact evaluation supports that statement.

Distinguish:

    zero-shot baseline
    supervised development result
    internal OOD result
    external OOD result
    locked benchmark result

A model that scores extremely high on a contaminated split but
collapses on unseen generators is inferior to a slightly weaker model
that genuinely generalizes.

============================================================
27. DO NOT CHASE COMPLEXITY
============================================================

Do NOT assume:

    more models = better
    larger model = better
    more parameters = better
    more datasets = automatically better
    more forensic branches = better

Every component must earn its place empirically.

The final architecture should ideally be:

    small enough to deploy
    strong enough to generalize
    robust enough for post-processing
    calibrated enough for real decisions
    explainable enough to demonstrate the evidence sources

============================================================
28. FINAL EXECUTION ORDER
============================================================

Follow this order unless a scientific/engineering issue requires a
documented deviation:

CURRENT REPOSITORY AUDIT
        ↓
AIDE / DDA DIAGNOSTIC
        ↓
SUPERVISED REPRESENTATION PROBES
        ↓
ERROR / COMPLEMENTARITY ANALYSIS
        ↓
SELECT PROMISING REPRESENTATIONS
        ↓
LARGE-SCALE TRAINING
        ↓
HARD-EXAMPLE MINING
        ↓
ROBUSTNESS TRAINING
        ↓
FUSION ABLATIONS
        ↓
RELIABILITY ROUTER
        ↓
CALIBRATION
        ↓
FREEZE FINAL CHECKPOINT
        ↓
PARAMETER / VRAM / LATENCY AUDIT
        ↓
EXTERNAL OOD EVALUATION
        ↓
LOCKED HACKATHON VALIDATION
        ↓
FINAL SUBMISSION PACKAGE

============================================================
29. CRITICAL STOP CONDITIONS
============================================================

STOP and report to the user before proceeding if:

    a checkpoint cannot be verified
    a dataset role is ambiguous
    an external benchmark may have leaked into development
    generator exposure cannot be established
    labels are inconsistent
    model outputs are constant
    P(AIGC) polarity is ambiguous
    parameter count approaches/exceeds 2B
    VRAM behavior is unexpectedly unsafe
    an experiment contradicts its declared methodology
    a result cannot be reproduced

Do not silently make assumptions.

============================================================
30. REQUIRED DECISION-GATE OUTPUTS
============================================================

Before final architecture training, provide:

    reports/aide_dda_diagnostic.json
    reports/supervised_representation_benchmark.json
    reports/error_complementarity_matrix.json
    reports/pairwise_fusion_benchmark.json
    reports/generator_contamination_audit.json

and a human-readable summary containing:

    strongest standalone expert
    strongest learned representation
    strongest forensic representation
    strongest pairwise fusion
    strongest compact fusion
    error-rescue findings
    robustness findings
    calibration findings
    latency/VRAM findings
    unresolved risks

THEN STOP.

Do NOT automatically train the final architecture.

============================================================
31. FINAL PRINCIPLE
============================================================

The project is NOT:

    "throw every model at the problem."

It is:

    semantic evidence
          +
    structural evidence
          +
    forensic evidence
          +
    specialized pretrained detectors
          ↓
    experimentally validated complementarity
          ↓
    lightweight learned fusion
          ↓
    robustness training
          ↓
    generator-held-out validation
          ↓
    genuinely generalizable AIGC detection

The 400-image Stage-1 experiment tells us WHICH SIGNALS are worth
investigating.

The hundreds of GB of approved datasets are where the detector should
actually LEARN.

The untouched external datasets are where we determine whether it
actually GENERALIZES.

Do not confuse those three purposes.

Execute carefully, preserve reproducibility, and stop at every required
decision gate rather than making irreversible architecture decisions
automatically.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-08-28T17:57:45+08:00.
</ADDITIONAL_METADATA>