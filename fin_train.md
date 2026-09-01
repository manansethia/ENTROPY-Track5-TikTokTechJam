# ======================================================================
# FINAL MASTER EXECUTION PROTOCOL
# FRESH 260K TRAINING + HARD FP/FN LEARNING +
# ACTUAL MULTIMODAL FORENSIC EXPLANATION +
# INDEPENDENT VERIFICATION +
# AI CRITIQUE +
# REWARD/PENALTY +
# REAL FEEDBACK-DRIVEN BACKPROPAGATION
# ======================================================================

IMPORTANT:

THIS IS AN EXECUTION DIRECTIVE.

DO NOT TREAT THIS AS:
    a planning exercise
    a model-selection exercise
    a report-generation exercise
    a feature-extraction exercise
    a benchmark-only exercise

The goal is to ACTUALLY TRAIN the detector and ACTUALLY TEACH IT from
verified hard false-positive and false-negative cases.

The previous runs incorrectly collapsed:
    feature-head training
    heuristic masking
    report generation

into claims of "final training" and "explanation learning."

DO NOT REPEAT THAT.

Every mandatory stage below must correspond to actual computation.

======================================================================
0. MASTER OBJECTIVE
======================================================================

Build a highly accurate AIGC detector whose practical objective is:

    VERY LOW FALSE POSITIVE RATE
    +
    VERY LOW FALSE NEGATIVE RATE
    +
    MAXIMUM TPR AT EXTREME LOW FPR
    +
    STRONG GENERATOR GENERALIZATION
    +
    STRONG REAL-DOMAIN GENERALIZATION
    +
    ROBUSTNESS
    +
    CALIBRATION
    +
    REASONABLE INFERENCE EFFICIENCY

The current project target is NOT simply:

    highest AUROC

The primary operating objective is:

    maximize TPR subject to a very low FPR constraint.

Explicitly measure:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

======================================================================
1. AUTHORITATIVE DOCUMENTS
======================================================================

Read completely:

    AUTH_PHASE1.md
    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md
    reports/final_reconciliation_v2.json
    reports/final_reconciliation_v2.md

These define the current project state and data governance.

Historical experiment artifacts may be inspected for context, but MUST
NOT silently become training targets or starting checkpoints.

======================================================================
2. FRESH START
======================================================================

START A NEW EXPERIMENT NAMESPACE.

Do NOT use any previous TRAINED detector checkpoint as initialization.

Do NOT use previous:

    fusion weights
    fusion heads
    optimizer state
    scheduler state
    hard-example scores
    hard-example labels
    prediction arrays
    explanation outputs
    critic outputs
    reward values
    feedback values
    calibration
    thresholds
    derived feature arrays

as training inputs.

Allowed:

    original pretrained foundation-model checkpoints

The following pretrained foundations are allowed:

    CLIP-ViT-L/14
    SigLIP-SO400M-224
    DINOv2-Registers
    SRM-DWT
    Edge-Specialist

Pretrained foundation weights are NOT previous detector training.

======================================================================
3. AUTHORITATIVE DATASET
======================================================================

Use the VERIFIED TRAINING PARTITION:

    260,184 UNIQUE IMAGES

Composition:

    REAL = 149,000
    AIGC = 111,184

The isolated evaluation partitions are:

    DEV         = 10,000
    CALIBRATION = 4,000
    INTERNAL TEST = 10,316

TOTAL ISOLATED CORPUS:

    284,500

The 260,184 training samples are the authoritative training population.

DO NOT replace this with:

    all files found in directories
    a directory scan
    a convenience subset
    an arbitrary 50K subset
    an arbitrary 100K subset

Every training sample must come from the authoritative training
manifest.

======================================================================
4. STRICT DATA ISOLATION
======================================================================

NEVER use for training:

    DEV
    CALIBRATION
    INTERNAL TEST
    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

If the DataLoader opens a locked OOD file:

    STOP THE RUN.

Mark the current run:

    INVALID_DUE_TO_DATA_GOVERNANCE_VIOLATION

Do not repair the report afterward and call the run valid.

======================================================================
5. VERIFY THE TRAINING MANIFEST BEFORE TRAINING
======================================================================

Before starting optimization:

verify:

    manifest SHA256
    exact sample count
    REAL count
    AIGC count
    split assignments
    SHA256 uniqueness
    train/dev overlap
    train/calibration overlap
    train/test overlap

Every training row should contain:

    image_id
    path
    label
    source_dataset
    domain/generator
    SHA256
    split

The DataLoader MUST be manifest-driven.

DO NOT recursively scan dataset directories and infer training
membership from file presence.

======================================================================
6. PRIMARY DETECTOR
======================================================================

Stage 1 detector:

    CLIP-ViT-L/14
    +
    SigLIP-SO400M-224
    +
    SRM-DWT

Feature representation:

    CLIP
    +
    SigLIP
    +
    SRM

Fusion:

    Structured Branch Dropout MLP

Initial configuration:

    hidden_dim = 256
    LayerNorm
    GELU
    dropout = 0.15

Start with the foundation backbones frozen.

Initialize a NEW fusion head.

Do NOT load any previous fusion head.

======================================================================
7. OPTIONAL LIGHTWEIGHT BACKBONE ADAPTATION
======================================================================

The RTX 3050 has limited VRAM.

Therefore:

START WITH FROZEN FOUNDATION BACKBONES.

If validation later demonstrates a representation ceiling, the system MAY
test small adaptation mechanisms:

    LoRA
    adapters
    last-block tuning
    projection tuning

Do NOT immediately full-fine-tune CLIP/SigLIP.

Any adaptation must produce measurable held-out improvement.

======================================================================
8. CONDITIONAL FORENSIC VERIFIER
======================================================================

Stage 2 forensic experts:

    DINOv2-Registers
    +
    Edge-Specialist

Optional evidence:

    SRM
    frequency analysis
    gradient analysis
    localization
    counterfactual masking

Preferred topology:

    IMAGE
      |
      v
    STAGE 1
      |
      v
    P(AIGC)
      |
      +--------------------+
      |                    |
      v                    v
 confident             uncertain/suspicious
 decision                   |
                            v
                      DINO + EDGE
                            |
                            v
                      final decision

Do not run expensive specialists on every image unless measurement
shows that doing so gives a meaningful improvement.

======================================================================
9. CRITICAL SEPARATION OF ROLES
======================================================================

There are THREE different roles.

ROLE A — DETECTOR

    CLIP
    SigLIP
    SRM
    Fusion head

ROLE B — FORENSIC EVIDENCE

    DINO
    Edge
    SRM
    frequency/gradient/counterfactual analysis

ROLE C — FORENSIC REASONING AI

    ACTUAL MULTIMODAL VLM

Do NOT call CLIP, SigLIP, DINO or Edge the explanation VLM.

Those models provide visual/forensic evidence.

The VLM is responsible for explaining the evidence in human-readable
terms.

======================================================================
10. MANDATORY VLM GATE
======================================================================

BEFORE FULL TRAINING BEGINS:

inspect the actual machine environment.

Find ONE actual multimodal vision-language model capable of:

    accepting an actual image
    reasoning about the image
    producing an explanation

The VLM must be tested on real images.

Verify:

    model name
    checkpoint path
    parameter count
    dtype/quantization
    GPU/RAM requirements
    successful image ingestion
    successful response generation

Do NOT spend the experiment exploring dozens of VLMs.

Use ONE practical model.

Prefer an already available local VLM.

If necessary and technically safe, install/download ONE suitable
quantized VLM, provided this does not contaminate the training corpus.

======================================================================
11. NO VLM = HARD STOP FOR THE EXPLANATION EXPERIMENT
======================================================================

If no actual VLM is available:

    DO NOT SIMULATE ONE.

DO NOT use:

    templates
    regex
    handcrafted text
    CLIP captions
    feature names
    rules
    masking output
    generated JSON

and call them "AI explanations."

Report:

    REQUIRED_FORENSIC_VLM_UNAVAILABLE

The explanation/feedback experiment is then NOT COMPLETE.

Do NOT declare explanation learning complete.

Do not fabricate VLM telemetry.

The classifier may only proceed under a separately labeled
classification-only experiment.

======================================================================
12. IMPORTANT VLM TRAINING DESIGN
======================================================================

The VLM does NOT need to be fine-tuned.

Preferred design:

    VLM = FROZEN TEACHER / REASONER / CRITIC
    DETECTOR = LEARNER

This is intentional.

The VLM provides forensic hypotheses and critique.

The detector learns from VERIFIED feedback.

Do not waste RTX 3050 capacity trying to fine-tune the VLM unless a
separate small adaptation experiment is demonstrably feasible.

======================================================================
13. PHASE A — ACTUAL DETECTOR TRAINING
======================================================================

This MUST be actual gradient-based training.

For each actual training batch:

    manifest IDs
        ->
    raw image load
        ->
    image decode
        ->
    preprocessing
        ->
    CLIP forward
        ->
    SigLIP forward
        ->
    SRM
        ->
    fusion head
        ->
    P(AIGC)
        ->
    loss
        ->
    backward()
        ->
    optimizer.step()

This is the minimum real training path.

DO NOT perform:

    feature extraction only

and call that final training.

DO NOT create a giant feature matrix first and then silently rename
fusion-head optimization "raw-image training."

An in-run cache may be used for efficiency, but it MUST be produced from
the current run and provenance-bound.

======================================================================
14. RAW IMAGE TRAINING vs FEATURE CACHING
======================================================================

Caching is allowed for COMPUTATIONAL EFFICIENCY.

However the experiment must distinguish:

    RAW IMAGE FORWARD PASSES
    FEATURE-CACHE TRAINING

in telemetry.

If you generate fresh features from raw images in the current run and
then train a head on those features:

    report:
    FRESH FEATURE-SPACE TRAINING

Do NOT label that:

    FULL END-TO-END RAW-IMAGE BACKPROPAGATION

If the foundation models are frozen, this is still valid detector
training, but the report must describe it accurately.

======================================================================
15. REAL TRAINING PARAMETERS
======================================================================

Initial loss:

    asymmetric binary cross entropy

    lambda_FP = 2.5

Optimizer:

    AdamW

Scheduler:

    cosine annealing or another validated smooth schedule

Use:

    mixed precision
    gradient accumulation if beneficial
    gradient clipping if required

Batch size must be selected from actual RTX 3050 telemetry.

Do not select batch size merely to make VRAM look full.

======================================================================
16. TRAINING DURATION
======================================================================

Run REAL multi-epoch training.

Initial target:

    several complete epochs

Use early stopping only if independently justified by DEV.

Do not perform only:

    one feature-generation pass

and call the resulting checkpoint final.

Record:

    epochs
    batches
    samples
    optimizer steps
    backward passes
    forward passes
    learning rate
    gradient norms
    elapsed time

======================================================================
17. PARAMETER UPDATE PROOF
======================================================================

Before training:

hash every trainable parameter.

After every epoch:

measure:

    parameter delta
    L2 delta
    max absolute delta
    changed tensors
    changed parameter count

Required:

    optimizer_steps > 0
    backward_passes > 0
    parameter_delta > 0

If not:

    TRAINING_FAILED

Do not generate a final model.

======================================================================
18. TRAINING TELEMETRY
======================================================================

Continuously record:

    timestamp
    epoch
    batch
    samples_seen
    unique_samples_seen
    loss
    validation_loss
    learning_rate
    optimizer_steps
    backward_passes
    gradient_norm

Hardware:

    GPU utilization
    VRAM
    RAM
    swap
    CPU
    NVMe throughput
    images/sec

DO NOT fabricate telemetry.

======================================================================
19. DEV EVALUATION AFTER BASE TRAINING
======================================================================

Use DEV for model selection.

Do NOT use:

    INTERNAL TEST
    OOD

for tuning.

Report:

    AUROC
    AUPRC
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
    ECE
    Brier

Also report:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

======================================================================
20. PHASE B — HARD ERROR DISCOVERY
======================================================================

After actual base training:

MINE ONLY FROM TRAIN.

For REAL:

    rank by highest P(AIGC)

These are candidate HARD FALSE POSITIVES.

For AIGC:

    rank by lowest P(AIGC)

These are candidate HARD FALSE NEGATIVES.

Do NOT simply select all top-scoring samples.

Preserve diversity across:

    source dataset
    domain
    generator
    resolution
    image type

Create:

    HARD_FP_POOL
    HARD_FN_POOL

Record:

    image ID
    label
    score
    source
    generator/domain
    resolution
    transformation
    expert outputs

======================================================================
21. HARD FP LEARNING OBJECTIVE
======================================================================

The central lesson for REAL false positives is:

    unusual REAL appearance != AIGC

For each important hard FP, the VLM must inspect the actual image and
answer:

    WHY did the detector think this REAL image was AIGC?

Then:

    What visual characteristic triggered suspicion?

    Which characteristic is actually present?

    Which characteristic is misleading?

    What evidence supports REAL?

    What would distinguish a legitimate camera/artistic artifact from
    a synthetic artifact?

Potential examples to investigate:

    bokeh
    macro texture
    studio flash
    foliage
    hair/fur
    high-frequency natural texture
    sensor noise
    HDR
    sharpening
    JPEG
    resampling
    legitimate painting texture
    legitimate digital art
    CGI/3D imagery

These are hypotheses only.

The VLM must inspect the actual image.

======================================================================
22. HARD FN LEARNING OBJECTIVE
======================================================================

The central lesson for AIGC false negatives is:

    subtle synthetic evidence must be learned.

For each important hard FN, ask the VLM:

    WHY did this AIGC image look REAL?

Then:

    What synthetic evidence was missed?

    Where is that evidence?

    Which characteristics distinguish it from REAL?

    Which forensic signals support the conclusion?

    What should the detector have learned?

Investigate actual difficult examples from:

    SID
    Quality Paradox
    FLUX
    SDXL
    SD3
    Midjourney
    PixArt
    HFCF
    Defactify
    other represented generators

======================================================
23. HOW MANY HARD EXAMPLES?
======================================================================

Do NOT send all 260,184 images to the VLM.

That would waste enormous compute.

Use the detector to identify a bounded hard subset.

Initial target per feedback round:

    up to 2,000 REAL hard FP
    up to 2,000 AIGC hard FN

Prefer smaller if evidence saturates.

Maintain generator/domain diversity.

The exact count may be adjusted based on actual compute.

Record the exact number.

======================================================================
24. VLM INPUT
======================================================================

For each selected hard case, provide the VLM:

    actual original image
    true label
    detector prediction
    detector probability
    relevant specialist evidence
    relevant localization evidence
    counterfactual evidence where available

The VLM must know whether it is analyzing:

    HARD FP
    HARD FN

because the goal is error diagnosis.

======================================================================
25. VLM QUESTIONS — HARD FP
======================================================================

Prompt the VLM with questions equivalent to:

    This is a verified REAL image.

    The detector classified it as AIGC.

    Explain why the detector may have made this false-positive error.

    Identify the visual characteristics that may have triggered the
    detector.

    Identify which characteristics are genuinely present.

    Explain which characteristics are legitimate REAL-image properties
    rather than reliable evidence of AIGC.

    Identify what evidence, if any, contradicts the AIGC hypothesis.

    Identify the region(s) supporting your explanation.

    State uncertainty and alternative explanations.

======================================================================
26. VLM QUESTIONS — HARD FN
======================================================================

Prompt the VLM with questions equivalent to:

    This is a verified AIGC image.

    The detector classified it as REAL.

    Explain why the synthetic image was difficult to detect.

    Identify subtle evidence that supports AIGC.

    Identify the region(s) containing that evidence.

    Explain what the detector appears to have missed.

    Identify generator/post-processing characteristics if supported.

    State uncertainty and alternative explanations.

======================================================================
27. STRUCTURED EXPLANATION OUTPUT
======================================================================

The VLM MUST produce structured information before prose.

Required:

    case_id
    predicted_class
    confidence
    evidence_tags
    evidence_regions
    explanation
    alternative_hypothesis
    uncertainty

Potential ontology:

ANATOMY:

    hand/finger anomaly
    facial geometry
    eye/teeth anomaly
    limb/object geometry

TEXT:

    malformed text
    inconsistent characters
    impossible typography
    repeated glyphs

GEOMETRY:

    perspective inconsistency
    reflection inconsistency
    shadow inconsistency
    impossible geometry

TEXTURE:

    repeated texture
    unnatural smoothness
    local texture inconsistency
    brushstroke inconsistency

FORENSIC:

    edge anomaly
    residual anomaly
    high-frequency anomaly
    periodic artifact
    spectral anomaly
    upsampling artifact

IMAGE PROCESSING:

    compression inconsistency
    resampling anomaly
    sharpening anomaly

SEMANTIC:

    semantic contradiction
    local/global inconsistency

These are evidence hypotheses, not automatically true labels.

======================================================================
28. EXPLANATION = HYPOTHESIS
======================================================================

NEVER treat the VLM explanation as ground truth.

The verified classification label comes from the governed dataset.

The explanation is a hypothesis about WHY the model succeeded or failed.

The system must distinguish:

    LABEL TRUTH

from:

    EXPLANATION TRUTH

======================================================================
29. INDEPENDENT FORENSIC VERIFICATION
======================================================================

Every important explanation must be checked independently.

Use as available:

    DINO
    Edge
    SRM
    frequency analysis
    image gradients
    localization
    segmentation/detection
    counterfactual masking

For each explanation produce:

    forensic_support
    spatial_support
    counterfactual_support
    causal_support
    contradiction
    confidence

Classify:

    VERIFIED_SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    UNDETERMINED

Do NOT force uncertain cases into binary labels.

======================================================================
30. COUNTERFACTUAL / ABLATION TEST
======================================================================

When the VLM claims:

    "region X caused the suspicious prediction"

perform where computationally reasonable:

    ORIGINAL IMAGE
       ->
    P(AIGC)

then:

    MASK CLAIMED REGION
       ->
    P(AIGC)

optionally:

    KEEP ONLY CLAIMED REGION
       ->
    P(AIGC)

Record:

    original_score
    masked_score
    isolated_score where available
    absolute delta
    direction of change

IMPORTANT:

Counterfactual change is evidence of influence, not automatic proof of
semantic correctness.

======================================================================
31. AI CRITIC
======================================================================

Run an ACTUAL critic pass.

If a second multimodal model exists:

    use the second model as critic.

If only one VLM exists:

    use a separate adversarial critic pass with a fresh context,
    while explicitly marking it:

        CRITIC_NOT_INDEPENDENT_FROM_VLM

Do NOT claim two-model independence when there is only one model.

The critic receives:

    original image
    VLM explanation
    evidence tags
    evidence regions
    specialist evidence
    counterfactual results

======================================================================
32. CRITIC TASK — THE IMPORTANT PART
======================================================================

The critic MUST explicitly answer:

    "What is wrong with this explanation?"

or:

    "Why is this explanation adequately supported?"

For unsupported explanations, it must identify:

    incorrect claim
    missing evidence
    contradictory evidence
    misleading visual characteristic
    unsupported causal claim

For correct explanations:

    identify what evidence supports it

The critic must output:

    supported
    unsupported
    uncertain
    contradiction
    missing_evidence
    causal_support
    evidence_quality
    critique
    confidence

======================================================================
33. DO NOT TRUST THE CRITIC BLINDLY
======================================================================

The critic is NOT ground truth.

If:

    VLM says yes
    critic says yes
    independent evidence says no

then:

    explanation = REJECTED

If:

    VLM says no
    critic says no
    evidence says yes

then:

    explanation = REJECTED

Final explanation validity is based primarily on independent evidence.

======================================================================
34. EXPLANATION QUALITY LABEL
======================================================================

Every explanation must end in one:

    VERIFIED_SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    UNDETERMINED

This status is derived from:

    evidence
    specialist outputs
    counterfactuals
    critic analysis

not from model confidence alone.

======================================================================
35. REWARD / PENALTY SYSTEM
======================================================================

Use bounded feedback.

INITIAL SCALE:

    VERIFIED_SUPPORTED
        +1.0

    PARTIALLY_SUPPORTED
        +0.25

    UNDETERMINED
         0.0

    UNSUPPORTED
        -0.50

    CONTRADICTED
        -1.0

    CONFIDENTLY_FABRICATED
        -1.0

These are starting values.

Do NOT automatically treat them as optimal.

======================================================================
36. IMPORTANT:
# REWARD/PENALTY MUST HAVE DIFFERENT EFFECTS
======================================================================

Do NOT simply multiply the classification label by the reward.

Separate:

    CLASSIFICATION ERROR

from:

    EXPLANATION ERROR

Example:

A REAL image is correctly classified as REAL but the explanation says:

    "six fingers"

The model should NOT be punished for the REAL label.

It should be punished for the incorrect evidence attribution.

Likewise:

AIGC image is correctly detected but the VLM explanation is wrong.

Keep:

    classification = correct

while correcting:

    evidence attribution / explanation.

This prevents the explanation loop from corrupting class labels.

======================================================================
37. FEEDBACK LEARNING SIGNAL
======================================================================

Convert verified forensic feedback into trainable signals.

Use a combination such as:

    classification loss
    +
    evidence consistency loss
    +
    localization/evidence alignment loss
    +
    counterfactual consistency loss
    +
    feedback/reward loss

Conceptually:

    L_total =
        L_class
        +
        λ_e * L_evidence
        +
        λ_loc * L_localization
        +
        λ_cf * L_counterfactual
        +
        λ_fb * L_feedback

Start auxiliary terms SMALL.

Classification remains primary.

======================================================================
38. HARD FP FEEDBACK SIGNAL
======================================================================

For a REAL false positive:

    Ground truth = REAL

If VLM/expert verification establishes that the suspected characteristic
is legitimate and misleading:

    increase loss against false AIGC prediction

AND:

    penalize attribution toward the misleading feature

AND:

    encourage evidence alignment with genuinely non-discriminative or
    legitimate characteristics where applicable.

The model should learn:

    "This visual pattern does not justify accusing REAL images."

======================================================================
39. HARD FN FEEDBACK SIGNAL
======================================================================

For an AIGC false negative:

    Ground truth = AIGC

If VLM + forensic experts establish valid synthetic evidence:

    increase loss against false REAL prediction

AND:

    encourage the evidence mechanism toward the verified synthetic
    region/signal

AND:

    increase sensitivity to the validated subtle signature.

The model should learn:

    "This subtle synthetic evidence matters."

======================================================================
40. CRITICAL:
# FEEDBACK MUST CAUSE REAL BACKPROPAGATION
======================================================================

The feedback loop is NOT complete when:

    VLM writes explanation
    critic writes critique
    JSON is saved

The loop is complete ONLY when:

    hard case
       ->
    VLM explanation
       ->
    independent verification
       ->
    critic
       ->
    reward/penalty
       ->
    trainable loss
       ->
    backward()
       ->
    optimizer.step()
       ->
    changed parameters

Record:

    feedback_optimizer_steps
    feedback_backward_passes
    feedback_gradient_norm
    feedback_parameter_delta

Required:

    feedback_optimizer_steps > 0

AND:

    feedback_parameter_delta > 0

Otherwise:

    EXPLANATION_FEEDBACK_LEARNING_DID_NOT_EXECUTE

======================================================================
41. FEEDBACK PARAMETER CHANGE PROOF
======================================================================

Before feedback update:

    hash trainable detector parameters

After feedback update:

    hash trainable detector parameters

Record:

    changed tensor count
    changed parameter count
    L2 delta
    max absolute delta

If feedback supposedly happened but:

    parameter delta = 0

then:

    feedback learning FAILED.

Do not report it as successful.

======================================================================
42. IMPORTANT:
# THE VLM DOES NOT AUTOMATICALLY TRAIN THE DETECTOR
======================================================================

The VLM's prose does not directly become detector weights.

The system must convert verified feedback into actual machine-learning
targets.

Acceptable mechanisms include:

    reward-weighted classification
    pairwise ranking
    hard-example weighted BCE
    evidence classification
    localization loss
    contrastive evidence learning
    counterfactual consistency
    attribution regularization

Use whichever combination is actually implementable and differentiable.

Document the exact implementation.

======================================================================
43. NO FAKE RL
======================================================================

Do NOT implement fake reinforcement learning where:

    reward is written to a file

but:

    no trainable policy/model receives the reward.

Do NOT call a reward table "reinforcement learning."

If reward is used:

    it must influence a differentiable or explicitly trainable objective.

======================================================================
44. PHASE C — FEEDBACK RETRAINING ROUND 1
======================================================================

After base training:

    mine HARD FP/FN
        ->
    VLM explanation
        ->
    independent evidence
        ->
    critic
        ->
    verified feedback
        ->
    feedback loss
        ->
    actual backward()
        ->
    actual optimizer.step()
        ->
    new checkpoint

Then re-evaluate DEV.

Record:

    BEFORE_FEEDBACK
    AFTER_FEEDBACK_ROUND_1

Compare:

    FP
    FN
    FPR
    FNR
    TPR
    AUROC
    AUPRC
    low-FPR TPR

======================================================================
45. PHASE D — FEEDBACK ROUND 2
======================================================================

After Round 1:

mine NEW remaining hard FP/FN.

Do NOT simply reuse exactly the same cases.

Repeat:

    hard mining
    explanation
    verification
    critic
    feedback
    optimization

Then evaluate DEV again.

Record:

    AFTER_FEEDBACK_ROUND_2

Maximum:

    2 major feedback rounds

Do not create an endless self-training loop.

======================================================================
46. PROVE WHETHER EXPLANATION FEEDBACK HELPED
======================================================================

Compare:

    A = fresh base training

    B = after hard-example training

    C = after explanation feedback round 1

    D = after explanation feedback round 2

For every stage report:

    AUROC
    AUPRC
    FP
    FN
    FPR
    FNR
    TPR
    TNR
    ECE
    Brier

And:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

It is completely acceptable if:

    explanation feedback does NOT improve classification.

If it hurts:

    report the degradation.

Do not force the system into production just because it is more complex.

======================================================================
47. EXPLANATION FEEDBACK IS NOT SELF-LABELING
======================================================================

NEVER perform:

    detector prediction
       ->
    detector explanation
       ->
    explanation treated as truth
       ->
    detector trained to agree with itself

Required:

    detector prediction
       ->
    VLM hypothesis
       ->
    independent forensic evidence
       ->
    counterfactual evidence
       ->
    critic
       ->
    verified feedback
       ->
    detector update

Ground truth remains external.

======================================================================
48. HARD-EXAMPLE SAMPLING
======================================================================

Do not train exclusively on hard examples.

Use:

    NORMAL CORPUS
        +
    HARD REAL
        +
    HARD AIGC

Hard-example weight should be bounded.

Initial values may be:

    HARD REAL = 2.0x
    HARD AIGC = 2.0x

Then validate.

Do not allow hard samples to overwhelm the generator/domain distribution.

======================================================================
49. GENERATOR-AWARE SAMPLING
======================================================================

Preserve diversity across:

    Quality Paradox
    SDXL
    Midjourney
    FLUX/SD3
    SID
    PixArt
    HFCF
    Defactify
    other approved generators

Do not allow a single generator to dominate gradient exposure.

Record:

    raw corpus distribution
    sampled distribution
    samples/epoch
    effective exposures
    repeat factor

======================================================================
50. REAL-DOMAIN SAMPLING
======================================================================

Preserve:

    COCO
    WikiArt
    Web Photography
    Archival
    Hard Macro/Bokeh

Especially preserve legitimate images that trigger false AIGC signals.

======================================================================
51. FULL DATA REQUIREMENT
======================================================================

Use the full verified:

    260,184 UNIQUE TRAINING IMAGES

Do not reduce to:

    50K
    100K

for convenience.

The point of this run is to learn from the complete approved training
population.

If additional approved images exist elsewhere in the 400–600+ GB storage
pool but are NOT part of the verified 260,184-image partition:

    inventory them separately.

DO NOT silently add them.

They require their own provenance/deduplication audit before inclusion.

======================================================================
52. MEMORY / STORAGE ARCHITECTURE
======================================================================

Hardware:

    RTX 3050 6GB
    ~31GB RAM
    ~24GB swap
    ~400GB+ NVMe

Use:

    NVMe staging
        ->
    asynchronous prefetch
        ->
    pinned RAM
        ->
    non-blocking GPU transfer

Do NOT load hundreds of GB into RAM.

Do NOT intentionally use swap as a dataset cache.

Target:

    sustained swap I/O ≈ 0

Use RAM aggressively as a bounded hot cache.

======================================================================
53. VLM COMPUTE SCHEDULING
======================================================================

Do NOT run VLM explanation inference concurrently with the main GPU
training if that causes GPU contention.

Preferred schedule:

    detector training
       ->
    hard-example mining
       ->
    VLM forensic analysis
       ->
    feedback target construction
       ->
    detector feedback training
       ->
    next round

This makes the compute budget predictable.

======================================================================
54. EXPLANATION COMPUTE BUDGET
======================================================================

The VLM should run primarily on:

    hard FP
    hard FN
    uncertain examples
    high-confidence mistakes
    specialist disagreement cases

Do NOT run the VLM on every training image.

Record:

    VLM calls
    images analyzed
    average VLM latency
    total VLM time
    failed calls
    retries

======================================================================
55. EXPLANATION OUTPUT EXAMPLES
======================================================================

Example HARD FP:

    Ground truth:
        REAL

    Detector:
        P(AIGC) = 0.97

    VLM:
        "The detector may be reacting to strong macro texture and
         shallow optical bokeh."

    Evidence region:
        background blur / macro texture

    Independent verification:
        consistent with authentic optical characteristics

    Critic:
        "The explanation incorrectly treats optical bokeh as evidence
         of synthesis."

    Feedback:
        reduce reliance on misleading cue

This is a VALID feedback case.

------------------------------------------------------------

Example HARD FN:

    Ground truth:
        AIGC

    Detector:
        P(AIGC) = 0.14

    VLM:
        "The image appears photographic globally, but local texture
         repetition and edge continuity are inconsistent in this region."

    Independent verification:
        supports the claimed region

    Critic:
        "The detector missed a subtle local synthetic signature."

    Feedback:
        increase sensitivity to the verified signal

This is a VALID feedback case.

======================================================================
56. EXPLANATION MODEL TRAINING — OPTIONAL
======================================================================

The preferred system is:

    VLM = frozen teacher

    DETECTOR = learner

If a lightweight explanation model itself can be trained safely:

    it may learn from VERIFIED_SUPPORTED explanations.

However:

    VLM fine-tuning is NOT required for detector feedback learning.

If VLM parameters do not change:

    report:

        VLM_FROZEN = TRUE

Do not falsely claim that the VLM learned.

======================================================================
57. CALIBRATION
======================================================================

After the final feedback-trained detector is selected:

use:

    CALIBRATION = 4,000

Fit fresh calibration.

Compare:

    temperature scaling
    Platt scaling
    isotonic where statistically justified

Do not blindly reuse historical temperature values.

Inspect the high-confidence tail:

    p >= 0.95
    p >= 0.99

======================================================================
58. THRESHOLD OPTIMIZATION
======================================================================

Perform dense threshold evaluation.

Do NOT assume:

    tau = 0.80

is final.

Find maximum TPR satisfying:

    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

For every selected point report:

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

======================================================================
59. STATISTICAL HONESTY
======================================================================

If:

    FP = 0

report:

    0 / N_REAL

Do NOT claim:

    population FPR = 0

The evaluation sample must be large enough to resolve the target rate.

Where sample size is insufficient:

    explicitly report:

        INSUFFICIENT STATISTICAL RESOLUTION

======================================================================
60. ROBUSTNESS
======================================================================

Evaluate the final candidate under:

    Clean
    JPEG90
    JPEG70
    JPEG50
    JPEG30
    Blur
    Resize
    Noise
    Crop
    ColorJitter
    Sharpening
    Recompression

Report:

    AUROC
    AUPRC
    FPR
    FNR
    TPR
    TNR

per condition.

Calculate:

    RI
    worst AUROC
    clean-to-worst degradation

======================================================================
61. GENERATOR BREAKDOWN
======================================================================

Report performance by:

    Quality Paradox
    SDXL
    Midjourney
    FLUX/SD3
    SID
    PixArt
    HFCF
    Defactify
    every sufficiently represented additional generator

Do not hide generator-specific failures inside aggregate metrics.

======================================================================
62. REAL-DOMAIN BREAKDOWN
======================================================================

Report FPR by:

    COCO
    WikiArt
    Web Photography
    Archival
    Hard Macro/Bokeh

Identify which legitimate real domains generate the most FP.

======================================================================
63. TRUE END-TO-END LATENCY
======================================================================

Separate:

    image decode
    preprocessing
    CLIP
    SigLIP
    SRM
    Stage-1 fusion
    routing
    DINO
    Edge
    Stage-2 fusion
    calibration
    VLM explanation path where invoked

Report:

    average
    P95
    P99
    worst-case
    Stage-2 invocation rate
    VLM invocation rate
    VRAM
    RAM

Do NOT report:

    cached feature throughput

as:

    raw-image inference throughput.

======================================================================
64. INTERNAL TEST
======================================================================

Keep:

    INTERNAL TEST = 10,316

completely locked during training.

Only AFTER:

    architecture
    final weights
    feedback loop
    routing
    calibration
    threshold
    review policy

are frozen:

    evaluate the COMPLETE final system ONCE.

If Stage 2 is part of production:

    use actual Stage 1 -> Stage 2 routing.

Do NOT perform additional tuning after seeing the test result.

======================================================================
65. OOD
======================================================================

Only after final freezing evaluate approved OOD:

    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

Do NOT use these for:

    training
    hard mining
    explanation feedback
    threshold tuning
    calibration tuning

Evaluate once.

======================================================================
66. CHECKPOINTING
======================================================================

Save separate checkpoints for:

    fresh initialization
    best base model
    after hard-example Round 1
    after feedback Round 1
    after feedback Round 2
    best low-FPR model
    final frozen model

Every checkpoint must contain:

    model
    optimizer
    scheduler
    epoch
    step
    RNG state
    sampler state
    manifest SHA
    architecture
    preprocessing
    loss
    calibration configuration
    routing configuration
    feedback configuration

Use atomic checkpoint writes.

Never overwrite the only known-good checkpoint.

======================================================================
67. REQUIRED TRAINING PROOF
======================================================================

Produce:

    reports/final_actual_training_telemetry.json
    reports/final_parameter_update_proof.json

Must contain:

    epochs
    batches
    samples_processed
    unique_images_processed
    forward_passes
    backward_passes
    optimizer_steps
    scheduler_steps
    gradient_norms
    training_duration
    images_per_second

and:

    initial_parameter_hash
    per-epoch_parameter_hash
    final_parameter_hash
    changed_parameter_count
    cumulative_parameter_delta

======================================================================
68. REQUIRED FORENSIC FEEDBACK PROOF
======================================================================

Produce:

    reports/final_hard_fp_round1.json
    reports/final_hard_fn_round1.json
    reports/final_hard_fp_round2.json
    reports/final_hard_fn_round2.json

For every hard example record:

    image_id
    label
    detector_score
    explanation
    evidence_tags
    evidence_regions
    independent_verification
    counterfactual_result
    critic_output
    explanation_quality
    reward
    feedback_target

======================================================================
69. REQUIRED VLM PROOF
======================================================================

Produce:

    reports/final_vlm_execution.json
    reports/final_explanation_generation.json
    reports/final_explanation_verification.json
    reports/final_explanation_critic.json
    reports/final_explanation_feedback.json

Include:

    VLM name
    VLM checkpoint
    VLM hash
    calls
    successful calls
    failed calls
    explanations generated
    supported
    partially supported
    unsupported
    contradicted
    undetermined

    critic calls
    critic results

    counterfactual tests
    supported counterfactuals

    reward distribution
    penalty distribution

======================================================================
70. REQUIRED FEEDBACK OPTIMIZATION PROOF
======================================================================

Produce:

    reports/final_feedback_parameter_updates.json

It MUST include:

    feedback_backward_passes
    feedback_optimizer_steps
    feedback_gradient_norms
    before_parameter_hash
    after_parameter_hash
    parameter_delta
    changed_parameter_count
    feedback_loss

If:

    feedback_optimizer_steps == 0

then:

    FINAL_FEEDBACK_LEARNING = NOT_EXECUTED

If:

    parameter_delta == 0

then:

    FINAL_FEEDBACK_LEARNING = FAILED

======================================================================
71. REQUIRED A/B SCIENTIFIC COMPARISON
======================================================================

Produce a table comparing:

    BASE
    HARD-EXAMPLE
    FEEDBACK ROUND 1
    FEEDBACK ROUND 2
    FINAL

Metrics:

    AUROC
    AUPRC
    FP
    FN
    FPR
    FNR
    TPR
    TNR
    ECE
    Brier

and:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

This is mandatory.

======================================================================
72. FINAL FORENSIC EXPLANATION VALIDATION
======================================================================

Measure:

    explanation accuracy
    evidence support rate
    contradiction rate
    unsupported rate
    causal-support rate
    critic agreement
    critic disagreement

Also categorize common mistakes:

    correct class + wrong explanation
    wrong class + plausible explanation
    wrong class + wrong explanation
    correct class + correct explanation

The system must learn specifically from these edge cases.

======================================================================
73. FINAL EXPLANATION LEARNING TELEMETRY
======================================================================

Produce:

    reports/final_explanation_learning_telemetry.json

Fields:

    VLM_available
    VLM_name
    VLM_checkpoint
    VLM_frozen

    explanations_generated
    explanations_verified
    supported
    partially_supported
    unsupported
    contradicted
    undetermined

    critic_calls
    critic_rejections
    critic_uncertain

    counterfactual_tests
    counterfactual_supported

    feedback_samples
    positive_rewards
    penalties

    classification_feedback_steps
    evidence_feedback_steps
    counterfactual_feedback_steps
    total_feedback_optimizer_steps

    feedback_gradient_norm
    feedback_parameter_delta

Do not fabricate zeros.

A zero means the corresponding operation did not happen.

======================================================================
74. HARD FAILURE CONDITIONS
======================================================================

STOP and mark the run FAILED if:

    locked OOD data is accessed during training

OR:

    train/test leakage occurs

OR:

    wrong manifest is used

OR:

    training images cannot be reconciled

OR:

    optimizer steps = 0

OR:

    trainable parameters never change

OR:

    VLM is unavailable when explanation learning is required

OR:

    explanations are only templates/rules

OR:

    critic is claimed without actual critic execution

OR:

    feedback is claimed without feedback optimizer steps

OR:

    feedback optimizer steps occur but parameter delta = 0

OR:

    metrics cannot be reproduced from predictions

OR:

    report contradicts raw telemetry

Do NOT silently repair the report.

======================================================================
75. ABSOLUTE NO-SIMULATION RULE
======================================================================

NEVER simulate:

    VLM outputs
    explanations
    evidence
    critic responses
    reward
    penalty
    optimizer steps
    parameter changes
    training duration
    images processed
    hard examples
    calibration
    threshold results

NEVER generate a JSON file first and then execute nothing.

Computation comes first.

Reports come second.

======================================================================
76. NO REPORT-ONLY COMPLETION
======================================================================

The system may NOT declare:

    TRAINING_COMPLETE

until:

    actual detector training
    +
    actual optimizer updates
    +
    actual hard-example mining
    +
    actual VLM explanations
    +
    actual independent verification
    +
    actual critic
    +
    actual reward/penalty
    +
    actual feedback backward pass
    +
    actual feedback optimizer step
    +
    measurable parameter change

have all occurred.

======================================================================
77. FINAL EXECUTION STATE MACHINE
======================================================================

The pipeline MUST follow this state machine:

    STATE 0
    DATA / ENVIRONMENT VALIDATION
          |
          v
    STATE 1
    VLM VALIDATION
          |
          v
    STATE 2
    FRESH MODEL INITIALIZATION
          |
          v
    STATE 3
    REAL BASE TRAINING
          |
          v
    STATE 4
    DEV EVALUATION
          |
          v
    STATE 5
    HARD FP/FN MINING
          |
          v
    STATE 6
    VLM FORENSIC EXPLANATION
          |
          v
    STATE 7
    INDEPENDENT FORENSIC VERIFICATION
          |
          v
    STATE 8
    AI CRITIC
          |
          v
    STATE 9
    REWARD / PENALTY
          |
          v
    STATE 10
    FEEDBACK LOSS CONSTRUCTION
          |
          v
    STATE 11
    BACKWARD
          |
          v
    STATE 12
    OPTIMIZER UPDATE
          |
          v
    STATE 13
    PARAMETER-CHANGE VERIFICATION
          |
          v
    STATE 14
    DEV RE-EVALUATION
          |
          v
    STATE 15
    SECOND HARD FP/FN ROUND
          |
          v
    STATE 16
    SECOND EXPLANATION / CRITIC / FEEDBACK ROUND
          |
          v
    STATE 17
    SECOND PARAMETER UPDATE
          |
          v
    STATE 18
    FINAL CALIBRATION
          |
          v
    STATE 19
    FINAL THRESHOLD OPTIMIZATION
          |
          v
    STATE 20
    ROBUSTNESS / GENERATOR / DOMAIN ANALYSIS
          |
          v
    STATE 21
    FREEZE
          |
          v
    STATE 22
    INTERNAL TEST ONCE
          |
          v
    STATE 23
    OOD ONCE
          |
          v
    STATE 24
    FINAL REPORT
          |
          v
    STATE 25
    STOP

Do NOT skip states.

======================================================================
78. FINAL TRAINING PIPELINE IN PLAIN ENGLISH
======================================================================

The system must learn like this:

    TRAIN ON REAL + AIGC
             |
             v
       MAKE PREDICTIONS
             |
             v
       FIND WRONG CASES
             |
        +----+----+
        |         |
        v         v
      HARD FP   HARD FN
        |         |
        +----+----+
             |
             v
       ASK ACTUAL VLM
             |
             v
       "WHY WAS THIS WRONG?"
             |
             v
     IDENTIFY EVIDENCE
             |
             v
    INDEPENDENTLY VERIFY
             |
             v
        ASK CRITIC
             |
             v
    "WHY IS THIS EXPLANATION
        WRONG OR RIGHT?"
             |
             v
       REWARD / PENALTY
             |
             v
      TRAINING OBJECTIVE
             |
             v
         BACKPROP
             |
             v
       OPTIMIZER UPDATE
             |
             v
        MODEL CHANGES
             |
             v
      LEARN FROM MISTAKE
             |
             v
      FIND NEW MISTAKES
             |
             v
         REPEAT ONCE
             |
             v
          CALIBRATE
             |
             v
           FREEZE
             |
             v
           TEST

This is the actual research objective.

======================================================================
79. FINAL PRODUCTION EXPLANATION
======================================================================

For difficult images, the final system should be able to output:

    Classification:
        AIGC / REAL

    Confidence:
        numerical probability

    Evidence:
        structured forensic evidence tags

    Region:
        relevant image region(s)

    Explanation:
        concise natural-language explanation

    Alternative hypothesis:
        plausible REAL or AIGC alternative

    Verification:
        independent support / contradiction

    Counterfactual:
        whether masking the claimed region changes the prediction

    Explanation confidence:
        calibrated confidence

For REAL false positives the explanation should be capable of saying:

    "The detector was misled by a legitimate characteristic such as
     optical bokeh or high-frequency photographic texture."

For AIGC false negatives it should be capable of saying:

    "The detector missed subtle synthetic evidence in this region."

======================================================================
80. FINAL PERFORMANCE OBJECTIVE
======================================================================

The final system should aim for:

    extremely low FPR

while retaining:

    maximum TPR

and minimizing:

    FNR

Do NOT sacrifice recall solely to create a visually attractive
zero-FP headline.

Report the complete operating frontier.

======================================================================
81. FINAL REQUIRED REPORTS
======================================================================

Produce:

DATA:

    reports/final_training_dataset_audit.json
    reports/final_training_manifest_audit.json

TRAINING:

    reports/final_actual_training_telemetry.json
    reports/final_training_loss_curve.json
    reports/final_parameter_update_proof.json

HARD EXAMPLES:

    reports/final_hard_fp_round1.json
    reports/final_hard_fn_round1.json
    reports/final_hard_fp_round2.json
    reports/final_hard_fn_round2.json
    reports/final_fp_fn_forensics.json

VLM:

    reports/final_vlm_execution.json
    reports/final_explanation_generation.json
    reports/final_explanation_verification.json
    reports/final_explanation_critic.json
    reports/final_explanation_feedback.json
    reports/final_explanation_learning_telemetry.json
    reports/final_feedback_parameter_updates.json

MODEL:

    reports/final_conditional_verifier.json
    reports/final_calibration.json
    reports/final_thresholds.json
    reports/final_robustness.json

GENERALIZATION:

    reports/final_generator_breakdown.json
    reports/final_domain_breakdown.json

EFFICIENCY:

    reports/final_latency.json
    reports/final_hardware_telemetry.json

FINAL:

    reports/final_internal_test.json
    reports/final_ood.json
    reports/FINAL_TRAINING_MASTER_REPORT.md

Update:

    docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

======================================================================
82. FINAL REPORT MUST SEPARATE THESE FIVE STATES
======================================================================

The final report MUST separately state:

    ACTUAL_DETECTOR_TRAINING = YES/NO

    ACTUAL_HARD_EXAMPLE_LEARNING = YES/NO

    ACTUAL_VLM_EXPLANATIONS = YES/NO

    ACTUAL_FORENSIC_FEEDBACK_LEARNING = YES/NO

    ACTUAL_FINAL_EVALUATION = YES/NO

Never collapse them into one "training complete" flag.

======================================================================
83. FINAL CHECKLIST
======================================================================

Before declaring COMPLETE:

[ ] 260,184 authorized training images used
[ ] zero train/dev/calibration/test overlap
[ ] zero OOD contamination
[ ] fresh detector initialization
[ ] actual raw-image training occurred
[ ] optimizer steps > 0
[ ] backward passes > 0
[ ] detector parameters changed
[ ] hard FP mining occurred
[ ] hard FN mining occurred
[ ] actual VLM found and tested
[ ] actual VLM explanations generated
[ ] explanation evidence independently verified
[ ] critic actually executed
[ ] explanation quality assigned
[ ] rewards/penalties computed
[ ] feedback entered trainable loss
[ ] feedback backward pass occurred
[ ] feedback optimizer steps > 0
[ ] feedback parameter delta > 0
[ ] feedback round 1 completed
[ ] feedback round 2 completed
[ ] before/after metrics compared
[ ] calibration completed
[ ] threshold frontier computed
[ ] robustness completed
[ ] generator breakdown completed
[ ] real-domain breakdown completed
[ ] complete model frozen
[ ] internal test evaluated once
[ ] OOD evaluated once
[ ] all telemetry reproducible
[ ] knowledge base updated

======================================================================
84. FINAL AUTHORITY RULE
======================================================================

If the evidence says:

    the explanation system did not improve the detector

then say:

    EXPLANATION_FEEDBACK_DID_NOT_IMPROVE_CLASSIFICATION

That is a valid scientific outcome.

If DINO/Edge do not help:

    drop them.

If hard-example training hurts:

    document it.

If an explanation type is unreliable:

    mark it unreliable.

If a VLM claim cannot be verified:

    do not train on it as truth.

If a report conflicts with raw data:

    raw data wins.

If something did not execute:

    say NOT_EXECUTED.

======================================================================
85. FINAL COMMAND
======================================================================

STOP WRITING SUMMARIES.

STOP EXPLORING FOUNDATION MODELS.

STOP CREATING REPORTS BEFORE COMPUTATION.

START THE ACTUAL EXECUTION.

The required learning loop is:

    260K APPROVED TRAINING IMAGES
        ->
    REAL DETECTOR TRAINING
        ->
    HARD FP/FN MINING
        ->
    ACTUAL VLM EXPLANATION
        ->
    INDEPENDENT FORENSIC VERIFICATION
        ->
    ACTUAL CRITIC
        ->
    REWARD / PENALTY
        ->
    FEEDBACK LOSS
        ->
    BACKPROPAGATION
        ->
    OPTIMIZER UPDATE
        ->
    PARAMETER CHANGE
        ->
    NEW HARD FP/FN MINING
        ->
    SECOND FEEDBACK ROUND
        ->
    CALIBRATION
        ->
    THRESHOLD
        ->
    FREEZE
        ->
    INTERNAL TEST
        ->
    OOD
        ->
    FINAL REPORT

EVERY ARROW MUST ACTUALLY EXECUTE.

NO SIMULATION.
NO REPORT-ONLY MODE.
NO FEATURE-HEAD-ONLY CLAIMS DISGUISED AS FULL TRAINING.
NO FAKE VLM.
NO FAKE CRITIC.
NO FAKE REWARD.
NO FAKE PARAMETER UPDATES.
NO SELF-GENERATED GROUND TRUTH.
NO DATA LEAKAGE.

BEGIN WITH:

    STEP 1 — VERIFY THE EXACT TRAINING MANIFEST
    STEP 2 — LOCATE AND TEST THE ACTUAL VLM
    STEP 3 — INITIALIZE A FRESH DETECTOR
    STEP 4 — START REAL TRAINING

DO NOT DECLARE COMPLETION UNTIL THE CHECKLIST PASSES.