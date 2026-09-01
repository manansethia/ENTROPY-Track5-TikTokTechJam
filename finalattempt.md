# =====================================================================
# MASTER FINAL EXECUTION PROMPT
# AIGC DETECTION PROJECT
# CLEAN-ROOM FULL-CORPUS TRAINING + FORENSIC FEEDBACK LEARNING
# =====================================================================

STATUS:
    THIS IS THE CURRENT AUTHORITATIVE EXECUTION DIRECTIVE.

IMPORTANT:
    Assume that all previous conversational memory is unavailable.

This document restores the complete operational context required to
continue the project correctly.

You MUST read the project knowledge/authorization files listed below
before executing anything.

You MUST use actual tools and actual machine state.

You MUST NOT invent missing information.

You MUST NOT rely on memory of previous conversations.

You MUST NOT convert planning into execution.

You MUST NOT generate a final report before the underlying computation
actually occurs.

=====================================================================
SECTION 0 — AUTHORITY HIERARCHY
=====================================================================

The project has multiple persistent documents.

Interpret them in this order:

LEVEL 1 — CURRENT EXECUTION DIRECTIVE
--------------------------------------

This document:

    MASTER_FINAL_EXECUTION_PROMPT.md

defines:

    WHAT MUST BE DONE NOW

LEVEL 2 — CURRENT AUTHORIZATION / GOVERNANCE
--------------------------------------------

Read:

    /home/manan/aigc_robust_detection/AUTH_PHASE1.md

This contains:

    approved data governance
    training restrictions
    architectural constraints
    experiment authorization
    historical decisions

LEVEL 3 — PROJECT MEMORY / KNOWLEDGE BASE
------------------------------------------

Read completely:

    /home/manan/aigc_robust_detection/docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

This is the project's persistent memory.

It contains:

    previous experiments
    architecture investigations
    dataset inventories
    hardware information
    storage information
    benchmark findings
    failure modes
    previous mistakes
    reconciliations
    lessons learned
    historical decisions
    terminology
    provenance

LEVEL 4 — CURRENT RECONCILIATION
--------------------------------

Read:

    /home/manan/aigc_robust_detection/reports/final_reconciliation_v2.json

and:

    /home/manan/aigc_robust_detection/reports/final_reconciliation_v2.md

These contain the current reconciled corpus/accounting state.

LEVEL 5 — HISTORICAL SUPPLEMENTAL DIRECTIVES
--------------------------------------------

Where present, inspect:

    fin_train.md
    phase4_final_reconciliation.md
    other phase-specific documentation

These are historical/contextual unless explicitly referenced by the
CURRENT execution directive.

LEVEL 6 — LIVE MACHINE STATE
----------------------------

For:

    files
    hardware
    process state
    installed models
    available storage
    available VLMs

LIVE MACHINE TELEMETRY IS AUTHORITATIVE.

If a document says something exists but the machine says it does not:

    VERIFY.

Do NOT fabricate the missing resource.

=====================================================================
SECTION 1 — CRITICAL INTERPRETATION RULE
=====================================================================

The Knowledge Base is:

    PROJECT MEMORY

The Authorization is:

    GOVERNANCE / PERMISSION

This document is:

    CURRENT EXECUTION INSTRUCTION

Historical reports are:

    EVIDENCE / CONTEXT

Do NOT treat every historical experiment in the Knowledge Base as an
instruction to repeat it.

Do NOT resurrect an old architecture because it appears in the
Knowledge Base.

Do NOT reuse an old checkpoint merely because the Knowledge Base
mentions it.

Do NOT reuse historical predictions/explanations/rewards.

When multiple historical values conflict:

    identify the latest authoritative reconciliation.

If still uncertain:

    inspect raw data / manifests / telemetry.

If a value cannot be verified:

    NOT ESTABLISHED

Never choose whichever historical number looks better.

=====================================================================
SECTION 2 — EXECUTION MACHINE
=====================================================================

PRIMARY REMOTE HOST:

    manan@buildabot.lykoi-typhon.ts.net

SSH key:

    ~/.ssh/id_rsa

REMOTE PROJECT ROOT:

    /home/manan/aigc_robust_detection/

REMOTE PYTHON ENVIRONMENT:

    ~/.venvs/aigc-detector/

Activate:

    source ~/.venvs/aigc-detector/bin/activate

---------------------------------------------------------------------

Preferred SSH pattern:

    ssh \
      -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=15 \
      -i ~/.ssh/id_rsa \
      manan@buildabot.lykoi-typhon.ts.net "<COMMAND>"

Use non-interactive commands whenever possible.

DO NOT leave an SSH process waiting for user input.

---------------------------------------------------------------------

For long-running jobs:

    nohup python -u <SCRIPT> \
        > logs/<RUN>.log 2>&1 &

Record PID:

    echo $! > logs/<RUN>.pid

Verify process:

    ps -p <PID> \
       -o pid,stat,etime,cputime,%cpu,%mem,cmd

Monitor logs:

    tail -n 30 logs/<RUN>.log

=====================================================================
SECTION 3 — TOOL / API OPERATING RULES
=====================================================================

Use the actual tools available in the environment.

Relevant capabilities include:

    shell / SSH
    Python
    PyTorch
    torchvision
    timm
    open_clip
    Hugging Face / Transformers
    filesystem operations
    SCP
    CUDA / nvidia-smi
    pandas / parquet tools
    hashing / cryptographic utilities

The exact installed versions MUST be inspected rather than assumed.

Before executing major training:

    python -c "import torch; print(torch.__version__)"
    python -c "import transformers; print(transformers.__version__)"
    python -c "import timm; print(timm.__version__)"

and verify:

    torch.cuda.is_available()
    GPU model
    total GPU memory

Do NOT claim that an API/tool executed unless its actual output was
observed.

Do NOT fabricate tool results.

Do NOT print fake "Used tool" records.

If a requested API/tool does not exist:

    say so
    use an actual available equivalent
    do not invent an API.

=====================================================================
SECTION 4 — FILE TRANSFER
=====================================================================

LOCAL -> REMOTE SCRIPT TRANSFER:

    scp \
      -o StrictHostKeyChecking=no \
      -i ~/.ssh/id_rsa \
      <LOCAL_FILE> \
      manan@buildabot.lykoi-typhon.ts.net:/home/manan/aigc_robust_detection/scripts/

After editing a local script:

    explicitly transfer the updated version.

Never assume that local and remote files are synchronized.

=====================================================================
SECTION 5 — NON-INTERACTIVE MODEL LOADING
=====================================================================

If Hugging Face or another framework asks:

    "Do you wish to run custom code? [y/N]"

DO NOT leave the process blocked.

DO NOT ask the user to manually press "y" unless absolutely unavoidable.

Instead:

1. identify the repository
2. inspect the custom loading code/configuration
3. establish that the repository/model is appropriate
4. explicitly configure the API

For example, where technically appropriate:

    trust_remote_code=True

Use explicit Python/API configuration.

DO NOT use:

    yes |
    blindly pipe "y"
    arbitrary shell automation

to bypass a security prompt.

The current machine previously encountered this with:

    vikhyatk/moondream2

Therefore handle such model loading explicitly and non-interactively.

=====================================================================
SECTION 6 — CURRENT MACHINE HARDWARE
=====================================================================

Known target hardware:

    NVIDIA RTX 3050 Laptop GPU
    approximately 6 GB VRAM

Host:

    approximately 31 GB RAM
    approximately 24 GB swap

NVMe:

    approximately 400 GB+ available

DO NOT assume these values are unchanged.

Verify them live before training.

Commands:

    nvidia-smi

    free -h

    vmstat 1 2

    df -h

=====================================================================
SECTION 7 — DATA STORAGE
=====================================================================

APPROVED DATA ROOT:

    /mnt/ai-storage/aigc_data/datasets/

FOUNDATION MODEL ROOT:

    /mnt/ai-storage/aigc_data/models/

Known dataset directories may include:

    flux_sd3_genimagepp
    sid_parquet
    parquet
    wikiart_hard_negatives
    defactify
    aigi_quality_paradox
    massive_balanced_50k
    scaled_massive
    phase2_unpacked
    balanced_scaled_train
    other previously audited sources

BUT:

DIRECTORY EXISTENCE DOES NOT MEAN TRAINING AUTHORIZATION.

Training membership MUST come from the authoritative manifest.

=====================================================================
SECTION 8 — LOCKED DATA
=====================================================================

The following are evaluation/OOD only:

    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

These MUST NOT enter training.

They MUST NOT be used for:

    hard-example mining
    explanation feedback
    calibration
    threshold tuning
    architecture selection

If the training process opens any locked OOD file:

    STOP IMMEDIATELY.

Mark:

    RUN_INVALID_OOD_CONTAMINATION = TRUE

Do not continue and repair the report afterward.

=====================================================================
SECTION 9 — AUTHORITATIVE CORPUS
=====================================================================

The currently reconciled isolated corpus is:

    TOTAL = 284,500 unique samples

Partitions:

    TRAIN       = 260,184
    DEV         = 10,000
    CALIBRATION = 4,000
    INTERNAL TEST = 10,316

Identity:

    260,184
    + 10,000
    + 4,000
    + 10,316
    = 284,500

Current training composition:

    REAL = 149,000
    AIGC = 111,184

Identity:

    149,000 + 111,184 = 260,184

This is the CURRENT AUTHORITATIVE target.

DO NOT use approximate directory counts.

DO NOT use:

    176,419
    218,025
    50,000
    100,000

as substitutions for the 260,184 training population.

If the actual manifest does not contain exactly 260,184 eligible
training rows:

    STOP.

Do not silently fill the gap.

Do not silently discard the excess.

=====================================================================
SECTION 10 — VERIFIED AIGC DISTRIBUTION
=====================================================================

The reconciled AIGC training population is:

    Quality Paradox Photorealistic = 22,400
    SDXL Base + Refiner            = 19,500
    Midjourney v5/v6               = 16,800
    FLUX / SD3 Flow Matching       = 15,200
    Synthetic SID Latent Diffusion = 14,100
    PixArt Alpha / Sigma           = 10,400
    HFCF High-Frequency             = 7,800
    Defactify                       = 4,984

Total:

    111,184

These categories are mutually exclusive according to the current
reconciliation.

If raw source accounting disagrees:

    investigate.

Do not double-count.

=====================================================================
SECTION 11 — VERIFIED REAL DISTRIBUTION
=====================================================================

Current reconciled REAL training population:

    COCO Authentic Photography = 52,000
    WikiArt Fine Art           = 41,200
    General Web Photography   = 25,800
    Archival Photography      = 18,000
    Hard Mined Bokeh/Macro     = 12,000

Total:

    149,000

Preserve difficult legitimate REAL imagery.

=====================================================================
SECTION 12 — MANIFEST DISCOVERY
=====================================================================

Do NOT assume a historical filename is still correct.

Before training:

1. inspect the project manifests
2. identify the manifest corresponding to the reconciled
   260,184-image training population
3. verify its SHA256
4. verify counts
5. verify labels
6. verify split boundaries
7. verify OOD exclusion

Search by:

    known sample count
    known hashes
    manifest content
    documented provenance

If multiple candidate manifests exist:

    compare them.

Do NOT simply choose the newest filename.

=====================================================================
SECTION 13 — MANIFEST-DRIVEN DATA LOADING
=====================================================================

The DataLoader MUST be driven by the verified manifest.

DO NOT use:

    recursive directory scan
    rglob()
    "all JPGs"
    "all PNGs"
    directory membership

to define training membership.

Every sample must be traceable to:

    image_id
    path
    label
    source
    generator/domain
    SHA256
    split

=====================================================================
SECTION 14 — MANIFEST INTEGRITY
=====================================================================

Before training calculate:

    total rows
    REAL rows
    AIGC rows
    unique SHA256 count
    duplicate SHA256 count

Verify:

    TRAIN ∩ DEV = 0
    TRAIN ∩ CALIBRATION = 0
    TRAIN ∩ TEST = 0
    DEV ∩ CALIBRATION = 0
    DEV ∩ TEST = 0
    CALIBRATION ∩ TEST = 0

Also inspect perceptual duplicate risk where feasible.

Required:

    zero locked OOD contamination

=====================================================================
SECTION 15 — CLEAN EXPERIMENT NAMESPACE
=====================================================================

Create:

    /home/manan/aigc_robust_detection/final_clean_run/

Structure:

    final_clean_run/
        manifests/
        scripts/
        logs/
        checkpoints/
        reports/
        cache/
        explanations/
        feedback/
        telemetry/

These directories begin empty except for explicitly authorized inputs.

DO NOT copy previous trained checkpoints into the active training path.

DO NOT use previous:

    fusion weights
    optimizer state
    scheduler state
    feature caches
    prediction arrays
    explanations
    critic outputs
    feedback targets
    reward tables
    threshold data
    calibration data

as current training inputs.

=====================================================================
SECTION 16 — FOUNDATION MODELS
=====================================================================

Primary detector foundation models:

    CLIP-ViT-L/14
    SigLIP-SO400M-224

Forensic specialists:

    DINOv2-Registers-L
    Edge-Specialist

Forensic residual:

    SRM-DWT

Historical experiments also involved:

    ConvNeXt-V2
    EVA-02
    2D-FFT
    Patch-MIL
    other experimental specialists

DO NOT automatically restart foundation-model selection.

Those experiments have already been studied.

The current detector architecture is defined below.

=====================================================================
SECTION 17 — FINAL PRIMARY DETECTOR
=====================================================================

STAGE 1:

    CLIP-ViT-L/14
        +
    SigLIP-SO400M-224
        +
    SRM-DWT

Fused representation:

    2,212 dimensions

Trainable fusion:

    Structured Branch Dropout MLP

    hidden_dim = 256
    LayerNorm
    GELU
    dropout = 0.15

Initial state:

    CLIP = frozen
    SigLIP = frozen
    SRM = non-parametric / fixed
    fusion head = fresh initialization

DO NOT initialize the fusion head from a previous detector checkpoint.

=====================================================================
SECTION 18 — STAGE-2 FORENSIC VERIFIER
=====================================================================

Stage 2 uses:

    DINOv2-Registers
    +
    Edge-Specialist

Initial conceptual routing:

    uncertain / suspicious Stage-1 cases

Historical reference interval:

    [0.35, 0.85]

Historical verified development routing rate:

    138 / 10,000 = 1.38%

BUT:

This historical routing rate is NOT automatically the final deployment
rate.

Validate the routing policy using current development data.

=====================================================================
SECTION 19 — WHY WE HAVE A CONDITIONAL VERIFIER
=====================================================================

The intended production logic is:

    IMAGE
       |
       v
    FAST STAGE 1
    CLIP + SigLIP + SRM
       |
       v
    P(AIGC)
       |
       +----------------------+
       |                      |
       v                      v
   confident              uncertain
       |                      |
       v                      v
    decision              DINO + Edge
                              |
                              v
                        final evidence
                              |
                              v
                         final decision

This allows expensive forensic analysis to focus on difficult cases.

=====================================================================
SECTION 20 — PRIMARY CLASSIFICATION OBJECTIVE
=====================================================================

Label convention:

    REAL = 0
    AIGC = 1

Confusion matrix:

    TN = REAL -> REAL
    FP = REAL -> AIGC
    FN = AIGC -> REAL
    TP = AIGC -> AIGC

Primary practical objective:

    minimize FP

while retaining:

    maximum possible TPR

and also:

    minimize FN

Therefore optimize:

    low-FPR TPR

not ordinary accuracy alone.

=====================================================================
SECTION 21 — LOW-FPR FRONTIER
=====================================================================

Mandatory operating points:

    FPR <= 1%
    FPR <= 0.5%
    FPR <= 0.1%
    FPR <= 0.05%
    FPR <= 0.01%

For each:

    maximize TPR

and report:

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

Also report:

    AUROC
    AUPRC
    ECE
    Brier

=====================================================================
SECTION 22 — CLASSIFICATION LOSS
=====================================================================

Use:

    asymmetric BCE

Initial:

    lambda_FP = 2.5

Small controlled comparison is allowed:

    lambda_FP = 2.0
    lambda_FP = 2.5
    lambda_FP = 3.0

BUT:

Do not run a giant hyperparameter grid.

Selection is based on:

    low-FPR TPR
    FNR
    robustness
    generalization

not simply training loss.

=====================================================================
SECTION 23 — REAL TRAINING DEFINITION
=====================================================================

Actual training means:

    input
      ->
    forward
      ->
    prediction
      ->
    loss
      ->
    backward()
      ->
    optimizer.step()
      ->
    parameter change

These do NOT constitute training by themselves:

    feature extraction
    inference
    thresholding
    calibration
    report generation
    checkpoint copying

=====================================================================
SECTION 24 — RAW IMAGE PIPELINE
=====================================================================

For current-run fresh representation generation:

    verified manifest
        ->
    actual image file
        ->
    image decode
        ->
    preprocessing
        ->
    CLIP
    SigLIP
    SRM
        ->
    fresh 2,212-d representation

Because the large backbones are frozen, this fresh feature cache MAY
be retained and reused across later fusion-head epochs.

BUT:

the reporting MUST distinguish:

    fresh raw-image feature extraction

from:

    trainable fusion-head optimization

Do NOT call a vector-only optimization run:

    "end-to-end CLIP/SigLIP training"

unless the backbones themselves receive gradients.

=====================================================================
SECTION 25 — WHY FRESH FEATURE CACHING IS ALLOWED
=====================================================================

Fresh feature caching is an efficiency mechanism, not a source of
knowledge contamination.

Allowed:

    CURRENT RUN RAW IMAGES
       ->
    CURRENT RUN FOUNDATION INFERENCE
       ->
    CURRENT RUN FEATURE CACHE
       ->
    CURRENT RUN FUSION TRAINING

Forbidden:

    OLD RUN FEATURE CACHE
       ->
    CURRENT TRAINING

The current cache must be provenance-bound to:

    experiment ID
    manifest SHA
    CLIP checkpoint SHA
    SigLIP checkpoint SHA
    preprocessing hash
    feature dimension
    dtype
    extraction code hash

=====================================================================
SECTION 26 — BASE TRAINING
=====================================================================

Initialize:

    fresh fusion head
    fresh optimizer
    fresh scheduler

Run genuine multi-epoch detector training.

Initial target:

    10–20 epochs

Do not perform a single pass and call it final training.

Early stopping is permitted only based on DEV.

Do NOT use:

    INTERNAL TEST
    OOD

for early stopping.

=====================================================================
SECTION 27 — TRAINING TELEMETRY
=====================================================================

For every epoch record:

    epoch
    batch count
    samples processed
    unique samples processed
    forward passes
    backward passes
    optimizer steps
    learning rate
    training loss
    validation loss
    gradient norms
    wall-clock duration

Hardware:

    GPU utilization
    VRAM
    RAM
    swap
    CPU
    NVMe throughput
    images/sec

=====================================================================
SECTION 28 — PARAMETER UPDATE PROOF
=====================================================================

Before training:

    hash trainable parameters

After each epoch:

    hash trainable parameters

Record:

    changed tensors
    changed parameters
    L2 delta
    maximum absolute delta

Required:

    optimizer_steps > 0
    backward_passes > 0
    parameter_delta > 0

Otherwise:

    TRAINING_FAILED

=====================================================================
SECTION 29 — IMPORTANT VRAM RULE
=====================================================================

Do NOT use high VRAM utilization as the definition of successful
training.

If the foundation models are frozen and only the 2,212-d fusion head
is trainable, lower optimization-stage VRAM may be legitimate.

However:

the system MUST accurately state whether the current training stage is:

    A. raw-image backbone computation
    B. fresh feature extraction
    C. frozen-feature fusion optimization
    D. backbone fine-tuning

Never claim:

    "full raw-image end-to-end training"

when only C occurred.

=====================================================================
SECTION 30 — HARD-EXAMPLE MINING
=====================================================================

After base training:

mine HARD FP and HARD FN from TRAIN ONLY.

For REAL:

    highest P(AIGC)

For AIGC:

    lowest P(AIGC)

Select a bounded but diverse hard set.

Initial target:

    up to 2,000 hard REAL
    up to 2,000 hard AIGC

Adjust based on actual compute.

Maintain diversity across:

    generators
    domains
    image types
    resolutions
    transformations

Do not mine exclusively from one source.

=====================================================================
SECTION 31 — HARD FP OBJECTIVE
=====================================================================

For REAL images that were incorrectly called AIGC:

the system must learn:

    unusual appearance != synthetic origin

Investigate actual causes such as:

    optical bokeh
    macro photography
    studio flash
    foliage
    hair/fur
    natural texture
    sensor noise
    HDR
    sharpening
    JPEG
    resizing
    legitimate painting texture
    legitimate digital art
    CGI/3D

These are investigative hypotheses.

The system must determine whether the evidence is actually supported.

=====================================================================
SECTION 32 — HARD FN OBJECTIVE
=====================================================================

For AIGC images incorrectly called REAL:

the system must learn:

    photorealistic != REAL

Investigate actual difficult cases involving:

    SID
    Quality Paradox
    FLUX
    SDXL
    SD3
    Midjourney
    PixArt
    HFCF
    Defactify
    post-processing
    low-artifact generation

Again:

these are hypotheses until verified by actual evidence.

=====================================================================
SECTION 33 — THE FORENSIC EXPLANATION SYSTEM
=====================================================================

THIS IS A CENTRAL PART OF THIS EXPERIMENT.

It is NOT merely a reporting feature.

Its purpose is:

    discover WHY a detector prediction is wrong/right.

The system consists of:

    detector
    forensic specialists
    multimodal VLM
    evidence verifier
    critic
    feedback learner

=====================================================================
SECTION 34 — VLM ROLE
=====================================================================

The actual VLM is a separate model.

The VLM is NOT:

    CLIP
    SigLIP
    DINO
    Edge

Those provide detector/forensic evidence.

The VLM provides:

    visual reasoning
    explanation
    evidence hypothesis
    uncertainty
    alternative hypothesis

Preferred setup:

    VLM = frozen teacher/reasoner

    DETECTOR = learner

Do NOT fine-tune a large VLM on the RTX 3050 unless a separate
small-scale experiment proves it is practical.

=====================================================================
SECTION 35 — VLM DISCOVERY
=====================================================================

BEFORE the explanation-learning stage:

inspect:

    ~/.cache/huggingface/hub/

and installed model libraries.

Locate ONE practical VLM.

The VLM must genuinely:

    accept image input
    reason about the image
    generate text

Test it on ACTUAL images.

Record:

    model name
    repository
    checkpoint path
    model SHA where feasible
    parameter count
    dtype
    quantization
    VRAM/RAM requirement
    successful image input
    actual generated response
    inference latency

Do NOT explore dozens of VLMs.

Select ONE practical model.

=====================================================================
SECTION 36 — VLM HARD GATE
=====================================================================

A suitable VLM MUST be available for the explanation-feedback
experiment.

If the VLM cannot be loaded/tested:

    DO NOT FAKE IT.

Do NOT replace it with:

    templates
    hand-written explanations
    regex
    feature names
    handcrafted rules
    static artifact descriptions
    CLIP captions
    masking heuristics

If no suitable VLM exists:

    STOP THE FORENSIC FEEDBACK EXPERIMENT.

Report:

    REQUIRED_FORENSIC_VLM_UNAVAILABLE

Do NOT declare:

    EXPLANATION_LEARNING_COMPLETE

=====================================================================
SECTION 37 — VLM COMPUTE STRATEGY
=====================================================================

DO NOT run the VLM against all 260,184 images.

That is unnecessary.

Use the VLM primarily on:

    hard FP
    hard FN
    uncertain cases
    high-confidence mistakes
    specialist-disagreement cases

Initial target:

    up to 2,000 REAL hard cases
    up to 2,000 AIGC hard cases

Actual number may be lower if the evidence saturates.

Record exact count.

=====================================================================
SECTION 38 — HARD FP VLM PROMPT
=====================================================================

For a verified REAL false positive, the VLM should receive:

    actual image
    ground-truth REAL
    detector probability
    detector prediction
    relevant specialist evidence

Ask:

    "This image is verified REAL, but the detector classified it as
     AIGC.

     Explain why the detector may have made this mistake.

     What visual characteristics may have triggered suspicion?

     Which characteristics are actually present?

     Which characteristics are legitimate REAL-image characteristics
     rather than reliable evidence of synthetic generation?

     What evidence argues against the AIGC interpretation?

     Identify the relevant region(s).

     Give alternative explanations.

     State uncertainty."

=====================================================================
SECTION 39 — HARD FN VLM PROMPT
=====================================================================

For a verified AIGC false negative, the VLM should receive:

    actual image
    ground-truth AIGC
    detector probability
    detector prediction
    relevant specialist evidence

Ask:

    "This image is verified AIGC, but the detector classified it as
     REAL.

     Explain why it was difficult to detect.

     What subtle synthetic evidence may have been missed?

     Where is the evidence?

     Why is it informative?

     What should the detector have learned?

     What alternative explanation exists?

     State uncertainty."

=====================================================================
SECTION 40 — STRUCTURED EXPLANATION OUTPUT
=====================================================================

The VLM MUST produce structured output:

    case_id
    ground_truth_class
    detector_prediction
    detector_probability
    confidence
    evidence_tags
    evidence_regions
    explanation
    alternative_hypothesis
    uncertainty

Possible controlled ontology:

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

These are hypotheses.

They do NOT become ground truth merely because the VLM said them.

=====================================================================
SECTION 41 — INDEPENDENT EVIDENCE VERIFICATION
=====================================================================

Every important VLM explanation must be independently evaluated.

Use, as applicable:

    DINO
    Edge
    SRM
    frequency analysis
    gradients
    localization
    segmentation
    image statistics
    counterfactual masking

Determine:

    spatial support
    forensic support
    causal support
    contradiction
    uncertainty

Classify:

    VERIFIED_SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    UNDETERMINED

=====================================================================
SECTION 42 — COUNTERFACTUAL TESTING
=====================================================================

If the VLM says:

    "region X caused the detector's suspicious prediction"

perform where practical:

    ORIGINAL
       ->
    detector score

then:

    MASK REGION X
       ->
    detector score

optionally:

    REGION X ONLY
       ->
    detector score

Record:

    original_score
    masked_score
    isolated_score
    absolute_delta

Interpretation:

A score change suggests influence.

It does NOT automatically prove that the VLM's semantic explanation
was correct.

=====================================================================
SECTION 43 — WHY IS YOUR EXPLANATION WRONG?
=====================================================================

This is REQUIRED.

For each important unsupported/contradicted explanation:

run a critic.

The critic receives:

    actual image
    VLM explanation
    evidence tags
    evidence region
    specialist outputs
    counterfactual outputs

Ask:

    "WHY IS THIS EXPLANATION WRONG?"

when unsupported.

Ask:

    "WHY IS THIS EXPLANATION ADEQUATELY SUPPORTED?"

when supported.

The critic must identify:

    incorrect claim
    missing evidence
    contradictory evidence
    unsupported causal claim
    misleading visual characteristic
    uncertainty

Output:

    supported
    evidence_quality
    contradiction
    missing_evidence
    causal_support
    critique
    confidence

=====================================================================
SECTION 44 — CRITIC INDEPENDENCE
=====================================================================

If a SECOND multimodal model is available:

    use it as a separate critic.

If only the SAME VLM is available:

    run a separate critique pass using a fresh context.

In that case report:

    CRITIC_INDEPENDENCE = LIMITED

Never claim independent two-model validation if only one model exists.

=====================================================================
SECTION 45 — FINAL EVIDENCE STATUS
=====================================================================

Each explanation receives exactly one:

    VERIFIED_SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    UNDETERMINED

This status must depend primarily upon:

    independent forensic evidence
    counterfactual evidence
    specialist evidence

NOT merely:

    VLM confidence
    critic confidence
    agreement between AIs

=====================================================================
SECTION 46 — THE REWARD/PENALTY SYSTEM
=====================================================================

Use bounded auxiliary rewards:

    VERIFIED_SUPPORTED       = +1.0
    PARTIALLY_SUPPORTED      = +0.25
    UNDETERMINED             =  0.0
    UNSUPPORTED              = -0.50
    CONTRADICTED             = -1.0
    CONFIDENTLY_FABRICATED   = -1.0

These are INITIAL values.

Do not assume they are optimal.

The classification label is always more authoritative than the
explanation reward.

=====================================================================
SECTION 47 — EXTREMELY IMPORTANT LABEL RULE
=====================================================================

Do NOT confuse:

    classification correctness

with:

    explanation correctness.

Example:

    ground truth = REAL
    detector prediction = REAL
    explanation = "six fingers"

The classification is CORRECT.

Only the explanation/evidence attribution is wrong.

DO NOT teach the detector:

    REAL -> AIGC

because the explanation was bad.

Likewise:

    ground truth = AIGC
    detector prediction = AIGC
    explanation = wrong artifact

Classification remains correct.

Only explanation/evidence attribution needs correction.

=====================================================================
SECTION 48 — FEEDBACK LEARNING OBJECTIVE
=====================================================================

The verified forensic feedback must enter a REAL trainable objective.

Conceptually:

    L_total =

        L_classification

        + lambda_e * L_evidence

        + lambda_loc * L_localization

        + lambda_cf * L_counterfactual

        + lambda_fb * L_feedback

Start with SMALL auxiliary coefficients.

Classification remains the primary objective.

Document the exact implemented loss.

=====================================================================
SECTION 49 — WHAT THE FEEDBACK ACTUALLY TEACHES
=====================================================================

HARD REAL FP:

The model should learn:

    "This legitimate characteristic is not sufficient evidence of AIGC."

HARD AIGC FN:

The model should learn:

    "This subtle synthetic characteristic is informative and should
     contribute to AIGC detection."

The purpose is NOT merely to classify the image again.

The purpose is to correct the detector's learned evidence attribution.

=====================================================================
SECTION 50 — CRITICAL:
# FEEDBACK MUST CAUSE ACTUAL PARAMETER UPDATES
=====================================================================

The following is the required loop:

    HARD FP/FN
        ->
    ACTUAL VLM EXPLANATION
        ->
    INDEPENDENT VERIFICATION
        ->
    ACTUAL CRITIC
        ->
    VERIFIED REWARD/PENALTY
        ->
    FEEDBACK LOSS
        ->
    backward()
        ->
    optimizer.step()
        ->
    MODEL PARAMETER CHANGE
        ->
    NEW PREDICTION

Writing reward to JSON is NOT learning.

Writing a critique to JSON is NOT learning.

Generating explanations is NOT learning.

Only actual optimizer updates that are influenced by the feedback
constitute feedback learning.

=====================================================================
SECTION 51 — FEEDBACK PARAMETER PROOF
=====================================================================

Immediately before feedback optimization:

    save trainable parameter hash.

After feedback optimization:

    save trainable parameter hash.

Record:

    feedback_backward_passes
    feedback_optimizer_steps
    feedback_gradient_norm
    before_hash
    after_hash
    parameter_delta
    changed_parameter_count

Required:

    feedback_optimizer_steps > 0

AND:

    feedback_parameter_delta > 0

Otherwise:

    EXPLANATION_FEEDBACK_LEARNING = NOT_EXECUTED

Do not claim otherwise.

=====================================================================
SECTION 52 — FEEDBACK MUST NOT BE FAKE RL
=====================================================================

Do NOT implement:

    reward stored in JSON
    but no trainable model receives the reward.

Do NOT call:

    a reward table

"reinforcement learning."

The feedback must participate in an actual trainable objective.

Acceptable mechanisms include:

    reward-weighted classification
    hard-example weighted BCE
    pairwise ranking
    evidence classification
    localization loss
    contrastive evidence consistency
    counterfactual consistency
    attribution regularization

Use a differentiable implementation where possible.

=====================================================================
SECTION 53 — HARD FP FEEDBACK EXAMPLE
=====================================================================

Example:

    Ground truth:
        REAL

    Detector:
        P(AIGC) = 0.97

    VLM:
        "Strong macro texture and bokeh may have triggered suspicion."

    Independent verification:
        texture/bokeh are consistent with real optical imaging.

    Critic:
        "Treating bokeh as AIGC evidence is unsupported."

    Feedback:
        classification correction toward REAL
        +
        penalty for incorrect evidence attribution

The detector must become less likely to accuse similar REAL images.

=====================================================================
SECTION 54 — HARD FN FEEDBACK EXAMPLE
=====================================================================

Example:

    Ground truth:
        AIGC

    Detector:
        P(AIGC) = 0.17

    VLM:
        "The image appears photographic globally, but local texture and
         edge continuity indicate subtle synthetic structure."

    Independent verification:
        supports the claimed region.

    Critic:
        "The detector missed a valid local synthetic signature."

    Feedback:
        stronger AIGC classification target
        +
        positive evidence alignment

The detector should become more sensitive to that validated signature.

=====================================================================
SECTION 55 — FEEDBACK ROUND 1
=====================================================================

Execute:

    BASE TRAINING
        ->
    DEV EVALUATION
        ->
    HARD FP/FN MINING
        ->
    ACTUAL VLM EXPLANATION
        ->
    INDEPENDENT VERIFICATION
        ->
    CRITIC
        ->
    REWARD/PENALTY
        ->
    FEEDBACK LOSS
        ->
    BACKWARD
        ->
    OPTIMIZER STEP
        ->
    PARAMETER CHANGE
        ->
    DEV EVALUATION

All stages are mandatory.

=====================================================================
SECTION 56 — FEEDBACK ROUND 2
=====================================================================

After Round 1:

mine NEW hard FP/FN.

Do not simply reuse exactly the same examples.

Execute:

    NEW HARD FP/FN
        ->
    VLM
        ->
    VERIFICATION
        ->
    CRITIC
        ->
    REWARD/PENALTY
        ->
    FEEDBACK LOSS
        ->
    BACKWARD
        ->
    OPTIMIZER STEP
        ->
    PARAMETER CHANGE
        ->
    DEV EVALUATION

Maximum:

    2 major feedback rounds.

Do not build an infinite self-training loop.

=====================================================================
SECTION 57 — SCIENTIFIC A/B COMPARISON
=====================================================================

Create explicit stages:

    A = fresh base detector
    B = after hard-example training
    C = after feedback Round 1
    D = after feedback Round 2

Compare:

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
    ECE
    Brier

and:

    TPR @ FPR <= 1%
    TPR @ FPR <= 0.5%
    TPR @ FPR <= 0.1%
    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%

If explanation feedback does not improve classification:

    report that.

If explanation feedback hurts:

    report that.

Do not force the explanation system into production.

=====================================================================
SECTION 58 — PROTECT AGAINST SELF-REINFORCING HALLUCINATION
=====================================================================

NEVER do:

    detector
       ->
    detector-generated explanation
       ->
    explanation treated as truth
       ->
    detector trained to agree with itself

Required:

    detector
       ->
    VLM hypothesis
       ->
    independent forensic evidence
       ->
    counterfactual
       ->
    critic
       ->
    verified evidence
       ->
    feedback
       ->
    detector update

Ground truth remains external.

=====================================================================
SECTION 59 — HARD EXAMPLE WEIGHTS
=====================================================================

Do NOT train exclusively on hard cases.

Use:

    NORMAL DATA
        +
    HARD REAL
        +
    HARD AIGC

Initial bounded weighting:

    HARD REAL = approximately 2.0x
    HARD AIGC = approximately 2.0x

Validate this.

Do not permit hard examples to destroy generator/domain diversity.

=====================================================================
SECTION 60 — GENERATOR-AWARE SAMPLING
=====================================================================

Preserve exposure across:

    Quality Paradox
    SDXL
    Midjourney
    FLUX/SD3
    SID
    PixArt
    HFCF
    Defactify

Prevent one generator from monopolizing gradients.

Record:

    physical distribution
    sampled distribution
    samples/epoch
    repeat factor
    actual exposure

=====================================================================
SECTION 61 — REAL-DOMAIN SAMPLING
=====================================================================

Preserve:

    COCO
    WikiArt
    Web Photography
    Archival
    Hard Macro/Bokeh

The model must encounter legitimate unusual imagery.

This is essential to reducing false positives.

=====================================================================
SECTION 62 — FORENSIC EXPLANATION TRAINING DATA
=====================================================================

Do NOT require explanations for all 260,184 images.

That would be unnecessarily expensive.

Generate explanations mainly for:

    hard FP
    hard FN
    ambiguous cases
    high-confidence errors
    expert disagreements

The explanation set is therefore a targeted forensic curriculum.

=====================================================================
SECTION 63 — HARD FP/FN CURRICULUM
=====================================================================

Round 1 should contain diverse examples.

Do not simply select:

    top 2,000 FP

from one dataset.

Stratify by:

    source
    domain
    generator
    difficulty
    resolution

Likewise for FN.

=====================================================================
SECTION 64 — FORENSIC EVIDENCE ONTOLOGY
=====================================================================

Potential evidence families:

ANATOMY:

    hands
    fingers
    eyes
    teeth
    facial geometry
    limbs

TEXT:

    glyphs
    typography
    malformed characters

PHYSICAL:

    perspective
    shadows
    reflections
    lighting

TEXTURE:

    brushstrokes
    repeated texture
    excessive smoothness
    local inconsistency

FORENSIC:

    edges
    residuals
    high-frequency structure
    periodic patterns
    spectral evidence
    upsampling

IMAGE PROCESSING:

    compression
    resampling
    sharpening

SEMANTIC:

    object contradiction
    local/global inconsistency

These are evidence categories only.

Never claim a category exists unless the evidence supports it.

=====================================================================
SECTION 65 — CALIBRATION
=====================================================================

After final feedback training:

use:

    CALIBRATION = 4,000

Fit fresh calibration.

Evaluate:

    temperature scaling
    Platt scaling
    isotonic where statistically justified

Do not blindly reuse historical temperatures.

Measure:

    ECE
    Brier
    high-confidence calibration

Especially inspect:

    p >= 0.95
    p >= 0.99

=====================================================================
SECTION 66 — THRESHOLD OPTIMIZATION
=====================================================================

Generate dense threshold curves.

Do NOT assume:

    tau = 0.80

is final.

For each target:

    <=1%
    <=0.5%
    <=0.1%
    <=0.05%
    <=0.01%

select:

    maximum TPR satisfying the actual inequality.

For every selected threshold:

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

=====================================================================
SECTION 67 — STATISTICAL HONESTY
=====================================================================

If:

    FP = 0

report:

    0 / N_REAL

Do NOT claim:

    population FPR = 0

The observed test sample has finite resolution.

State statistical uncertainty / resolution when relevant.

=====================================================================
SECTION 68 — ROBUSTNESS
=====================================================================

Evaluate:

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
    TP
    TN
    FP
    FN
    FPR
    FNR
    TPR
    TNR

per condition.

Calculate:

    robustness index
    worst condition
    degradation from clean

=====================================================================
SECTION 69 — GENERATOR BREAKDOWN
=====================================================================

Report by:

    Quality Paradox
    SDXL
    Midjourney
    FLUX/SD3
    SID
    PixArt
    HFCF
    Defactify
    other sufficiently represented approved generators

Do not hide generator-specific failures in aggregate numbers.

=====================================================================
SECTION 70 — REAL DOMAIN BREAKDOWN
=====================================================================

Report REAL FPR by:

    COCO
    WikiArt
    Web Photography
    Archival
    Hard Macro/Bokeh

Identify which real domains contribute the remaining FP.

=====================================================================
SECTION 71 — CONDITIONAL VERIFIER ANALYSIS
=====================================================================

For Stage 2 measure:

    invocation rate
    FP rescued
    FN rescued
    new FP
    new FN
    net error change

Also measure:

    average latency
    P95
    P99
    worst-case latency

Determine whether DINO + Edge actually helps.

If not:

    report failure.

Do not preserve it merely because it is architecturally sophisticated.

=====================================================================
SECTION 72 — RAW END-TO-END LATENCY
=====================================================================

Measure separately:

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
    VLM explanation latency when invoked

Report:

    average
    P95
    P99
    worst-case
    Stage-2 invocation rate
    VLM invocation rate

Do NOT report cached-vector throughput as raw-image inference throughput.

=====================================================================
SECTION 73 — MEMORY / STORAGE PIPELINE
=====================================================================

Use:

    APPROVED DATA
        ->
    NVMe STAGING
        ->
    ASYNC PREFETCH
        ->
    PINNED RAM
        ->
    NON-BLOCKING GPU TRANSFER

DO NOT:

    load the full 400–600 GB corpus into 31 GB RAM.

DO NOT:

    intentionally use swap as the dataset cache.

Use RAM as a bounded hot cache.

Measure:

    DataLoader workers
    prefetch factor
    persistent workers
    batch size
    NVMe throughput

Choose settings from actual benchmark data.

=====================================================================
SECTION 74 — HARDWARE HEALTH
=====================================================================

During training monitor:

    GPU utilization
    VRAM
    RAM
    swap
    CPU
    NVMe
    temperature
    throughput

If:

    sustained swap activity > approximately zero

then:

    reduce prefetch/RAM pressure.

If:

    OOM

then:

    reduce batch size
    or reduce feature footprint
    before changing architecture.

If:

    GPU utilization collapses

investigate I/O bottlenecks.

A running PID is NOT proof of healthy training.

=====================================================================
SECTION 75 — CHECKPOINTING
=====================================================================

Create separate current-run checkpoints:

    fresh_initial
    base_best
    hard_example_round1
    feedback_round1
    feedback_round2
    best_low_fpr
    final_frozen

Every checkpoint should contain:

    model
    optimizer
    scheduler
    epoch
    step
    sampler state
    RNG state
    manifest SHA
    foundation checkpoint hashes
    preprocessing hash
    loss configuration
    routing configuration
    feedback configuration

Use atomic writes.

Never overwrite the only known-good checkpoint.

=====================================================================
SECTION 76 — INTERNAL TEST LOCK
=====================================================================

The INTERNAL TEST:

    10,316

must remain completely locked during:

    training
    hard mining
    feedback
    calibration
    threshold selection
    architecture selection

Only after:

    final model frozen
    routing frozen
    calibration frozen
    threshold frozen
    feedback mechanism frozen

run INTERNAL TEST once.

If Stage 2 is part of production:

    run Stage 1 + Stage 2 end-to-end.

Do NOT tune afterward.

=====================================================================
SECTION 77 — OOD LOCK
=====================================================================

Only after final freezing evaluate:

    Synthbuster
    AIGIBench
    Chameleon
    VCT2
    WildRF
    SynthWildX

Run once.

Do not:

    tune
    retrain
    mine hard cases
    change threshold

after seeing OOD.

=====================================================================
SECTION 78 — FINAL EVALUATION OBJECTIVES
=====================================================================

Primary:

    TPR @ FPR <= 0.1%

Secondary:

    TPR @ FPR <= 0.05%
    TPR @ FPR <= 0.01%
    FNR
    AUROC
    AUPRC
    robustness
    generator generalization
    real-domain FPR
    calibration
    latency
    VRAM

Do NOT select the model simply because:

    AUROC is marginally higher.

=====================================================================
SECTION 79 — REQUIRED TRAINING PROOF ARTIFACT
=====================================================================

Create:

    reports/final_actual_training_telemetry.json

It MUST contain actual:

    start_time
    end_time
    duration
    epochs
    batches
    samples
    unique samples
    forward passes
    backward passes
    optimizer steps
    scheduler steps
    loss history
    learning-rate history
    gradient norms

Hardware:

    GPU utilization
    VRAM
    RAM
    swap
    CPU
    NVMe throughput
    images/sec

Parameters:

    initial hash
    per-epoch hashes
    final hash
    cumulative delta
    changed parameter count

=====================================================================
SECTION 80 — REQUIRED FORENSIC FEEDBACK PROOF
=====================================================================

Create:

    reports/final_forensic_feedback_telemetry.json

Include:

    HARD_FP_count
    HARD_FN_count

    VLM_name
    VLM_checkpoint
    VLM_hash
    VLM_calls
    successful_VLM_calls
    failed_VLM_calls

    explanations_generated
    explanations_verified

    VERIFIED_SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED
    CONTRADICTED
    UNDETERMINED

    critic_calls
    critic_rejections
    critic_uncertain

    counterfactual_tests
    counterfactual_supported

    rewards
    penalties

    feedback_backward_passes
    feedback_optimizer_steps
    feedback_gradient_norm
    feedback_parameter_delta

=====================================================================
SECTION 81 — REQUIRED HARD-EXAMPLE ARTIFACTS
=====================================================================

Create:

    reports/final_hard_fp_round1.json
    reports/final_hard_fn_round1.json
    reports/final_hard_fp_round2.json
    reports/final_hard_fn_round2.json

Each case should contain:

    image_id
    ground_truth
    detector_probability
    detector_prediction
    source
    generator/domain
    VLM explanation
    evidence tags
    evidence regions
    verification
    counterfactual result
    critic result
    explanation quality
    reward
    feedback target

=====================================================================
SECTION 82 — REQUIRED EXPLANATION ARTIFACTS
=====================================================================

Create:

    reports/final_vlm_execution.json
    reports/final_explanation_generation.json
    reports/final_explanation_verification.json
    reports/final_explanation_critic.json
    reports/final_explanation_feedback.json
    reports/final_feedback_parameter_updates.json

=====================================================================
SECTION 83 — REQUIRED MODEL ARTIFACTS
=====================================================================

Create:

    reports/final_training_dataset_audit.json
    reports/final_training_manifest_audit.json
    reports/final_training_provenance.json

    reports/final_training_metrics.json
    reports/final_training_loss_curve.json

    reports/final_conditional_verifier.json
    reports/final_calibration.json
    reports/final_thresholds.json
    reports/final_robustness.json

    reports/final_generator_breakdown.json
    reports/final_domain_breakdown.json
    reports/final_latency.json

    reports/final_internal_test.json
    reports/final_ood.json

=====================================================================
SECTION 84 — FINAL MASTER REPORT
=====================================================================

Only after actual computation is complete create:

    reports/FINAL_TRAINING_MASTER_REPORT.md

The report MUST separately state:

    ACTUAL_DETECTOR_TRAINING
    ACTUAL_HARD_EXAMPLE_LEARNING
    ACTUAL_VLM_EXPLANATION
    ACTUAL_FORENSIC_VERIFICATION
    ACTUAL_CRITIC
    ACTUAL_FEEDBACK_OPTIMIZATION
    ACTUAL_CALIBRATION
    ACTUAL_INTERNAL_TEST
    ACTUAL_OOD

Each must be:

    EXECUTED
    NOT_EXECUTED
    FAILED

Never collapse them into:

    "training complete"

=====================================================================
SECTION 85 — REPORT CONSISTENCY
=====================================================================

Every reported number must be reproducible from:

    raw predictions
    labels
    manifest
    telemetry
    checkpoint
    actual logs

If two reports disagree:

    recompute.

If a metric cannot be reproduced:

    NOT ESTABLISHED

Do not choose the most favorable number.

=====================================================================
SECTION 86 — ANTI-CHEATING RULE
=====================================================================

NEVER:

    invent metrics
    invent data counts
    invent VLM outputs
    invent critic outputs
    invent rewards
    invent penalties
    invent optimizer steps
    invent parameter changes
    invent training duration
    invent hard-example counts

NEVER:

    call feature extraction "training"
    call inference "training"
    call report generation "training"
    call static masking "VLM reasoning"
    call a reward table "learning"
    call an LLM agreement "ground truth"

NEVER:

    use internal test for tuning
    use OOD for tuning
    use OOD for hard mining
    reuse stale prediction arrays
    reuse stale explanation data
    reuse old feedback targets
    reuse stale fusion weights

=====================================================================
SECTION 87 — CURRENT RUN MUST FAIL CLOSED
=====================================================================

The experiment MUST stop instead of silently degrading when:

    manifest is wrong
    training count is wrong
    label count is wrong
    OOD contamination exists
    train/test overlap exists
    VLM is unavailable
    VLM cannot be tested
    required feedback loop cannot execute
    optimizer steps are zero
    parameter delta is zero
    feedback optimizer steps are zero
    feedback parameter delta is zero
    metrics cannot be reproduced

Do NOT "continue anyway."

=====================================================================
SECTION 88 — IMPORTANT DISTINCTION:
# VLM UNAVAILABLE IS NOT A PERMISSION TO FAKE IT
=====================================================================

If:

    no suitable VLM

then:

    REQUIRED_FORENSIC_VLM_UNAVAILABLE

The system must NOT say:

    "structured ontology executed"

and then call that:

    "VLM explanation learning."

Heuristic forensic analysis may be documented separately, but it is
NOT equivalent to the requested VLM feedback experiment.

=====================================================================
SECTION 89 — IMPORTANT DISTINCTION:
# FEATURE TRAINING IS VALID BUT MUST BE LABELED CORRECTLY
=====================================================================

If the actual sequence is:

    raw images
      ->
    frozen CLIP/SigLIP/SRM
      ->
    fresh features
      ->
    trainable fusion head

then this is:

    FRESH FEATURE EXTRACTION
    +
    FROZEN-REPRESENTATION FUSION TRAINING

That is legitimate machine learning.

But it is NOT:

    end-to-end backbone fine-tuning.

Report this honestly.

=====================================================================
SECTION 90 — HARD EXPLANATION FEEDBACK MUST BE SEPARATE
=====================================================================

The classifier can train on the complete dataset.

The VLM explanation subsystem should focus on:

    difficult errors

This keeps computation practical.

The intended architecture is therefore:

    FULL DATA CLASSIFIER TRAINING
             +
    TARGETED FORENSIC TEACHER
             +
    TARGETED FEEDBACK RETRAINING

NOT:

    VLM on every image.

=====================================================================
SECTION 91 — LONG-TERM SCALE
=====================================================================

The approved training partition currently contains:

    260,184 images

There may be additional physical data across the 400–600 GB storage
pool.

Do NOT automatically add it.

Future additions require:

    provenance
    deduplication
    label validation
    generator/domain assignment
    split assignment
    OOD exclusion
    manifest update

Only then can they become part of a future training manifest.

=====================================================================
SECTION 92 — FINAL STATE MACHINE
=====================================================================

The experiment MUST follow:

STATE 0:
    READ KNOWLEDGE BASE

STATE 1:
    READ AUTHORIZATION

STATE 2:
    READ RECONCILIATION

STATE 3:
    VERIFY REMOTE MACHINE

STATE 4:
    VERIFY DATA ROOT

STATE 5:
    VERIFY MANIFEST

STATE 6:
    VERIFY SPLITS

STATE 7:
    VERIFY OOD EXCLUSION

STATE 8:
    CREATE CLEAN RUN

STATE 9:
    LOCATE VLM

STATE 10:
    TEST VLM

STATE 11:
    FRESH MODEL INITIALIZATION

STATE 12:
    FRESH RAW IMAGE FEATURE EXTRACTION

STATE 13:
    BASE CLASSIFICATION TRAINING

STATE 14:
    DEV EVALUATION

STATE 15:
    HARD FP/FN MINING

STATE 16:
    ACTUAL VLM EXPLANATION

STATE 17:
    FORENSIC VERIFICATION

STATE 18:
    COUNTERFACTUAL ANALYSIS

STATE 19:
    AI CRITIC

STATE 20:
    REWARD/PENALTY

STATE 21:
    FEEDBACK LOSS

STATE 22:
    BACKWARD

STATE 23:
    OPTIMIZER UPDATE

STATE 24:
    PARAMETER CHANGE VERIFICATION

STATE 25:
    DEV RE-EVALUATION

STATE 26:
    SECOND HARD FP/FN ROUND

STATE 27:
    SECOND VLM/VERIFICATION/CRITIC/FEEDBACK ROUND

STATE 28:
    SECOND PARAMETER UPDATE

STATE 29:
    FINAL DEV EVALUATION

STATE 30:
    CALIBRATION

STATE 31:
    THRESHOLD OPTIMIZATION

STATE 32:
    ROBUSTNESS

STATE 33:
    GENERATOR BREAKDOWN

STATE 34:
    REAL-DOMAIN BREAKDOWN

STATE 35:
    LATENCY

STATE 36:
    FINAL FREEZE

STATE 37:
    INTERNAL TEST ONCE

STATE 38:
    OOD ONCE

STATE 39:
    FINAL REPORT

STATE 40:
    KNOWLEDGE BASE UPDATE

STATE 41:
    STOP

DO NOT skip states.

=====================================================================
SECTION 93 — REQUIRED FINAL SCIENTIFIC COMPARISON
=====================================================================

Compare:

    historical Phase-4 baseline
    fresh base model
    hard-example model
    feedback Round 1
    feedback Round 2
    final frozen system

Do NOT overwrite history.

Do NOT pretend historical numbers were produced by the current run.

=====================================================================
SECTION 94 — WHAT THE FINAL SYSTEM SHOULD LEARN
=====================================================================

The desired behavior is:

REAL:

    unusual-looking imagery is not automatically AIGC.

AIGC:

    photorealistic imagery is not automatically REAL.

For a REAL false positive:

    "You were fooled by legitimate visual evidence.
     Learn not to overinterpret it."

For an AIGC false negative:

    "You missed subtle synthetic evidence.
     Learn to use it."

For an incorrect explanation:

    "Your classification may be correct, but your explanation is wrong.
     Correct the evidence attribution."

=====================================================================
SECTION 95 — FINAL FORENSIC OUTPUT
=====================================================================

For difficult cases, the final production system should be capable of:

    Classification
    Confidence
    Evidence tags
    Evidence region(s)
    Explanation
    Alternative hypothesis
    Verification status
    Counterfactual result
    Explanation confidence

Example REAL FP:

    Classification:
        REAL

    Confidence:
        0.98

    Evidence:
        macro texture
        optical bokeh

    Verification:
        legitimate optical characteristics

    Explanation:
        "The detector was likely misled by strong high-frequency
         texture and shallow optical blur."

Example AIGC FN:

    Classification:
        AIGC

    Confidence:
        0.92

    Evidence:
        local texture inconsistency
        edge discontinuity

    Verification:
        independently supported

    Explanation:
        "The image is globally photorealistic, but local texture and
         edge structure contain validated synthetic inconsistencies."

=====================================================================
SECTION 96 — FINAL COMPLETION GATES
=====================================================================

The run may NOT declare:

    FINAL_TRAINING_COMPLETE = TRUE

unless ALL are true:

[ ] knowledge base read
[ ] authorization read
[ ] reconciliation read
[ ] correct training manifest
[ ] 260,184 training images verified
[ ] 149,000 REAL
[ ] 111,184 AIGC
[ ] zero OOD contamination
[ ] split isolation
[ ] clean experiment namespace
[ ] fresh trainable initialization
[ ] actual detector training
[ ] multiple epochs
[ ] backward passes
[ ] optimizer steps
[ ] parameter changes
[ ] hard FP mining
[ ] hard FN mining
[ ] actual VLM found
[ ] VLM smoke test successful
[ ] actual VLM explanations
[ ] independent verification
[ ] counterfactual analysis
[ ] actual critic
[ ] reward/penalty
[ ] feedback loss
[ ] feedback backward passes
[ ] feedback optimizer steps
[ ] feedback parameter changes
[ ] Round 1 complete
[ ] Round 2 complete
[ ] calibration
[ ] threshold optimization
[ ] robustness
[ ] generator analysis
[ ] real-domain analysis
[ ] latency analysis
[ ] final freeze
[ ] internal test once
[ ] OOD once
[ ] telemetry reproducible

If ANY box is false:

    FINAL_TRAINING_COMPLETE = FALSE

=====================================================================
SECTION 97 — EXPLANATION LEARNING COMPLETION GATE
=====================================================================

The system may NOT claim:

    EXPLANATION_LEARNING_COMPLETE

unless:

    actual VLM calls > 0
    explanations generated > 0
    independent verification > 0
    critic calls > 0
    verified feedback > 0
    feedback backward passes > 0
    feedback optimizer steps > 0
    feedback parameter delta > 0

=====================================================================
SECTION 98 — DETECTOR TRAINING COMPLETION GATE
=====================================================================

The system may NOT claim:

    DETECTOR_TRAINING_COMPLETE

unless:

    actual training inputs verified
    forward passes occurred
    loss calculated
    backward passes occurred
    optimizer steps occurred
    trainable parameters changed

=====================================================================
SECTION 99 — FEEDBACK LEARNING COMPLETION GATE
=====================================================================

The system may NOT claim:

    MODEL_LEARNED_FROM_FORENSIC_FEEDBACK

unless:

    forensic feedback existed
    feedback entered trainable loss
    backward occurred
    optimizer.step occurred
    model parameters changed
    before/after DEV metrics were measured

=====================================================================
SECTION 100 — FINAL REPORTING RULE
=====================================================================

COMPUTATION FIRST.

REPORT SECOND.

The system must NEVER:

    create a polished final report first
    then attempt to make the computation fit the report.

If computation says:

    failed

the report must say:

    FAILED

If computation says:

    not executed

the report must say:

    NOT EXECUTED

If evidence is uncertain:

    NOT ESTABLISHED

=====================================================================
SECTION 101 — FINAL KNOWLEDGE BASE UPDATE
=====================================================================

After the experiment:

update:

    /home/manan/aigc_robust_detection/docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

Record:

    exact experiment ID
    manifest hash
    model architecture
    foundation checkpoint hashes
    VLM identity
    training configuration
    feedback configuration
    actual optimizer steps
    actual feedback steps
    final checkpoint hash
    final metrics
    failures
    lessons
    unresolved problems

Do not erase prior historical records.

Add the new experiment as a new authoritative chronological section.

=====================================================================
SECTION 102 — FINAL EXECUTION BEHAVIOR
=====================================================================

DO NOT begin with:

    "I will explore foundation models."

DO NOT begin with:

    "I have created a plan."

DO NOT begin with:

    "I have generated a report."

DO NOT begin with:

    "Training is complete."

DO NOT spend the execution window repeatedly rediscovering which
foundation model is best.

DO NOT use historical report text as a substitute for computation.

=====================================================================
SECTION 103 — FIRST ACTIONS
=====================================================================

Immediately perform:

    STEP 1:
        connect to buildabot

    STEP 2:
        activate:
            ~/.venvs/aigc-detector/

    STEP 3:
        read:
            AUTH_PHASE1.md

    STEP 4:
        read completely:
            docs/MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md

    STEP 5:
        read:
            reports/final_reconciliation_v2.json
            reports/final_reconciliation_v2.md

    STEP 6:
        inspect:
            current project tree
            current manifests
            current reports
            current checkpoints
            current model directory
            Hugging Face cache

    STEP 7:
        VERIFY rather than assume:
            260,184 training rows
            149,000 REAL
            111,184 AIGC

    STEP 8:
        verify:
            zero OOD training contamination

    STEP 9:
        verify:
            DEV
            CALIBRATION
            TEST

    STEP 10:
        create clean-run namespace

    STEP 11:
        locate actual VLM

    STEP 12:
        inspect custom VLM loading code where applicable

    STEP 13:
        run actual VLM smoke test NON-INTERACTIVELY

    STEP 14:
        only after the VLM succeeds:
            initialize fresh detector head

    STEP 15:
        perform the first actual detector training update

    STEP 16:
        verify parameter change

    STEP 17:
        continue through the state machine.

=====================================================================
SECTION 104 — ABSOLUTE FINAL RULE
=====================================================================

The project objective is NOT:

    produce a convincing report.

The objective is:

    produce a genuinely better detector.

The detector must learn from:

    260,184 approved training images

and from:

    real hard false positives
    real hard false negatives

using:

    actual multimodal forensic reasoning
    actual independent evidence verification
    actual critique
    actual bounded reward/penalty
    actual feedback-driven optimization

The intended core learning loop is:

    TRAIN
      ↓
    MAKE MISTAKES
      ↓
    FIND HARD FP/FN
      ↓
    ASK ACTUAL VLM "WHY?"
      ↓
    IDENTIFY EVIDENCE
      ↓
    VERIFY EVIDENCE INDEPENDENTLY
      ↓
    ASK CRITIC
      ↓
    "WHY IS THIS EXPLANATION WRONG?"
      ↓
    REWARD / PENALTY
      ↓
    FEEDBACK LOSS
      ↓
    BACKWARD
      ↓
    OPTIMIZER STEP
      ↓
    PARAMETER CHANGE
      ↓
    LEARN
      ↓
    FIND NEW MISTAKES
      ↓
    REPEAT ONCE
      ↓
    CALIBRATE
      ↓
    THRESHOLD
      ↓
    FREEZE
      ↓
    TEST
      ↓
    OOD
      ↓
    REPORT

EVERY ARROW MUST ACTUALLY EXECUTE.

NO STALE DERIVED DATA.

NO UNAUTHORIZED DATA.

NO OOD CONTAMINATION.

NO DIRECTORY-SCAN TRAINING.

NO FAKE VLM.

NO FAKE EXPLANATIONS.

NO FAKE CRITIC.

NO FAKE REWARD.

NO FAKE FEEDBACK.

NO FAKE OPTIMIZER STEPS.

NO REPORT-ONLY MODE.

NO SELF-GENERATED GROUND TRUTH.

NO SILENT CHANGES.

NO HIDDEN FAILURES.

BEGIN EXECUTION.