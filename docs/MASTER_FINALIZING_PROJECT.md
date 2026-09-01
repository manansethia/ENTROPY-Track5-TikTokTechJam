PROJECT: AIGC FORENSICS

MISSION:

Take ownership of the entire existing project and turn it into the final hackathon-ready AIGC Forensics product.

This includes:

1. understanding the entire repository
2. understanding all historical model work
3. inspecting the actual model files and reports on the private compute machine
4. finalizing the strongest practical model
5. integrating it into the backend
6. finishing the website
7. making every forensic feature functional
8. preparing browser/local inference
9. preparing downloads and reproducibility
10. preparing the public GitHub repository
11. preparing the technical documentation needed for the hackathon submission

Do not treat this as a redesign from scratch.

A lot of work already exists.

First understand it.
Then improve and complete it.

==================================================
NON-NEGOTIABLE PUBLIC PRIVACY RULE
==================================================

THE PUBLIC WEBSITE MUST NEVER DISPLAY:

- the private machine name
- the word Buildabot
- server hostnames
- SSH addresses
- IP addresses
- GPU model
- RTX
- CUDA device name
- filesystem paths
- usernames
- backend infrastructure names
- internal deployment details

Do not put these anywhere in:

- header
- footer
- About
- Technology
- API responses shown to users
- model page
- metadata
- reports
- screenshots
- demo mode
- source-visible frontend constants

Public terminology should be generic:

ANALYSIS ENGINE ONLINE

FULL ANALYSIS MODE

LOCAL BROWSER MODE

MODEL READY

The infrastructure may of course be used internally.

==================================================
INTERNAL COMPUTE ACCESS
==================================================

For development and model inspection only, connect privately to:

root@100.69.97.120

Use SSH only from the development environment.

Never copy the address or credentials into public frontend code or public documentation.

Inspect:

/home/manan/aigc_robust_detection

and all relevant AI model/data locations under the available storage.

Read the actual:

- Markdown reports
- JSON reports
- training manifests
- model manifests
- benchmark results
- checkpoints
- scripts
- logs
- dataset registries
- heatmaps
- architecture files

Do not rely on previous prose summaries if the actual artifact can be inspected.

The artifact on disk is the source of truth.

==================================================
FIRST TASK: FULL PROJECT AUDIT
==================================================

Before making major changes, create:

reports/final_project_audit.md

Document:

FRONTEND
BACKEND
MODEL PIPELINE
MODEL HISTORY
DATASETS
TRAINING
VALIDATION
TESTING
ROBUSTNESS
PROVENANCE
REPORTING
DEPLOYMENT
GITHUB READINESS

Also produce:

reports/model_registry.json

with every model/checkpoint found.

For every checkpoint record:

name
version
architecture
parameters
checkpoint size
precision
SHA-256
purpose
training status
date
source dataset
validation dataset
current role
superseded_by
production_candidate
standalone_or_ensemble

Do not invent missing information.

==================================================
GIT SAFETY FIRST
==================================================

Before changing anything:

git add -A
git commit -m "checkpoint: before final product integration"

Create:

git checkout -b feat/final-hackathon-product

Use small logical commits.

Never commit:

SSH keys
API keys
Cloudflare tokens
Tailscale credentials
private paths
private datasets
large checkpoints unless explicitly intended
.env
credentials

Create a strong .gitignore before public release.

==================================================
CHALLENGE REQUIREMENTS
==================================================

The solution must address:

AI-generated vs authentic image detection

with robustness against realistic transformations.

The challenge transformation set includes:

JPEG:
quality 90
70
50
30

Gaussian Blur:
sigma 0.5
1.0
2.0

Resize:
0.5x then upscale
0.25x then upscale

Gaussian Noise:
sigma 0.02
0.05
0.10

Color Jitter:
brightness / contrast / saturation ±20%

Center Crop:
retain 80%

The model must remain below 2 billion parameters.

==================================================
STRICT BENCHMARK ISOLATION
==================================================

The organizer demonstration benchmark is:

REAL:
COCO val2017
4,998 images

AIGC:
DALL-E Advanced
8,843 images

TOTAL:
13,841 images

THIS DATA IS EVALUATION ONLY.

It must NEVER be used for:

training
fine-tuning
distillation
threshold tuning
calibration
hard-example mining
feature selection
checkpoint selection
teacher feedback
augmentation generation

Verify programmatically that no benchmark file appears in any training manifest.

Create:

reports/benchmark_isolation_audit.json

==================================================
MODEL HISTORY
==================================================

Reconstruct the actual model lineage from files.

Expected broad history includes work such as:

early V1 models
V2 spectral/frequency models
Triple-Hybrid Champion
V3 C0-C7 specialist work
CommunityForensics C3 correction
high-resolution training
hard-real remediation
V4 patch/localization work
V5 CAG spatial work
master teacher/fusion work
student distillation work

But do NOT copy this list blindly.

Verify every stage from disk.

Create:

docs/MODEL_HISTORY.md

For each generation explain in simple language:

what problem existed
what was changed
which data was added
how many parameters
what improved
what became worse
what was learned from the experiment
why the next version was created

Avoid unnecessary jargon.

==================================================
MODEL EVOLUTION DATA
==================================================

Create machine-readable:

public/data/model_history.json

Fields:

version
date
parameters
checkpoint_size
training_images
validation_images
test_images
precision
main_problem
main_change
accuracy
auc
fpr
partial_ai_ap
dice
latency
memory
status

Only populate values backed by actual reports.

The website will use this to visualize:

PARAMETER GROWTH
DATASET GROWTH
ACCURACY CHANGES
FALSE POSITIVE CHANGES
LOCALIZATION IMPROVEMENT
MODEL COMPRESSION

==================================================
FINAL MODEL DECISION
==================================================

Do NOT automatically assume the latest checkpoint is best.

Compare actual candidates.

At minimum inspect:

large teacher / master systems
master intelligent fusion
standalone distilled student models
older reliable production champion

Evaluate practical trade-offs.

The final deployment may use:

A. FULL ANALYSIS MODEL
for highest-quality server-side analysis

and

B. LOCAL BROWSER MODEL
for private browser-only analysis

These do not have to be identical.

Be honest if they differ.

==================================================
FINAL MODEL GOAL
==================================================

The ideal production model is a true standalone trained model.

One checkpoint.
One model class.
One inference pipeline.
No teacher networks required at runtime.

If the existing standalone distilled model is clearly too weak compared with the teacher, do not falsely declare it final.

Determine whether a stronger distilled student is needed.

A reasonable target may be a larger student with enough capacity for:

visual features
spectral features
multi-scale features
partial-AI localization
classification

Use the existing teacher knowledge if training is required.

Do not retrain blindly.

First benchmark.

==================================================
MODEL DISTILLATION / MERGING IMPLEMENTATION GUIDE
==================================================

If a stronger standalone model is required:

Use the existing expert system only as TEACHERS.

Teachers may provide:

class logits
soft probabilities
global features
frequency evidence
specialist outputs
gating outputs
patch probabilities
spatial embeddings
localization masks

The student must contain its own:

visual backbone
spectral branch
multi-scale feature fusion
classification head
localization head

Training loss should combine appropriate forms of:

ground-truth classification
teacher logit distillation
feature distillation
patch/localization supervision
ground-truth mask supervision

Do not simply package all teachers into one .pt file.

Do not call a container an amalgamated model.

Do not call a weighted average a distilled model.

A true distilled model must run after teacher checkpoints are unavailable.

Test this in an isolated process.

==================================================
DATASET GOVERNANCE
==================================================

Create:

docs/DATASETS.md
public/data/datasets.json

Inventory EVERY dataset actually used.

Include:

dataset
source
official link
license
category
real count
AIGC count
partial-AI count
mask availability
resolution range
training / validation / testing role
transformations
reason for use

Known external challenge resources include:

SID_Set
https://huggingface.co/datasets/saberzl/SID_Set

CIFAKE
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

WildFake
https://modelscope.cn/datasets/hy2628982280/WildFake/summary

Also inventory all other data actually used in this project.

Do not say a dataset was used if it was merely downloaded or considered.

==================================================
DOWNLOAD / OPEN FORENSICS PAGE
==================================================

Create a dedicated page:

DOWNLOAD & RUN LOCALLY

It should include:

MODEL DOWNLOADS

FP32
FP16
INT8
browser-compatible model if available

For every model:

file size
parameters
recommended hardware
SHA-256
accuracy caveat
download link
license

Also provide:

Python inference script
CLI
requirements
sample config
API example

Example:

python infer.py --input-dir ./images --output predictions.json

Required output:

[
  {
    "image_path": "images/example.jpg",
    "pred": 0.8312
  }
]

Also show DATASETS.

Do not upload enormous datasets to GitHub.

Link to the official Hugging Face, Kaggle, ModelScope, or other provider.

If redistribution is not permitted, provide only the official source.

==================================================
BROWSER-ONLY PRIVATE MODE
==================================================

Implement an optional mode where analysis happens entirely in the user's browser.

This must be a separate selectable mode.

Possible implementation:

ONNX
ONNX Runtime Web
WebGPU
WASM fallback

Use the compact standalone student model suitable for browser execution.

The user should see:

PRIVATE BROWSER MODE

Image remains on this device.

No image is uploaded.

If browser hardware or WebGPU is unsupported, explain clearly.

Allow the user to:

START LOCAL MODEL
STOP LOCAL MODEL
UNLOAD MODEL FROM MEMORY

Opening a separate tab should allow an independent local session.

Do not imply browser mode has identical performance to full analysis unless testing proves it.

==================================================
ANALYSIS MODES
==================================================

Provide:

FULL ANALYSIS

and

PRIVATE BROWSER MODE

Possibly also:

DEMO MODE

No private infrastructure terminology.

==================================================
UPLOAD WARNING
==================================================

Before upload, show a short useful note:

WHAT TO EXPECT

AIGC Forensics estimates whether an image appears authentic, partly AI-edited, or fully synthetic.

Results are probabilistic, not proof.

Detection can be harder for:

very small images
heavily compressed images
screenshots
strongly filtered images
unseen generators
extreme crops
images with little texture

For best analysis:

use the highest-quality original available
avoid screenshots when the original exists
keep metadata when possible

Do not make this scary or verbose.

==================================================
UPLOAD PIPELINE
==================================================

The user must be able to upload:

single image
multiple images
up to 100 images
maximum 500 MB per session

Each image becomes a forensic evidence card.

On ingest generate:

evidence ID
SHA-256
filename
MIME
format
width
height
megapixels
bytes
color mode
ingest timestamp

Store files only temporarily for the active session.

Provide:

CLEAR SESSION

==================================================
EVIDENCE INTEGRITY
==================================================

Create a chain-of-custody style record.

For each file:

original hash
current hash
transformations performed
whether original bytes were modified
analysis model version
analysis time

The original uploaded image must remain untouched.

Derived transformed images must be stored separately.

==================================================
CORE FORENSIC PIPELINE
==================================================

For every image run all available real analysis stages.

Conceptually:

INGEST
↓
FILE INTEGRITY
↓
IMAGE PROPERTIES
↓
MODEL INFERENCE
↓
SPATIAL ANALYSIS
↓
PATCH ANALYSIS
↓
FREQUENCY ANALYSIS
↓
METADATA
↓
PROVENANCE
↓
ROBUSTNESS TESTS
↓
REPORT

Only perform stages actually implemented.

Never generate fake forensic values.

==================================================
VERDICT SYSTEM
==================================================

Support:

REAL

PARTIAL-AI

FULL-AIGC

REVIEW REQUIRED

Do not force every image into an overconfident class.

Show probabilities if the model supports them:

Real
Partial-AI
Full-AIGC

==================================================
WHY THIS VERDICT
==================================================

Create a plain-English explanation.

Example:

The image is mostly consistent with a photograph, but two local regions differ strongly from the surrounding texture. No supporting AI provenance information was found.

Do not generate meaningless jargon.

Do not claim causality that the model cannot prove.

==================================================
FORENSIC VIEWER
==================================================

This must be one of the strongest parts of the website.

LEFT:

large interactive image

RIGHT:

forensic evidence

Viewer modes:

ORIGINAL

HEATMAP

PATCH GRID

SUSPICIOUS REGIONS

FREQUENCY

METADATA

PROVENANCE

COMPARE

==================================================
HEATMAP
==================================================

Use REAL spatial model outputs.

Provide:

opacity slider
original/overlay comparison
zoom
pan

Do not simply stretch a scalar probability into a heatmap.

If localization is unavailable:

SPATIAL LOCALIZATION NOT AVAILABLE

==================================================
PATCH GRID
==================================================

Show actual evaluated patches.

When a patch is selected:

patch crop
coordinates
scale
AI probability
neighboring scores
region overlap
model signal

Support scales actually used by the model.

==================================================
SUSPICIOUS REGIONS
==================================================

Display:

bounding box
region ID
score
affected area
source scale

Clicking a region should focus the image.

==================================================
AFFECTED AREA
==================================================

For PARTIAL-AI:

show estimated affected percentage.

Make clear it is an estimate.

==================================================
FREQUENCY FORENSICS
==================================================

If the pipeline genuinely computes:

FFT
SRM residual
wavelet
high-pass response
noise residual

allow the user to inspect them.

Do not create decorative fake spectrum graphics.

Explain simply:

Frequency analysis looks for patterns in fine image texture and noise that may differ between camera photographs and generated images.

==================================================
METADATA
==================================================

Extract real:

EXIF
XMP
IPTC

Display:

camera
camera model
lens
ISO
shutter
aperture
focal length
software
creation date
modify date
GPS availability
color space
orientation

Also report missing fields.

==================================================
METADATA INTERPRETATION
==================================================

Do not merely dump fields.

Add simple interpretation.

Example:

Camera information is present.

Adobe Lightroom appears in the software metadata.

This indicates editing software was used. It does not by itself indicate AI generation.

==================================================
PROVENANCE
==================================================

Check available support for:

C2PA
Content Credentials
manifest
issuer
signature
editing actions
generator tags
software identifiers
known AI metadata
visible watermark indicators
supported synthetic watermark schemes

Statuses:

VERIFIED
PRESENT
NOT DETECTED
INVALID
UNKNOWN
NOT SUPPORTED

Never say:

No metadata = AI

or:

No watermark = Real

==================================================
FORENSIC TIMELINE
==================================================

Use actual backend timings.

Example:

IMAGE RECEIVED
HASH CALCULATED
MODEL ANALYSIS
SPATIAL ANALYSIS
METADATA PARSED
PROVENANCE CHECKED
REPORT GENERATED

Show actual elapsed times.

No fabricated timestamps.

==================================================
TRANSFORMATION LAB
==================================================

This is a major feature.

Allow the user to create an evaluation-only transformed copy of any uploaded image.

Available transformations:

JPEG:
90 / 70 / 50 / 30

BLUR:
0.5 / 1.0 / 2.0

RESIZE:
0.5x
0.25x
then upscale

NOISE:
0.02
0.05
0.10

COLOR:
brightness
contrast
saturation
±20%

CROP:
80% center crop

Never modify the original evidence file.

==================================================
ROBUSTNESS COMPARISON
==================================================

After transformations, run the same model again.

Show:

ORIGINAL PREDICTION

TRANSFORMED PREDICTION

DELTA

Example:

Original       84%
JPEG Q70       82%
JPEG Q30       71%

Calculate:

ROBUSTNESS STABILITY

based on actual prediction changes.

==================================================
BATCH ROBUSTNESS
==================================================

Allow a robustness suite to run across a batch.

Create:

clean vs transformed comparison table.

Metrics:

accuracy
ROC-AUC
AP
FPR
TPR
F1

where ground-truth labels are available.

==================================================
RESULTS PAGE
==================================================

Create:

RESULTS & BENCHMARKS

Show only actual measured results.

Include:

clean performance
transformed performance
real FPR
AIGC recall
partial-AI AP
localization Dice
IoU
calibration
latency
memory

If a metric was never measured:

do not invent it.

==================================================
STRENGTHS PAGE
==================================================

Explain where the model performs well.

Examples must be supported by evaluation.

Potential categories:

camera photographs
portraits
landscapes
certain diffusion generators
compressed images
localized edits
high-resolution images

Use actual results to determine these.

==================================================
WHERE IT FAILS PAGE
==================================================

This page is mandatory-quality.

Explain representative:

false positives
false negatives
localization failures
high-resolution failures
compression failures
unseen generator failures
soft/low-detail synthetic failures

For each example show:

image
true label
prediction
confidence
heatmap
what likely confused the model

Explain limitations in plain English.

Example:

The model saw few examples with this combination of heavy denoising and smooth skin during training, so the texture looked closer to generated imagery than to the real photographs it learned from.

Do not hide weaknesses.

==================================================
ERROR ANALYSIS
==================================================

Create:

docs/ERROR_ANALYSIS.md

and corresponding website page.

Include representative FP/FN examples and trade-offs.

This is required by the challenge.

==================================================
MODEL PAGE
==================================================

Create an understandable model explanation.

Sections:

WHAT THE MODEL SEES

HOW IT LEARNS

GLOBAL FEATURES

FINE TEXTURE

FREQUENCY SIGNALS

MULTI-SCALE ANALYSIS

PARTIAL-AI LOCALIZATION

FUSION

DISTILLATION

CONFIDENCE

==================================================
SCIENCE / ALGORITHM PAGE
==================================================

Explain the actual mathematics without making it unreadable.

Include equations only where useful.

Explain concepts first.

Potential sections:

softmax probability

cross entropy

knowledge distillation

KL divergence

spectral filtering

SRM kernels

Fourier transforms

wavelets

feature pyramids

multi-scale patches

segmentation

Dice loss

IoU

calibration

temperature scaling

quantization

FP32
FP16
INT8

For every equation:

include a simple explanation immediately below it.

Avoid pointless jargon.

==================================================
PRECISION PAGE
==================================================

Compare actual production formats.

For each:

FP32
FP16
INT8
FP8 only if genuinely supported

Show:

checkpoint size
parameter count
RAM
VRAM
latency
accuracy delta
heatmap stability

Do not display private hardware names.

Use generic:

Reference GPU
Desktop CPU
Browser/WebGPU

if public hardware reporting is necessary.

==================================================
RESOURCE USAGE
==================================================

Show model resource requirements without exposing infrastructure.

Example:

FP32
Memory: ...
Latency: ...

FP16
Memory: ...
Latency: ...

INT8
Memory: ...
Latency: ...

Use measurements from actual benchmarks.

==================================================
MODEL PARAMETER HISTORY
==================================================

Create a visual history of:

parameter counts over time
checkpoint sizes
datasets used
training images
validation images
accuracy
FPR
localization performance

Explain why models grew and later shrank through distillation.

==================================================
MODEL INTELLIGENCE STORY
==================================================

Explain the project evolution simply.

Example:

At first the detector learned whole-image differences.

Then it struggled with edited real photographs.

We added hard-real examples.

Then it struggled with high-resolution images.

We added multi-scale patches.

Then localized AI edits were being diluted by authentic background pixels.

We introduced spatial supervision.

Later multiple specialist models were combined.

Finally their knowledge was distilled into a standalone model.

Write this using actual project history.

==================================================
FORENSIC REPLAY
==================================================

Allow the user to replay the analysis stages:

INGEST

MODEL

PATCHES

HEATMAP

METADATA

PROVENANCE

ROBUSTNESS

VERDICT

Selecting a stage changes the viewer to the relevant evidence.

==================================================
REPORT EXPORT
==================================================

Support:

JSON

PDF

Printable report

Report should include:

Evidence ID
SHA-256
filename
image preview
model/version
verdict
probabilities
confidence
heatmap
regions
affected area
metadata
provenance
robustness results
timeline
limitations

==================================================
CASE COMPARISON
==================================================

Allow two analyzed images to be compared side-by-side.

Useful for:

original vs edited

original vs transformed

real vs synthetic

before vs after model upgrade

==================================================
MODEL VERSION COMPARISON
==================================================

If older analysis exists:

show:

Previous model
Current model
Prediction change

This must use actual stored session results.

==================================================
SESSION PRIVACY
==================================================

Make it clear:

Uploads are temporary.

Provide:

CLEAR SESSION NOW

Delete temporary files after expiry.

Do not permanently retain user evidence unless explicitly requested.

==================================================
MAIN WEBSITE DESIGN
==================================================

PRESERVE THE CURRENT DESIGN.

The current:

green felt
black
warm ivory
gold/brass
physical evidence cards
forensic table
luxury visual language

is already good.

Do not redesign it.

Apply only targeted refinements.

==================================================
HEADER
==================================================

No infrastructure names.

Make the existing header:

sticky
smoked glass
frosted
premium

Use backdrop blur so content behind it visibly diffuses.

Do not make it a giant SaaS pill.

==================================================
TABLE
==================================================

Complete the existing curved green table.

It currently ends abruptly in some layouts.

Extend the felt surface naturally.

Preserve:

curvature
wood rim
existing texture
existing composition

Add subtle contact/shadow below the front lip.

Do not redesign it.

==================================================
CARDS
==================================================

Preserve the current card design.

Improve selection.

Cards should fan enough to choose individual images.

On hover/touch:

raise selected card
spread nearby cards slightly
increase z-index
show clear focus

For many images:

use controlled cycling/fanning rather than spreading 100 cards.

==================================================
SHUFFLE
==================================================

Make the existing shuffle more visible.

Sequence:

LIFT
SPLIT
INTERLEAVE
TOSS
REFORM
DEAL

Keep it believable, not chaotic.

==================================================
DEALING
==================================================

Cards MUST visibly travel from the central deck into:

REAL

PARTIAL-AI

FULL-AIGC

Each card:

lifts
travels in an arc
rotates slightly
lands
settles

Do not simply fade in the result decks.

==================================================
RESULT DECKS
==================================================

Allow users to inspect individual cards.

Hover:

fan top cards

Click:

pick actual card up

Continue into forensic inspection.

==================================================
TYPOGRAPHY
==================================================

Keep the large headings mostly as they are.

Replace overly condensed AI/HUD-looking small text.

Use:

normal-width serious grotesk
SF Pro Text / Inter / Helvetica-style stack

Use mono only for real technical data:

hashes
IDs
coordinates
timestamps

Reduce unnecessary letter spacing.

Avoid fake military HUD typography.

==================================================
FOOTER
==================================================

Refine existing footer only.

No infrastructure names.

Desktop:

LEFT
brand

CENTER
copyright / links

RIGHT
model version / open source

Mobile:

clean stack

No tiny unreadable text.

==================================================
MOBILE AND TABLET
==================================================

Test:

1440+
1024
768
430
390
320

Ensure:

upload works
card selection works
shuffle works
deal works
inspection works
heatmap works
patches work
Transformation Lab works
downloads work
browser mode works

Do not simply shrink desktop.

==================================================
ABOUT PAGE
==================================================

Explain:

problem
why AI detection matters
why robustness matters
why local edits matter
how the project evolved
what this system can and cannot prove

Easy language.

==================================================
HISTORY PAGE
==================================================

Use actual project records.

Timeline:

models
datasets
parameters
performance
failures
improvements

Interactive where practical.

==================================================
DATASETS PAGE
==================================================

Visual dataset registry.

Show:

dataset
purpose
size
class distribution
source
license
train/val/test role

Links to official source.

Mark benchmark isolation clearly.

==================================================
RESULTS & LIMITATIONS PAGE
==================================================

This should directly satisfy hackathon requirements.

Include:

Robustness Evaluation Summary

Error Analysis Note

Strengths

Weaknesses

False positives

False negatives

Trade-offs

Future improvements

==================================================
GITHUB FINALIZATION
==================================================

Prepare a clean public repository.

Required structure should be something like:

README.md
LICENSE
MODEL_CARD.md
DATASETS.md
ARCHITECTURE.md
RESULTS.md
LIMITATIONS.md
CONTRIBUTING.md if useful
requirements.txt or pyproject.toml
.env.example
infer.py
app/
frontend/
server/
scripts/
configs/
examples/
docs/
reports/public/
tests/

Do not expose internal private files.

==================================================
README
==================================================

README must include:

Project overview

Why it exists

Demo screenshot

Architecture

Model

Setup

Install

Local inference

Browser inference

API usage

Dataset sources

Evaluation

Robustness

Limitations

Reproduction

Credits

Hackathon information

==================================================
MODEL CARD
==================================================

Create:

MODEL_CARD.md

Include:

model name
architecture
parameters
checkpoint size
precisions
training data
validation data
test data
intended use
out-of-scope use
known limitations
performance
robustness
ethical considerations
license

==================================================
OFFICIAL INFERENCE CLI
==================================================

Required:

python infer.py --input-dir ./images --output predictions.json

Output exactly:

[
  {
    "image_path": "...",
    "pred": 0.83
  }
]

pred must mean AIGC probability.

Test it.

==================================================
PUBLIC CODE QUALITY
==================================================

Add:

typing
docstrings
clear config
error handling
tests
no dead scripts in main public path

Research/archive scripts may go under:

research/
archive/

Do not delete historical work unnecessarily.

==================================================
SECURITY
==================================================

Backend:

strict MIME checking
filename sanitization
image decoder safety
image decompression bomb protection
pixel limit
100 image limit
500 MB session limit
timeouts
rate limiting
CORS
temporary directories
session isolation
automatic deletion

Never expose shell execution.

Never expose arbitrary filesystem paths.

==================================================
PUBLIC INFRASTRUCTURE LANGUAGE
==================================================

Again:

NO:

Buildabot
RTX
CUDA device model
server hostname
IP
SSH
filesystem path

in the actual website.

Internal technical development may use them.

==================================================
API
==================================================

Stable public API:

GET /health
GET /model
POST /analyze
POST /analyze/batch
GET /session/{id}
DELETE /session/{id}

Potential:

POST /transform
POST /robustness

Return one stable schema.

==================================================
ANALYSIS RESPONSE
==================================================

Use a structure similar to:

{
  "evidence_id": "AF-0001",
  "verdict": "PARTIAL_AI",
  "probabilities": {
    "real": 0.12,
    "partial": 0.74,
    "full": 0.14
  },
  "confidence": 0.74,
  "affected_area_percentage": 8.4,
  "regions": [],
  "patches": [],
  "heatmap_url": "...",
  "metadata": {},
  "provenance": {},
  "timeline": [],
  "model": {
    "name": "...",
    "version": "...",
    "precision": "FP16"
  }
}

Use actual backend output fields where already implemented.

Do not duplicate incompatible schemas.

==================================================
COPY STYLE
==================================================

All public writing should be:

clear
natural
direct
easy to understand

Avoid excessive jargon.

Avoid em dashes.

Avoid unnecessary punctuation.

Avoid AI-marketing language.

Avoid:

revolutionary
cutting-edge
next-generation
unprecedented
groundbreaking

unless objectively justified.

Explain things as if talking to an intelligent person who is not an ML researcher.

==================================================
MODEL SCIENCE WRITING
==================================================

For technical pages:

First explain in simple language.

Then optionally show the technical equation.

Then explain the equation.

Do not dump mathematics without context.

==================================================
NO FAKE DATA RULE
==================================================

Never fabricate:

accuracy
AUC
FPR
parameters
latency
memory
training counts
dataset size
heatmaps
timestamps
metadata
provenance
watermarks
model evidence

If unknown:

UNKNOWN
NOT MEASURED
NOT AVAILABLE

==================================================
DEMO MODE
==================================================

Demo mode may contain predetermined examples.

Clearly label:

DEMO DATA

Never present canned examples as live inference.

==================================================
FINAL VERIFICATION
==================================================

Before declaring completion:

1. upload one image
2. upload a batch
3. analyze
4. view deck classification
5. pick individual card
6. inspect original
7. inspect heatmap
8. inspect patches
9. inspect suspicious regions
10. inspect frequency evidence
11. inspect metadata
12. inspect provenance
13. run transformation
14. compare transformed prediction
15. export JSON
16. export PDF
17. run browser inference
18. unload browser model
19. download model
20. run CLI inference
21. verify required JSON
22. verify mobile
23. verify tablet
24. verify desktop
25. verify benchmark isolation
26. verify no internal infrastructure appears publicly
27. verify repository has no secrets

==================================================
HACKATHON SUBMISSION OUTPUTS
==================================================

Prepare:

DEVPOST_DESCRIPTION.md

README.md

MODEL_CARD.md

DATASETS.md

RESULTS.md

LIMITATIONS.md

ERROR_ANALYSIS.md

ROBUSTNESS.md

ARCHITECTURE.md

DEMO_VIDEO_SCRIPT.md

SUBMISSION_CHECKLIST.md

==================================================
DEMO VIDEO SCRIPT
==================================================

Create a concise script showing:

problem

upload

analysis

card shuffle

three categories

open one case

heatmap

patch analysis

metadata/provenance

Transformation Lab

robustness results

model evolution

local browser mode

limitations

closing impact

==================================================
FINAL PRODUCT PHILOSOPHY
==================================================

This is not:

"an AI detector that outputs a percentage"

It is:

an image forensic investigation environment.

It should answer:

WHAT DOES THE MODEL THINK?

WHERE IS THE EVIDENCE?

WHY DID IT THINK THAT?

WHAT DOES THE FILE ITSELF SAY?

DOES THE RESULT SURVIVE COMPRESSION OR EDITING?

HOW CERTAIN IS THE SYSTEM?

WHERE DOES THE SYSTEM FAIL?

CAN I REPRODUCE THE RESULT?

CAN I RUN IT PRIVATELY?

==================================================
PRIORITY ORDER
==================================================

Do work in this order:

1. audit project and models
2. freeze current Git checkpoint
3. establish benchmark isolation
4. choose/finalize production model
5. stabilize API
6. make upload to forensic result fully functional
7. finish heatmap/patch/metadata/provenance pipeline
8. implement Transformation Lab
9. finish card shuffle/dealing interactions
10. finish browser private mode
11. build Results/Strengths/Weaknesses pages
12. build Model History and Dataset pages
13. build Download & Run Locally
14. finish responsive layouts
15. finalize GitHub repo
16. run complete QA
17. generate submission docs

Do not stop after producing plans.

Implement, test, verify, and commit the working result.