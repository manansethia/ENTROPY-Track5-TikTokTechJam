# MASTER PROMPT — 20-HOUR AUTONOMOUS AIGC DETECTOR REMEDIATION & PARALLEL COMPUTE SPRINT

## Mission

We have approximately 20 hours of wall-clock time remaining to produce the best scientifically validated AIGC detector we can, then freeze it and prepare deployment.

The current production detector has severe real-world failures:

- high-resolution REAL → AIGC
- professional/high-grade/color-corrected REAL → AIGC
- portraits/selfies/headshots → AIGC
- mildly cropped/edited REAL → AIGC
- brightness/contrast edited REAL → AIGC
- aggressively shrinking both REAL and AIGC often makes both look REAL

Therefore:

**Do not solve the problem by simply shrinking every image.**

The goal is to remove these shortcuts while preserving genuine AIGC detection, especially on unseen generators and realistic post-processing.

---

# 1. IMMUTABLE PRODUCTION CONTROL

Never modify, overwrite, or delete:

`/home/manan/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt`

Expected SHA-256:

`91a6a3814c30f0b62f7b63e63fa81fe595c83b9edf91733ef9a8d3358e133438`

The frozen production model remains the control for the entire experiment.

Every new candidate must start from a COPY.

Never tune specifically to one user's image.

---

# 2. AUTHORITATIVE MACHINE

Buildbot remains authoritative for:

- master manifests
- dataset governance
- final evaluation
- final model selection
- final calibration
- final checkpoint
- final deployment package
- provenance

Kaggle workers are independent experiment/benchmark workers.

Mac M1 is auxiliary only.

Do not create distributed gradient synchronization between Buildbot, Kaggle, TPU, or Mac.

---

# 3. CENTRAL ACCELERATOR BUDGET

Maintain:

`reports/accelerator_budget.json`

Hard ceilings:

- Kaggle GPU total: <= 30.0 GPU-hours
- Kaggle TPU total: <= 20.0 TPU-hours

Treat GPU and TPU as separate budgets.

For every worker track:

- worker_id
- Kaggle account
- kernel_id
- accelerator type/model
- start time
- stop time
- runtime
- GPU-hours used
- TPU-hours used
- task
- result
- checkpoint hash if produced

Before launching a worker, calculate projected usage.

Never launch a job that would push the relevant budget over its limit.

Optimize for useful work per hour, not maximum number of running workers.

---

# 4. CURRENT BUILDBOT STATUS

First inspect current processes and GPU usage.

If the current Buildbot RTX 3050 is running a useful diagnostic/training task, do not start a competing GPU process that causes VRAM contention.

If the current task is only waiting on CPU/network/disk, use the GPU for the highest-value available task.

Do not allow useful hardware to sit idle unnecessarily.

---

# 5. STORAGE FIRST, BUT DO NOT BLOCK FOREVER

The project has substantial free storage after cleanup.

Before each major dataset ingestion:

- check free disk
- check inode usage
- verify destination capacity

Delete only:

- verified temporary files
- incomplete downloads
- exact duplicates
- recreatable caches
- obsolete staging data

Never blindly delete:

- frozen model
- locked evaluation data
- required manifests
- provenance
- reproducibility-critical source material

Maintain safe free space.

If disk usage becomes dangerous, pause optional downloads first.

---

# 6. CURRENT DATA DOWNLOAD STRATEGY

Continue downloading datasets currently in flight.

Do not require every optional evaluation dataset to finish before training.

As soon as there is a sufficient, validated TRAIN-ELIGIBLE pool:

**start remediation training immediately.**

Optional evaluation-only datasets can continue downloading in parallel if they do not interfere with training.

---

# 7. DATASET ROLE CLASSIFICATION

Every dataset MUST be classified before use:

- `TRAIN-ELIGIBLE`
- `EVALUATION-ONLY`

Never put evaluation-only datasets into the training manifest.

In particular, if current dataset cards still designate them as evaluation-only:

- HiRes-50K → evaluation only
- AIGC Detection Benchmark → evaluation/testing only

Do not train on them.

For every dataset record:

- source
- dataset name
- version
- license
- URL
- training/evaluation role
- image count
- REAL/AIGC counts
- resolution distribution
- generator distribution

---

# 8. HIGH-RESOLUTION REAL DATA PRIORITY

Build a large, diverse authentic pool covering:

- 2K
- 4K
- 8K+
- high-megapixel images
- smartphone photography
- DSLR
- mirrorless
- professional photography
- portraits
- selfies
- headshots
- studio
- outdoor
- HDR
- bokeh
- high ISO
- color correction
- color grading
- retouching
- sharpening
- denoising
- JPEG/web compression
- multiple aspect ratios

Do not let one source dominate.

Avoid creating a new shortcut such as:

`dataset identity = REAL`

or:

`square face crop = REAL`.

---

# 9. HIGH-RESOLUTION AIGC DATA

Build or use TRAIN-ELIGIBLE synthetic data covering:

- high-resolution AIGC
- photorealistic AIGC
- portraits
- selfies/people
- professional-looking generations
- multiple generator families
- multiple aspect ratios
- realistic compression/processing

Match REAL and AIGC distributions across:

- resolution
- aspect ratio
- JPEG quality
- resize history where known
- visual quality
- processing characteristics

The model must not be able to solve the task through metadata/geometry shortcuts.

---

# 10. DATA CONTAMINATION & DEDUPLICATION

Before a sample enters training:

- SHA-256 exact duplicate check
- perceptual hash / near-duplicate check where appropriate
- contamination check against locked benchmarks
- validate image decoding
- validate image dimensions

Check against all locked/internal evaluation pools relevant to this project.

Maintain a manifest with:

- path
- label
- source
- license/provenance
- width
- height
- megapixels
- aspect ratio
- format
- SHA-256
- perceptual hash
- transformation
- split

---

# 11. PORTRAIT/EDITED REAL HARD NEGATIVES

Create a dedicated real hard-negative pool prioritizing:

- high-res REAL falsely classified as AIGC
- portraits
- selfies
- headshots
- studio images
- high-grade camera output
- color-corrected images
- HDR
- bokeh
- mild retouching
- mild sharpening
- mild denoising
- cropped images
- mild brightness changes
- mild contrast changes
- JPEG/web-compressed images

Keep the original and transformed variants linked with a common `source_image_id`.

Do not train on the user's personal test image.

---

# 12. REALISTIC TRANSFORMATIONS

Use mild transformations that a real user may perform:

- crop 5–15%
- brightness ±5–10%
- contrast ±5–10%
- mild color adjustment
- resize
- JPEG Q90–95
- mild sharpening
- mild denoising

Do not immediately use destructive blur/extreme compression/extreme downsampling.

Those can destroy the very forensic signals we need.

The intended behavior is:

`mild ordinary edit → same underlying class`

while preserving:

`genuine synthetic evidence → still detectable`

---

# 13. MULTI-CROP / HIGH-RES INPUT STRATEGY

The earlier diagnostic tested:

1. full image → 224
2. native-resolution local crops → 224
3. global 224 + native local crops

If current/available evidence supports the multi-crop approach, use:

`global context + native high-resolution local crops → shared encoder → lightweight aggregation`

Do not duplicate the entire backbone per crop.

Do not make inference depend on users manually shrinking images.

---

# 14. MAIN REMEDIATION CANDIDATE

Create:

`PORTRAIT-REM-1`

Start from a copy of the frozen production model.

Initial remediation should prioritize:

- corrected real data distribution
- hard-negative weighting
- geometry matching
- high-resolution coverage
- realistic mild post-processing exposure
- multi-crop only where empirically justified

Do not simultaneously introduce:

- another VLM feedback loop
- multiple new loss functions
- another architecture search
- many new augmentation systems

unless a current experiment exposes a genuine correctness problem.

---

# 15. EPOCH-BY-EPOCH TRAINING

Train in explicit epochs.

After every epoch:

1. save checkpoint
2. evaluate immediately
3. record metrics
4. decide whether to continue

Do not choose based on training loss alone.

Use early stopping when a candidate clearly degrades.

---

# 16. REQUIRED EPOCH METRICS

For every epoch:

### Competition metrics
- Clean ROC-AUC
- Robust ROC-AUC
- `0.50 * Clean AUC + 0.50 * Robust AUC`

### Standard metrics
- accuracy
- FP
- FN
- AUPRC

### Low-FPR diagnostics
- TPR @ 0.10% FPR
- TPR @ 0.01% FPR

### Real-world robustness
- high-res REAL FPR
- portrait FPR
- selfie FPR
- headshot FPR
- edited-real FPR
- crop-real FPR
- brightness-real FPR
- contrast-real FPR

### Generalization
- SID/LDM
- unseen-generator AUC
- pseudo-OOD macro AUC
- hard-edge accuracy

---

# 17. PRIMARY MODEL-SELECTION RULE

The competition score is primary:

`FINAL_SCORE = 0.50 * AUC_clean + 0.50 * AUC_robust`

But do not accept a candidate that merely wins on aggregate AUC while producing an unacceptable real-world failure.

Use secondary safeguards:

- high-res REAL FPR
- portrait FPR
- selfie FPR
- unseen-generator AUC
- SID/LDM
- low-FPR TPR
- hard-edge performance

Compare candidates against the immutable frozen production control.

---

# 18. STATISTICAL COMPARISON

For key real-photo FPR improvements calculate:

- absolute ΔFPR
- relative ΔFPR
- bootstrap 95% confidence interval

For paired transformed-image tests calculate:

- probability shift
- median absolute probability shift
- P95 absolute probability shift
- classification flip rate

Do not claim statistical significance without computing it.

---

# 19. KAGGLE HEADLESS WORKFLOW

Use Kaggle as a remote batch worker through the API/CLI.

Do not require interactive browser work after setup.

Use:

- `kaggle kernels push`
- `kaggle kernels status`
- `kaggle kernels logs`
- `kaggle kernels output`

Each kernel is an independent job.

Buildbot submits jobs and collects outputs.

---

# 20. KAGGLE GPU WORKER VERIFICATION

Every GPU worker MUST first run:

- `torch.cuda.is_available()`
- `torch.cuda.device_count()`
- actual GPU name
- actual GPU VRAM
- a real CUDA tensor operation

If CUDA is unavailable:

**terminate the worker immediately.**

Do not waste accelerator quota on CPU-only Kaggle sessions.

Do not trust the requested accelerator name; verify the actual assigned hardware.

Prefer a GPU with sufficient VRAM for large-model work.

For P100, verify that the Kaggle PyTorch/CUDA environment actually supports the GPU before using meaningful quota.

---

# 21. KAGGLE DATA/ARTIFACT STORAGE

Use Kaggle persistent Dataset storage for large reusable datasets rather than `/kaggle/working`.

Where currently available, verify the actual account limits before assuming capacity.

Use meaningful persistent datasets such as:

- `aigc-highres-real-v1`
- `aigc-highres-synthetic-v1`
- `aigc-robustness-transforms-v1`
- `aigc-diagnostic-benchmark-v1`

Do not create unnecessary duplicate datasets.

For large data, shard by meaningful category.

Keep manifests, provenance, and licenses with the data.

---

# 22. KAGGLE MODEL STORAGE

For large experimental checkpoints, use Kaggle model artifacts where appropriate.

Do not repeatedly transfer large optimizer-state checkpoints back to Buildbot.

Return only:

- best candidate
- final candidate
- required provenance
- metrics/logs

Buildbot remains authoritative.

---

# 23. KAGGLE MULTI-ACCOUNT USE

If the user has additional legitimate Kaggle accounts, they may be used as independent workers only within Kaggle's rules.

Do not:

- bypass quotas
- circumvent restrictions
- create fake accounts
- misrepresent identity
- evade enforcement

The project budget ledger must account for all worker runtime.

Use different accounts for different independent jobs when permitted.

Do not duplicate identical experiments without a scientific reason.

---

# 24. GPU WORKER ASSIGNMENTS

Use independent workers such as:

### GPU-1
Main high-resolution remediation candidate.

### GPU-2
Independent remediation candidate.

### GPU-3
SPAI / any-resolution detector benchmark.

### GPU-4
CommunityForensics ViT-Small benchmark.

### GPU-5
High-resolution robustness benchmark.

### GPU-6
Unseen-generator benchmark.

### GPU-7
Multi-crop/native-resolution ablation if needed.

Stop low-value workers early and reassign their unused budget.

---

# 25. TPU USE

TPU quota is separate.

Do not port the main 735M CUDA-oriented detector to TPU unless the framework is already compatible.

Use TPU only for a low-engineering-cost independent experiment such as:

- compatible lightweight model training
- distillation
- a clearly TPU-compatible alternative

Before spending real TPU quota:

1. initialize TPU
2. run one tensor operation
3. run one training step
4. measure throughput

If setup/debugging exceeds 15 minutes:

**stop the TPU job.**

Never let TPU migration consume the project deadline.

---

# 26. MAC M1 USE

The Mac M1 can perform lightweight auxiliary work:

- image hashing
- metadata extraction
- manifest generation
- lightweight preprocessing
- download/decompression

Do not use large amounts of Mac RAM.

Do not build distributed training across Mac + Buildbot.

Do not move large training tensors over the network.

---

# 27. PARALLEL DATA FLOW

The system should operate as:

`SOURCE DATA`
↓
`DOWNLOAD / STREAM`
↓
`PERSISTENT STORAGE`
↓
`VALIDATE`
↓
`DEDUPLICATE`
↓
`MASTER MANIFEST`
↓
`SPLIT TRAIN / DEV / EVAL`
↓
`KAGGLE WORKER DATASETS`
↓
`INDEPENDENT WORKER JOBS`
↓
`RESULTS / CHECKPOINTS`
↓
`BUILDBOT AGGREGATION`
↓
`FINAL COMPARISON`
↓
`FINAL MODEL`
↓
`FREEZE`

Do not repeatedly download the same dataset for every worker.

---

# 28. STREAMING & SHARDING

Where datasets are too large to materialize repeatedly:

- stream or shard them
- use bounded prefetch
- use persistent caches
- process each shard once where possible
- produce reusable processed artifacts

Never load an entire huge dataset into RAM unnecessarily.

Use streaming for ingestion/benchmarking when it reduces disk pressure or setup time.

Use local cached shards for repeated training passes where that is faster.

Choose based on measured throughput.

---

# 29. RESOURCE UTILIZATION

Use available hardware aggressively but safely.

### Buildbot GPU
Use the largest stable batch size.

### Buildbot CPU
Use available cores for:

- image decoding
- preprocessing
- hashing
- manifest work

### Buildbot RAM
Use bounded staging/cache.

### Kaggle GPUs
Run independent useful workloads.

### TPU
Use only if technically cheap and productive.

### Mac
Use lightweight preprocessing only.

Never intentionally:

- cause CUDA OOM
- induce sustained swap thrashing
- fill disk
- starve the training GPU with unnecessary CPU work

Optimize for throughput, not utilization percentage.

---

# 30. FAILURE HANDLING

If a worker encounters:

- no GPU
- missing dependencies
- unavailable network
- incompatible CUDA
- unavailable dataset
- broken environment

and it cannot be repaired quickly:

1. record exact failure
2. terminate worker
3. release budget
4. reassign task if worthwhile
5. continue without blocking the main pipeline

Never spend more than 15 minutes fighting a worker environment.

---

# 31. COAGGREGATION OF RESULTS

Every worker writes:

- `metrics.json`
- `metrics.md`
- `config.json`
- `environment.json`
- `worker_manifest.json`
- checkpoint metadata
- model hash
- dataset hash

Store under:

`/home/manan/aigc_robust_detection/reports/parallel_workers/<worker_id>/`

Buildbot aggregates all worker results.

Never let workers overwrite each other's output.

---

# 32. TIME-BASED EXECUTION CONTROL

### T+0–2h
- finish/validate current downloads
- complete current diagnostics
- verify Kaggle workers
- build training manifest
- start training as soon as enough valid data exists

### T+2–10h
- main remediation training
- independent candidate
- robustness tests
- high-res tests
- multi-crop where useful

### T+10–15h
- compare top candidates
- evaluate unseen generators
- external high-res evaluation

### T+15–17h
- select final candidate
- final robustness evaluation
- calibration preparation

### T+17h
STOP all optional research.

Only:
- final evaluation
- calibration
- thresholding
- model freeze

### T+19h
Deployment/package only.

### T+20h
Final artifact must exist.

---

# 33. EARLY-STOP POLICY

Stop an experiment early if it is clearly failing on:

- clean AUC
- robust AUC
- real high-res FPR
- portrait FPR
- unseen-generator performance
- SID/LDM

Do not spend remaining accelerator budget on a clearly inferior candidate.

Reallocate remaining compute to the best candidate or highest-value validation.

---

# 34. FINAL MODEL SELECTION

Compare:

- frozen production
- PORTRAIT-REM-1 best epoch
- independent remediation candidate(s)
- multi-crop candidate if viable
- any strong external-model baseline if relevant

Primary:

`0.50 * Clean AUC + 0.50 * Robust AUC`

Secondary:

- real high-res FPR
- portrait FPR
- selfie FPR
- edited-real FPR
- unseen-generator AUC
- SID/LDM
- low-FPR TPR
- edge cases

Do not select by training loss.

---

# 35. FINAL CALIBRATION

After selecting the best candidate:

- fit temperature scaling on the designated CAL split
- recompute exact operational thresholds
- never use test labels for calibration
- record calibration temperature
- record threshold values
- record calibration dataset/version

---

# 36. FINAL FREEZE

Create final production checkpoint only after selection.

Verify:

- SHA-256
- parameter count
- trainable parameter count
- parameter hash
- fresh reload
- inference sanity

Then freeze.

No more training after freeze.

---

# 37. FINAL DEPLOYMENT VALIDATION

Verify:

- Buildbot CUDA inference
- Buildbot CPU inference
- Mac MPS inference if supported
- preprocessing parity
- output parity within reasonable numerical tolerance

Return:

- probability
- predicted class
- threshold used
- model version

Do not optimize TensorRT/CoreML/ONNX before the model is frozen.

After freeze, optimize runtime as a separate deployment task.

---

# 38. FINAL REPORTS

Produce:

- `reports/final_model_comparison.json`
- `reports/final_model_comparison.md`
- `reports/accelerator_budget.json`
- `reports/dataset_inventory.json`
- `reports/training_manifest_summary.json`
- `reports/robustness_evaluation.json`
- `reports/unseen_generator_evaluation.json`

The final report must clearly state:

- final model
- checkpoint SHA-256
- parameter count
- calibration temperature
- thresholds
- clean AUC
- robust AUC
- competition score
- high-res REAL FPR
- portrait FPR
- edited-real FPR
- unseen-generator metrics
- SID/LDM
- remaining accelerator budget

---

# 39. AUTONOMOUS OPERATION RULE

Do not wait for user approval between routine stages.

Automatically proceed through:

`download → validate → manifest → benchmark → train → epoch evaluation → candidate selection → final evaluation → calibration → freeze`

Only stop and request human input when:

- a genuine scientific ambiguity cannot be resolved from the protocol
- credentials are required
- legal/licensing status is unclear
- a destructive action would be required
- a major architecture change would be necessary
- the immutable production checkpoint would otherwise be modified

Otherwise continue autonomously.

---

# 40. FINAL PRIORITY

The priority is:

**correctness > generalization > competition score > speed > polish**

But within the 20-hour deadline, maximize useful throughput.

Do not endlessly research.

Do not endlessly tune.

Do not endlessly download.

Do not endlessly train.

Build the best validated candidate available, freeze it, and ship.

---

# 41. FIRST ACTION NOW

Immediately:

1. inspect current Buildbot processes
2. inspect current download state
3. inspect current storage
4. inspect current Kaggle workers
5. verify actual Kaggle GPU assignment
6. verify accelerator budget
7. classify downloaded datasets
8. finish enough TRAIN-ELIGIBLE data to build the remediation manifest
9. start PORTRAIT-REM-1 as soon as its minimum valid training pool exists
10. keep useful evaluation-only downloads in parallel
11. continuously aggregate results
12. stop weak jobs early
13. freeze the final champion before the 20-hour deadline

Do not wait for every optional dataset.

Do not let Kaggle setup become the bottleneck.

Do not let benchmark-only data leak into training.

Do not overwrite the frozen production control.

**Operate autonomously and continuously until the final model is frozen.**
