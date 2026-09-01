# Final Submission Checklist

## Code

- [ ] Public GitHub repository
- [ ] Clean install tested from a fresh environment
- [ ] Training script runs
- [ ] Inference script runs
- [ ] JSON contains `image_path` and `pred`
- [ ] Robustness evaluator runs
- [ ] No hard-coded local paths
- [ ] No secrets/API keys committed

## Data

- [ ] Exact training datasets documented
- [ ] Licenses checked
- [ ] COCO val2017 benchmark isolated
- [ ] WildFake DALL-E Advanced benchmark isolated
- [ ] No benchmark leakage

## Model

- [ ] Exact total parameter count recorded
- [ ] Under 2B parameters
- [ ] Checkpoint loads successfully
- [ ] Inference tested on CPU if feasible
- [ ] CUDA inference tested
- [ ] Seed recorded

## Evaluation

- [ ] Clean benchmark score
- [ ] JPEG 90/70/50/30
- [ ] Blur 0.5/1.0/2.0
- [ ] Down/up 0.5×/0.25×
- [ ] Noise 0.02/0.05/0.10
- [ ] Color jitter
- [ ] Center crop 80%
- [ ] Accuracy
- [ ] Balanced accuracy
- [ ] F1
- [ ] AUROC
- [ ] False positives documented
- [ ] False negatives documented

## Devpost

- [ ] Problem
- [ ] Solution
- [ ] Architecture
- [ ] Tools
- [ ] Models
- [ ] Libraries
- [ ] Datasets
- [ ] Results
- [ ] Limitations
- [ ] Team contributions
- [ ] GitHub link
- [ ] YouTube link

## Video

- [ ] End-to-end inference shown
- [ ] Robustness result shown
- [ ] Failure case shown
- [ ] Public YouTube visibility
- [ ] No unauthorized third-party copyrighted/trademarked material
