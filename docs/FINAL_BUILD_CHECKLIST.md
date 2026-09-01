# Final Build Checklist

## Server
- [ ] Fedora confirmed
- [ ] NVIDIA driver / nvidia-smi works
- [ ] PyTorch CUDA works
- [ ] GPU smoke test passes
- [ ] `/mnt/ai-storage` has sufficient free space
- [ ] Hugging Face cache points to HDD
- [ ] Git LFS installed

## Models
- [ ] Candidate checkpoints downloaded
- [ ] Each checkpoint loads
- [ ] Exact parameter count recorded
- [ ] `<2B` verified for final model
- [ ] Model licenses recorded

## Data
- [ ] Community Forensics / selected shards downloaded
- [ ] SID_Set downloaded
- [ ] GenImage selected/training subset available
- [ ] WildFake training data separated
- [ ] CIFAKE optional
- [ ] Challenge validation locked
- [ ] No validation images in training manifests

## Memory
- [ ] Expert extraction runs sequentially
- [ ] Outputs moved off GPU
- [ ] Python cleanup tested
- [ ] subprocess reset tested
- [ ] peak VRAM recorded
- [ ] no unexplained VRAM accumulation

## Modeling
- [ ] individual baselines benchmarked
- [ ] robustness benchmark run
- [ ] complementary errors analyzed
- [ ] teacher fusion tested
- [ ] distillation tested only if justified
- [ ] final model frozen

## Submission
- [ ] inference script emits image_path/pred JSON
- [ ] README reproducible
- [ ] robustness table included
- [ ] error analysis included
- [ ] Devpost text complete
- [ ] demo video public
- [ ] GitHub public
