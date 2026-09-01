# Implementation Notes

This repository is an implementation-ready refinement of the supplied project specification.

## Important refinements made

### 1. Raw forensic input is kept separate from normalized semantic inputs

The earlier specification passed a normalized SigLIP tensor into the residual branch. The repository instead keeps the RGB image in `[0,1]` for SRM/DWT while separately creating normalized tensors for CLIP and SigLIP.

This prevents semantic preprocessing constants from becoming part of the forensic signal.

### 2. Checkpoint format is self-describing

Training saves:

```python
{
    "state_dict": ...,
    "epoch": ...,
    "config": ...,
    "parameter_report": ...
}
```

Inference accepts both this format and a plain PyTorch state dictionary.

### 3. The parameter count is measured

The supplied specification estimates a fused model around 415M parameters. Because the exact count can depend on the instantiated model revisions, the implementation calculates the actual count at runtime and verifies it is below 2B.

### 4. Evaluation reports more than accuracy

The robustness script reports accuracy, balanced accuracy, F1 and AUROC. This makes class imbalance and probability ranking easier to inspect.

### 5. Benchmark isolation is structural

The training loader only reads `data/train/real` and `data/train/synthetic`. The validation benchmark is read only by the separate robustness evaluator.

### 6. Noise parameters use the brief's normalized sigma convention

The supplied challenge describes Gaussian noise using σ values 0.02, 0.05 and 0.10. The evaluation implementation treats those values as standard deviations in the normalized `[0,1]` pixel domain.

## Remaining experimental work

The repository is code-complete, but the following require the team's actual data and compute:

- exact training dataset composition
- model checkpoint training
- parameter-count capture from the final environment
- robustness results
- false-positive/false-negative examples
- probability calibration
- final runtime benchmark
