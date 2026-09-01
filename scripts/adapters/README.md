# Candidate adapters

The model pool is intentionally heterogeneous. Add one adapter per candidate under this directory.
Each adapter should expose:

```python
load(device) -> model
predict(images) -> probability
parameter_count() -> int
peak_vram_mb() -> float
```

The benchmark runner should evaluate every candidate on the same probe images and transformations, then write JSON/CSV outputs.
