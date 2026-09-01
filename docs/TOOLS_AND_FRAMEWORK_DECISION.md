# Tools / Framework Decision Record

| Tool | Decision | Why |
|---|---|---|
| PyTorch | REQUIRED | Core model/training/inference stack |
| Transformers | REQUIRED | SigLIP2/DINO and HF ecosystem |
| timm | REQUIRED | ConvNeXt and vision backbones |
| Accelerate | RECOMMENDED | device placement, CPU/disk offload experiments |
| OverflowML | OPTIONAL / RECOMMENDED EXPERIMENT | automatic memory strategy planning and guards |
| DeepSpeed | OPTIONAL | ZeRO/NVMe training experiments after baseline |
| Lightning | OPTIONAL | clean training/checkpointing/experiment organization |
| Unsloth | NOT REQUIRED | primarily LLM-oriented; no reason to add it without a specific compatible vision experiment |
| bitsandbytes | OPTIONAL | quantized inference where compatible; benchmark quality before using |
| safetensors | REQUIRED | safer/fast model weight loading |

## Memory hierarchy

1. Direct GPU load if it safely fits with headroom.
2. Model CPU offload / Accelerate if useful.
3. Layer/hybrid GPU+RAM if a model adapter supports it reliably.
4. Sequential expert execution.
5. Subprocess isolation between experts for hard CUDA-context reset.
6. NVMe feature cache / offload when RAM becomes limiting.

Do not assume a generic offload library works perfectly with every custom AIDE/DDA checkpoint. Use adapters and verify memory/accuracy/latency.
