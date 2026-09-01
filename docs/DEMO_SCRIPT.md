# Demo Video Script

Target length: 2–3 minutes.

## Scene 1 — Problem (15–20 sec)

Show a clean AI-generated image, then show the same image after JPEG compression, resizing and blur.

Narration:

> "AI-generated images can look convincing, but online images rarely stay pristine. Re-encoding, resizing and filtering can destroy the artifacts that a detector relies on."

## Scene 2 — Architecture (25–35 sec)

Show the architecture diagram.

Narration:

> "Our detector combines three evidence sources: CLIP and SigLIP semantic representations, plus a forensic residual stream built from SRM filtering, Haar wavelet details and ConvNeXt-Tiny. A learned gate dynamically combines them."

## Scene 3 — Training (20 sec)

Show the training command and a short terminal output.

Highlight:

- stochastic corruptions
- frozen semantic backbones
- trainable forensic branch and fusion head
- parameter count

## Scene 4 — Inference (25 sec)

Run:

```bash
python scripts/run_inference.py   --image_dir ./demo_images   --checkpoint ./checkpoints/master_model_epoch_15.pth   --output ./demo_results.json   --device cuda
```

Show the JSON output and a simple visualization/dashboard if available.

## Scene 5 — Robustness (30–40 sec)

Show the final clean-vs-transformed table.

Narration:

> "We evaluate the same isolated benchmark under clean images and controlled transformations, including JPEG compression, blur, down/up-scaling, noise, color jitter and center cropping."

Only display measured results.

## Scene 6 — Failure cases (20 sec)

Show one false positive and one false negative.

Narration:

> "The detector is not perfect. Authentic HDR photography and CGI can look synthetic, while localized edits or severe compression can hide synthetic evidence. We treat these as explicit boundary conditions rather than hiding them."

## Scene 7 — Closing (10–15 sec)

Narration:

> "The goal is a practical forensic signal that remains useful after the image has passed through the messy transformations of the real internet."
