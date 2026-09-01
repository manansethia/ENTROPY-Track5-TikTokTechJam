# ENTROPY — AI-Generated Image & Inpainting Forensic Detector

## Inspiration

Modern AI image generators like FLUX.1, Midjourney v6, SDXL, and DALL-E 3 have gotten so good that the human eye can no longer tell what is real and what is fake. 

When looking at existing AI detectors, we noticed three big problems:
1. **Most detectors only say "Real" or "Fake"**: In the real world, people rarely replace an entire image. They use tools like Photoshop Generative Fill or Google Magic Editor to edit just 2% to 5% of an authentic photo (like changing a face, removing an object, or adding a weapon). Standard detectors look at the whole image and miss these small edits completely.
2. **Social media breaks existing tools**: When an image is shared on TikTok, Instagram, or WhatsApp, it gets compressed, resized, and blurred. Detectors that rely on fragile pixel patterns break immediately.
3. **Zero explanation**: Giving a single number like "85% Fake" isn't useful for journalists, researchers, or platforms. People need to see *where* the image was changed and *why*.

We built **ENTROPY** to solve these problems by creating a forensic platform that accurately detects full AI images, spots localized partial edits, highlights the exact manipulated areas with heatmaps and bounding boxes, and survives real-world compression.

---

## What It Does

- **3-Way Classification**: Categorizes images into three clear verdicts:
  - `REAL`: Authentic camera photograph.
  - `PARTIAL-AI`: An authentic photo with localized AI inpainting, generative fill, or face edits.
  - `FULL-AIGC`: 100% synthetic AI generation.
- **Heatmap & Bounding Boxes**: Generates a spatial heatmap showing exactly which parts of the image were generated or modified, and draws bounding boxes around the edited regions.
- **Noise & Frequency Analysis**: Analyzes camera sensor noise patterns (PRNU) and Fourier frequency decay to catch generative upsampling artifacts.
- **Image Provenance**: Reads EXIF, camera metadata, and C2PA Content Credentials to trace image history.
- **Interactive Web Desk**: A web application featuring a physical desk layout with 3D cards, real-time laser scanning, and instant inspection.

---

## How We Built It

1. **Multi-Stream AI Architecture**:
   - **CLIP (ViT-L/14)**: Looks at overall scene semantics, lighting consistency, and natural anatomy.
   - **SigLIP (SO400M)**: Checks fine-grained textures and photorealism details.
   - **ConvNeXt + SRM Noise Filter**: Strips away the image content to look directly at the underlying pixel noise and mathematical residuals.
   - **Learned Gating Router**: Automatically decides whether to trust semantic features or noise patterns depending on how heavily compressed the image is.
2. **Knowledge Distillation**:
   - We trained a large multi-expert system across 11 specialist models.
   - We then distilled all that knowledge into a single standalone model (96M parameters) that runs in just **17 milliseconds** on a GPU ($73\times$ faster) with zero extra dependencies.
3. **Dataset & Training**:
   - Trained on an audited dataset of **over 103,000 images** across Midjourney v4/v5/v6, FLUX.1, SDXL, Stable Diffusion 1.5/2.1/3, DALL-E 2/3, Google Imagen, Firefly, StyleGAN, and real DSLR camera photos.
   - Strictly balanced image resolutions from thumbnails up to 4K/8K to prevent the model from cheating on image sharpness.
   - Cryptographically isolated all test benchmarks so the model never saw evaluation data during training.
4. **Full-Stack Application**:
   - **Backend**: Python, PyTorch, and FastAPI.
   - **Frontend**: HTML5, WebGL / Three.js, and vanilla JavaScript.

---

## Challenges We Faced

- **Models Cheating on Resolution**: Early on, the model learned that high-resolution photos were real and low-resolution photos were fake. We had to rebuild our dataset with resolution tiers so both real and fake images had equal distributions of sizes.
- **Portrait False Alarms**: Sharp skin pores in authentic DSLR studio portraits produced high-frequency noise that tricked basic filters into thinking they were AI. We trained a dedicated portrait specialist to learn the difference between natural skin texture and generative artifacts.
- **Spotting Tiny Partial Edits**: When someone edits only 3% of an image, standard neural networks drown out the edit with the 97% real background. We added coordinate-aware cross-attention and mask-based Dice loss so the model specifically focuses on small edited patches.
- **Speed & Server Costs**: Our initial multi-expert system was 1.82B parameters and took over 1.2 seconds per image. Through knowledge distillation, we compressed it down to a 184MB standalone model that runs in 17ms while cutting false alarms on real photos in half.

---

## What We Learned

- Pure deep learning alone isn't enough—combining classical noise filters with vision transformers makes detection much more resilient to compression.
- Knowing *where* an image is fake is much more valuable than just getting a single confidence score.
- Model distillation and INT8 quantization can deliver huge speedups (up to $167\times$) with almost no drop in accuracy.

---

## What's Next for ENTROPY

- **Video AI Detection**: Extending our spatial engine to detect AI-generated videos from Sora, Runway Gen-3, and Kling.
- **Mobile & Browser Extension**: Running the lightweight 4.8MB INT8 model directly inside the browser and on mobile devices for instant on-device verification.
- **C2PA Integration**: Adding automated cryptographic signing so authenticated photos can be permanently certified against tampering.
