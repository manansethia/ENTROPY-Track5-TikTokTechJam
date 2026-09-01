#!/usr/bin/env python3
"""
scripts/benchmark_highres_external_detectors.py
Authoritative 3-Way High-Resolution Detector Benchmark on Buildabot:
1. ScientificVisionDetector-ConfigA (735M Champion Model)
2. SPAI / TFG-model (Any-Resolution Detector, aminasifar1/TFG-model)
3. CommunityForensics ViT-Small (21.8M Model, buildborderless/CommunityForensics-DeepfakeDet-ViT)

Evaluates on:
- Suite A: Problematic High-Res Real Portraits & Studio Headshots (including user test portrait)
- Suite B: Selfies & Smartphone Photos
- Suite C: 2K, 4K, 8K+ DSLR Real Photos
- Suite D: Color-Graded, HDR & Retouched Photos
- Suite E: Cropped, Brightness, Contrast & JPEG Edited Real Images
- Suite F: Comparable High-Res Synthetic Images (SDXL, Flux, Midjourney, LDM)
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import sys
import io
import gc
import json
import time
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SPAI path setup
SPAI_DIR = Path("/mnt/ai-storage/aigc_data/models/spai_tfg")
if str(SPAI_DIR) not in sys.path:
    sys.path.insert(0, str(SPAI_DIR))

from deployment.portable_model import load_portable_champion_model, portable_eval_transform

CHAMPION_CHECKPOINT = REPO_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt"
OUTPUT_REPORT_JSON = REPO_ROOT / "reports" / "external_detector_benchmark_comparison.json"
OUTPUT_REPORT_MD = REPO_ROOT / "reports" / "external_detector_benchmark_comparison.md"

def collect_evaluation_suites():
    """Collects diverse evaluation sets across the 6 diagnostic categories."""
    suites = {
        "A_Studio_Portraits_HighRes": [],
        "B_Selfies_Smartphone_Photos": [],
        "C_2K_4K_8K_DSLR_Photos": [],
        "D_ColorGraded_HDR_Retouched": [],
        "E_Cropped_Edited_Compressed_Real": [],
        "F_HighRes_Synthetic_AIGC": []
    }
    
    # 1. User test portrait & Studio portraits
    user_img = REPO_ROOT / "user_test_portrait.png"
    if user_img.exists():
        suites["A_Studio_Portraits_HighRes"].append(str(user_img))
        
    celeba_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_portrait")
    if celeba_dir.exists():
        suites["A_Studio_Portraits_HighRes"].extend([str(p) for p in list(celeba_dir.glob("*.jpg"))[:50]])
        
    # 2. Selfies & Smartphone Photos
    phone_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_smartphone")
    if phone_dir.exists():
        suites["B_Selfies_Smartphone_Photos"].extend([str(p) for p in list(phone_dir.glob("*.jpg"))[:50]])
    coco_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real")
    if coco_dir.exists():
        suites["B_Selfies_Smartphone_Photos"].extend([str(p) for p in list(coco_dir.glob("coco_*.jpg"))[:50]])
        
    # 3. 2K/4K/8K DSLR Photos (DIV2K & Wikimedia)
    dslr_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_dslr")
    if dslr_dir.exists():
        suites["C_2K_4K_8K_DSLR_Photos"].extend([str(p) for p in list(dslr_dir.glob("*.png"))[:50] + list(dslr_dir.glob("*.jpg"))[:50]])
        
    # 4. Color Graded / HDR / Retouched
    hdr_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_hdr")
    if hdr_dir.exists():
        suites["D_ColorGraded_HDR_Retouched"].extend([str(p) for p in list(hdr_dir.glob("*.jpg"))[:50]])
    sid_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_real")
    if sid_dir.exists():
        suites["D_ColorGraded_HDR_Retouched"].extend([str(p) for p in list(sid_dir.glob("*.jpg"))[:50]])
        
    # 5. Cropped / Edited / Compressed Real Images (Created from base real images)
    base_real_samples = list(suites["A_Studio_Portraits_HighRes"])[:25] + list(suites["C_2K_4K_8K_DSLR_Photos"])[:25]
    for p in base_real_samples:
        suites["E_Cropped_Edited_Compressed_Real"].append(p)
        
    # 6. High-Res Synthetic AIGC (Quality Paradox & Massive Balanced Synthetic)
    paradox_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/synthetic")
    if paradox_dir.exists():
        for p in list(paradox_dir.glob("*.jpg"))[:35] + list(paradox_dir.glob("*.png"))[:35]:
            suites["F_HighRes_Synthetic_AIGC"].append(str(p))
            if len(suites["F_HighRes_Synthetic_AIGC"]) >= 50:
                break
    sid_synth_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_synthetic")
    if sid_synth_dir.exists():
        suites["F_HighRes_Synthetic_AIGC"].extend([str(p) for p in list(sid_synth_dir.glob("*.jpg"))[:50]])
        
    return suites

def evaluate_model_on_suites(model_name: str, predict_fn, suites: Dict[str, List[str]]):
    """Evaluates a detector model across all suites and computes performance statistics."""
    print(f"\n--- Benchmarking Model: {model_name} ---")
    results = {}
    
    for suite_name, paths in suites.items():
        if not paths:
            continue
            
        probs = []
        latencies = []
        
        for path_str in paths:
            p = Path(path_str)
            if not p.exists():
                continue
            try:
                with Image.open(p) as raw_img:
                    img = ImageOps.exif_transpose(raw_img).convert("RGB")
                    
                    if suite_name == "E_Cropped_Edited_Compressed_Real":
                        w, h = img.size
                        img = img.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)))
                        img = ImageEnhance.Brightness(img).enhance(1.08)
                        img = ImageEnhance.Contrast(img).enhance(1.05)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        buf.seek(0)
                        img = Image.open(buf).convert("RGB")
                        
                    t_start = time.perf_counter()
                    prob = predict_fn(img)
                    t_lat = (time.perf_counter() - t_start) * 1000.0
                    
                probs.append(prob)
                latencies.append(t_lat)
            except Exception as e:
                continue
                
        if probs:
            mean_p = float(np.mean(probs))
            median_p = float(np.median(probs))
            p95_p = float(np.percentile(probs, 95))
            mean_lat = float(np.mean(latencies))
            
            if suite_name == "F_HighRes_Synthetic_AIGC":
                tpr = (sum(1 for pr in probs if pr >= 0.50) / len(probs)) * 100.0
                metric_label = f"TPR @ 0.50: {tpr:.1f}%"
            else:
                fpr = (sum(1 for pr in probs if pr >= 0.50) / len(probs)) * 100.0
                metric_label = f"FPR @ 0.50: {fpr:.1f}%"
                
            results[suite_name] = {
                "sample_count": len(probs),
                "mean_p_aigc": mean_p,
                "median_p_aigc": median_p,
                "p95_p_aigc": p95_p,
                "mean_latency_ms": mean_lat,
                "metric_label": metric_label
            }
            print(f"  {suite_name:32s} | N={len(probs):3d} | Mean P={mean_p:7.4f}% | Median={median_p:7.4f}% | P95={p95_p:7.4f}% | Latency={mean_lat:6.1f}ms | {metric_label}")
            
    return results

def run_comparative_benchmark():
    print("=" * 90)
    print("  AUTHORITATIVE 3-WAY DETECTOR BENCHMARK (735M vs SPAI vs 21.8M ViT)")
    print("=" * 90)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suites = collect_evaluation_suites()
    for s_name, s_paths in suites.items():
        print(f"  Suite '{s_name}': {len(s_paths)} images")
        
    all_benchmark_results = {}
    
    # -------------------------------------------------------------
    # 1. ScientificVisionDetector-ConfigA (735M Champion)
    # -------------------------------------------------------------
    print("\nLoading Model 1: ScientificVisionDetector-ConfigA (735M Frozen Champion)...")
    champion_model, champ_meta = load_portable_champion_model(CHAMPION_CHECKPOINT, device=device)
    T = champ_meta.get("temperature", 1.5230212761606914)
    
    def predict_champion(img: Image.Image) -> float:
        tensor = portable_eval_transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            logit = float(champion_model(tensor).cpu().item())
        return float(torch.sigmoid(torch.tensor(logit / T)).item())
        
    all_benchmark_results["ScientificVisionDetector_735M"] = {
        "architecture": "Dual CLIP ViT-L/14 + SigLIP SO400M + SRM Residual",
        "parameters": "735,038,561 (735M)",
        "vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "results": evaluate_model_on_suites("ScientificVisionDetector-ConfigA (735M)", predict_champion, suites)
    }
    
    del champion_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # -------------------------------------------------------------
    # 2. SPAI / TFG-model (Any-Resolution Detector)
    # -------------------------------------------------------------
    print("\nLoading Model 2: SPAI / TFG-model (Any-Resolution Detector)...")
    os.chdir(str(SPAI_DIR))
    from inference import EndpointHandler
    spai_handler = EndpointHandler(str(SPAI_DIR))
    
    def predict_spai(img: Image.Image) -> float:
        # Scale to max 1280px to avoid quadratic attention OOM on 4K/8K images
        w, h = img.size
        max_d = max(w, h)
        if max_d > 1280:
            scale = 1280.0 / max_d
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        res = spai_handler({"inputs": img})
        return float(res.get("score", 0.0))
        
    all_benchmark_results["SPAI_TFG_AnyResolution"] = {
        "architecture": "SPAI Spatial-Frequency Swin/ViT Artifact Detector",
        "parameters": "88,000,000 (~88M)",
        "vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "results": evaluate_model_on_suites("SPAI / TFG Any-Resolution", predict_spai, suites)
    }
    
    del spai_handler
    torch.cuda.empty_cache()
    gc.collect()
    
    # -------------------------------------------------------------
    # 3. CommunityForensics ViT-Small (21.8M Model)
    # -------------------------------------------------------------
    print("\nLoading Model 3: CommunityForensics ViT-Small (21.8M)...")
    cf_id = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    cf_processor = AutoImageProcessor.from_pretrained(cf_id)
    cf_model = AutoModelForImageClassification.from_pretrained(cf_id).to(device)
    cf_model.eval()
    
    def predict_community_forensics(img: Image.Image) -> float:
        inputs = cf_processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = cf_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            fake_prob = float(probs[1].cpu().item())
        return fake_prob
        
    all_benchmark_results["CommunityForensics_ViT_Small_21M"] = {
        "architecture": "Timm ViT-Small / Patch-16",
        "parameters": "21,811,969 (21.8M)",
        "vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "results": evaluate_model_on_suites("CommunityForensics ViT-Small (21.8M)", predict_community_forensics, suites)
    }
    
    # -------------------------------------------------------------
    # 4. Calculate AUROC on Mixed Labeled Evaluation Sets
    # -------------------------------------------------------------
    print("\n--- AUROC / AUPRC on High-Res Test Sets ---")
    labeled_real = suites["A_Studio_Portraits_HighRes"][:25] + suites["C_2K_4K_8K_DSLR_Photos"][:25]
    labeled_fake = suites["F_HighRes_Synthetic_AIGC"][:50]
    
    y_true = [0] * len(labeled_real) + [1] * len(labeled_fake)
    all_eval_paths = labeled_real + labeled_fake
    
    champion_model, _ = load_portable_champion_model(CHAMPION_CHECKPOINT, device=device)
    champ_scores = []
    for p in all_eval_paths:
        with Image.open(p) as im:
            champ_scores.append(predict_champion(ImageOps.exif_transpose(im).convert("RGB")))
    del champion_model
    torch.cuda.empty_cache()
    
    spai_handler = EndpointHandler(str(SPAI_DIR))
    spai_scores = []
    for p in all_eval_paths:
        with Image.open(p) as im:
            spai_scores.append(predict_spai(ImageOps.exif_transpose(im).convert("RGB")))
    del spai_handler
    torch.cuda.empty_cache()
    
    cf_scores = []
    for p in all_eval_paths:
        with Image.open(p) as im:
            cf_scores.append(predict_community_forensics(ImageOps.exif_transpose(im).convert("RGB")))
            
    champ_auroc = float(roc_auc_score(y_true, champ_scores))
    spai_auroc = float(roc_auc_score(y_true, spai_scores))
    cf_auroc = float(roc_auc_score(y_true, cf_scores))
    
    all_benchmark_results["Comparative_AUROC"] = {
        "test_sample_count": len(y_true),
        "scientific_detector_735m_auroc": champ_auroc,
        "spai_tfg_auroc": spai_auroc,
        "community_forensics_21m_auroc": cf_auroc
    }
    print(f"  Scientific Vision Detector (735M) AUROC: {champ_auroc:.4f}")
    print(f"  SPAI / TFG Any-Resolution AUROC:         {spai_auroc:.4f}")
    print(f"  CommunityForensics ViT-Small (21.8M) AUROC: {cf_auroc:.4f}")
    
    OUTPUT_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(all_benchmark_results, f, indent=2)
        
    md_content = f"""# High-Resolution Detector Comparative Benchmark Report

## 1. Executive Model Comparison Matrix
| Detector Model | Architecture | Parameters | VRAM (MB) | Studio Portrait Real FPR | 2K/4K DSLR Real FPR | Edited Real FPR | High-Res AIGC TPR | Benchmark AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Scientific Vision Detector`** | Dual CLIP ViT-L + SigLIP + SRM | **`735M`** | `{all_benchmark_results['ScientificVisionDetector_735M']['vram_mb']}` | **`{all_benchmark_results['ScientificVisionDetector_735M']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['ScientificVisionDetector_735M']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['ScientificVisionDetector_735M']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['ScientificVisionDetector_735M']['results']['F_HighRes_Synthetic_AIGC']['mean_p_aigc']:.2f}%`** | **`{champ_auroc:.4f}`** |
| **`SPAI / TFG Detector`** | Spatial-Frequency Swin/ViT Artifact | **`~88M`** | `{all_benchmark_results['SPAI_TFG_AnyResolution']['vram_mb']}` | **`{all_benchmark_results['SPAI_TFG_AnyResolution']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['SPAI_TFG_AnyResolution']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['SPAI_TFG_AnyResolution']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['SPAI_TFG_AnyResolution']['results']['F_HighRes_Synthetic_AIGC']['mean_p_aigc']:.2f}%`** | **`{spai_auroc:.4f}`** |
| **`CommunityForensics ViT-Small`** | Timm ViT-Small Patch-16 | **`21.8M`** | `{all_benchmark_results['CommunityForensics_ViT_Small_21M']['vram_mb']}` | **`{all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%`** | **`{all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['F_HighRes_Synthetic_AIGC']['mean_p_aigc']:.2f}%`** | **`{cf_auroc:.4f}`** |

---

## 2. Forensic Failure Mode Diagnostic Breakdown
### A. Studio Portraits & High-End Headshots
- **Scientific Vision Detector (735M)**: Mean P(AIGC) = {all_benchmark_results['ScientificVisionDetector_735M']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%, P95 = {all_benchmark_results['ScientificVisionDetector_735M']['results']['A_Studio_Portraits_HighRes']['p95_p_aigc']:.2f}%.
- **SPAI / TFG Any-Resolution**: Mean P(AIGC) = {all_benchmark_results['SPAI_TFG_AnyResolution']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%, P95 = {all_benchmark_results['SPAI_TFG_AnyResolution']['results']['A_Studio_Portraits_HighRes']['p95_p_aigc']:.2f}%.
- **CommunityForensics ViT-Small**: Mean P(AIGC) = {all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['A_Studio_Portraits_HighRes']['mean_p_aigc']:.2f}%, P95 = {all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['A_Studio_Portraits_HighRes']['p95_p_aigc']:.2f}%.

### B. 2K / 4K / 8K DSLR Clean Photography
- **Scientific Vision Detector (735M)**: Mean P(AIGC) = {all_benchmark_results['ScientificVisionDetector_735M']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%.
- **SPAI / TFG Any-Resolution**: Mean P(AIGC) = {all_benchmark_results['SPAI_TFG_AnyResolution']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%.
- **CommunityForensics ViT-Small**: Mean P(AIGC) = {all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['C_2K_4K_8K_DSLR_Photos']['mean_p_aigc']:.2f}%.

### C. Post-Processed & Edited Real Images (Crop + Brightness + Contrast + JPEG)
- **Scientific Vision Detector (735M)**: Mean P(AIGC) = {all_benchmark_results['ScientificVisionDetector_735M']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%.
- **SPAI / TFG Any-Resolution**: Mean P(AIGC) = {all_benchmark_results['SPAI_TFG_AnyResolution']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%.
- **CommunityForensics ViT-Small**: Mean P(AIGC) = {all_benchmark_results['CommunityForensics_ViT_Small_21M']['results']['E_Cropped_Edited_Compressed_Real']['mean_p_aigc']:.2f}%.
"""

    with open(OUTPUT_REPORT_MD, "w") as f:
        f.write(md_content)
        
    print(f"\nSaved benchmark reports to:\n  - {OUTPUT_REPORT_JSON}\n  - {OUTPUT_REPORT_MD}")

if __name__ == "__main__":
    run_comparative_benchmark()
