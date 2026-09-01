#!/usr/bin/env python3
"""
scripts/finalize_production_champion.py
Ultra-Fast Parallel Production Finalization & Calibration Engine
Uses 16 worker threads for network storage I/O and PyTorch AMP GPU inference.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, accuracy_score
from scipy.optimize import minimize_scalar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHAMPION_CKPT_PATH = Path("/home/manan/aigc_robust_detection/checkpoints/ood_remediation/champion_remediation_base.pt")
MANIFEST_PATH = Path("/home/manan/aigc_robust_detection/manifests/ood_remediation_manifest_v1.jsonl")
PROD_CKPT_DIR = Path("/home/manan/aigc_robust_detection/checkpoints/production")
REPORT_DIR = Path("/home/manan/aigc_robust_detection/reports")
DEPLOY_CONFIG_PATH = Path("/home/manan/aigc_robust_detection/deployment/config.py")

PROD_CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.execute_final_forensic_feedback_pipeline import ScientificVisionDetector, eval_transform

def get_param_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        if p.requires_grad:
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def load_single_image(rec):
    path, label, domain, img_id = rec
    try:
        with Image.open(path) as img:
            t = eval_transform(img.convert("RGB"))
            return t, label, domain, img_id
    except Exception:
        return torch.zeros(3, 224, 224), label, domain, img_id

def evaluate_split_parallel(model, records, batch_size=64, num_workers=16, desc="Split"):
    model.eval()
    all_probs, all_labels, all_domains = [], [], []
    t0 = time.perf_counter()
    print(f"  >>> Evaluating {len(records):,} records for {desc} (batch_size={batch_size}, workers={num_workers})...", flush=True)
    
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i in range(0, len(records), batch_size):
            batch_recs = records[i:i+batch_size]
            results = list(pool.map(load_single_image, batch_recs))
            
            batch_tensors = torch.stack([r[0] for r in results]).to(device)
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch_tensors).squeeze(-1)
                probs = torch.sigmoid(logits.to(torch.float32)).cpu().tolist()
                
            all_probs.extend(probs)
            all_labels.extend([r[1] for r in results])
            all_domains.extend([r[2] for r in results])
            
            if (i // batch_size) % 25 == 0 or (i + batch_size) >= len(records):
                print(f"      [{desc}] {min(i+batch_size, len(records)):,}/{len(records):,} ({min(i+batch_size, len(records))/len(records)*100:.1f}%) in {time.perf_counter()-t0:.1f}s", flush=True)
                
    return np.array(all_labels, dtype=np.int32), np.array(all_probs, dtype=np.float32), all_domains

def main():
    print("=" * 80, flush=True)
    print("  PRODUCTION CHAMPION FINALIZATION & FREEZE ENGINE (ULTRA-FAST)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    
    # 1. Load Champion Detector
    print("\n[STEP 1] Instantiating ScientificVisionDetector (Config A)...", flush=True)
    detector = ScientificVisionDetector().to(device)
    base_data = torch.load(CHAMPION_CKPT_PATH, map_location="cpu", weights_only=False)
    detector.load_state_dict(base_data.get("model_state_dict", base_data), strict=False)
    detector.eval()
    
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_hash = get_param_hash(detector)
    
    print(f"  - Total Parameters:     {total_params:,}", flush=True)
    print(f"  - Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"  - Frozen Parameters:    {frozen_params:,}", flush=True)
    print(f"  - Parameter Hash:       {param_hash}", flush=True)
    
    # 2. Index CAL & DEV Splits
    print("\n[STEP 2] Indexing CAL (4,000) and DEV (10,000) splits...", flush=True)
    cal_records, dev_records = [], []
    with open(MANIFEST_PATH) as f:
        for line in f:
            item = json.loads(line)
            rec = (
                item.get("canonical_path", item.get("image_path", "")),
                int(item["label"]),
                item.get("generator_or_domain", item.get("domain", "general")),
                item.get("image_id", "img")
            )
            if item.get("split") in ("CAL", "CALIBRATION"):
                cal_records.append(rec)
            elif item.get("split") == "DEV":
                dev_records.append(rec)
    print(f"  - CAL Split: {len(cal_records):,} records | DEV Split: {len(dev_records):,} records", flush=True)
    
    # 3. Fit Temperature Scaling on CAL Split
    print("\n[STEP 3] Fitting Temperature Scaling T on CAL split...", flush=True)
    cal_labels, cal_probs, _ = evaluate_split_parallel(detector, cal_records, batch_size=64, num_workers=16, desc="CAL")
    
    cal_p = np.clip(cal_probs, 1e-12, 1.0 - 1e-12)
    cal_logits = np.log(cal_p / (1.0 - cal_p))
    cal_labels_f = cal_labels.astype(np.float64)
    
    def nll_obj(t_val):
        t_val = max(0.01, float(t_val))
        scaled_logits = cal_logits / t_val
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return -np.mean(cal_labels_f * np.log(probs) + (1.0 - cal_labels_f) * np.log(1.0 - probs))
        
    res_opt = minimize_scalar(nll_obj, bounds=(0.1, 5.0), method="bounded")
    fitted_temp = float(res_opt.x)
    print(f"  - Fitted Calibration Temperature: T = {fitted_temp:.6f}", flush=True)
    
    # 4. Evaluate DEV Split & Compute Exact Calibrated Operating Thresholds
    print("\n[STEP 4] Evaluating DEV Split & Computing Calibrated Operating Thresholds...", flush=True)
    dev_labels, dev_raw_probs, dev_domains = evaluate_split_parallel(detector, dev_records, batch_size=64, num_workers=16, desc="DEV")
    
    # Apply Temperature Scaling
    dev_p = np.clip(dev_raw_probs, 1e-12, 1.0 - 1e-12)
    dev_logits = np.log(dev_p / (1.0 - dev_p))
    dev_cal_probs = 1.0 / (1.0 + np.exp(-(dev_logits / fitted_temp)))
    
    dev_acc = float(accuracy_score(dev_labels, (dev_cal_probs >= 0.5).astype(int)))
    dev_auroc = float(roc_auc_score(dev_labels, dev_cal_probs))
    dev_auprc = float(average_precision_score(dev_labels, dev_cal_probs))
    dev_fp = int(np.sum((dev_labels == 0) & (dev_cal_probs >= 0.5)))
    dev_fn = int(np.sum((dev_labels == 1) & (dev_cal_probs < 0.5)))
    
    fprs, tprs, thresholds = roc_curve(dev_labels, dev_cal_probs)
    
    exact_thresholds = {}
    for target_fpr in [0.0100, 0.0050, 0.0010, 0.0005, 0.0001]:
        valid_idx = np.where(fprs <= target_fpr)[0]
        if len(valid_idx) > 0:
            idx = valid_idx[-1]
            exact_thresholds[f"FPR<={target_fpr*100:.2f}%"] = {
                "target_fpr": target_fpr,
                "achieved_fpr": float(fprs[idx]),
                "empirical_tpr": float(tprs[idx]),
                "calibrated_threshold": float(thresholds[idx])
            }
            
    print(f"\n  - DEV Accuracy:  {dev_acc*100:.2f}% (FP: {dev_fp}, FN: {dev_fn})", flush=True)
    print(f"  - DEV AUROC:     {dev_auroc:.6f}", flush=True)
    print(f"  - DEV AUPRC:     {dev_auprc:.6f}", flush=True)
    print("\n--- Exact Calibrated Operating Thresholds ---", flush=True)
    for k, v in exact_thresholds.items():
        print(f"  - {k:15s}: Threshold = {v['calibrated_threshold']:.6f} | Empirical TPR = {v['empirical_tpr']*100:.2f}% (FPR = {v['achieved_fpr']*100:.4f}%)", flush=True)
        
    # 5. Save and Freeze Production Model Checkpoint
    print("\n[STEP 5] Freezing Production Champion Checkpoint...", flush=True)
    frozen_ckpt_path = PROD_CKPT_DIR / "final_champion_frozen_model.pt"
    
    prod_payload = {
        "model_name": "ScientificVisionDetector-ConfigA",
        "champion_origin": "REM-A_Epoch3",
        "architecture": {
            "backbones": ["CLIP-ViT-L/14", "SigLIP-SO400M-14"],
            "fusion": "BottleneckFusionHead (2212 -> 512 -> 128 -> 1)",
            "evidence": "ForensicEvidenceHead (512 -> 128 -> 36)",
            "wavelet_srm": "GPU-Wavelet SRM Residual Block"
        },
        "parameter_counts": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params
        },
        "parameter_hash": param_hash,
        "calibration": {
            "temperature": fitted_temp,
            "calibration_split_size": len(cal_records),
            "nll_optimized": True
        },
        "operating_thresholds": exact_thresholds,
        "metrics_summary": {
            "dev_accuracy": dev_acc,
            "dev_auroc": dev_auroc,
            "dev_auprc": dev_auprc,
            "dev_fp": dev_fp,
            "dev_fn": dev_fn,
            "tpr_at_01_fpr": exact_thresholds["FPR<=0.10%"]["empirical_tpr"],
            "tpr_at_001_fpr": exact_thresholds["FPR<=0.01%"]["empirical_tpr"]
        },
        "frozen_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_state_dict": detector.state_dict()
    }
    
    torch.save(prod_payload, frozen_ckpt_path)
    file_sha256 = hashlib.sha256(open(frozen_ckpt_path, "rb").read()).hexdigest()
    
    print(f"  >>> FROZEN MODEL SAVED TO: {frozen_ckpt_path}", flush=True)
    print(f"  >>> FILE SHA-256:          {file_sha256}", flush=True)
    
    # 6. Save JSON Freeze Report
    report_path = REPORT_DIR / "final_production_freeze_report.json"
    clean_report = {k: v for k, v in prod_payload.items() if k != "model_state_dict"}
    clean_report["file_sha256"] = file_sha256
    clean_report["checkpoint_file"] = str(frozen_ckpt_path)
    
    with open(report_path, "w") as f:
        json.dump(clean_report, f, indent=2)
    print(f"  >>> PRODUCTION REPORT SAVED TO: {report_path}", flush=True)
    
    # 7. Update Deployment Config
    print("\n[STEP 6] Updating deployment/config.py with Frozen Parameters...", flush=True)
    t_01_thresh = exact_thresholds["FPR<=0.10%"]["calibrated_threshold"]
    t_001_thresh = exact_thresholds["FPR<=0.01%"]["calibrated_threshold"]
    t_1_thresh = exact_thresholds["FPR<=1.00%"]["calibrated_threshold"]
    t_05_thresh = exact_thresholds["FPR<=0.50%"]["calibrated_threshold"]
    t_005_thresh = exact_thresholds["FPR<=0.05%"]["calibrated_threshold"]
    
    deploy_config_content = f'''"""
Production Deployment Configuration for AIGC Robust Detector
AUTOGENERATED BY finalize_production_champion.py
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

DEFAULT_CHECKPOINT_PATH = Path("/home/manan/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt")

class DeploymentConfig(BaseModel):
    # Model Identification & Governance Integrity
    model_name: str = "ScientificVisionDetector-ConfigA"
    champion_origin: str = "REM-A_Epoch3"
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)
    expected_model_sha256: str = "{file_sha256}"
    parameter_hash: str = "{param_hash}"
    total_parameters: int = {total_params}
    trainable_parameters: int = {trainable_params}
    frozen_parameters: int = {frozen_params}
    preprocessing_version: str = "dual_vit_224_standard"
    class_mapping: dict = {{"0": "REAL", "1": "AIGC"}}
    
    # Inference Defaults
    device: str = Field(default_factory=lambda: "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and os.path.exists("/dev/nvidia0") else "cpu")
    torch_dtype: str = "bfloat16"
    batch_size: int = 32
    max_image_dimension: int = 4096
    
    # Calibration & Operating Thresholds
    temperature_scaling: float = {fitted_temp:.6f}
    threshold_standard: float = 0.500000
    threshold_low_fpr_10: float = {t_1_thresh:.6f}
    threshold_low_fpr_05: float = {t_05_thresh:.6f}
    threshold_low_fpr_01: float = {t_01_thresh:.6f}
    threshold_low_fpr_005: float = {t_005_thresh:.6f}
    threshold_low_fpr_001: float = {t_001_thresh:.6f}
    
    # Operating Points
    operating_thresholds: dict = {json.dumps(exact_thresholds, indent=4)}
    
    # Normalization
    input_resolution: int = 224
    norm_mean: list = [0.48145466, 0.4578275, 0.40821073]
    norm_std: list = [0.26862954, 0.26130258, 0.27577711]

config = DeploymentConfig()
'''
    with open(DEPLOY_CONFIG_PATH, "w") as f:
        f.write(deploy_config_content)
    print(f"  >>> Updated {DEPLOY_CONFIG_PATH}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print("  PRODUCTION FREEZE & CALIBRATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
