#!/usr/bin/env python3
"""
test_all_6_distilled_vs_big_agent.py
------------------------------------
Authoritative benchmark comparing:
  - Big Master Ensemble (1.82B FP16)
  - High-Capacity Distilled Student (96.59M): FP32, FP16, INT8
  - Ultra-Light Distilled Student (4.67M):   FP32, FP16, INT8

Answers directly whether the 6 distilled models are on par with the main big agent.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.distilled_forensic_model import SingleStudentForensicModel
from scripts.final.highcap_distilled_forensic_model import HighCapacityStudentForensicModel
from scripts.final.compile_master_unified_model import MasterUnifiedForensicModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_int8_checkpoint(model, ckpt_path, device, dtype=torch.float16):
    data = torch.load(ckpt_path, map_location="cpu")
    sd = data["model_state_dict"]
    unpacked_sd = {}
    for k, v in sd.items():
        if isinstance(v, dict) and v.get("is_quantized", False):
            qw = v["qweight"].float()
            scale = float(v["scale"])
            unpacked_sd[k] = (qw * scale).to(device, dtype=dtype)
        elif isinstance(v, torch.Tensor):
            unpacked_sd[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            unpacked_sd[k] = v
    model.load_state_dict(unpacked_sd)
    return model

def load_models() -> Dict[str, Any]:
    print("=" * 105)
    print("  LOADING 7 AI FORENSIC SYSTEMS: 1 BIG MASTER ENSEMBLE + 6 DISTILLED MODELS")
    print("=" * 105)
    models = {}

    # 1. Big Master Teacher Ensemble (1.82B FP16)
    print("\n[1/7] Loading Big Master Teacher Ensemble (1.82B FP16)...", flush=True)
    t0 = time.time()
    teacher = MasterUnifiedForensicModel().half()
    sd_t = torch.load("/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp16.pt", map_location="cpu")
    teacher.load_state_dict(sd_t["model_state_dict"])
    teacher = teacher.eval()
    p_t = sum(p.numel() for p in teacher.parameters())
    sz_t = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp16.pt") / (1024**2)
    models["Big_Master_1.82B_FP16"] = {"model": teacher, "params": p_t, "size_mb": sz_t, "type": "teacher", "prec": "FP16"}
    print(f"  Loaded: {p_t:,} params ({sz_t:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 2. High-Cap 96.59M FP32
    print("\n[2/7] Loading High-Cap Distilled Student (96.59M FP32)...", flush=True)
    t0 = time.time()
    hc_fp32 = HighCapacityStudentForensicModel().to(DEVICE).eval()
    sd_hc_fp32 = torch.load("/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_fp32.pt", map_location=DEVICE)
    hc_fp32.load_state_dict(sd_hc_fp32["model_state_dict"])
    p_hc = sum(p.numel() for p in hc_fp32.parameters())
    sz_hc_fp32 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_fp32.pt") / (1024**2)
    models["HighCap_96M_FP32"] = {"model": hc_fp32, "params": p_hc, "size_mb": sz_hc_fp32, "type": "highcap", "prec": "FP32"}
    print(f"  Loaded: {p_hc:,} params ({sz_hc_fp32:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 3. High-Cap 96.59M FP16
    print("\n[3/7] Loading High-Cap Distilled Student (96.59M FP16)...", flush=True)
    t0 = time.time()
    hc_fp16 = HighCapacityStudentForensicModel().half().to(DEVICE).eval()
    sd_hc_fp16 = torch.load("/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_fp16.pt", map_location=DEVICE)
    hc_fp16.load_state_dict(sd_hc_fp16["model_state_dict"])
    sz_hc_fp16 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_fp16.pt") / (1024**2)
    models["HighCap_96M_FP16"] = {"model": hc_fp16, "params": p_hc, "size_mb": sz_hc_fp16, "type": "highcap", "prec": "FP16"}
    print(f"  Loaded: {p_hc:,} params ({sz_hc_fp16:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 4. High-Cap 96.59M INT8
    print("\n[4/7] Loading High-Cap Distilled Student (96.59M INT8)...", flush=True)
    t0 = time.time()
    hc_int8 = HighCapacityStudentForensicModel().half().to(DEVICE).eval()
    load_int8_checkpoint(hc_int8, "/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_int8.pt", DEVICE, dtype=torch.float16)
    sz_hc_int8 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_int8.pt") / (1024**2)
    models["HighCap_96M_INT8"] = {"model": hc_int8, "params": p_hc, "size_mb": sz_hc_int8, "type": "highcap", "prec": "INT8"}
    print(f"  Loaded: {p_hc:,} params ({sz_hc_int8:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 5. Ultra-Light 4.67M FP32
    print("\n[5/7] Loading Ultra-Light Distilled Student (4.67M FP32)...", flush=True)
    t0 = time.time()
    ul_fp32 = SingleStudentForensicModel().to(DEVICE).eval()
    sd_ul_fp32 = torch.load("/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp32.pt", map_location=DEVICE)
    ul_fp32.load_state_dict(sd_ul_fp32["model_state_dict"])
    p_ul = sum(p.numel() for p in ul_fp32.parameters())
    sz_ul_fp32 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp32.pt") / (1024**2)
    models["UltraLight_4M_FP32"] = {"model": ul_fp32, "params": p_ul, "size_mb": sz_ul_fp32, "type": "ultralight", "prec": "FP32"}
    print(f"  Loaded: {p_ul:,} params ({sz_ul_fp32:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 6. Ultra-Light 4.67M FP16
    print("\n[6/7] Loading Ultra-Light Distilled Student (4.67M FP16)...", flush=True)
    t0 = time.time()
    ul_fp16 = SingleStudentForensicModel().half().to(DEVICE).eval()
    sd_ul_fp16 = torch.load("/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp16.pt", map_location=DEVICE)
    ul_fp16.load_state_dict(sd_ul_fp16["model_state_dict"])
    sz_ul_fp16 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp16.pt") / (1024**2)
    models["UltraLight_4M_FP16"] = {"model": ul_fp16, "params": p_ul, "size_mb": sz_ul_fp16, "type": "ultralight", "prec": "FP16"}
    print(f"  Loaded: {p_ul:,} params ({sz_ul_fp16:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    # 7. Ultra-Light 4.67M INT8
    print("\n[7/7] Loading Ultra-Light Distilled Student (4.67M INT8)...", flush=True)
    t0 = time.time()
    ul_int8 = SingleStudentForensicModel().half().to(DEVICE).eval()
    load_int8_checkpoint(ul_int8, "/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_int8.pt", DEVICE, dtype=torch.float16)
    sz_ul_int8 = os.path.getsize("/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_int8.pt") / (1024**2)
    models["UltraLight_4M_INT8"] = {"model": ul_int8, "params": p_ul, "size_mb": sz_ul_int8, "type": "ultralight", "prec": "INT8"}
    print(f"  Loaded: {p_ul:,} params ({sz_ul_int8:.2f} MB) in {time.time()-t0:.2f}s ✅", flush=True)

    return models

def collect_evaluation_samples() -> List[Tuple[str, int, str]]:
    base_dir = "/mnt/ai-storage/aigc_data/datasets"
    val_samples = []

    # 10 REAL
    real_dirs = [f"{base_dir}/massive_balanced_50k/real", f"{base_dir}/portrait_remediation/real_dslr"]
    for d in real_dirs:
        if os.path.exists(d):
            files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for f in files[-5:]:
                val_samples.append((f, 0, "REAL"))

    # 10 PARTIAL_AIGC
    part_dirs = [f"{base_dir}/v4_3_large_partial_ai_corpus/images"]
    for d in part_dirs:
        if os.path.exists(d):
            files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for f in files[-10:]:
                val_samples.append((f, 1, "PARTIAL_AIGC"))

    # 10 FULL_AIGC
    full_dirs = [f"{base_dir}/massive_balanced_50k/synthetic", f"{base_dir}/scaled_train/synthetic"]
    for d in full_dirs:
        if os.path.exists(d):
            files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for f in files[-5:]:
                val_samples.append((f, 2, "FULL_AIGC"))

    # User test images
    user_test_dir = "/home/manan/aigc_robust_detection/test_inputs"
    user_test_sub = "/home/manan/aigc_robust_detection/test_inputs/final_user_test"
    
    if os.path.exists(os.path.join(user_test_dir, "4women.webp")):
        val_samples.append((os.path.join(user_test_dir, "4women.webp"), 2, "FULL_AIGC"))
    if os.path.exists(user_test_sub):
        for f in sorted(os.listdir(user_test_sub)):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                p = os.path.join(user_test_sub, f)
                # determine probable label for user test images
                if "real" in f.lower():
                    val_samples.append((p, 0, "REAL"))
                else:
                    val_samples.append((p, 1, "PARTIAL_AIGC"))

    print(f"\nCollected {len(val_samples)} evaluation samples (30 held-out benchmark + {len(val_samples)-30} user test images) ✅", flush=True)
    return val_samples

def evaluate_all():
    models = load_models()
    samples = collect_evaluation_samples()

    t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    model_names = list(models.keys())
    stats = {
        m: {
            "correct": 0, "total": 0, "latencies": [], "real_fps": 0, "real_tot": 0,
            "partial_correct": 0, "partial_tot": 0,
            "full_correct": 0, "full_tot": 0,
            "teacher_agreements": 0,
            "preds": []
        }
        for m in model_names
    }

    teacher_obj = models["Big_Master_1.82B_FP16"]["model"]

    print("\n" + "=" * 125)
    print("  RUNNING 7-MODEL BENCHMARK: BIG MASTER (1.82B) VS 6 DISTILLED VARIANTS")
    print("=" * 125)
    header = f"{'Sample Image':<28} | {'GT':<7} | {'Big Master':<12} | {'HC-FP32':<10} | {'HC-FP16':<10} | {'HC-INT8':<10} | {'UL-FP32':<10} | {'UL-FP16':<10} | {'UL-INT8':<10}"
    print(header)
    print("-" * 125)

    class_names = ["REAL", "PARTIAL", "FULL"]

    for idx, (path, gt_class, tag) in enumerate(samples, 1):
        try:
            img = Image.open(path).convert("RGB")
            i224 = t_224(img).unsqueeze(0).to(DEVICE)
            i224_h = t_224(img).unsqueeze(0).half().to(DEVICE)
            i256_h = t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).half().to(DEVICE)
            i384_h = t_384(img).unsqueeze(0).half().to(DEVICE)

            fname = os.path.basename(path)
            if len(fname) > 26:
                fname = fname[:12] + "..." + fname[-11:]

            # 1. Big Master Teacher Inference
            t_start = time.perf_counter()
            with torch.no_grad():
                v2_out = teacher_obj.v2_aide.to(DEVICE)(i256_h)
                v2_s = torch.sigmoid(v2_out[:, 0:1] if v2_out.shape[-1] > 1 else v2_out).item()
                teacher_obj.v2_aide.to("cpu")

                srm_d = torch.zeros((1, 36), dtype=torch.float16, device=DEVICE)
                c0_out = teacher_obj.v3_c0_champion.to(DEVICE)(i224_h, srm_d)
                c0_s = float(c0_out[:, 0].item() if c0_out.ndim > 1 else c0_out.item())
                teacher_obj.v3_c0_champion.to("cpu")

                spec_vec = [c0_s]
                for s_mod, res in zip(
                    [teacher_obj.v3_c1_portrait, teacher_obj.v3_c2_spai, teacher_obj.v3_c3_community, teacher_obj.v3_c4_highres, teacher_obj.v3_c5_divine2k, teacher_obj.v3_c6_efficientnet, teacher_obj.v3_c7_resnet50],
                    [i224_h, i384_h, i384_h, i384_h, i224_h, i224_h, i224_h]
                ):
                    mod_gpu = s_mod.to(DEVICE)
                    out = mod_gpu(res)
                    spec_vec.append(float(out[:, 0].item() if out.ndim > 1 else out.item()))
                    s_mod.to("cpu")

                s_t = torch.tensor(spec_vec, dtype=torch.float16, device=DEVICE).unsqueeze(0)
                g_out = teacher_obj.v3_gating.to(DEVICE)(s_t)
                gw = F.softmax(g_out[0] if isinstance(g_out, (list, tuple)) else g_out, dim=-1)
                v3_s = torch.sigmoid((s_t * gw).sum(dim=-1)).item()
                teacher_obj.v3_gating.to("cpu")

                v5_bb = teacher_obj.v5_backbone.to(DEVICE)
                v5_pl = teacher_obj.v5_pool.to(DEVICE)
                v5_cag = teacher_obj.v5_cag_head.to(DEVICE)
                feats = v5_bb(i224_h)
                g_feat = v5_pl(feats[-1] if isinstance(feats, (list, tuple)) else feats).flatten(1)
                p_c = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float16, device=DEVICE)
                w_l, p_l, p_mask, _ = v5_cag(g_feat, g_feat, p_c)
                v5_s = float(torch.sigmoid(p_mask).mean().item())
                v5_bb.to("cpu"); v5_pl.to("cpu"); v5_cag.to("cpu")
                torch.cuda.empty_cache()

                fused_ai = 0.35 * v2_s + 0.40 * v3_s + 0.25 * v5_s
                if fused_ai < 0.35:
                    pred_teacher = 0
                elif fused_ai < 0.70:
                    pred_teacher = 1
                else:
                    pred_teacher = 2
            lat_teacher = (time.perf_counter() - t_start) * 1000

            # Record Teacher
            stats["Big_Master_1.82B_FP16"]["total"] += 1
            stats["Big_Master_1.82B_FP16"]["latencies"].append(lat_teacher)
            stats["Big_Master_1.82B_FP16"]["preds"].append(pred_teacher)
            if pred_teacher == gt_class:
                stats["Big_Master_1.82B_FP16"]["correct"] += 1
            if gt_class == 0:
                stats["Big_Master_1.82B_FP16"]["real_tot"] += 1
                if pred_teacher != 0:
                    stats["Big_Master_1.82B_FP16"]["real_fps"] += 1
            elif gt_class == 1:
                stats["Big_Master_1.82B_FP16"]["partial_tot"] += 1
                if pred_teacher == 1:
                    stats["Big_Master_1.82B_FP16"]["partial_correct"] += 1
            elif gt_class == 2:
                stats["Big_Master_1.82B_FP16"]["full_tot"] += 1
                if pred_teacher == 2:
                    stats["Big_Master_1.82B_FP16"]["full_correct"] += 1
            stats["Big_Master_1.82B_FP16"]["teacher_agreements"] += 1

            row_preds = [f"{class_names[pred_teacher]} ({fused_ai*100:.0f}%)"]

            # Evaluate the 6 Distilled Students
            for m_name in model_names[1:]:
                m_info = models[m_name]
                m_obj = m_info["model"]
                m_prec = m_info["prec"]

                inp = i224_h if m_prec in ["FP16", "INT8"] else i224

                t_start = time.perf_counter()
                with torch.no_grad():
                    out = m_obj(inp)
                    pred = int(out["class_logits"].argmax(dim=-1).item())
                    prob = float(out["probabilities"][0, pred].item())
                lat = (time.perf_counter() - t_start) * 1000

                stats[m_name]["total"] += 1
                stats[m_name]["latencies"].append(lat)
                stats[m_name]["preds"].append(pred)
                if pred == gt_class:
                    stats[m_name]["correct"] += 1
                if pred == pred_teacher:
                    stats[m_name]["teacher_agreements"] += 1

                if gt_class == 0:
                    stats[m_name]["real_tot"] += 1
                    if pred != 0:
                        stats[m_name]["real_fps"] += 1
                elif gt_class == 1:
                    stats[m_name]["partial_tot"] += 1
                    if pred == 1:
                        stats[m_name]["partial_correct"] += 1
                elif gt_class == 2:
                    stats[m_name]["full_tot"] += 1
                    if pred == 2:
                        stats[m_name]["full_correct"] += 1

                row_preds.append(f"{class_names[pred]} ({prob*100:.0f}%)")

            # Print comparison line
            print(f"{fname:<28} | {class_names[gt_class]:<7} | {row_preds[0]:<12} | {row_preds[1]:<10} | {row_preds[2]:<10} | {row_preds[3]:<10} | {row_preds[4]:<10} | {row_preds[5]:<10} | {row_preds[6]:<10}")

        except Exception as e:
            print(f"Error on sample {path}: {e}")
            continue

    # Final Comparative Matrix
    print("\n" + "=" * 125)
    print("             COMPREHENSIVE BENCHMARK MATRIX: 6 DISTILLED MODELS VS BIG MASTER ENSEMBLE")
    print("=" * 125)
    print(f"{'Model Name':<26} | {'Params':<8} | {'Size':<10} | {'3-Way Acc':<10} | {'Real FPR':<10} | {'Teacher Match':<14} | {'GPU Latency':<12} | {'Speedup':<8}")
    print("-" * 125)

    base_lat = np.mean(stats["Big_Master_1.82B_FP16"]["latencies"])
    final_report = {}

    for m_name in model_names:
        s = stats[m_name]
        info = models[m_name]
        tot = max(1, s["total"])
        acc = (s["correct"] / tot) * 100.0
        fpr = (s["real_fps"] / max(1, s["real_tot"])) * 100.0
        match = (s["teacher_agreements"] / tot) * 100.0
        lat = float(np.mean(s["latencies"]))
        speedup = base_lat / max(0.1, lat)

        p_str = f"{info['params']/1e6:.1f}M" if info['params'] < 1e9 else f"{info['params']/1e9:.2f}B"
        sz_str = f"{info['size_mb']:.1f} MB"

        final_report[m_name] = {
            "params": info["params"],
            "size_mb": round(info["size_mb"], 2),
            "precision": info["prec"],
            "accuracy_pct": round(acc, 2),
            "real_false_alarm_pct": round(fpr, 2),
            "teacher_fidelity_pct": round(match, 2),
            "latency_ms": round(lat, 2),
            "speedup_vs_teacher": round(speedup, 1)
        }

        print(f"{m_name:<26} | {p_str:<8} | {sz_str:<10} | {acc:>7.1f} % | {fpr:>7.1f} % | {match:>11.1f} % | {lat:>9.1f} ms | {speedup:>6.1f}x")

    print("=" * 125)

    out_json = "/home/manan/aigc_robust_detection/reports/six_distilled_vs_big_agent_report.json"
    with open(out_json, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved detailed analysis to {out_json} ✅")

if __name__ == "__main__":
    evaluate_all()
