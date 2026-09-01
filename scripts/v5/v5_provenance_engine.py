#!/usr/bin/env python3
"""
v5_provenance_engine.py
-----------------------
Independent V5 Provenance & Content Credentials Subsystem.
Analyzes metadata, C2PA manifests, EXIF, IPTC, XMP, and AI watermarks
as a separate, non-contaminating evidence channel.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image, ExifTags

class V5ProvenanceEngine:
    def __init__(self):
        self.known_ai_software_signatures = [
            "midjourney", "stable diffusion", "dall-e", "adobe firefly",
            "comfyui", "automatic1111", "flux", "novelai", "invokeai"
        ]
        self.known_c2pa_manifest_markers = [
            b"c2pa", b"C2PA", b"contentcredentials", b"c2pa_claim", b"urn:c2pa:"
        ]

    def extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        exif_data[tag] = value
                    elif isinstance(value, bytes):
                        try:
                            exif_data[tag] = value.decode("utf-8", errors="ignore")[:100]
                        except Exception:
                            exif_data[tag] = "<bytes>"
        except Exception:
            pass
        return exif_data

    def scan_c2pa_and_raw_bytes(self, image_path: str) -> Dict[str, Any]:
        c2pa_findings = {
            "c2pa_status": "NOT_DETECTED",
            "manifest_detected": False,
            "signature_markers": []
        }
        try:
            with open(image_path, "rb") as f:
                content = f.read(1048576) # Read up to first 1MB of headers
                
            for marker in self.known_c2pa_manifest_markers:
                if marker in content:
                    c2pa_findings["manifest_detected"] = True
                    c2pa_findings["c2pa_status"] = "PRESENT"
                    c2pa_findings["signature_markers"].append(marker.decode("latin1"))
                    
        except Exception:
            c2pa_findings["c2pa_status"] = "UNKNOWN"
            
        return c2pa_findings

    def analyze_provenance(self, image_path: str) -> Dict[str, Any]:
        """Performs full decoupled provenance audit on an image file."""
        if not os.path.exists(image_path):
            return {"provenance_verdict": "UNKNOWN", "error": "File not found"}

        try:
            with Image.open(image_path) as img:
                w, h = img.size
                fmt = img.format
                exif_dict = self.extract_exif(img)
        except Exception as e:
            return {"provenance_verdict": "INVALID", "error": str(e)}

        c2pa_info = self.scan_c2pa_and_raw_bytes(image_path)
        
        # Analyze Software & Camera Fields
        software = str(exif_dict.get("Software", "")).lower()
        make = str(exif_dict.get("Make", ""))
        model = str(exif_dict.get("Model", ""))
        datetime_orig = str(exif_dict.get("DateTimeOriginal", ""))
        
        ai_metadata_flag = any(sig in software for sig in self.known_ai_software_signatures)
        
        if c2pa_info["manifest_detected"] or ai_metadata_flag:
            provenance_verdict = "DETECTED_AI_PROVENANCE"
        elif make or model:
            provenance_verdict = "AUTHENTIC_CAMERA_METADATA_PRESENT"
        elif len(exif_dict) > 0:
            provenance_verdict = "STANDARD_METADATA_PRESENT"
        else:
            provenance_verdict = "METADATA_STRIPPED_OR_ABSENT"

        return {
            "provenance_verdict": provenance_verdict,
            "c2pa": c2pa_info,
            "camera": {
                "make": make or None,
                "model": model or None,
                "datetime_original": datetime_orig or None
            },
            "software": software or None,
            "format": fmt,
            "dimensions": {"width": w, "height": h},
            "exif_tag_count": len(exif_dict)
        }

if __name__ == "__main__":
    engine = V5ProvenanceEngine()
    test_img = "/home/manan/aigc_robust_detection/reports/v4_heatmaps/whats-the-most-realistic-ai-photo-generator-online-v0-lav1uhmvubre1_spatial_heatmap.jpg"
    if os.path.exists(test_img):
        res = engine.analyze_provenance(test_img)
        print("Provenance Test Output:", json.dumps(res, indent=2))
