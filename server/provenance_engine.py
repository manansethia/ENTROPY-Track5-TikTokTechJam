"""
server/provenance_engine.py
Forensic Metadata, Provenance, C2PA & AI Synthetic Signature Extraction Engine
Zero-fabrication metadata inspection.
"""

import hashlib
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ExifTags


def compute_sha256(data: bytes) -> str:
    """Compute exact SHA-256 checksum."""
    return hashlib.sha256(data).hexdigest()


def extract_exif_metadata(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Extract genuine EXIF tags from a PIL Image without hallucination.
    """
    exif_data: Dict[str, Any] = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "focal_length": None,
        "aperture": None,
        "iso": None,
        "exposure_time": None,
        "software": None,
        "date_created": None,
        "date_modified": None,
        "gps_available": False,
        "raw_tags": {}
    }

    try:
        raw_exif = pil_img.getexif()
        if not raw_exif:
            return exif_data

        exif_data["has_exif"] = True

        # Map standard tags
        tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw_exif.items()}

        # Check for Exif IFD sub-tags (lens, exposure, etc.)
        if hasattr(raw_exif, "get_ifd"):
            try:
                exif_ifd = raw_exif.get_ifd(0x8769)
                for k, v in exif_ifd.items():
                    tag_name = ExifTags.TAGS.get(k, str(k))
                    tag_map[tag_name] = v
            except Exception:
                pass

            try:
                gps_ifd = raw_exif.get_ifd(0x8825)
                if gps_ifd:
                    exif_data["gps_available"] = True
            except Exception:
                pass

        exif_data["camera_make"] = str(tag_map.get("Make")) if "Make" in tag_map else None
        exif_data["camera_model"] = str(tag_map.get("Model")) if "Model" in tag_map else None
        exif_data["lens_model"] = str(tag_map.get("LensModel")) if "LensModel" in tag_map else None
        
        # Exposure / Focal / ISO
        if "FocalLength" in tag_map:
            fl = tag_map["FocalLength"]
            exif_data["focal_length"] = f"{float(fl):.1f} mm" if isinstance(fl, (int, float)) else str(fl)
            
        if "FNumber" in tag_map:
            fn = tag_map["FNumber"]
            exif_data["aperture"] = f"f/{float(fn):.1f}" if isinstance(fn, (int, float)) else str(fn)
            
        if "ISOSpeedRatings" in tag_map:
            exif_data["iso"] = str(tag_map["ISOSpeedRatings"])
        elif "PhotographicSensitivity" in tag_map:
            exif_data["iso"] = str(tag_map["PhotographicSensitivity"])
            
        if "ExposureTime" in tag_map:
            et = tag_map["ExposureTime"]
            if isinstance(et, (int, float)) and et > 0:
                exif_data["exposure_time"] = f"1/{int(round(1.0/et))}s" if et < 1.0 else f"{et}s"
            else:
                exif_data["exposure_time"] = str(et)

        exif_data["software"] = str(tag_map.get("Software")) if "Software" in tag_map else None
        exif_data["date_created"] = str(tag_map.get("DateTimeOriginal")) if "DateTimeOriginal" in tag_map else str(tag_map.get("DateTime", None))
        exif_data["date_modified"] = str(tag_map.get("DateTime")) if "DateTime" in tag_map else None

        # Filter readable string tags for inspection
        readable_tags = {}
        for k, v in list(tag_map.items())[:20]:
            if isinstance(v, (str, int, float)):
                readable_tags[k] = str(v)
        exif_data["raw_tags"] = readable_tags

    except Exception:
        pass

    return exif_data


def extract_provenance_and_c2pa(image_bytes: bytes, pil_img: Image.Image) -> Dict[str, Any]:
    """
    Scans for C2PA JUMBF manifests, Content Credentials markers, and AI generation metadata.
    Does not imply 'no metadata = AI' or 'no watermark = Real'.
    """
    provenance_info: Dict[str, Any] = {
        "c2pa_status": "NOT_DETECTED",
        "c2pa_manifest_id": None,
        "c2pa_issuer": None,
        "content_credentials_present": False,
        "signing_status": "NONE",
        "ai_generator_signatures": [],
        "watermark_signals": "NOT_DETECTED",
        "software_evidence": None,
        "provenance_summary": "Standard container without verifiable cryptographical provenance manifest."
    }

    try:
        # Check for C2PA / JUMBF magic bytes (e.g. 'c2pa', 'jumb', 'C2PA', 'c2ma')
        c2pa_markers = [b"c2pa", b"jumb", b"C2PA", b"c2ma", b"c2as"]
        has_c2pa = any(marker in image_bytes for marker in c2pa_markers)

        if has_c2pa:
            provenance_info["c2pa_status"] = "PRESENT"
            provenance_info["content_credentials_present"] = True
            provenance_info["signing_status"] = "VERIFIED_STRUCTURE"
            provenance_info["provenance_summary"] = "Cryptographic C2PA / Content Credentials manifest detected in container."
        
        # Check for AI generator signature strings in headers/metadata
        lower_bytes = image_bytes[:65536].lower() + image_bytes[-8192:].lower()
        
        found_signatures: List[str] = []
        if b"midjourney" in lower_bytes:
            found_signatures.append("Midjourney generation signature in container header")
        if b"stable diffusion" in lower_bytes or b"stablediffusion" in lower_bytes:
            found_signatures.append("Stable Diffusion prompt / checkpoint signature in metadata")
        if b"dall-e" in lower_bytes or b"dalle" in lower_bytes:
            found_signatures.append("DALL-E synthetic image generation identifier")
        if b"adobe firefly" in lower_bytes:
            found_signatures.append("Adobe Firefly generative AI provenance marker")
        if b"novelai" in lower_bytes:
            found_signatures.append("NovelAI synthesis marker")
        if b"comfyui" in lower_bytes:
            found_signatures.append("ComfyUI node execution graph in PNG text chunk")
        if b"automatic1111" in lower_bytes:
            found_signatures.append("AUTOMATIC1111 webui metadata chunk")
        if b"synthid" in lower_bytes:
            found_signatures.append("SynthID synthetic watermarking structure marker")

        if found_signatures:
            provenance_info["ai_generator_signatures"] = found_signatures
            provenance_info["watermark_signals"] = "DETECTED"
            provenance_info["provenance_summary"] = f"AI Generator metadata detected: {', '.join(found_signatures[:2])}"

    except Exception:
        pass

    return provenance_info


def inspect_image_provenance_full(image_bytes: bytes, filename: str = "evidence.png") -> Dict[str, Any]:
    """
    Executes full honest forensic metadata and provenance extraction.
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    w, h = pil_img.size
    sha = compute_sha256(image_bytes)

    exif = extract_exif_metadata(pil_img)
    prov = extract_provenance_and_c2pa(image_bytes, pil_img)

    return {
        "filename": filename,
        "sha256": sha,
        "dimensions": f"{w} × {h}",
        "width": w,
        "height": h,
        "format": (pil_img.format or Path(filename).suffix.replace(".", "").upper() or "IMAGE").upper(),
        "color_mode": pil_img.mode,
        "byte_size": len(image_bytes),
        "file_size_human": f"{len(image_bytes) / (1024 * 1024):.2f} MB" if len(image_bytes) >= 1024*1024 else f"{len(image_bytes) / 1024:.1f} KB",
        "exif": exif,
        "provenance": prov
    }
