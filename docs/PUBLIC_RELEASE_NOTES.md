# AIGC Forensics release notes

## Product integrity

The public interface exposes model status, forensic results, and reproducible evidence without disclosing infrastructure details. A verdict is returned only when a verified model is loaded; the service does not substitute heuristics or demonstration scores for an unavailable model.

## Evidence returned for an upload

- calibrated model probability and classification
- EXIF, XMP, IPTC, and C2PA inspection where present
- frequency-spectrum, SRM residual, and localization heatmap artifacts
- SHA-256 file identifier and processing timeline

Analysis results are retained as session records for a limited server-defined period. Original uploads are processed transiently and are not retained by this session mechanism.

## Training and evaluation

Candidate training runs require a content-hash manifest and an official benchmark hash list. A run is blocked if an overlap is detected or if the selected pipeline contains synthetic data, tensor, or logit fallbacks. Final performance claims are published only after independent reconciliation of the selected checkpoint, data split, and held-out evaluation.
