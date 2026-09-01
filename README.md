# ENTROPY

ENTROPY is an image-authenticity workstation for examining camera images,
localized AI edits, and fully synthetic imagery. The public interface is served
at `techjam.manansethia.com`; analysis is provided by the separate API origin.

## Current model state

The deployed service uses the audited C0 forensic model. HighCap FP16 is a
96,590,564-parameter standalone candidate that now strictly loads and performs
real-image inference at its required 384px resolution. Its release status is
evaluation pending; its measured performance must be published before it is
advertised as an equivalent replacement.

## Data governance

Candidate training sources are inventoried with provenance. The organizer
benchmark remains excluded from training, calibration, threshold selection, and
model promotion.

## Project author

Manan Sethia
