#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
LOCK="$AI_ROOT/validation_LOCKED"
mkdir -p "$LOCK"

cat > "$LOCK/DO_NOT_TRAIN.txt" <<'TXT'
CHALLENGE VALIDATION DATA — DO NOT TRAIN ON THIS DIRECTORY.

Required benchmark:
  Real: COCO val2017, 4,998 images
  Fake: WildFake DALL-E Advanced, 8,843 images

This directory is intentionally separate from datasets/.
Training scripts should refuse paths under validation_LOCKED.
TXT

chmod -R a-w "$LOCK" || true
printf 'Validation lock prepared at %s\n' "$LOCK"
printf 'Place the exact challenge-provided COCO/WildFake subsets here only after download/translation instructions are satisfied.\n'
