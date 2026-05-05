#!/usr/bin/env bash
# Download Amazon Reviews 2023
# Source: https://amazon-reviews-2023.github.io/  (McAuley Lab, UCSD)
#
# Usage:
#   bash scripts/download_amazon.sh                    # Video_Games — 917 MB download, 4.6M reviews
#   bash scripts/download_amazon.sh All_Beauty         # 90 MB download  — good for limited disk space
#   bash scripts/download_amazon.sh Musical_Instruments # 437 MB download

set -euo pipefail

CATEGORY="${1:-Video_Games}"
BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
OUT="data/raw/amazon"

mkdir -p "$OUT"

echo "Downloading Amazon Reviews 2023 — $CATEGORY"

echo "[1/2] Reviews..."
curl -L --progress-bar \
  "$BASE/review_categories/${CATEGORY}.jsonl.gz" \
  -o "$OUT/${CATEGORY}.jsonl.gz"

echo "[2/2] Metadata..."
curl -L --progress-bar \
  "$BASE/meta_categories/meta_${CATEGORY}.jsonl.gz" \
  -o "$OUT/meta_${CATEGORY}.jsonl.gz"

echo ""
echo "Done. Files saved to $OUT/"
echo "  Reviews  : $OUT/${CATEGORY}.jsonl.gz"
echo "  Metadata : $OUT/meta_${CATEGORY}.jsonl.gz"
echo ""
echo "Next: python main.py --dataset amazon --category $CATEGORY"
