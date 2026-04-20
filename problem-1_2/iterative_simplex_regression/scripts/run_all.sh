#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-../../../data/limestone_data_challenge_2026.data.csv}"
OUTDIR="${2:-../results}"

python 01_validate_known_decomps.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 02_stability_validation.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 03_farmer_only_stepwise.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 04_fill_and_rerun.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 05_exact_fill_all.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 06_refine_col30_col46.py --csv "$CSV_PATH" --outdir "$OUTDIR"
python 07_classify_remaining.py --csv "$OUTDIR/filled_matrix.csv" --outdir "$OUTDIR"
