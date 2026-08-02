#!/usr/bin/env bash
#
# LEGACY / NON-PAPER / HISTORICAL COMPATIBILITY ONLY.
# This is not the canonical paper-results pipeline.
# Measured 3.5 GHz point-level validation pipeline.
#
# Converts the measured 3.5 GHz indoor CSVs to a point table (if needed) and runs
# leakage-safe grouped cross-validation. This is standalone and intentionally NOT
# part of the dense radio-map pipeline.
#
# Prerequisite: place the raw measured CSV files under a directory and either
# point MEASURED_RAW_ROOT at it or pass the default below. Conversion command:
#   python scripts/converters/convert_measured_3p5ghz.py \
#       --raw-root <RAW_CSV_DIR> --out-root data/measured_3p5ghz_processed
#
# Usage:  bash scripts/run_measured_validation_pipeline.sh
set -euo pipefail

# Run from the repository root regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/config_measured_3p5ghz.yaml"
PROCESSED_CSV="data/measured_3p5ghz_processed/measured_points.csv"
OUT_DIR="results/measured_3p5ghz_validation"
MEASURED_RAW_ROOT="${MEASURED_RAW_ROOT:-data/measured_3p5ghz_raw}"

step() { printf '\n========== %s ==========\n' "$1"; }

step "1/2 Convert measured CSVs if needed"
if [[ -f "${PROCESSED_CSV}" ]]; then
  echo "Processed table already exists: ${PROCESSED_CSV}"
else
  if [[ ! -d "${MEASURED_RAW_ROOT}" ]]; then
    echo "ERROR: ${PROCESSED_CSV} not found and raw CSV directory is missing." >&2
    echo "Place the measured 3.5 GHz CSV files under a directory, then either set" >&2
    echo "MEASURED_RAW_ROOT to it or use the default '${MEASURED_RAW_ROOT}', and run:" >&2
    echo "  python scripts/converters/convert_measured_3p5ghz.py \\" >&2
    echo "      --raw-root ${MEASURED_RAW_ROOT} --out-root data/measured_3p5ghz_processed" >&2
    exit 1
  fi
  echo "Converting raw CSVs from ${MEASURED_RAW_ROOT}"
  python scripts/converters/convert_measured_3p5ghz.py \
    --raw-root "${MEASURED_RAW_ROOT}" \
    --out-root data/measured_3p5ghz_processed
fi

step "2/2 Run point-level validation"
python scripts/analysis/evaluate_measured_3p5ghz.py --config "${CONFIG}"

step "Measured validation pipeline complete"
echo "Outputs are saved under: ${OUT_DIR}"
echo "  - Cross-validation metrics and per-fold results live in this directory."
echo "  - Figures (if enabled) are written alongside the metric CSVs."
