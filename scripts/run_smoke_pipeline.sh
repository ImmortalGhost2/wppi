#!/usr/bin/env bash
#
# Synthetic smoke pipeline.
#
# Reproduces the first end-to-end WallPath-PI result on the bundled synthetic
# data: generate data, build a scene-disjoint split, verify the repository,
# train one sparse-rate/seed debug experiment, render plots, and run the tests.
#
# WARNING: synthetic smoke results are a wiring check, NOT paper evidence.
#
# Usage:  bash scripts/run_smoke_pipeline.sh
set -euo pipefail

# Run from the repository root regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/config.yaml"
EXPERIMENT="wallpath_pi_synthetic"

step() { printf '\n========== %s ==========\n' "$1"; }

step "1/6 Generate synthetic data"
python scripts/generate_synthetic_data.py --config "${CONFIG}"

step "2/6 Make scene-disjoint train/val split"
python scripts/make_internal_split.py --config "${CONFIG}"

step "3/6 Verify repository is ready"
python scripts/verify_repo_ready.py --config "${CONFIG}"

step "4/6 Train debug experiment (sparse_rate=0.05, seed=11)"
python scripts/train.py --config "${CONFIG}" --sparse_rates 0.05 --sparse_seeds 11

RUN_DIR="$(ls -dt "results/${EXPERIMENT}"/run_* 2>/dev/null | head -1 || true)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: no run directory found under results/${EXPERIMENT}." >&2
  exit 1
fi

step "5/6 Evaluate and generate plots"
python scripts/evaluate.py --run_dir "${RUN_DIR}" --csv val_split.csv
python scripts/analysis/plot_sparse_curves.py --run_dir "${RUN_DIR}"
python scripts/analysis/plot_model_comparison.py --run_dir "${RUN_DIR}" --sparse_rate 0.05
python scripts/analysis/plot_prediction_maps.py --run_dir "${RUN_DIR}"
python scripts/analysis/plot_feature_importance.py --run_dir "${RUN_DIR}"
python scripts/analysis/bootstrap_metric_ci.py --run_dir "${RUN_DIR}" --metric rmse --n_boot 200

step "6/6 Run test suite"
python -m pytest -q

step "Smoke pipeline complete"
echo "Outputs are saved under: ${RUN_DIR}"
echo "  - Aggregate metrics : ${RUN_DIR}/final_evaluation_results.csv"
echo "  - Per-sample metrics : ${RUN_DIR}/per_sample_metrics.csv"
echo "  - Evaluation arrays  : ${RUN_DIR}/eval_outputs_primary.npz"
echo "  - Bootstrap RMSE CI  : ${RUN_DIR}/bootstrap_rmse_ci.csv"
echo "  - Figures            : ${RUN_DIR}/plots/"
echo
echo "NOTE: synthetic smoke results are a wiring check, not paper evidence."
