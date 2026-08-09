# Run commands

Full commands for every workflow. The README has the short version. Every script takes `--help`, and the config-driven ones take `--config path/to/config.yaml`.

## Smoke test

Tiny generated dataset. It checks the wiring, nothing more. The numbers mean nothing.

```bash
python scripts/generate_synthetic_data.py --config configs/config.yaml
python scripts/make_internal_split.py     --config configs/config.yaml
python scripts/verify_repo_ready.py       --config configs/config.yaml
python scripts/train.py                   --config configs/config.yaml --sparse_rates 0.05 --sparse_seeds 11
```

Or in one go:

```bash
bash scripts/run_smoke_pipeline.sh
```

Score the run it produced:

```bash
RUN_DIR=$(ls -dt results/wallpath_pi_synthetic/run_* | head -1)
python scripts/evaluate.py --run_dir "$RUN_DIR" --csv val_split.csv
```

## ICASSP 2025 dense maps

Convert the release. Use `--dry-run` first to see what the converter found without writing anything.

```bash
python scripts/converters/convert_icassp2025_indoor.py --raw-root data/icassp2025_indoor_raw --dry-run
python scripts/converters/convert_icassp2025_indoor.py --raw-root data/icassp2025_indoor_raw \
  --task 1 --out-root data/icassp2025_indoor_converted_task1_full --overwrite
```

The converter builds the material raster from the reflectance and transmittance channels and clips a few out-of-bounds transmitter coordinates. Both are explained in `preprocessing_icassp2025.md`.

Build the B1-B20 / B21-B25 split once and check it:

```bash
python scripts/make_internal_split.py --config configs/config_icassp2025_task1_final_sparse_seed11.yaml
python scripts/verify_repo_ready.py   --config configs/config_icassp2025_task1_final_sparse_seed11.yaml
```

Task 1 benchmark. Each seed config sweeps all four sparse rates (0.005, 0.01, 0.05, 0.10) in one run.

```bash
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed11.yaml
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed22.yaml
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed33.yaml
```

Aggregate the three seeds into the mean and standard deviation table. The script groups by method and sparse rate.

```bash
python scripts/analysis/summarize_sparse_seed_runs.py \
  results/wallpath_pi_task1_final_sparse_seed*/run_1 \
  --out-dir results/summaries/task1_final
```

Confidence intervals on a single run:

```bash
python scripts/analysis/bootstrap_metric_ci.py --run_dir "$RUN_DIR" --metric rmse --n_boot 200
```

## Frozen transfer to Tasks 2 and 3

The transfer reuses the 1 percent Task 1 model, so it needs `models/*.joblib` from a training run. Those are written locally and not committed, so train the source run first.

```bash
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed11.yaml
```

Then point the transfer at the run directory that training printed:

```bash
python scripts/analysis/evaluate_transfer.py \
  --source-run results/wallpath_pi_task1_final_sparse_seed11/run_1 \
  --target-config configs/config_icassp2025_task2_transfer_unseen_buildings.yaml \
  --out-dir results/transfer/task1_to_task2_unseen_buildings
```

Swap in `config_icassp2025_task3_transfer_unseen_buildings.yaml` for Task 3, and the seed 22 and seed 33 source runs for the other two seeds.

Summarize both targets:

```bash
python scripts/analysis/summarize_task2_task3_transfer.py --help
python scripts/analysis/analyze_transfer_statistics.py --help
```

## Measured 3.5 GHz

Converted table:

```bash
python scripts/converters/convert_measured_3p5ghz.py --raw-root data/measured_3p5ghz_raw --out-root data/measured_3p5ghz_processed
python scripts/analysis/evaluate_measured_3p5ghz.py --config configs/config_measured_3p5ghz.yaml
```

Or set `MEASURED_RAW_ROOT` and run `bash scripts/run_measured_validation_pipeline.sh`.

Point-wise external validation, leave-one-scenario-out:

```bash
python scripts/analysis/evaluate_external_3p5ghz.py \
  --data-root data/external/3p5ghz_measured \
  --out-dir results/external_3p5ghz_measured_validation \
  --split leave_scenario_out
```

Few-shot variant. It reveals a small fraction of target-scenario anchors and scores only the non-anchor points.

```bash
python scripts/analysis/evaluate_external_3p5ghz.py \
  --data-root data/external/3p5ghz_measured \
  --out-dir results/external_3p5ghz_measured_validation_fewshot \
  --split leave_scenario_fewshot --anchor-fractions 0.01 0.05 0.10 --anchor-seeds 11 22 33
```

Fixed anchor counts instead of fractions:

```bash
python scripts/analysis/evaluate_external_3p5ghz.py \
  --data-root data/external/3p5ghz_measured \
  --out-dir results/external_3p5ghz_measured_validation_fewshot_counts \
  --split leave_scenario_fewshot --anchor-counts 5 10 20 --anchor-seeds 11 22 33
```

## Tables

```bash
python scripts/analysis/generate_paper_tables.py --help
python scripts/analysis/summarize_external_3p5ghz_grouped.py --help
python scripts/analysis/summarize_external_3p5ghz_fewshot.py --help
```

## Leakage audit

```bash
python scripts/analysis/audit_leakage.py --help
pytest tests/test_audit_leakage.py tests/test_splits.py
```
