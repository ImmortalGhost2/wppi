# WallPath-PI

Code for the paper *WallPath-PI: Propagation-Guided Sparse Residual Correction for Wall-Aware Indoor Path-Loss Maps*.

The idea is simple. Fit a multi-wall propagation model to whatever sparse measurements you have in a building, then train a tree ensemble to predict only what that model gets wrong. At transfer time the learned model stays frozen while the propagation prior and the anchor-derived features are recomputed for the new building.

## Layout

```
src/wallpath_pi/     library code: geometry, features, physics, models, evaluation
scripts/             entry points for training, transfer, analysis
configs/             the YAML configs used for the reported runs
tests/               unit tests plus the leakage and split audits
docs/                data contract, preprocessing, split protocol, run commands
paper_artifacts/     the tables and figures that appear in the paper
```

Raw datasets are not included. See `THIRD_PARTY_DATA.md` for where to get them and what the licence allows.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Check it works

This runs the whole pipeline on generated data in a couple of minutes. The numbers are meaningless, it only proves the wiring is intact.

```bash
bash scripts/run_smoke_pipeline.sh
pytest
```

## Reproduce the paper

You need the ICASSP 2025 indoor path-loss release first. Convert it, make the split, then run the three seeds.

```bash
python scripts/converters/convert_icassp2025_indoor.py --help
python scripts/make_internal_split.py --help

python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed11.yaml
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed22.yaml
python scripts/train.py --config configs/config_icassp2025_task1_final_sparse_seed33.yaml
```

Frozen transfer to Tasks 2 and 3:

```bash
python scripts/train.py --config configs/config_icassp2025_task2_transfer_unseen_buildings.yaml
python scripts/train.py --config configs/config_icassp2025_task3_transfer_unseen_buildings.yaml
```

Measured 3.5 GHz validation:

```bash
python scripts/analysis/evaluate_external_3p5ghz.py \
  --data-root data/external/3p5ghz_measured \
  --split leave_scenario_out
```

Every command with its full argument list is in `docs/RUN_COMMANDS.md`.

## Splits

Buildings B1 to B20 train, B21 to B25 evaluate. No building appears on both sides. Sparse anchor masks are fixed by seed, so all methods in a run see exactly the same anchors. Dense labels away from the anchors are only ever touched during scoring. `docs/experimental_splits_icassp2025.md` has the details and `tests/` has the audits that enforce it.

## What this is not

It is not a physics-informed neural network, not a field solver, and not a leaderboard entry. The wall and interface features come from the reflectance and transmittance channels, so they are material proxies rather than annotated material types, see `docs/preprocessing_icassp2025.md`. Results are on a held-out split of the public release, not on the challenge blind test.

## Citing

The paper is under review. Citation details will go here once it is out.

## Licence

MIT, see `LICENSE`. Datasets keep their own terms, see `THIRD_PARTY_DATA.md`.
