# Splits and transfer protocol

Map counts, the fixed scene-disjoint split, and how the cross-task transfer is scored. All numbers here were checked against the converted manifests.

## The split

Every ICASSP 2025 experiment uses the same fixed split:

- Train on B1 to B20, twenty buildings.
- Evaluate on B21 to B25, five buildings.
- Same lists for Task 1, Task 2 and Task 3.
- No building on both sides.

`scripts/make_internal_split.py` builds it from the explicit `split.train_scenes` and `split.val_scenes` lists in each config, so there is no seed involved and it comes out the same every time.

## Why building-level and not random

All 50, 150 or 1110 maps of a building share one floor plan. Split those maps randomly and you put near-identical geometry on both sides of the boundary, which leaks building-level information into the evaluation and inflates the score. Holding out whole buildings avoids that, and the reported error then measures generalization to layouts the model has never seen.

Inside a map there is a second rule. Before prediction, only the path-loss labels at the declared sparse-anchor pixels are readable. Dense labels at the remaining evaluation pixels are used for scoring and nothing else. Building disjointness plus sparse-anchor observation is the whole protocol.

## Sizes

| Task | Antennas | Frequencies | Maps/building | Total | Train (B1-B20) | Eval (B21-B25) |
|------|----------|-------------|--------------:|------:|---------------:|---------------:|
| 1 | 1 (Ant1) | 1 (f1) | 50 | 1250 | 1000 | 250 |
| 2 | 1 (Ant1) | 3 (f1-f3) | 150 | 3750 | 3000 | 750 |
| 3 | 5 (Ant1-Ant5) | 3 (f1-f3) | 1110 | 27750 | 22200 | 5550 |

Where those come from:

- Task 1: `25 x 1 x 1 x 50 = 1250`, so 50 per building, 1000 train and 250 eval.
- Task 2: `25 x 1 x 3 x 50 = 3750`, so 150 per building, 3000 train and 750 eval.
- Task 3: Ant1 gives `1 x 3 x 50 = 150` and Ant2 to Ant5 give `4 x 3 x 80 = 960`, so 1110 per building, `25 x 1110 = 27750` total, 22200 train and 5550 eval.

## Making the split

Configs carry the scene lists directly:

```yaml
split:
  group_column: scene_id
  train_scenes: [B1, ..., B20]
  val_scenes:   [B21, B22, B23, B24, B25]
```

```bash
python scripts/make_internal_split.py --config configs/config_icassp2025_task2_transfer_unseen_buildings.yaml
```

When both lists are present the splitter assigns by those lists and checks three things: the lists are disjoint, every listed scene exists in `manifest.csv`, and together they cover every scene in the manifest. So each map lands in exactly one side and nothing is dropped or duplicated.

## Cross-task transfer

The transfer takes a Task 1 model, trained on B1 to B20 at 1 percent sparse anchors, freezes it, and evaluates it on Task 2 and Task 3 maps with no retraining. `scripts/analysis/evaluate_transfer.py` does this.

Since B21 to B25 sit in every task's `val_split.csv`, the transfer configs just set `eval_csv: val_split.csv`. That makes the evaluation cross-task and unseen-building at once: Task 2 adds frequencies f2 and f3, Task 3 adds antennas Ant2 to Ant5 on top, and none of the five evaluation buildings were ever trained on.

| Task | Buildings | Maps |
|------|-----------|-----:|
| 2 | B21-B25 | 750 |
| 3 | B21-B25 | 5550 |

The transfer needs a serialized source model at `<source-run>/models/*.joblib`. Training writes those locally. They are not committed, so from a fresh clone you train the source run first and then point `--source-run` at the directory it prints. Commands are in `RUN_COMMANDS.md`.

Only the analytic baseline and the anchor-derived components are refitted on the target maps. The learned estimator weights are untouched, and `run_summary.json` records `retrained = false` along with both dataset names. The Task 2 and Task 3 `train_split.csv` files go unused.

Split CSVs (`train_split.csv`, `val_split.csv`, `split_meta.json`, `split_summary.json`) are written under `data/`, which is gitignored. Regenerate them with the command above.
