# ICASSP 2025 preprocessing

What `scripts/converters/convert_icassp2025_indoor.py` does to the raw release on the way into the NPZ schema described in `DATA_CONTRACT.md`. Two things need explaining: how the material raster is built, and why a few transmitter coordinates get clipped.

## Where the data lives

The raw release goes in `data/icassp2025_indoor_raw` and the converter writes to `data/icassp2025_indoor_converted_task1_full`. Both sit under `data/`, which is gitignored, so neither is redistributed here and the converted tensors are just a cache you can rebuild at any time.

```bash
python scripts/converters/convert_icassp2025_indoor.py \
  --raw-root data/icassp2025_indoor_raw --task 1 \
  --out-root data/icassp2025_indoor_converted_task1_full --overwrite
```

## Material classes are proxies, not labels

The release gives normal-incidence reflectance and transmittance channels. It does not give material categories. So the converter builds `material_map` itself: for each wall pixel it takes the interface strength `|reflectance| + |transmittance|` and bins it into tertiles, 1 for weak, 2 for medium, 3 for strong.

That means the `mat_*_count` features downstream count interface-strength classes accumulated along the transmitter-to-receiver ray. They are not brick, glass and drywall counts. Calling them material types would overstate what the data supports, which is why the paper and the README both say material proxies.

## Out-of-bounds transmitter coordinates

A handful of the official Task 2 and Task 3 transmitter coordinates land just outside the raster. Before changing anything in the converter we audited every coordinate with `scripts/analysis/audit_icassp_tx_bounds.py`.

| Task | Tx coordinates | Out of bounds | Beyond 1 px | Worst overflow |
|------|---------------:|--------------:|------------:|----------------|
| 2 | 3750 | 17 | 6 | 5 px (1.25 m) |
| 3 | 27750 | 273 | 170 | 5 px (1.25 m) |

The worst case is 5 pixels, which at 0.25 m per pixel is 1.25 m. Almost all of them are on the bottom and left edges. Two examples:

- Task 2 `B19_Ant1_f2_S33`: `tx_xy = (-3, 165)` on a 180x219 map, 3 px past the left edge.
- Task 3 `B1_Ant2_f1_S74`: `tx_xy = (277, 468)` on a 348x464 map, 5 px past the bottom edge.

The pattern matters. If the coordinates had been axis-swapped we would expect overflows scattered across both axes with large magnitudes. Instead they cluster at two edges and never exceed 5 px, which reads as rounding in the official position tables. So the fix is a clip, not a transpose.

## What the converter does about it

The official convention is kept exactly as published, `tx_x = row["Y"]` and `tx_y = row["X"]`. On top of that, per axis:

1. In-bounds coordinates are left alone.
2. A coordinate up to `--tx-boundary-tol` pixels outside is snapped to the nearest valid index, so `-3` becomes `0` and `132` becomes `131` on a height-132 axis.
3. Anything further out raises `ValueError`.

The third rule is the point of the whole design. The tolerance defaults to 5 px, which covers every audited case and leaves no room for anything else, so a genuinely broken coordinate fails loudly instead of being quietly moved. Task 1 has no out-of-bounds coordinates at all, so none of this touches it.

The conversion summary tells you what happened:

```
Tx boundary corrections : 17 coordinate(s) clipped within tolerance 5 px; 0 exceeded tolerance.
```

Pass `--tx-boundary-tol 1` for a stricter run, or `--tx-boundary-tol 0` to switch the correction off and reject every out-of-bounds coordinate.

## Re-running the audit

```bash
python scripts/analysis/audit_icassp_tx_bounds.py \
  --raw-root data/icassp2025_indoor_raw --task 2 --out-dir results/diagnostics
python scripts/analysis/audit_icassp_tx_bounds.py \
  --raw-root data/icassp2025_indoor_raw --task 3 --out-dir results/diagnostics
```

It writes one row per coordinate: sample id, antenna, frequency, raster size, the raw `tx_x` and `tx_y`, the per-edge overflow, and the clip distance in pixels and metres. The audit only reads, it never writes dataset files, and it is where the 5 px number came from.
