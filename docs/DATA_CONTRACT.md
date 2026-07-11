# Data contract

Two kinds of input, two contracts.

Dense maps come as a CSV index plus one compressed NPZ per transmitter. Each NPZ holds a full `H x W` path-loss raster and the geometry that goes with it. The measured 3.5 GHz release is different: discrete point measurements, no rasters, so it gets its own tidy table instead. Keeping them apart means the training and evaluation code works on any dense-map dataset without special cases, and the measured campaign is scored separately.

The loader takes a few column and array aliases so data prepared by different tools loads without renaming. Arrays are read with `np.load(..., allow_pickle=False)`, so the files are safe to pass around.

## Dense-map index (CSV)

One row per transmitter map. Three columns are required, the rest are read from the NPZ if absent.

| Column | Requirement | Description |
| --- | --- | --- |
| `scene_id` | Required | Building or layout identifier. This is the grouping key for scene-disjoint splitting. |
| `sample_id` | Required | Unique transmitter or map identifier. |
| `scene_path` | Required | Path to the NPZ, relative to `paths.data_root`. `map_path` is accepted as an alias. |
| `tx_x`, `tx_y` | Recommended | Transmitter pixel coordinates, used when the NPZ has none. |
| `frequency_hz` | Recommended | Carrier frequency in hertz. |
| `resolution_m` or `cell_size_m` | Recommended | Metres per pixel. |
| `antenna_id`, `split`, `notes` | Optional | Metadata, carried through untouched. |

## Scene arrays (NPZ)

Every raster in one file shares the same `H x W` grid. Masks are boolean or 0/1 integer. The transmitter position, frequency and resolution can come from either the NPZ or the manifest row.

| Array | Requirement | Description |
| --- | --- | --- |
| `path_loss` or `path_loss_db` | Required | `H x W` path loss in dB. Aliases `target`, `target_db`. |
| `valid_mask` | Required | `H x W` receiver-valid mask. Only valid cells are scored. Aliases `mask`, `receiver_mask`. |
| `material_map` | Required | `H x W` integer material raster, `0` for free space, positive integers for walls and obstacles. Aliases `materials`, `building_map`. |
| `tx_xy`, or scalar `tx_x` and `tx_y` | Required | Transmitter pixel coordinate. |
| `frequency_hz` | Required | Carrier frequency in hertz. |
| `resolution_m` or `cell_size_m` | Required | Metres per pixel. |
| `wall_mask` | Optional | `H x W` obstacle mask. Derived as `material_map > 0` when missing. |
| `transmittance` | Optional | `H x W` channel, summed along the transmitter ray for line-integral features. |
| `reflectance` | Optional | `H x W` channel, summed along the transmitter ray for line-integral features. |
| `scene_id`, `sample_id` | Optional | Scalar strings, used when the manifest does not supply them. |

## Material identifiers

The default config uses `[1, 2, 3]`, which the synthetic generator reads as light, medium and heavy wall classes. For a real dataset, keep the original mapping in a separate metadata file and set `dataset.material_ids` to match, otherwise per-material wall counts lose their meaning.

## Point-level measured data

`scripts/converters/convert_measured_3p5ghz.py` flattens the raw CSVs into one table, `measured_points.csv`.

| Field group | Columns |
| --- | --- |
| Identity | `scenario`, `campaign_id`, `tx_id`, `rx_id` |
| Measurement | `measured_path_loss_db`, `distance_m`, `frequency_hz`, `los_nlos` |
| Wall counts | `brick_wall_count`, `wood_wall_count`, `glass_wall_count`, `drywall_count`, `column_count` |
| Provenance | `source_file` |

Original columns that do not map onto this schema are kept as they are. A column whose name would clash with a canonical field is kept under an `extra_` prefix. Only `scripts/analysis/evaluate_measured_3p5ghz.py` reads this table. It never touches the dense-map pipeline.
