# External 3.5 GHz Generated Results

This directory tree preserves the complete generated outputs used for the
final independent 3.5 GHz validation of WallPath-PI.

## Directory roles

- `results/external_3p5ghz_*` contains the original run, prediction,
  preflight, grouped-summary, few-shot, and integrity outputs.
- `paper_artifacts/external_3p5ghz/` contains curated copies used for
  manuscript tables, reporting, and automated artifact verification.

Some files are therefore deliberately duplicated byte-for-byte. The copies
under `results/` preserve the complete output structure produced by the
evaluation scripts; the copies under `paper_artifacts/` form the compact
publication-facing evidence package.

## Row-level outputs

The `per_sample_metrics.csv` files contain predictions, errors, split
metadata, and source-row identifiers. They also repeat target path-loss
values needed to verify the reported errors. Original source dataset files
are not included in this repository.

Dataset source:

- Perdomo-Reyes et al., *Path Loss Dataset for Fifth Generation of Wireless
  Communications in Indoor*, DOI: 10.17605/OSF.IO/T9EDP.

This is third-party data licensed CC BY 4.0 (distinct from the associated
article's licence). See [`THIRD_PARTY_DATA.md`](../THIRD_PARTY_DATA.md) for
attribution and redistribution terms before redistributing any dataset-derived
artifact in this directory.

## Integrity

`external_3p5ghz_results_manifest.json` records the path, byte size, SHA-256
hash, classification, and any exact publication-artifact counterpart for
every archived generated file.

## Hash basis

The primary `bytes`, `sha256`, and aggregate values in
`external_3p5ghz_results_manifest.json` describe the exact blobs committed
to Git after normal text-file line-ending normalisation.

Each entry also records `source_working_tree_bytes` and
`source_working_tree_sha256`, identifying the original Windows-generated
file before Git converted CRLF line endings to LF. This conversion changes
only line endings; it does not change predictions, metrics, identifiers, or
scientific results.
