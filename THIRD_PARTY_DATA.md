# Third-party data

The code in this repository is MIT licensed. The datasets it consumes are not ours and are not redistributed here. You have to fetch them yourself.

## ICASSP 2025 Indoor Pathloss Radio Map Prediction Challenge

The dense simulated maps used for Tasks 1, 2 and 3.

- Record: <https://dx.doi.org/10.21227/c0ec-cw74> (IEEE DataPort, "Indoor Radio Map Dataset")
- Challenge: <https://IndoorRadioMapChallenge.github.io/>

Use it under the terms stated on the IEEE DataPort record.

## Measured 3.5 GHz indoor path loss

The point measurements used for the external validation.

- Title: Path Loss Dataset for Fifth Generation of Wireless Communications in Indoor
- Creators: Perdomo-Reyes, P.; Galvan-Tejada, G. M.; Meneses-Viveros, A.
- Record: <https://doi.org/10.17605/OSF.IO/T9EDP> (Open Science Framework)
- Files used: `PL_Comms_C1.csv`, `PL_Comms_C2.csv`, `PL_Library_C1.csv`, `PL_Library_C2.csv`, `PL_SSE_C1.csv`, `PL_SSE_C2.csv`

The OSF record is CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). If you redistribute the data or values derived from it, credit the creators, cite the DOI, link the licence, and say what you changed.

One thing worth spelling out: the CC BY 4.0 licence applies to the OSF dataset, not to any journal article associated with it. The article is a separate work under its own, possibly more restrictive, terms. This work uses only the dataset.

## Derived values in this repository

The aggregate numbers in `paper_artifacts/final_manuscript/tables/table4_*` and `table5_*` are computed from the measured 3.5 GHz dataset. They are summary statistics, not source rows, but redistributing them still falls under CC BY 4.0.
