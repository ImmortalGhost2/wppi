# Table 4: External measured 3.5 GHz grouped evaluation

Pointwise RMSE in dB. Values are mean ± standard deviation across seeds 11, 22, and 33; lower is better. Pooled metrics weight measured points equally, while the scenario macro metric weights held-out scenarios equally.

| Method | Random pooled | Leave-config pooled | Leave-scenario pooled | Leave-scenario macro |
| --- | ---: | ---: | ---: | ---: |
| WallPath-PI | 6.493 ± 0.124 | 7.040 ± 0.006 | 10.957 ± 0.010 | 10.820 ± 0.005 |
| WallPath-RF | 6.450 ± 0.117 | **6.974 ± 0.005** | 11.100 ± 0.050 | 10.877 ± 0.041 |
| Direct RF | **6.364 ± 0.195** | 7.097 ± 0.006 | 10.892 ± 0.059 | 10.948 ± 0.050 |
| Direct ExtraTrees | 6.643 ± 0.174 | 7.148 ± 0.005 | 11.143 ± 0.043 | 10.932 ± 0.040 |
| Multi-wall | 7.059 ± 0.119 | 7.383 ± 0.000 | **10.083 ± 0.000** | **10.185 ± 0.000** |
| Log-distance | 9.997 ± 0.179 | 10.149 ± 0.000 | 14.995 ± 0.000 | 12.890 ± 0.000 |

Grouped-protocol standard deviations measure estimator randomness on fixed folds. The random-split variability also includes partition variation.
