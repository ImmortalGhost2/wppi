# Table 1: Task1 mean per-map RMSE

Scene-disjoint evaluation on held-out buildings B21-B25. Values are mean ± standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result at each sparse rate is bold.

| Method | 0.005 | 0.010 | 0.050 | 0.100 |
| --- | ---: | ---: | ---: | ---: |
| WallPath-PI | 3.328 ± 0.015 | 2.967 ± 0.008 | 2.249 ± 0.007 | 1.951 ± 0.003 |
| Direct RF | 3.222 ± 0.013 | 2.890 ± 0.009 | 2.228 ± 0.008 | 1.949 ± 0.001 |
| Direct ExtraTrees | **3.205 ± 0.017** | **2.878 ± 0.015** | **2.214 ± 0.006** | **1.938 ± 0.002** |
| Multi-wall residual-IDW | 4.094 ± 0.031 | 3.681 ± 0.022 | 2.987 ± 0.011 | 2.748 ± 0.005 |
| IDW | 4.774 ± 0.006 | 4.244 ± 0.015 | 3.371 ± 0.005 | 3.079 ± 0.003 |
| Multi-wall | 6.059 ± 0.010 | 5.954 ± 0.009 | 5.870 ± 0.001 | 5.861 ± 0.000 |
| Log-distance | 7.502 ± 0.003 | 7.468 ± 0.001 | 7.444 ± 0.000 | 7.442 ± 0.000 |

**Metric:** RMSE averaged across evaluation maps (dB).
