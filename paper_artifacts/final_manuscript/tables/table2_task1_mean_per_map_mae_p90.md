# Table 2: Task1 mean per-map MAE and P90 absolute error

Scene-disjoint evaluation on held-out buildings B21-B25. Values are mean ± standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result for each metric and sparse rate is bold.

## Panel A: MAE

| Method | 0.005 | 0.010 | 0.050 | 0.100 |
| --- | ---: | ---: | ---: | ---: |
| WallPath-PI | 2.228 ± 0.002 | 1.921 ± 0.006 | **1.324 ± 0.003** | **1.084 ± 0.001** |
| Direct RF | **2.156 ± 0.006** | **1.898 ± 0.004** | 1.369 ± 0.004 | 1.137 ± 0.002 |
| Direct ExtraTrees | 2.168 ± 0.011 | 1.908 ± 0.010 | 1.395 ± 0.004 | 1.180 ± 0.004 |
| Multi-wall residual-IDW | 2.734 ± 0.004 | 2.458 ± 0.004 | 1.985 ± 0.005 | 1.796 ± 0.000 |
| IDW | 3.511 ± 0.001 | 3.119 ± 0.005 | 2.470 ± 0.003 | 2.224 ± 0.003 |
| Multi-wall | 4.596 ± 0.011 | 4.559 ± 0.003 | 4.545 ± 0.003 | 4.544 ± 0.002 |
| Log-distance | 6.029 ± 0.002 | 6.009 ± 0.004 | 5.993 ± 0.001 | 5.991 ± 0.001 |

## Panel B: P90 absolute error

| Method | 0.005 | 0.010 | 0.050 | 0.100 |
| --- | ---: | ---: | ---: | ---: |
| WallPath-PI | 5.422 ± 0.009 | 4.808 ± 0.016 | **3.480 ± 0.006** | **2.906 ± 0.000** |
| Direct RF | 5.357 ± 0.028 | **4.776 ± 0.016** | 3.527 ± 0.010 | 2.978 ± 0.007 |
| Direct ExtraTrees | **5.355 ± 0.031** | 4.781 ± 0.030 | 3.553 ± 0.011 | 3.027 ± 0.010 |
| Multi-wall residual-IDW | 6.147 ± 0.003 | 5.586 ± 0.019 | 4.578 ± 0.011 | 4.232 ± 0.003 |
| IDW | 7.813 ± 0.014 | 6.938 ± 0.016 | 5.496 ± 0.008 | 5.038 ± 0.007 |
| Multi-wall | 9.417 ± 0.019 | 9.372 ± 0.011 | 9.336 ± 0.019 | 9.343 ± 0.005 |
| Log-distance | 12.172 ± 0.013 | 12.126 ± 0.003 | 12.095 ± 0.009 | 12.089 ± 0.004 |

**Metrics:** MAE and P90 absolute error averaged across evaluation maps (dB).
