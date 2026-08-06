# Table 3: Frozen transfer at 1% sparse anchors

Frozen Task1-to-Task2 and Task1-to-Task3 transfer at sparse rate 0.01. Values are mean ± standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result for each task and metric is bold.

| Method | Task 2 RMSE | Task 2 MAE | Task 2 P90 | Task 3 RMSE | Task 3 MAE | Task 3 P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WallPath-PI | **4.819 ± 0.006** | **3.254 ± 0.006** | **7.978 ± 0.007** | **5.320 ± 0.018** | **3.735 ± 0.019** | **8.805 ± 0.029** |
| WallPath-RF | 5.240 ± 0.036 | 3.460 ± 0.026 | 8.594 ± 0.101 | 6.300 ± 0.084 | 4.261 ± 0.061 | 10.345 ± 0.308 |
| WallPath-Calibrated | 5.134 ± 0.040 | 3.677 ± 0.036 | 8.487 ± 0.075 | 6.276 ± 0.079 | 4.804 ± 0.071 | 10.238 ± 0.122 |
| Direct RF | 4.927 ± 0.003 | 3.427 ± 0.006 | 8.272 ± 0.009 | 5.871 ± 0.019 | 4.314 ± 0.017 | 9.803 ± 0.030 |
| Direct ExtraTrees | 5.519 ± 0.043 | 4.025 ± 0.044 | 9.249 ± 0.072 | 7.767 ± 0.097 | 6.197 ± 0.111 | 12.562 ± 0.137 |
| Multi-wall residual-IDW | 5.627 ± 0.012 | 3.927 ± 0.006 | 8.856 ± 0.023 | 5.856 ± 0.003 | 4.158 ± 0.001 | 9.286 ± 0.003 |
| IDW | 6.161 ± 0.004 | 4.581 ± 0.004 | 10.155 ± 0.006 | 6.296 ± 0.003 | 4.670 ± 0.002 | 10.320 ± 0.004 |
| Multi-wall | 9.338 ± 0.007 | 7.355 ± 0.007 | 14.888 ± 0.009 | 10.152 ± 0.001 | 8.097 ± 0.000 | 16.255 ± 0.004 |
| Log-distance | 10.865 ± 0.002 | 8.819 ± 0.004 | 17.596 ± 0.013 | 11.500 ± 0.001 | 9.367 ± 0.001 | 18.595 ± 0.003 |

**Metrics:** RMSE, MAE, and P90 absolute error averaged across evaluation maps (dB). Learned estimator weights remain frozen on the target tasks.
