# Table 5: External measured 3.5 GHz few-shot adaptation

Leave-scenario-out pooled RMSE in dB on non-anchor target observations. Values are mean ± standard deviation across anchor-selection seeds 11, 22, and 33 with model seed fixed at 11; lower is better.

| Method | 1% anchors | 5% anchors | 10% anchors |
| --- | ---: | ---: | ---: |
| WallPath-PI + anchors | 9.857 ± 0.072 | 8.246 ± 0.323 | 7.539 ± 0.064 |
| WallPath-RF + anchors | 9.834 ± 0.168 | 8.294 ± 0.272 | 7.590 ± 0.041 |
| Direct RF + anchors | 9.293 ± 0.260 | 8.047 ± 0.287 | 7.557 ± 0.093 |
| Direct ExtraTrees + anchors | 9.044 ± 0.191 | **7.882 ± 0.203** | **7.432 ± 0.062** |
| Bias-calibrated multi-wall | **9.036 ± 0.549** | 8.345 ± 0.011 | 8.275 ± 0.048 |
| Bias-calibrated log-distance | 9.203 ± 0.197 | 8.487 ± 0.106 | 8.482 ± 0.134 |
| Multi-wall (zero-shot) | 10.083 ± 0.005 | 10.067 ± 0.046 | 10.068 ± 0.034 |
| Log-distance (zero-shot) | 14.974 ± 0.012 | 15.013 ± 0.001 | 14.974 ± 0.083 |

Anchor fractions refer to measured rows in each held-out scenario. All target anchors are excluded from scoring.
