# Paired Transfer Statistics

This report analyzes Task1-to-Task2 and Task1-to-Task3 frozen transfer using paired per-map metrics. Each map is first averaged over seeds 11, 22, and 33. Positive paired differences mean that WallPath Extra has lower error than the comparison method.

The building-cluster bootstrap resamples the five unseen target buildings (B21-B25). Because there are only five independent building clusters, its confidence intervals and sign tests should be described as exploratory rather than definitive.

## Provenance audit

| Task | Seed | Runtime seed | Unique maps | Rows | Status |
|---|---:|---|---:|---:|---|
| TASK2 | 11 | [11] | 750 | 8250 | PASS |
| TASK2 | 22 | [22] | 750 | 8250 | PASS |
| TASK2 | 33 | [33] | 750 | 8250 | PASS |
| TASK3 | 11 | [11] | 5550 | 61050 | PASS |
| TASK3 | 22 | [22] | 5550 | 61050 | PASS |
| TASK3 | 33 | [33] | 5550 | 61050 | PASS |

## TASK2: WallPath Extra versus Direct RF

| Metric | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Valid-map wins | Building wins | 95% cluster CI for difference | Seed wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmse` | 4.8185 | 4.9269 | +2.20% | 49.9% (374/750) | 4/5 | [0.0005, 0.2248] | 3/3 |
| `non_anchor_rmse` | 4.8394 | 4.9460 | +2.16% | 49.7% (373/750) | 4/5 | [-0.0010, 0.2232] | 3/3 |
| `mae` | 3.2538 | 3.4274 | +5.07% | 68.8% (516/750) | 5/5 | [0.0831, 0.2836] | 3/3 |
| `p90_ae` | 7.9781 | 8.2721 | +3.55% | 44.3% (332/750) | 4/5 | [0.0716, 0.5301] | 3/3 |
| `los_rmse` | 2.8238 | 2.2682 | -24.50% | 25.2% (179/711) | 1/5 | [-1.0856, -0.1555] | 0/3 |
| `nlos_rmse` | 4.9552 | 5.0770 | +2.40% | 50.7% (380/750) | 4/5 | [0.0064, 0.2477] | 3/3 |
| `high_wall_rmse` | 5.1456 | 5.2801 | +2.55% | 50.9% (382/750) | 4/5 | [0.0077, 0.2735] | 3/3 |
| `free_space_rmse` | 4.6663 | 4.8017 | +2.82% | 53.5% (401/750) | 4/5 | [0.0268, 0.2479] | 3/3 |
| `wall_region_rmse` | 6.9216 | 6.7511 | -2.53% | 32.9% (247/750) | 1/5 | [-0.2690, -0.0564] | 0/3 |

### Transfer-condition breakdown for RMSE

| Condition | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Valid-map wins | Building wins | Cluster CI |
|---|---:|---:|---:|---:|---:|---:|
| Ant1-f1 source-like | 2.9671 | 2.8898 | -2.67% | 24.0% (60/250) | 2/5 | [-0.1688, 0.0069] |
| Ant1-f2-f3 frequency shift | 5.7443 | 5.9455 | +3.38% | 62.8% (314/500) | 4/5 | [0.0280, 0.3833] |

### Per-building RMSE

| Building | Scope | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Winner |
|---|---|---:|---:|---:|---|
| B21 | overall | 6.7055 | 6.8732 | +2.44% | WallPath |
| B22 | overall | 4.1116 | 4.0435 | -1.68% | Direct RF |
| B23 | overall | 3.2048 | 3.5226 | +9.02% | WallPath |
| B24 | overall | 4.5336 | 4.6189 | +1.85% | WallPath |
| B25 | overall | 5.5373 | 5.5765 | +0.70% | WallPath |

## TASK3: WallPath Extra versus Direct RF

| Metric | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Valid-map wins | Building wins | 95% cluster CI for difference | Seed wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmse` | 5.3195 | 5.8714 | +9.40% | 87.3% (4844/5550) | 5/5 | [0.3555, 0.7577] | 3/3 |
| `non_anchor_rmse` | 5.3402 | 5.8898 | +9.33% | 87.2% (4840/5550) | 5/5 | [0.3537, 0.7603] | 3/3 |
| `mae` | 3.7354 | 4.3143 | +13.42% | 94.2% (5230/5550) | 5/5 | [0.4085, 0.7603] | 3/3 |
| `p90_ae` | 8.8048 | 9.8030 | +10.18% | 86.3% (4792/5550) | 5/5 | [0.6675, 1.3766] | 3/3 |
| `los_rmse` | 4.2678 | 5.0053 | +14.73% | 78.5% (4132/5267) | 5/5 | [0.4574, 0.9589] | 3/3 |
| `nlos_rmse` | 5.4168 | 5.9412 | +8.83% | 84.9% (4714/5550) | 5/5 | [0.3163, 0.7441] | 3/3 |
| `high_wall_rmse` | 5.5997 | 6.1132 | +8.40% | 81.3% (4511/5550) | 5/5 | [0.2806, 0.7589] | 3/3 |
| `free_space_rmse` | 5.1908 | 5.7844 | +10.26% | 89.4% (4963/5550) | 5/5 | [0.3919, 0.7972] | 3/3 |
| `wall_region_rmse` | 7.1871 | 7.2353 | +0.67% | 45.0% (2499/5550) | 3/5 | [-0.1082, 0.2023] | 3/3 |

### Transfer-condition breakdown for RMSE

| Condition | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Valid-map wins | Building wins | Cluster CI |
|---|---:|---:|---:|---:|---:|---:|
| Ant1-f1 source-like | 2.9671 | 2.8898 | -2.67% | 24.0% (60/250) | 2/5 | [-0.1621, 0.0069] |
| Ant1-f2-f3 frequency shift | 5.7443 | 5.9455 | +3.38% | 62.8% (314/500) | 4/5 | [0.0280, 0.3833] |
| Ant2-Ant5 additional antennas across frequencies | 5.3978 | 6.0189 | +10.32% | 93.1% (4470/4800) | 5/5 | [0.4102, 0.8459] |

### Per-building RMSE

| Building | Scope | Mean per-map WallPath | Mean per-map Direct RF | Improvement | Winner |
|---|---|---:|---:|---:|---|
| B21 | overall | 7.2448 | 7.9375 | +8.73% | WallPath |
| B22 | overall | 4.5553 | 4.8358 | +5.80% | WallPath |
| B23 | overall | 3.7824 | 4.7053 | +19.61% | WallPath |
| B24 | overall | 4.9566 | 5.4664 | +9.33% | WallPath |
| B25 | overall | 6.0586 | 6.4119 | +5.51% | WallPath |

## Interpretation limits

- Task-level values in this report are **means of per-map metrics**, not the pooled pixel-level metrics stored in `final_evaluation_results.csv`.
- The three seeds are repeated matched runs over the same target maps. They are used for stability checks and are first averaged at map level for the paired analysis.
- Map win rates use only finite paired values for the stated metric. Maps with undefined region-specific metrics are reported as missing pairs and excluded from that metric's denominator.
- Maps within a building are not assumed independent. The cluster bootstrap uses 20,000 replicates over the five target buildings.
- With only five buildings, an exact two-sided sign test cannot reach p < 0.05 even when WallPath wins all five buildings (minimum p = 0.0625). Avoid overstating formal significance.
