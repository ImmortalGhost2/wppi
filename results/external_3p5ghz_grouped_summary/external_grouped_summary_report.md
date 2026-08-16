# External Measured 3.5 GHz Grouped Evaluation

## Scope

This report summarizes six grouped evaluations: leave-configuration-out and leave-scenario-out under random seeds 11, 22, and 33. The models are trained from scratch on the measured pointwise dataset; this is not frozen transfer of the ICASSP radio-map model.

Seed standard deviations measure estimator randomness over the same observations. The dataset contains only two configuration groups and three scenario groups, so the external results should be interpreted descriptively as secondary evidence.

## Six-run audit

| protocol | random_seed | fold_rows | sample_rows | unique_test_points | status |
| --- | ---: | ---: | ---: | ---: | --- |
| leave_config_out | 11 | 12 | 13728 | 2288 | PASS |
| leave_config_out | 22 | 12 | 13728 | 2288 | PASS |
| leave_config_out | 33 | 12 | 13728 | 2288 | PASS |
| leave_scenario_out | 11 | 18 | 13728 | 2288 | PASS |
| leave_scenario_out | 22 | 18 | 13728 | 2288 | PASS |
| leave_scenario_out | 33 | 18 | 13728 | 2288 | PASS |

## Three-seed pooled point-level RMSE ranking

| protocol | rank | method | rmse_mean | rmse_std | rmse_min | rmse_max |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| leave_config_out | 1 | wallpath_residual_rf | 6.9735 | 0.0051 | 6.9703 | 6.9794 |
| leave_config_out | 2 | wallpath_residual_extra | 7.0395 | 0.0055 | 7.0331 | 7.043 |
| leave_config_out | 3 | direct_rf | 7.0975 | 0.0064 | 7.0938 | 7.1049 |
| leave_config_out | 4 | direct_extra | 7.1477 | 0.0047 | 7.1433 | 7.1526 |
| leave_config_out | 5 | multi_wall_linear | 7.3825 | 0.0 | 7.3825 | 7.3825 |
| leave_config_out | 6 | log_distance | 10.1488 | 0.0 | 10.1488 | 10.1488 |
| leave_scenario_out | 1 | multi_wall_linear | 10.0829 | 0.0 | 10.0829 | 10.0829 |
| leave_scenario_out | 2 | direct_rf | 10.8921 | 0.0592 | 10.8357 | 10.9538 |
| leave_scenario_out | 3 | wallpath_residual_extra | 10.9569 | 0.0104 | 10.949 | 10.9686 |
| leave_scenario_out | 4 | wallpath_residual_rf | 11.1002 | 0.0496 | 11.0551 | 11.1533 |
| leave_scenario_out | 5 | direct_extra | 11.1432 | 0.0432 | 11.0935 | 11.1718 |
| leave_scenario_out | 6 | log_distance | 14.9953 | 0.0 | 14.9953 | 14.9953 |

## Three-seed macro equal-fold RMSE ranking

| protocol | rank | method | rmse_mean | rmse_std | rmse_min | rmse_max |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| leave_config_out | 1 | wallpath_residual_rf | 6.9751 | 0.0051 | 6.9718 | 6.9809 |
| leave_config_out | 2 | wallpath_residual_extra | 7.041 | 0.0055 | 7.0346 | 7.0445 |
| leave_config_out | 3 | direct_rf | 7.0989 | 0.0064 | 7.0952 | 7.1064 |
| leave_config_out | 4 | direct_extra | 7.1493 | 0.0047 | 7.1449 | 7.1542 |
| leave_config_out | 5 | multi_wall_linear | 7.3812 | 0.0 | 7.3812 | 7.3812 |
| leave_config_out | 6 | log_distance | 10.151 | 0.0 | 10.151 | 10.151 |
| leave_scenario_out | 1 | multi_wall_linear | 10.185 | 0.0 | 10.185 | 10.185 |
| leave_scenario_out | 2 | wallpath_residual_extra | 10.8196 | 0.005 | 10.8164 | 10.8253 |
| leave_scenario_out | 3 | wallpath_residual_rf | 10.8771 | 0.0415 | 10.8429 | 10.9233 |
| leave_scenario_out | 4 | direct_extra | 10.9322 | 0.0405 | 10.9033 | 10.9785 |
| leave_scenario_out | 5 | direct_rf | 10.9476 | 0.0496 | 10.9027 | 11.0009 |
| leave_scenario_out | 6 | log_distance | 12.8902 | 0.0 | 12.8902 | 12.8902 |

## Fold-specific RMSE

| protocol | fold | method | rmse_mean | rmse_std | rmse_min | rmse_max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| leave_config_out | holdout_C1 | wallpath_residual_rf | 6.8366 | 0.0143 | 6.8202 | 6.8462 |
| leave_config_out | holdout_C1 | wallpath_residual_extra | 6.8627 | 0.0087 | 6.8528 | 6.8692 |
| leave_config_out | holdout_C1 | direct_rf | 6.986 | 0.0101 | 6.9751 | 6.9949 |
| leave_config_out | holdout_C1 | direct_extra | 7.0073 | 0.0101 | 7.0005 | 7.0189 |
| leave_config_out | holdout_C1 | multi_wall_linear | 7.017 | 0.0 | 7.017 | 7.017 |
| leave_config_out | holdout_C1 | log_distance | 9.9419 | 0.0 | 9.9419 | 9.9419 |
| leave_config_out | holdout_C2 | wallpath_residual_rf | 7.1135 | 0.0125 | 7.1 | 7.1249 |
| leave_config_out | holdout_C2 | direct_rf | 7.2119 | 0.015 | 7.1954 | 7.2247 |
| leave_config_out | holdout_C2 | wallpath_residual_extra | 7.2193 | 0.0026 | 7.2164 | 7.2215 |
| leave_config_out | holdout_C2 | direct_extra | 7.2912 | 0.0031 | 7.2894 | 7.2949 |
| leave_config_out | holdout_C2 | multi_wall_linear | 7.7454 | 0.0 | 7.7454 | 7.7454 |
| leave_config_out | holdout_C2 | log_distance | 10.3602 | 0.0 | 10.3602 | 10.3602 |
| leave_scenario_out | holdout_Comms | multi_wall_linear | 9.4822 | 0.0 | 9.4822 | 9.4822 |
| leave_scenario_out | holdout_Comms | direct_rf | 10.0506 | 0.0924 | 9.9509 | 10.1334 |
| leave_scenario_out | holdout_Comms | wallpath_residual_extra | 10.6903 | 0.0274 | 10.6681 | 10.7209 |
| leave_scenario_out | holdout_Comms | direct_extra | 10.8444 | 0.083 | 10.7878 | 10.9397 |
| leave_scenario_out | holdout_Comms | wallpath_residual_rf | 10.9734 | 0.0611 | 10.9207 | 11.0404 |
| leave_scenario_out | holdout_Comms | log_distance | 15.6434 | 0.0 | 15.6434 | 15.6434 |
| leave_scenario_out | holdout_Library | multi_wall_linear | 11.2769 | 0.0 | 11.2769 | 11.2769 |
| leave_scenario_out | holdout_Library | wallpath_residual_rf | 11.6688 | 0.0363 | 11.6285 | 11.6989 |
| leave_scenario_out | holdout_Library | wallpath_residual_extra | 11.7367 | 0.0295 | 11.7148 | 11.7702 |
| leave_scenario_out | holdout_Library | direct_extra | 12.0697 | 0.1042 | 12.0059 | 12.1899 |
| leave_scenario_out | holdout_Library | direct_rf | 12.6027 | 0.0386 | 12.5604 | 12.6359 |
| leave_scenario_out | holdout_Library | log_distance | 15.382 | 0.0 | 15.382 | 15.382 |
| leave_scenario_out | holdout_SSE | log_distance | 7.6453 | 0.0 | 7.6453 | 7.6453 |
| leave_scenario_out | holdout_SSE | multi_wall_linear | 9.7958 | 0.0 | 9.7958 | 9.7958 |
| leave_scenario_out | holdout_SSE | direct_extra | 9.8824 | 0.0741 | 9.7987 | 9.9397 |
| leave_scenario_out | holdout_SSE | wallpath_residual_rf | 9.9892 | 0.0374 | 9.9576 | 10.0305 |
| leave_scenario_out | holdout_SSE | wallpath_residual_extra | 10.0319 | 0.0162 | 10.0132 | 10.0421 |
| leave_scenario_out | holdout_SSE | direct_rf | 10.1895 | 0.0439 | 10.1455 | 10.2333 |

## WallPath comparisons

| protocol | method | baseline | pooled_method_rmse_mean | pooled_method_rmse_std | pooled_baseline_rmse_mean | pooled_baseline_rmse_std | pooled_rmse_improvement_mean_percent | seed_wins_pooled_rmse | fold_wins_after_seed_average | num_folds | mean_point_win_percent |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leave_config_out | wallpath_residual_extra | direct_rf | 7.0395 | 0.0055 | 7.0975 | 0.0064 | 0.8168 | 3 | 1 | 2 | 51.6171 |
| leave_config_out | wallpath_residual_rf | direct_rf | 6.9735 | 0.0051 | 7.0975 | 0.0064 | 1.7461 | 3 | 2 | 2 | 54.0064 |
| leave_config_out | wallpath_residual_extra | multi_wall_linear | 7.0395 | 0.0055 | 7.3825 | 0.0 | 4.6464 | 3 | 2 | 2 | 52.433 |
| leave_config_out | wallpath_residual_rf | multi_wall_linear | 6.9735 | 0.0051 | 7.3825 | 0.0 | 5.5398 | 3 | 2 | 2 | 53.2197 |
| leave_scenario_out | wallpath_residual_extra | direct_rf | 10.9569 | 0.0104 | 10.8921 | 0.0592 | -0.5963 | 0 | 2 | 3 | 50.641 |
| leave_scenario_out | wallpath_residual_rf | direct_rf | 11.1002 | 0.0496 | 10.8921 | 0.0592 | -1.911 | 0 | 2 | 3 | 51.1801 |
| leave_scenario_out | wallpath_residual_extra | multi_wall_linear | 10.9569 | 0.0104 | 10.0829 | 0.0 | -8.668 | 0 | 0 | 3 | 45.7896 |
| leave_scenario_out | wallpath_residual_rf | multi_wall_linear | 11.1002 | 0.0496 | 10.0829 | 0.0 | -10.0891 | 0 | 0 | 3 | 47.0134 |

## Interpretation limits

- Pooled metrics weight each measured point equally; macro metrics weight each held-out configuration or scenario equally.
- Seeds reuse the same measured observations and grouped splits. Seed consistency is a stability check, not an independent-sample significance test.
- With only two configurations and three scenarios, cluster-based confidence intervals or formal group-level significance tests would be unstable and should not be used as strong evidence.
- Pointwise win rates are descriptive because measurements within the same environment are not guaranteed to be independent.
