# External Measured 3.5 GHz Few-Shot Summary

## Scope

The few-shot run uses model seed 11 and three independent target-anchor selections (11, 22, 33). Metrics exclude all target anchors. Matched adaptation gains are computed against the seed-11 zero-shot leave-scenario-out predictions on the exact same non-anchor observations.

## Pooled few-shot ranking

| anchor_fraction | rank | method | rmse_mean | rmse_std | mae_mean | p90_mean |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.01 | 1 | bias_calibrated_multi_wall | 9.0358 | 0.5486 | 7.1495 | 14.878 |
| 0.01 | 2 | direct_extra_with_anchors | 9.0444 | 0.1912 | 7.1743 | 15.0828 |
| 0.01 | 3 | bias_calibrated_log_distance | 9.2032 | 0.1966 | 7.5743 | 15.0076 |
| 0.01 | 4 | direct_rf_with_anchors | 9.2933 | 0.2603 | 7.3568 | 15.5401 |
| 0.01 | 5 | wallpath_fewshot_residual_rf | 9.8342 | 0.1677 | 7.7901 | 15.9797 |
| 0.01 | 6 | wallpath_fewshot_residual_extra | 9.8573 | 0.0719 | 7.8047 | 15.98 |
| 0.01 | 7 | multi_wall_linear | 10.0825 | 0.0047 | 8.1058 | 16.203 |
| 0.01 | 8 | log_distance | 14.974 | 0.0116 | 12.6777 | 24.067 |
| 0.05 | 1 | direct_extra_with_anchors | 7.8821 | 0.2034 | 6.2552 | 12.8796 |
| 0.05 | 2 | direct_rf_with_anchors | 8.047 | 0.2869 | 6.4237 | 13.4045 |
| 0.05 | 3 | wallpath_fewshot_residual_extra | 8.2457 | 0.3228 | 6.4942 | 13.5805 |
| 0.05 | 4 | wallpath_fewshot_residual_rf | 8.294 | 0.272 | 6.5787 | 13.6467 |
| 0.05 | 5 | bias_calibrated_multi_wall | 8.3445 | 0.0107 | 6.5735 | 13.9239 |
| 0.05 | 6 | bias_calibrated_log_distance | 8.487 | 0.1061 | 6.994 | 13.8852 |
| 0.05 | 7 | multi_wall_linear | 10.0672 | 0.0457 | 8.0934 | 16.2085 |
| 0.05 | 8 | log_distance | 15.0129 | 0.0012 | 12.7188 | 24.1087 |
| 0.1 | 1 | direct_extra_with_anchors | 7.432 | 0.0622 | 5.8409 | 12.0742 |
| 0.1 | 2 | wallpath_fewshot_residual_extra | 7.5391 | 0.0644 | 5.9142 | 12.3083 |
| 0.1 | 3 | direct_rf_with_anchors | 7.5566 | 0.0932 | 5.9783 | 12.2894 |
| 0.1 | 4 | wallpath_fewshot_residual_rf | 7.5903 | 0.0412 | 5.9851 | 12.4401 |
| 0.1 | 5 | bias_calibrated_multi_wall | 8.2746 | 0.0479 | 6.4963 | 13.7825 |
| 0.1 | 6 | bias_calibrated_log_distance | 8.4817 | 0.1339 | 6.9545 | 13.8929 |
| 0.1 | 7 | multi_wall_linear | 10.0679 | 0.0339 | 8.0855 | 16.192 |
| 0.1 | 8 | log_distance | 14.9742 | 0.0827 | 12.6766 | 24.0667 |

## Matched adaptation gains

| anchor_fraction | adapted_method | baseline_method | adapted_rmse_mean | adapted_rmse_std | baseline_rmse_mean | baseline_rmse_std | rmse_improvement_mean_percent | rmse_improvement_std_percent | adapted_mae_mean | baseline_mae_mean | mae_improvement_mean_percent | anchor_seed_rmse_wins | scenario_rmse_wins | num_scenarios | mean_point_win_percent |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | bias_calibrated_multi_wall | multi_wall_linear | 9.0358 | 0.5486 | 10.0825 | 0.0047 | 10.3826 | 5.4237 | 7.1495 | 8.1058 | 11.7988 | 3 | 2 | 3 | 59.4307 |
| 0.01 | direct_extra_with_anchors | direct_extra | 9.0444 | 0.1912 | 11.1529 | 0.0214 | 18.9063 | 1.6216 | 7.1743 | 8.9788 | 20.1004 | 3 | 3 | 3 | 60.7627 |
| 0.01 | bias_calibrated_log_distance | log_distance | 9.2032 | 0.1966 | 14.974 | 0.0116 | 38.5379 | 1.3598 | 7.5743 | 12.6777 | 40.2545 | 3 | 2 | 3 | 64.1076 |
| 0.01 | direct_rf_with_anchors | direct_rf | 9.2933 | 0.2603 | 10.879 | 0.0183 | 14.577 | 2.3181 | 7.3568 | 8.7649 | 16.0688 | 3 | 3 | 3 | 60.6157 |
| 0.01 | wallpath_fewshot_residual_rf | wallpath_residual_rf | 9.8342 | 0.1677 | 11.0551 | 0.0047 | 11.0443 | 1.4956 | 7.7901 | 8.7581 | 11.053 | 3 | 3 | 3 | 64.3991 |
| 0.01 | wallpath_fewshot_residual_extra | wallpath_residual_extra | 9.8573 | 0.0719 | 10.9558 | 0.0068 | 10.026 | 0.7114 | 7.8047 | 8.7471 | 10.7738 | 3 | 3 | 3 | 60.7286 |
| 0.05 | direct_extra_with_anchors | direct_extra | 7.8821 | 0.2034 | 11.1809 | 0.0207 | 29.506 | 1.7107 | 6.2552 | 8.9982 | 30.486 | 3 | 3 | 3 | 66.9588 |
| 0.05 | direct_rf_with_anchors | direct_rf | 8.047 | 0.2869 | 10.9033 | 0.0203 | 26.1969 | 2.6173 | 6.4237 | 8.7815 | 26.8503 | 3 | 3 | 3 | 68.7475 |
| 0.05 | wallpath_fewshot_residual_extra | wallpath_residual_extra | 8.2457 | 0.3228 | 10.9366 | 0.0783 | 24.6152 | 2.4266 | 6.4942 | 8.7349 | 25.6658 | 3 | 3 | 3 | 64.367 |
| 0.05 | wallpath_fewshot_residual_rf | wallpath_residual_rf | 8.294 | 0.272 | 11.0395 | 0.0814 | 24.8791 | 1.9167 | 6.5787 | 8.7525 | 24.8463 | 3 | 3 | 3 | 66.6942 |
| 0.05 | bias_calibrated_multi_wall | multi_wall_linear | 8.3445 | 0.0107 | 10.0672 | 0.0457 | 17.1105 | 0.4646 | 6.5735 | 8.0934 | 18.7778 | 3 | 3 | 3 | 62.5919 |
| 0.05 | bias_calibrated_log_distance | log_distance | 8.487 | 0.1061 | 15.0129 | 0.0012 | 43.4684 | 0.7047 | 6.994 | 12.7188 | 45.011 | 3 | 3 | 3 | 69.2699 |
| 0.1 | direct_extra_with_anchors | direct_extra | 7.432 | 0.0622 | 11.1416 | 0.0657 | 33.2948 | 0.3869 | 5.8409 | 8.9623 | 34.8273 | 3 | 3 | 3 | 67.8909 |
| 0.1 | wallpath_fewshot_residual_extra | wallpath_residual_extra | 7.5391 | 0.0644 | 10.9651 | 0.0376 | 31.2428 | 0.807 | 5.9142 | 8.7503 | 32.4089 | 3 | 3 | 3 | 66.8702 |
| 0.1 | direct_rf_with_anchors | direct_rf | 7.5566 | 0.0932 | 10.8742 | 0.0535 | 30.5093 | 0.6845 | 5.9783 | 8.7543 | 31.7114 | 3 | 3 | 3 | 68.4161 |
| 0.1 | wallpath_fewshot_residual_rf | wallpath_residual_rf | 7.5903 | 0.0412 | 11.0655 | 0.0406 | 31.4051 | 0.4824 | 5.9851 | 8.7627 | 31.6961 | 3 | 3 | 3 | 70.1921 |
| 0.1 | bias_calibrated_multi_wall | multi_wall_linear | 8.2746 | 0.0479 | 10.0679 | 0.0339 | 17.8115 | 0.5203 | 6.4963 | 8.0855 | 19.6538 | 3 | 3 | 3 | 62.5771 |
| 0.1 | bias_calibrated_log_distance | log_distance | 8.4817 | 0.1339 | 14.9742 | 0.0827 | 43.3536 | 1.197 | 6.9545 | 12.6766 | 45.1321 | 3 | 2 | 3 | 70.0436 |

## Best pooled method by fraction

| anchor_fraction | method | rmse_mean | rmse_std |
| ---: | --- | ---: | ---: |
| 0.01 | bias_calibrated_multi_wall | 9.0358 | 0.5486 |
| 0.05 | direct_extra_with_anchors | 7.8821 | 0.2034 |
| 0.1 | direct_extra_with_anchors | 7.432 | 0.0622 |

## Interpretation limits

- The three anchor seeds vary target-anchor selection, while model randomness is fixed.
- Only three held-out scenarios are available. Scenario-win counts are descriptive, not formal significance tests.
- The external experiment is pointwise measured-data adaptation and is not frozen transfer of the ICASSP radio-map model.
