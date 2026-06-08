# CACC Table 1 Reproduction Report

Source table: Main paper Table 1

| Row | Paper O/V/F | Release ref O/V/F | Release delta | Fallback ref O/V/F | Fallback delta | Bucket | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gsm8k/base | 0.4882 / 0.5885 / 0.2873 | - | - | - | - | artifact | artifact_reproduced |
| gsm8k/spp | 0.6471 / 0.5714 / 0.3698 | 0.8514 / 0.4604 / 0.3920 | +0.0222 | - | - | higher_final | fresh_rerun_close_final_different_decomposition |
| gsm8k/cacc | 0.6391 / 0.5682 / 0.3632 | - | - | - | - | artifact | artifact_reproduced |
| gsm8k/cacc_spp | 0.6962 / 0.6017 / 0.4189 | 0.6876 / 0.6130 / 0.4215 | +0.0026 | - | - | close | fresh_rerun_close_final_different_decomposition |
| compmath/base | 0.3210 / 0.4732 / 0.1519 | - | - | - | - | artifact | artifact_reproduced |
| compmath/spp | 0.4616 / 0.6014 / 0.2776 | 0.5505 / 0.4458 / 0.2454 | -0.0322 | - | - | watch | fresh_rerun_lower_than_paper |
| compmath/cacc | 0.4436 / 0.5814 / 0.2579 | - | - | - | - | artifact | artifact_reproduced |
| compmath/cacc_spp | 0.5024 / 0.6173 / 0.3101 | 0.4636 / 0.5421 / 0.2513 | -0.0588 | 0.5320 / 0.5499 / 0.2926 | -0.0175 | fallback_close | fresh_rerun_large_gap_fallback_available |
| mmlu_pro/base | 0.4072 / 0.5800 / 0.2362 | - | - | - | - | partial | partial_provenance_prediction_logs_missing |
| mmlu_pro/spp | 0.4921 / 0.5411 / 0.2663 | 0.5962 / 0.5530 / 0.3297 | +0.0634 | 0.6320 / 0.4285 / 0.2708 | +0.0045 | higher_final | fresh_rerun_higher_final_different_decomposition |
| mmlu_pro/cacc | 0.4767 / 0.4777 / 0.2277 | - | - | - | - | artifact | artifact_reproduced |
| mmlu_pro/cacc_spp | 0.5527 / 0.5312 / 0.2936 | - | - | - | - | artifact | artifact_reproduced_display_ratio |
| gpqa/base | 0.1919 / 0.2897 / 0.0556 | - | - | - | - | artifact | artifact_reproduced_display_ratio |
| gpqa/spp | 0.4724 / 0.4197 / 0.1983 | 0.3939 / 0.5000 / 0.1970 | -0.0013 | - | - | close | fresh_rerun_close_final_different_decomposition |
| gpqa/cacc | 0.4545 / 0.4222 / 0.1919 | - | - | - | - | artifact | artifact_reproduced |
| gpqa/cacc_spp | 0.4899 / 0.5258 / 0.2576 | - | - | - | - | artifact | artifact_reproduced |

Buckets are derived from final accuracy only: artifact, partial, pending, close (`<=0.02`), higher_final, watch (`<=0.05` below paper), or large_gap.
Rows labeled fallback_close or fallback_watch have a large fresh-rerun gap but a closer documented fallback reference.
Report oracle coverage, verifier efficiency given oracle, and final accuracy together when interpreting any rerun.
