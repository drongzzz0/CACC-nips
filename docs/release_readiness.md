# CACC Release Readiness

Practical readiness checks for large final-accuracy gaps in the Table 1 release manifest.

- Practical ready: `true`
- Strict ready: `false`
- Rows: `16` total, `0` blocker, `0` pending, `5` caveat, `11` ok.

| Severity | Row | Bucket | Release delta | Fallback delta | Status |
| --- | --- | --- | ---: | ---: | --- |
| caveat | compmath/cacc_spp | fallback_close | -0.0588 | -0.0175 | fresh_rerun_large_gap_fallback_available |
| caveat | compmath/spp | watch | -0.0322 | - | fresh_rerun_lower_than_paper |
| caveat | gsm8k/spp | higher_final | +0.0222 | - | fresh_rerun_close_final_different_decomposition |
| caveat | mmlu_pro/base | partial | - | - | partial_provenance_prediction_logs_missing |
| caveat | mmlu_pro/spp | higher_final | +0.0634 | +0.0045 | fresh_rerun_higher_final_different_decomposition |
| ok | compmath/base | artifact | - | - | artifact_reproduced |
| ok | compmath/cacc | artifact | - | - | artifact_reproduced |
| ok | gpqa/base | artifact | - | - | artifact_reproduced_display_ratio |
| ok | gpqa/cacc | artifact | - | - | artifact_reproduced |
| ok | gpqa/cacc_spp | artifact | - | - | artifact_reproduced |
| ok | gpqa/spp | close | -0.0013 | - | fresh_rerun_close_final_different_decomposition |
| ok | gsm8k/base | artifact | - | - | artifact_reproduced |
| ok | gsm8k/cacc | artifact | - | - | artifact_reproduced |
| ok | gsm8k/cacc_spp | close | +0.0026 | - | fresh_rerun_close_final_different_decomposition |
| ok | mmlu_pro/cacc | artifact | - | - | artifact_reproduced |
| ok | mmlu_pro/cacc_spp | artifact | - | - | artifact_reproduced_display_ratio |

Default practical readiness passes when there are no `large_gap` or `unknown` rows.
`pending`, `partial`, `watch`, `higher_final`, and fallback rows still need release notes or follow-up evidence.
Use `--strict` to fail on any pending row or caveat.
