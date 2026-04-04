# Epic #22: Cross-Category Comparison Report

## Goal
Generate a consolidated comparison report that separates evaluation contexts
(test-split, phone-complete, phone-sample) instead of conflating them.

## Status
- [x] `compare_models.py` extended with multi-context report functions
- [x] `_load_csv_by_model()` generic CSV loader
- [x] `build_multi_context_rows()` — merges 3 CSV sources with eval_context field
- [x] `save_multi_context_csv()` — writes multi_context_comparison.csv
- [x] `generate_multi_context_html()` — multi-section HTML with per-context tables
- [x] `main()` auto-generates multi-context report when phone CSVs exist
- [x] Tests added (19 new tests in test_compare_models.py)
- [x] Executed — 2026-04-04, generates 21-row multi-context report (full A/B/C/D + phone contexts)

## Outputs
Running `python scripts/compare_models.py` after syncing results produces:
- `results/comparison.csv` + `comparison.html` (backward-compatible, test-split only)
- `results/multi_context_comparison.csv` + `multi_context_comparison.html`
  (only generated when phone_eval_summary.csv or phone_sample_eval_summary.csv exist)

## Active Context
DONE. Reports at results/comparison.html and results/multi_context_comparison.html.
