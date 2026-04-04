# Epic #20: D1 Hybrid Pipeline Formalization

## Goal
Formally label and document the A4-sscd → LightGlue hybrid as **D1**.
The algorithm was already implemented in `--mode test` (story #13) but was
labelled as `C2-superpoint-lightglue` in summary.csv.

## Status
- [x] `--run-label` added to `evaluate_local_features.py`
- [x] `_mode_test` now writes `pipeline: D1-sscd-lightglue` to config.json
- [x] Default label for `--mode test` is `D1-sscd-lightglue-k{top_k}`
- [x] State file created

## What Changed
- `evaluate_local_features.py`: added `--run-label` CLI arg
  - Default for `--mode test`: `D1-sscd-lightglue-k{top_k}`
  - Default for other modes: `LOCAL_FEATURE_MODEL_ID` (backward-compatible)
- `_mode_test`: uses `run_label` in summary.csv `model_id` column, run_dir name,
  config.json `model_id`, and console banner

## Reference Result (K=50 from story #13)
- Label: `D1-sscd-lightglue-k50`
- R@1=0.875, R@5=0.900, mAP@5=0.886, MRR=0.887
- 855 gallery, 4494 queries, 0.33s/query (RTX 4090)
