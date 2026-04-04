# Epic #53: A-Series vs C2 Phone-Sample Comparison

## Goal
Run A1-cls, A1-gem, A2, A4 on the same 55-photo / 50-matched-query sample used
by C2 sample-mode evaluation (story #11/#13), with the same gallery construction:
canonical test-split images plus any extra albums referenced by the matched CSV
(match_score_min=60), to enable direct apples-to-apples comparison.

## Status
- [x] Code: `--eval-set sample` added to `evaluate_phone_photos.py`
- [x] Remote run: executed on RTX 4090 (CUDA) — 2026-04-04
- [x] Results synced back locally
- [x] State updated with results

## Implementation
`evaluate_phone_photos.py --eval-set sample` reads `test_sample_matched.csv`
and `test_sample` directory.  Results written to:
- `results/{model_id}-phone-sample/{timestamp}/` (metrics.json, config.json)
- `results/phone_sample_eval_summary.csv`

## Remote Run Command
```
python scripts/evaluate_phone_photos.py \
    --config configs/dataset-remote.yaml \
    --eval-set sample \
    --models A1-dinov2-cls,A1-dinov2-gem,A2-openclip,A4-sscd
```

## Active Context
Remote CUDA run is complete and results are recorded below. Current takeaway:
A4-sscd matches the C2 sample-mode baseline on this sample (R@1=0.880), while
A1/A2 remain meaningfully behind.

## Latest Results
50 matched queries, 870 gallery images, match_score_min=60

| Model        | R@1   | R@5   | mAP@5 | MRR   | lat      |
|--------------|-------|-------|-------|-------|----------|
| A4-sscd      | 0.880 | 0.960 | 0.908 | 0.909 | 0.03s/q  |
| A1-dinov2-cls| 0.500 | 0.600 | 0.542 | 0.553 | 0.03s/q  |
| A2-openclip  | 0.420 | 0.560 | 0.477 | 0.495 | 0.03s/q  |
| A1-dinov2-gem| 0.360 | 0.500 | 0.409 | 0.427 | 0.03s/q  |

C2 sample-mode reference (same gallery/queries): R@1=0.880 (from summary.csv)
→ **A4-sscd matches C2 on this sample; A1/A2 significantly behind.**

## Reference
- C2 sample-mode result: `results/summary.csv` rows with model_id=C2-superpoint-lightglue
- Phone-complete eval (#15): `results/phone_eval_summary.csv`
