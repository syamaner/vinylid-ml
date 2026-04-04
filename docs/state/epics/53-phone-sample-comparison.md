# Epic #53: A-Series vs C2 Phone-Sample Comparison

## Goal
Run A1-cls, A1-gem, A2, A4 on the same 55-photo / 50-matched-query sample used
by C2 sample-mode evaluation (story #11/#13), with the same gallery (857 images,
match_score_min=60), to enable direct apples-to-apples comparison.

## Status
- [x] Code: `--eval-set sample` added to `evaluate_phone_photos.py`
- [ ] Remote run: execute on RTX 4090 (CUDA)
- [ ] Results synced back locally
- [ ] State updated with results

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
Waiting for remote CUDA run.  After results arrive, update this file with
Recall@1 values and compare with C2 sample-mode result (R@1=0.XXX from #11).

## Latest Results
*(pending remote run)*

## Reference
- C2 sample-mode result: `results/summary.csv` rows with model_id=C2-superpoint-lightglue
- Phone-complete eval (#15): `results/phone_eval_summary.csv`
