# Epic #21: D1 K-Sweep (top_k = 5, 10, 20)

## Goal
Sweep `top_k` values 5, 10, 20 for the D1 hybrid pipeline on the RTX 4090 to
understand the trade-off between recall and latency.  K=50 from story #13 is
the reference (R@1=0.875, 0.33s/query).

## Status
- [x] Code: `--run-label` in `evaluate_local_features.py` enables labelled sweeps
- [ ] Remote run K=5
- [ ] Remote run K=10
- [ ] Remote run K=20
- [ ] Results synced back locally
- [ ] State updated with results

## Expected Latency
Gallery feature extraction is constant (~105s).  Re-ranking scales with K:
- K=5:  ~4-5 min total
- K=10: ~7-8 min total
- K=20: ~12-15 min total
(All 3 combined: ~25 min on RTX 4090 vs K=50 alone at ~24 min)

## Remote Run Commands
```bash
for K in 5 10 20; do
  python scripts/evaluate_local_features.py \
      --config configs/dataset-remote.yaml \
      --mode test \
      --top-k $K \
      --run-label D1-sscd-lightglue-k${K}
done
```

## Active Context
Waiting for remote CUDA runs.  After results arrive, update with R@1 and
latency per K value.

## Latest Results
*(pending remote run)*

| K  | R@1 | R@5 | mAP@5 | MRR | lat (s/query) |
|----|-----|-----|-------|-----|---------------|
| 5  | ?   | ?   | ?     | ?   | ?             |
| 10 | ?   | ?   | ?     | ?   | ?             |
| 20 | ?   | ?   | ?     | ?   | ?             |
| 50 | 0.875 | 0.900 | 0.886 | 0.887 | 0.33     |
