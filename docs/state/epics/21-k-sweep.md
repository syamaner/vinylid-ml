# Epic #21: D1 K-Sweep (top_k = 5, 10, 20)

## Goal
Sweep `top_k` values 5, 10, 20 for the D1 hybrid pipeline on the RTX 4090 to
understand the trade-off between recall and latency.  K=50 from story #13 is
the reference (R@1=0.875, 0.33s/query).

## Status
- [x] Code: `--run-label` in `evaluate_local_features.py` enables labelled sweeps
- [x] Remote run K=5 — DONE 2026-04-04
- [x] Remote run K=10 — DONE 2026-04-04
- [x] Remote run K=20 — DONE 2026-04-04
- [x] Results synced back locally
- [x] State updated with results

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
855 gallery, 4494 queries (full test split), RTX 4090

| K  | R@1   | R@5   | mAP@5 | MRR   | lat (s/query) |
|----|-------|-------|-------|-------|---------------|
| 5  | 0.868 | 0.890 | 0.878 | 0.879 | 0.07          |
| 10 | 0.869 | 0.891 | 0.879 | 0.881 | 0.10          |
| 20 | 0.872 | 0.895 | 0.883 | 0.884 | 0.16          |
| 50 | 0.875 | 0.900 | 0.886 | 0.887 | 0.33          |

Key finding: R@1 degrades only 0.7pp from K=50→K=5, but latency drops 4.7×.
K=10 is the sweet spot for production (R@1=0.869, 0.10s/query).
