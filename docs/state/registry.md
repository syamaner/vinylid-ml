# Active Epics Registry

Read this file first to identify which epic your current task belongs to.
Then open the linked state file for full context.

## Sprint 3: Fine-Tuning + Hybrid Pipeline

- **[#19] B3: SSCD + SupCon** — DONE (closed: SSCD fine-tuning did not beat zero-shot)
  - State: `docs/state/epics/19-b3-sscd-supcon.md`

## Sprint 3: Remaining Items (branch: feature/20-21-22-23-53-sprint3-remaining)

- **[#53] A-series vs C2 phone-sample comparison** — DONE
  - State: `docs/state/epics/53-phone-sample-comparison.md`
  - A4-sscd R@1=0.880 matches C2; A1-cls=0.500, A2=0.420, A1-gem=0.360 (50 queries, 870 gallery)
- **[#20] D1 hybrid formalization** — DONE
  - State: `docs/state/epics/20-d1-hybrid.md`
  - `--run-label D1-sscd-lightglue-k{K}` in evaluate_local_features.py
- **[#21] D1 K-sweep (K=5,10,20)** — DONE
  - State: `docs/state/epics/21-k-sweep.md`
  - K=5: R@1=0.868/0.07s, K=10: R@1=0.869/0.10s, K=20: R@1=0.872/0.16s, K=50: R@1=0.875/0.33s
  - Sweet spot: K=10 (0.869 R@1, 0.10s/query, 3.3× faster than K=50)
- **[#22] Cross-category comparison report** — DONE
  - State: `docs/state/epics/22-cross-category-report.md`
  - `results/multi_context_comparison.html` generated (7 rows, 3 contexts)
- **[#23] HF Hub push** — PENDING (run `python scripts/push_to_hub.py`)
  - State: `docs/state/epics/23-hf-hub-push.md`
  - `scripts/push_to_hub.py` ready

## Sprint 2: Remaining Items (deferred)

- **[#13] C2 full test-split eval** — DONE
  - State: `docs/state/epics/13-c2-local-features.md`
  - Result: R@1=0.875, R@5=0.900, mAP@5=0.886, MRR=0.887 (855 gallery, 4494 queries)
- **[#15] Real phone photo eval (243 labeled)** — DONE
  - State: `docs/state/epics/15-phone-photo-eval.md`
  - Result: A4-sscd R@1=0.778 leads; C2 R@1=0.621 (cross-domain gap confirmed)

## Sprint 4: CoreML + Index + E2E Integration

- Status: NOT STARTED
