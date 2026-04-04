# Active Epics Registry

Read this file first to identify which epic your current task belongs to.
Then open the linked state file for full context.

## Sprint 3: Fine-Tuning + Hybrid Pipeline

- **[#19] B3: SSCD + SupCon** — DONE (closed: SSCD fine-tuning did not beat zero-shot)
  - State: `docs/state/epics/19-b3-sscd-supcon.md`

## Sprint 3: Remaining Items (branch: feature/20-21-22-23-53-sprint3-remaining)

- **[#53] A-series vs C2 phone-sample comparison** — IN PROGRESS (code done, awaiting remote run)
  - State: `docs/state/epics/53-phone-sample-comparison.md`
  - `evaluate_phone_photos.py --eval-set sample` enabled
- **[#20] D1 hybrid formalization** — IN PROGRESS (code done)
  - State: `docs/state/epics/20-d1-hybrid.md`
  - `--run-label D1-sscd-lightglue-k{K}` added to evaluate_local_features.py
- **[#21] D1 K-sweep (K=5,10,20)** — IN PROGRESS (code done, awaiting remote run)
  - State: `docs/state/epics/21-k-sweep.md`
  - Reference: K=50 R@1=0.875 (from #13)
- **[#22] Cross-category comparison report** — IN PROGRESS (code done, awaiting data)
  - State: `docs/state/epics/22-cross-category-report.md`
  - `compare_models.py` extended with `build_multi_context_rows`, `generate_multi_context_html`
- **[#23] HF Hub push** — IN PROGRESS (script done, awaiting #22)
  - State: `docs/state/epics/23-hf-hub-push.md`
  - `scripts/push_to_hub.py` created

## Sprint 2: Remaining Items (deferred)

- **[#13] C2 full test-split eval** — DONE
  - State: `docs/state/epics/13-c2-local-features.md`
  - Result: R@1=0.875, R@5=0.900, mAP@5=0.886, MRR=0.887 (855 gallery, 4494 queries)
- **[#15] Real phone photo eval (243 labeled)** — DONE
  - State: `docs/state/epics/15-phone-photo-eval.md`
  - Result: A4-sscd R@1=0.778 leads; C2 R@1=0.621 (cross-domain gap confirmed)

## Sprint 4: CoreML + Index + E2E Integration

- Status: NOT STARTED
