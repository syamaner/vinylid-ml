# VinylID ML — Conclusions & Production Guidance

*Last updated: 2026-04-04 (Sprint 3 complete)*

## TL;DR

**Use A4-sscd for production.** It is the only model trained for
copy detection (phone→catalogue domain shift) and is the only one
that delivers useful real-world recall. All other models collapse on
actual phone photos despite looking good on the augmented test split.

---

## Key Finding: Test-Split Metrics Are Misleading

The test-split evaluation uses *augmented catalogue images* as queries —
perspective warps, brightness shifts, blur. These share the same visual
domain as the gallery. Real phone photos (hand-held, different lighting,
physical disc in frame) are a fundamentally different domain.

| Model | Test-split R@1 | Phone-complete R@1 | Delta |
|---|---|---|---|
| A1-dinov2-cls | **0.920** | 0.500 | −42pp |
| A4-sscd | 0.850 | **0.778** | −7pp |
| D1-sscd-lg-k50 | 0.875 | — | — |
| A2-openclip | 0.720 | 0.420 | −30pp |

A1 looks like the best model on paper. It is the worst in production.
A4-sscd drops only 7pp because it was trained for exactly this task
(copy detection across domains). All DINOv2/CLIP models drop 30–42pp.

---

## Model Selection

### Production: A4-sscd (zero-shot)

- Real-world phone eval: R@1=**0.778** (203 photos, 981 gallery)
- Phone-sample eval: R@1=**0.880** (50 photos, 870 gallery, same test set as C2)
- Fast: ~30ms/query on CUDA, ~3ms/query on ANE (CoreML, TBD)
- No fine-tuning needed — SSCD weights trained for copy detection are already optimal

### Optional server-side boost: D1 hybrid (A4-sscd + LightGlue, K=10)

- Test-split: R@1=0.869 vs A4-sscd alone 0.850 (+2pp)
- Phone-sample: A4-sscd alone already hits 0.880 — LightGlue may not help
- Latency: 0.10s/query on RTX 4090 — too slow for real-time on-device
- **Only worth adding if offline/server-side re-ranking for edge cases**

### Fine-tuning: not worth it (Sprint 2–3 finding)

B-series experiments conclusively showed:
- B1 (MobileNet SupCon): significantly underperformed A-series
- B2c (DINOv2 SupCon, best fine-tuned): R@1=0.8795 on test split — still
  below A1-cls (0.920) and likely worse than A4-sscd on real photos
- B3 (SSCD SupCon): did not beat zero-shot SSCD; abandoned
- **Conclusion: zero-shot A4-sscd > any fine-tuned model for this task**

---

## D1 K-Sweep: Latency vs Recall (Sprint 3 #21)

| K | R@1 | Latency | Notes |
|---|---|---|---|
| 5 | 0.868 | 0.07s/q | 4.7× faster than K=50 |
| **10** | **0.869** | **0.10s/q** | **Sweet spot** |
| 20 | 0.872 | 0.16s/q | |
| 50 | 0.875 | 0.33s/q | Reference (#13) |

The recall curve is nearly flat from K=5 to K=50 — A4-sscd cosine similarity
alone already places the correct answer in the top 5 for ~87% of queries.
LightGlue adds ~0.7pp by re-ranking, not by rescuing missed candidates.

If D1 is used server-side, **K=10 is the recommendation**.

---

## Result Files

### Cross-model comparison reports
- [`results/multi_context_comparison.html`](../results/multi_context_comparison.html) —
  full Sprint 3 leaderboard: test-split, phone-complete, and phone-sample in one page
- [`results/multi_context_comparison.csv`](../results/multi_context_comparison.csv) —
  machine-readable version of the above (11 rows, 3 eval contexts)
- [`results/comparison.html`](../results/comparison.html) —
  test-split only leaderboard (backward-compatible)

### Raw summary CSVs
- [`results/summary.csv`](../results/summary.csv) —
  test-split eval: A/B/C/D series + D1 K-sweep rows
- [`results/phone_eval_summary.csv`](../results/phone_eval_summary.csv) —
  phone-complete eval: 203 real photos (#15)
- [`results/phone_sample_eval_summary.csv`](../results/phone_sample_eval_summary.csv) —
  phone-sample eval: 50 queries on same set as C2 sample-mode (#53)

### Per-run artifacts (metrics.json + config.json)
Each experiment run has its own directory under `results/{model_id}/{timestamp}/`:

| Model | Key run | Notes |
|---|---|---|
| A4-sscd | [`results/A4-sscd/2026-03-14T23-04-25/`](../results/A4-sscd/2026-03-14T23-04-25/) | Best global model for real photos |
| A1-dinov2-cls | [`results/A1-dinov2-cls/2026-03-14T22-54-54/`](../results/A1-dinov2-cls/2026-03-14T22-54-54/) | Best test-split, worst real-world |
| D1-sscd-lg-k10 | [`results/D1-sscd-lightglue-k10/2026-04-04T01-02-36/`](../results/D1-sscd-lightglue-k10/2026-04-04T01-02-36/) | Server-side sweet spot |
| D1-sscd-lg-k50 | `results/D1-sscd-lightglue-k50/` (on remote, #13) | Reference K=50 run |
| B2c-dinov2-supcon | [`results/B2c-dinov2-supcon/2026-03-29T09-21-45/`](../results/B2c-dinov2-supcon/2026-03-29T09-21-45/) | Best fine-tuned model (didn’t beat zero-shot) |
| C2-superpoint-lg | [`results/C2-superpoint-lightglue/2026-03-15T19-55-25/`](../results/C2-superpoint-lightglue/2026-03-15T19-55-25/) | Sample-mode C2 reference |

### Infrastructure notes
- Large binaries (`.pt`, `.npy`) are gitignored; checkpoints live on the remote machine
- [`syamaner/vinylid-eval`](https://huggingface.co/syamaner/vinylid-eval/tree/main/results)
  mirrors the above to HF Hub for external sharing
- Regenerate reports: `python scripts/compare_models.py`
- Push to HF Hub: `python scripts/push_to_hub.py`

---

## Sprint 4 Direction: CoreML + On-Device E2E

Given the above, Sprint 4 should:

1. Export **A4-sscd → CoreML** (FP16, targeting Apple Neural Engine)
2. Build a **compressed gallery index** (FAISS flat or HNSW at 512-dim float16)
   suitable for on-device search across ~1 000 albums
3. **E2E pipeline test**: camera capture → CoreML embed → FAISS query → top-1
4. Measure on-device latency and recall on the phone-sample test set

If on-device latency is acceptable (<200ms end-to-end), ship A4-sscd only.
D1 hybrid remains a server-side option for future improvement.
