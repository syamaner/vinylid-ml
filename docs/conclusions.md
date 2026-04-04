# VinylID ML — Conclusions & Production Guidance

*Last updated: 2026-04-04 (Sprint 3 complete)*

## TL;DR

**Neither global embeddings nor local feature matching alone achieves
production-grade accuracy on real phone photos.** The honest ceiling
today is ~0.78 R@1 (A4-sscd global) or ~0.62 R@1 (C2 LightGlue).
SIFT reaches 1.00 on the curated 50-photo sample but this is a
best-case subset. The path to >0.90 R@1 requires either real-data
fine-tuning with phone→catalogue pairs or a faster SIFT-class matcher.

**For a deployable system today:** A4-sscd server-side is the
pragmatic choice — best real-world recall, fast, no on-device ML needed.

---

## Key Finding 1: Test-Split Metrics Are Misleading

The test-split uses *augmented catalogue images* as queries — same
domain as the gallery. Real phone photos are a different domain entirely.

| Model | Test-split R@1 | Phone-complete R@1 | Delta |
|---|---|---|---|
| A1-dinov2-cls | 0.866 | 0.650 | −22pp |
| A4-sscd | **0.872** | **0.778** | −9pp |
| C2 LightGlue (D1-style) | 0.880¹ | 0.621 | −26pp |
| A2-openclip | 0.861 | 0.601 | −26pp |

¹ Sample-mode R@1 on the curated 50-photo subset.
*Test-split R@1 from committed per-run metrics.json (855 gallery, 4564 queries).*

A4-sscd drops only 9pp because it was trained for copy detection
(domain shift by design). All other approaches still drop 22–26pp.

## Key Finding 2: Local Feature Matching Does Not Beat Global Embedding on Real Photos

SIFT achieves R@1=1.000 on the **curated 50-photo sample** — a
best-case set of clear, well-lit captures. C2 LightGlue gets 0.880
on the same sample. But on the **full 203-photo real-world set**:

| Method | Phone-sample R@1 | Phone-complete R@1 |
|---|---|---|
| SIFT | 1.000 (2.3–3.5s/q) | not measured |
| C2 SuperPoint+LightGlue | 0.880 | **0.621** |
| A4-sscd (global) | 0.880 | **0.778** |

LightGlue underperforms A4-sscd on real photos by 16pp. The geometric
verification that makes local matching powerful on clean pairs
breaks down on noisy, varied, real-world phone captures (rotation,
partial occlusion, glare, depth-of-field). Global copy-detection
embeddings are more robust to these conditions.

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

### Fine-tuning: not worth it with augmented data (Sprint 2–3 finding)

B-series experiments used augmented catalogue images as positives —
not real phone photos. Results:
- B1 (MobileNet SupCon): significantly underperformed A-series
- B2c (DINOv2 SupCon, best fine-tuned): R@1=0.8795 on test split — still
  below A1-cls (0.920) and likely worse than A4-sscd on real photos
- B3 (SSCD SupCon): did not beat zero-shot SSCD; abandoned
- **Conclusion: zero-shot A4-sscd > any B-series fine-tuned model**

Untried and potentially high-impact: **fine-tune A4-sscd on real
phone→catalogue pairs**. Even 200–300 labelled pairs could close the
0.78→0.90 gap given A4-sscd’s copy-detection pretraining.

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
  machine-readable version of the above (21 rows, 3 eval contexts)
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
| C2-phone (D1-style) | [`results/C2-superpoint-lightglue-phone/2026-04-04T01-50-55/`](../results/C2-superpoint-lightglue-phone/2026-04-04T01-50-55/) | Complete-mode: R@1=0.621 on 203 real photos |

### Infrastructure notes
- Large binaries (`.pt`, `.npy`) and `per_query.csv` files are gitignored
- Checkpoints live on the remote machine only
- Regenerate reports: `python scripts/compare_models.py`

---

## Sprint 4 Direction

### Accuracy first (recommended before CoreML work)

Before optimising for on-device, close the accuracy gap:

1. **Collect real phone→catalogue training pairs** — 100–300 pairs with
   confirmed album-id labels. Fine-tune A4-sscd with SupCon on these.
   This directly targets the 0.78 → 0.90 gap with in-domain data.
2. **Measure SIFT on the full 203-photo set** — we only know SIFT=1.000
   on the curated 50-photo sample. The real-world ceiling is unknown.

### If shipping at 0.78 R@1 is acceptable

1. Export **A4-sscd → CoreML** (FP16, targeting Apple Neural Engine)
2. Build a **FAISS flat/HNSW gallery index** (512-dim float16, ~1 000 albums)
3. **E2E test**: camera capture → CoreML embed → FAISS query → top-1
4. Target <200ms end-to-end on-device

### Server-side option (0.78 R@1, no on-device ML)

A4-sscd inference on a small server (RTX/CPU) — latency budget:
~30ms embed + ~5ms FAISS = ~35ms server, plus network round-trip.
Simpler to ship than CoreML and more accurate than C2 LightGlue.
