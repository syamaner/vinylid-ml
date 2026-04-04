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

## Evaluation Infrastructure Notes

- **`results/`** — small artifacts (metrics.json, config.json, CSVs, HTMLs)
  are tracked in git. Large binaries (`.pt`, `.npy`) are gitignored.
- **`syamaner/vinylid-eval`** (HF Hub) — eval artifacts mirrored there for
  external sharing. No model weights (A4-sscd is a public model; B-series
  fine-tuned checkpoints underperformed and are not worth publishing).
- HF Hub push: `python scripts/push_to_hub.py`
- Multi-context comparison report: `python scripts/compare_models.py`

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
