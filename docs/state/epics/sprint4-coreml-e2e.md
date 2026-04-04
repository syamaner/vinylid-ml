# Sprint 4 (Planned): CoreML + Index + E2E Integration

## Goal
Deliver the on-device album recognition pipeline:
`camera → detect+unwarp → CoreML embed → flat binary search → return album`
Target: < 500ms on iPhone 14+. Winning model: **A4-sscd** (R@1=0.778 real phone photos).

## Status: 🔄 #26 In Progress

Branch: `feature/26-rectangle-detection-unwarp`

## Latency Budget
| Stage | What | Target |
|-------|------|--------|
| 1 | Detect + unwarp (VNDetectRectanglesRequest + CIPerspectiveCorrection) | 10–20ms |
| 2 | CoreML forward pass (A4-sscd ResNet50, 4-bit palettized) | 20–50ms |
| 3 | Brute-force dot-product search (8,551 albums × 512 dims fp16) | 2–5ms |
| Total | | **40–95ms** |

## Story Status

### #26 — Rectangle Detection + Unwarp
**Status:** 🔄 In Progress
**Branch:** `feature/26-rectangle-detection-unwarp`

#### Python Prototype (gates CoreML input contract)
- [ ] OpenCV contour detection pipeline on 203 labeled phone photos
- [ ] Report detection success rate (target: ≥ 70%)
- [ ] Re-run `evaluate_phone_photos.py` with unwarped input
- [ ] Document R@1 delta vs. 0.778 baseline
- [ ] Decision gate: if improvement → unwarp is Stage 1; if not → CenterCrop stays

#### Swift Implementation
- [ ] `VNDetectRectanglesRequest` (aspect 0.85–1.0, quad tolerance 15°)
- [ ] `CIPerspectiveCorrection` → 224×224 output
- [ ] Fallback hierarchy:
  1. Auto-detect → silent correction
  2. Detection fails → draggable quad overlay with 4 corner handles (edge-snapping)
  3. User skips → center-crop
- [ ] Unit tests for corner-handle interaction

**Background:** Failure analysis on 45 A4-sscd failures found 89% are semantic
look-alike albums with clear phone photos — NOT degradation. However, the current
CenterCrop may include background context (shelf, other records) that contaminates
the embedding. Prototype will determine how much the background context contributes.

**Results:** TBD

---

### #24 — CoreML Conversion + Quantization
**Status:** ⏳ Pending (#26 result gates input contract)

- Model: `sscd_disc_mixup.torchscript.pt` → `sscd_disc_mixup.mlpackage`
- Format: `mlprogram` for ANE on iOS 17+
- Input: `(1, 3, 224, 224)` float32 ImageNet-normalized
  (with or without unwarp depending on #26 result)
- Output: `(1, 512)` float32 L2-normalized embedding
- Quantization: 4-bit palettization (kmeans) + fp16 compute
- Expected size: ~100MB unquantized → ~15–20MB quantized
- Validation gates:
  - Cosine similarity vs. PyTorch > 0.999 on 100 random inputs
  - Recall@1 degradation < 1pp on augmented test-split

**Risk:** TorchScript → CoreML conversion may hit unsupported ops. Verify early.
If ANE unavailable, fall back to GPU compute and re-measure latency.

---

### #25 — Gallery Index Build
**Status:** ⏳ Pending (can run parallel with #24)

- Format: `gallery.idx` (binary header + row-major fp16 matrix)
  + `gallery_meta.json` (album_id, artist, album, image_path per row)
- Size: 8,551 albums × 512 dims × 2 bytes = **8.75MB**
  (Note: story originally estimated 17.7MB for DINOv2 384-dim; corrected to A4-sscd 512-dim)
- Validation: brute-force fp32 Python search must match `embed_gallery.py`
  Recall@1 within 0.1pp

---

### #27 — E2E Evaluation + Latency Benchmarking
**Status:** ⏳ Pending (requires #24 + #25)

- Python: `scripts/evaluate_e2e.py`
  - Chain: detect+unwarp → A4-sscd embed → flat index search
  - Report Recall@1 with and without unwarp (delta documents #26's value)
  - 203 real phone photos, 979-album gallery
- Swift test app on physical iPhone 14+:
  - Per-stage wall-clock: VNDetectRectangles, CIPerspectiveCorrection, CoreML forward, dot-product
  - Target < 500ms total

---

### #28 — Confidence Threshold Calibration
**Status:** ⏳ Pending (requires #27)

Calibrate on val split (855 albums, ~4,500 augmented queries):
- `θ_high` — "confident match": auto-populate metadata (precision ≥ 99%)
- `θ_low` — "possible match": show top-5 candidates for user selection
- Plot precision-recall curve
- Document single-image album confidence (67% of catalog — hardest to distinguish)

## Decisions Made
- A4-sscd (SSCD ResNet50 disc_mixup, 512-dim) is the production model
- Gallery index is per-album (one canonical highest-res image), not per-image
- Fallback for failed rectangle detection: user-assisted 4-corner tap (not silent center-crop)
- Corner handles snap to detected edges to reduce adjustment friction
- Index format is flat binary fp16 (not FAISS) — sufficient for 8,551 items, zero dependencies
