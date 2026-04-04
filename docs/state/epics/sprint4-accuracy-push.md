# Sprint 4: Accuracy Push — Closing the Phone-Photo Gap to >0.90 R@1

## Goal
Reach R@1 > 0.90 on the 203-photo complete phone evaluation set (981-image gallery).
Current best: A4-sscd R@1=0.778 (zero-shot).

## Status: ❌ CLOSED — Target NOT reached; best achievable is R@1=0.783 with current data

Branch: `feature/sprint4-accuracy-push`

## Active Context
All three planned phases evaluated. Target R@1 > 0.90 not reached.
**Recommended production path: A4-sscd (zero-shot, R@1=0.778).** The TTA+QE
combination adds +0.5pp to 0.783 at the cost of 6x inference time.
Further gains require more labelled phone-album pairs (hundreds to thousands)
or calibrated synthetic augmentation of the full training corpus.

## Phase 1 Results

| Run Label | R@1 | R@5 | mAP@5 | MRR | Notes |
|-----------|-----|-----|-------|-----|-------|
| A4-sscd (Sprint 3 baseline) | 0.778 | — | — | — | Remote CUDA |
| A3-featureprint | 0.498 | 0.650 | 0.558 | 0.572 | Local MPS; gallery 979 |
| A5-sscd-blur | 0.611 | 0.704 | 0.645 | 0.653 | Local MPS; gallery 979 |
| A4-sscd-tta5 | 0.778 | 0.818 | 0.794 | 0.797 | TTA no change over baseline |
| A4-sscd-aqe5a0.5 | 0.773 | 0.798 | 0.785 | 0.788 | QE alone slightly hurts |
| **A4-sscd-tta5-aqe5a0.5** | **0.783** | **0.818** | **0.798** | **0.800** | Best Phase 1; +0.5pp |

**Key findings:**
- A4-sscd (mixup) is definitively the best SSCD variant; blur variant (A5) scores lower (0.611), suggesting blur augmentation does not match the phone-photo domain
- A3-featureprint is weakest (0.498); general Apple Vision model, not copy-detection
- TTA neither helped nor hurt alone; 5 mild augmentations do not reduce the domain gap
- Alpha-QE alone slightly hurts (wrong top-k neighbours pulled in at 0.778 baseline quality)
- TTA+QE combined ekes out +0.5pp to 0.783 by improving initial quality enough for QE

## Commands for Evaluation Runs

### Remote CUDA machine: A5-sscd-blur
```bash
# 1. Embed gallery with A5-sscd-blur
python scripts/embed_gallery.py \
    --config configs/dataset-remote.yaml \
    --model A5-sscd-blur \
    --split test

# 2. Evaluate on phone complete set
python scripts/evaluate_phone_photos.py \
    --config configs/dataset-remote.yaml \
    --model A5-sscd-blur
```

### Local Mac: A3-featureprint
```bash
# 1. Embed gallery (if not done yet)
python scripts/embed_featureprint.py --config configs/dataset.yaml

# 2. Evaluate on phone complete set
python scripts/evaluate_phone_photos.py \
    --config configs/dataset.yaml \
    --model A3-featureprint
```

### Local Mac: TTA and alpha-QE on A4-sscd
```bash
# TTA only
python scripts/evaluate_phone_photos.py \
    --config configs/dataset.yaml \
    --model A4-sscd --tta --tta-n-augs 5

# Alpha-QE only
python scripts/evaluate_phone_photos.py \
    --config configs/dataset.yaml \
    --model A4-sscd --alpha-qe --alpha-qe-k 5 --alpha-qe-alpha 0.5

# TTA + Alpha-QE combined
python scripts/evaluate_phone_photos.py \
    --config configs/dataset.yaml \
    --model A4-sscd --tta --alpha-qe
```

## Phase 2 Results: Frozen SSCD + MLP Projection Head

Script: `scripts/train_projection_head.py`
Config: 512->256->128, dropout=0.2, AdamW lr=1e-3, NT-Xent T=0.07, 300 epochs
Data: 161 train pairs / 42 val pairs (album-level 80/20 split), 979-image gallery

| Epoch | Train Loss | Val R@1 (out-of-sample) |
|-------|-----------|-------------------------|
| 10 | 0.0998 | **0.405** (peak) |
| 50 | 0.0460 | 0.333 |
| 100 | 0.0705 | 0.381 |
| 200 | 0.0412 | 0.357 |
| 300 | 0.0250 | 0.310 |

Full 203-photo R@1 (80% in-sample): 0.877 — **misleading, do not cite**.

**Result: FAILED to generalise.** Val R@1 peaks at 0.405 at epoch 10 (vs. 0.778 baseline)
then degrades to 0.310 by epoch 300 despite training loss dropping to 0.025.

Root cause: 161 training pairs vs. 162,560 head parameters. In-batch negatives from
161 samples provide insufficient diversity. The head memorises the training phone photos
rather than learning a domain-invariant alignment.

## Phase 3 Decision

Phase 3 (LoFTR detector-free re-ranking) is **not attempted**. Sprint 3 already
evaluated C2 LightGlue on the full 203-photo set and got R@1=0.621 — worse than
the 0.778 A4-sscd baseline. LoFTR faces the same fundamental issue: textureless,
foreshortened vinyl surfaces produce sparse and poorly-localised keypoints in
cross-domain (phone vs. flat catalog scan) pairs. Investing in Phase 3 has low
expected return given the Sprint 3 evidence.

## Final Sprint 4 Conclusions

The 0.90 target is not achievable with the current data and approach.

| Approach | R@1 | vs. Baseline | Recommendation |
|----------|-----|-------------|----------------|
| A4-sscd zero-shot (baseline) | 0.778 | — | **Production path** |
| A4-sscd + TTA+QE | 0.783 | +0.5pp | Optional; 6x slower |
| A5-sscd-blur | 0.611 | -17pp | Discard |
| A3-featureprint | 0.498 | -28pp | Discard |
| E1-sscd-projhead (Phase 2) | 0.405 val | -37pp | Overfit; discard |

The gap between A4-sscd (0.778) and target (0.900) is a data problem:
- 203 labelled pairs is insufficient for domain adaptation by any method tried
- The SSCD zero-shot representation is already near the ceiling for this data regime

**Paths to >0.90 (future sprints):**
1. Scale labelled pairs to 1000+ (label more phone captures per album across the collection)
2. Calibrated synthetic domain augmentation of the full 8551-album training corpus
   using glare/bokeh/perspective warp tuned to match the actual phone capture conditions
3. Dataset-side fix: encourage users to photograph covers flat-on under consistent lighting

## Decisions Made
- TTA uses 5 deterministic augmentations: identity, hflip, +5 rotation, -5 rotation, brightness+15%
- Alpha-QE: query expanded as q' = norm(q + alpha * mean(top-k gallery))
- Run label convention: `{model_id}-tta{n}-aqe{k}a{alpha}` (e.g. `A4-sscd-tta5-aqe5a0.5`)
- Results saved to `results/{run_label}-phone/{timestamp}/`
- Phase 2 checkpoint (projection_head.pt) gitignored per AGENTS.md large-file policy
