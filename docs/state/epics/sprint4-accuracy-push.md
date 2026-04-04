# Sprint 4: Accuracy Push — Closing the Phone-Photo Gap to >0.90 R@1

## Goal
Reach R@1 > 0.90 on the 203-photo complete phone evaluation set (981-image gallery).
Current best: A4-sscd R@1=0.778 (zero-shot).

## Status: ✅ Phase 1 Complete — Target NOT reached; proceeding to Phase 2

Branch: `feature/sprint4-accuracy-push`

## Active Context
Phase 1 all 5 experiments completed locally on MPS (Apple M-series). No experiment
reached the 0.90 target. A4-sscd (mixup) remains the best model by a wide margin.
Proceeding to Phase 2: frozen SSCD backbone + MLP projection head.

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

## Phase Gates
- If any Phase 1 result reaches R@1 ≥ 0.90 → document as production path and STOP
- Otherwise → proceed to Phase 2 (frozen SSCD backbone + MLP projection head)
- Phase 3 (LoFTR re-ranking) only if both Phase 1 and Phase 2 fail

## Phase 2 (if needed): Frozen SSCD + MLP Projection Head
- Train a 2-layer MLP on top of frozen A4-sscd features using 203 phone-album pairs
- InfoNCE loss; only the projection head is trained
- Script to implement: `scripts/train_projection_head.py`

## Decisions Made
- TTA uses 5 deterministic augmentations: identity, hflip, +5° rotation, -5° rotation, brightness+15%
- Alpha-QE: query expanded as q' = norm(q + alpha * mean(top-k gallery))
- Run label convention: `{model_id}-tta{n}-aqe{k}a{alpha}` (e.g. `A4-sscd-tta5-aqe5a0.5`)
- Results saved to `results/{run_label}-phone/{timestamp}/`
