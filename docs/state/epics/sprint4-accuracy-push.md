# Sprint 4: Accuracy Push — Closing the Phone-Photo Gap to >0.90 R@1

## Goal
Reach R@1 > 0.90 on the 203-photo complete phone evaluation set (981-image gallery).
Current best: A4-sscd R@1=0.778 (zero-shot).

## Status: 🔄 Phase 1 Implementing

Branch: `feature/sprint4-accuracy-push`

## Active Context
Phase 1 code implementation complete:
- `A5-sscd-blur` model added to `models.py` and `create_model()` factory
- `evaluate_phone_photos.py` extended with A3-featureprint support, TTA, alpha-QE
- All tests pass (318 unit), ruff clean, pyright clean
- Awaiting evaluation runs on remote CUDA (A5) and locally (A3, TTA, alpha-QE)

## Evaluation Runs Needed

### Phase 1a — Untested models on phone complete set (remote CUDA or local)
| Model | Requires | Status |
|-------|----------|--------|
| A5-sscd-blur | `embed_gallery.py --model A5-sscd-blur` then `evaluate_phone_photos.py --model A5-sscd-blur` | Pending |
| A3-featureprint | `embed_featureprint.py` then `evaluate_phone_photos.py --model A3-featureprint` | Pending (macOS) |

### Phase 1b — Inference-time improvements on A4-sscd (local)
| Experiment | Command | Status |
|------------|---------|--------|
| TTA-5 on A4-sscd | `evaluate_phone_photos.py --model A4-sscd --tta --tta-n-augs 5` | Pending |
| alpha-QE k=5 α=0.5 | `evaluate_phone_photos.py --model A4-sscd --alpha-qe` | Pending |
| TTA+QE combined | `evaluate_phone_photos.py --model A4-sscd --tta --alpha-qe` | Pending |

## Latest Results

| Run Label | R@1 | R@5 | mAP@5 | Notes |
|-----------|-----|-----|-------|-------|
| A4-sscd (baseline) | 0.778 | — | — | Sprint 3 best on real phones |
| A3-featureprint | TBD | — | — | Never run on phone set |
| A5-sscd-blur | TBD | — | — | Never run; blur variant |
| A4-sscd-tta5 | TBD | — | — | TTA not yet tried |
| A4-sscd-aqe5a0.5 | TBD | — | — | Alpha-QE not yet tried |

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
