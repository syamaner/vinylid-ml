# Story #13 — C2 SuperPoint+LightGlue: Full Test-Split Evaluation

## Status: IN PROGRESS

## Goal
Evaluate C2 (SuperPoint+LightGlue) on the 855-album test-split gallery using
A4-sscd pre-filtering (top-50) + LightGlue re-ranking. Cannot run against full
23k gallery (too slow). Ref: Phase 2 Plan §7 (C2).

## Approach
- Mode: `evaluate_local_features.py --mode test --top-k 50`
- Gallery: 855 canonical test-split images (1 per album, highest-res)
- Queries: ~4 564 remaining test-split images
- Pre-filter: A4-sscd cosine similarity → top-50 candidates per query
- Matching: LightGlue on top-50 candidates; non-candidates keep A4-sscd score
- Runtime estimate: ~20 min on RTX 4090

## Execution Environment
- Machine: Ubuntu RTX 4090 (192.168.0.158)
- Config: `configs/dataset-remote.yaml`
- Branch: `feature/13-15-c2-local-features-phone-eval`

## Active Context
Running on remote CUDA machine. Requires A4-sscd embeddings pre-computed for
test split (`data/A4-sscd/`). Results saved to `results/C2-superpoint-lightglue/{timestamp}/`.

## Steps
- [ ] git pull main + pip install -e .[dev] on remote
- [ ] Run prepare_dataset.py with dataset-remote.yaml
- [ ] Run embed_gallery.py --model A4-sscd --split test
- [ ] Run evaluate_local_features.py --mode test --top-k 50
- [ ] Record results below

## Latest Results
*(pending — evaluation not yet run)*

## Notes
- A4-sscd pre-filter needed because SuperPoint mean-descriptor gives only ~15%
  recall@50 on cross-domain pairs; A4-sscd gives >90%.
- `--precompute-gallery-tensors` flag available for faster matching at cost of
  more GPU memory (~24GB available on RTX 4090, so this should be safe).
