# VinylID ML — Album Cover Recognition

On-device album cover recognition for a vinyl record collection app. Camera → detect cover → extract embedding → match against gallery → return album identity.

**Target**: < 500ms end-to-end on iPhone 14+.

## Dataset

| Metric | Value |
|--------|-------|
| Gallery images | 23,098 |
| Albums | 8,555 |
| Artists | 352 |
| Single-image albums | 67% |
| Image sizes | 200–4,290px |
| Real-world test photos | 243 labeled iPhone captures |

Gallery images are sourced from [Cover Art Archive](https://coverartarchive.org/) via MusicBrainz, with metadata embedded in EXIF.

## Approach

We evaluate **10 model configurations** across 4 categories to find the best accuracy/latency tradeoff:

- **Category A — Zero-shot global embeddings**: DINOv2 (CLS + GeM), OpenCLIP, Apple FeaturePrint, SSCD
- **Category B — Fine-tuned global embeddings**: MobileNetV3 + ArcFace, DINOv2 + ArcFace, SSCD + ArcFace
- **Category C — Local feature matching**: DINOv2 patch-level, SuperPoint + LightGlue
- **Category D — Hybrid**: Best global model → LightGlue re-ranking

All models are converted to CoreML for on-device inference. Results are published as a blog series.

## Privacy

Real-world test photos (phone captures of vinyl covers) are **offline by default** — they may contain personal faces and GPS coordinates in EXIF. They are never committed to git or uploaded publicly.

Individual test photos may be shared only after **manual approval**: no personal faces, all metadata stripped. Aggregate evaluation metrics are always shareable.

See `AGENTS.md` for the full privacy policy.

## Project Structure

```
vinylid-ml/
├── AGENTS.md                # Project rules (coding standards, ML conventions, privacy)
├── pyproject.toml           # Python 3.11+, all dependencies
├── README.md
├── .skills/                 # Warp AI coding skills
├── configs/                 # dataset.yaml, models.yaml
├── scripts/                 # Dataset prep, evaluation, training, CoreML export
├── src/vinylid_ml/          # Library code (models, datasets, metrics, search)
├── notebooks/               # EDA and exploration
├── results/                 # Experiment outputs (git-ignored, large files go to HF Hub)
├── coreml/                  # Exported CoreML models + indexes
└── test_photos_approved/    # Manually approved, metadata-stripped test photos
```

## Quick Start

```bash
# Clone
git clone https://github.com/syamaner/vinylid-ml.git
cd vinylid-ml

# Create virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# Install package + dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint and type-check
ruff check src/ scripts/ tests/
pyright --project . src/

# Configure dataset paths (edit with your local paths)
# See configs/dataset.yaml — set gallery_root and sqlite_db

# Build dataset manifest and splits
python scripts/prepare_dataset.py

# Run evaluation (example: DINOv2 zero-shot)
python scripts/evaluate.py --model A1-dinov2-cls
```

> **Note**: All dependencies must be declared in `pyproject.toml` before installing.
> Never use bare `pip install <package>` — add the dep to `pyproject.toml` first,
> then `pip install -e ".[dev]"`. See `AGENTS.md` for the full dependency policy.

## Models & Results

Model checkpoints, evaluation metrics, and embedding indexes are hosted on Hugging Face Hub:

- **Evaluation staging**: [syamaner/vinylid-eval](https://huggingface.co/syamaner/vinylid-eval) (all models during development)
- **Production**: dedicated repos for winning model(s) after evaluation

## Links

- [Phase 2 Implementation Plan](docs/phase2-plan.md) *(coming soon)*
- Blog series *(coming soon)*

## License

TBD — model weights likely MIT or Apache-2.0. Dataset metadata references Cover Art Archive (separate consideration).
