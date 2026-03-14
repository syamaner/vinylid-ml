# AGENTS.md — VinylID ML Project Rules

## Language & Tooling
- Python 3.11+
- Type hints on all public functions
- Google-style docstrings on all public functions and classes
- `ruff` for linting and formatting (line length 100)
- `pyright` in strict mode for type checking

## Experiments & Reproducibility
- All experiments must be reproducible: seed all RNG (`torch.manual_seed`, `numpy.random.seed`, `random.seed`), log all hyperparameters
- Results go to `results/{model_name}/{timestamp}/` with `metrics.json` and `config.json`
- Evaluation scripts are idempotent — same config + seed = identical results

## Dependencies
- No commercial-license dependencies — verify before adding any new library
- Prefer `torch.hub` or Hugging Face Hub for model loading
- **All dependencies must be declared in `pyproject.toml`** — never install with bare `pip install`. Add the dep to `pyproject.toml` first (under `dependencies` for runtime or `[project.optional-dependencies] dev` for dev-only), then run `pip install -e '.[dev]'` to install.
- Type stubs (e.g., `pandas-stubs`, `types-PyYAML`) go under `dev` optional dependencies

## Testing
- Test with `pytest`
- ML evaluation scripts double as integration tests
- Run `ruff check` and `pyright` before considering code complete

## Large Files
- Models, embedding indexes, and datasets go to Hugging Face Hub — never committed to git
- Gallery images are referenced by MusicBrainz UUID, not redistributed
- See `configs/dataset.yaml` for dataset path configuration

## Commit Messages
- Reference the relevant plan section (e.g., "Section 4: implement manifest builder")
- Include `Co-Authored-By: Oz <oz-agent@warp.dev>` when AI-assisted

## Privacy — Real-World Test Photos
Real-world test photos (phone captures from `/Volumes/nvme1/vinyl/vinylid/test/`) are **offline by default**. They may contain:
- Personal faces visible in the background
- GPS coordinates and other location data in EXIF metadata

**Rules:**
- Never commit test photos to git
- Never upload unapproved test photos to Hugging Face Hub or any public location
- When loading test photos for evaluation, strip all EXIF metadata in-memory before processing — the pipeline only needs pixel data. Do NOT modify originals.
- Individual photos may be shared publicly **only after manual approval**: must contain no personal faces and must be fully metadata-stripped (all EXIF/XMP/IPTC removed)
- Approved photos go in `test_photos_approved/` (tracked by git)
- Aggregate metrics (Recall@1, mAP, latency, etc.) may always be shared

**`.gitignore` entries** enforce this:
- `test_photos/` — any local symlinks or copies of test data
- `results/` — experiment outputs (uploaded to HF Hub, not git)
- `coreml/` — exported models (uploaded to HF Hub, not git)
