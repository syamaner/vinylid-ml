# VinylID ML — Copilot Code Review Instructions

This is a Python 3.11+ ML project for on-device album cover recognition via embedding-based retrieval. We use PyTorch, torchvision, open_clip, and CoreML.

## Architecture
- Clean separation: library code in `src/vinylid_ml/`, scripts in `scripts/`, tests in `tests/`
- `EmbeddingModel` ABC in `models.py` — all model wrappers implement `embed()`, `get_transforms()`, `input_size`
- Each model has its own normalization (ImageNet vs CLIP) — callers must use `model.get_transforms()`
- Lazy imports in `__init__.py` via `__getattr__` to avoid loading torch for lightweight modules (e.g., `exif.py`)

## Coding Standards
- Type hints on all public functions; `Literal` for constrained params, `pathlib.Path` for file paths
- Google-style docstrings with `Args:`, `Returns:`, `Raises:`. Document tensor shapes `(B, C, H, W)`
- `structlog` for logging — no `print()` in `src/`. Custom exceptions inherit `VinylIDError`
- Configuration via dataclasses, not raw dicts. Constants in `UPPER_SNAKE_CASE`
- Absolute imports only. Define `__all__` in `__init__.py`
- `ruff` for lint/format (line-length 100), `pyright` strict mode

## ML-Specific Rules
- `torch.inference_mode()` for all evaluation/embedding extraction
- Batch processing — never embed one image at a time
- Pre-normalize embeddings to unit length (L2). Store as float16 `.npy`
- All metrics via `eval_metrics.py` — no ad-hoc calculations
- Random seeds set at script entry points for reproducibility
- MPS on macOS when available, CPU fallback. No crash if MPS unavailable

## Testing
- TDD: tests written before implementation
- Unit tests for pure functions, integration tests (marked `@pytest.mark.integration`) for model/IO
- No test fakes/mocks in production code
- xunit-style with pytest

## Security & Privacy
- No real-world test photo paths in code or shared results
- No secrets or API keys in code
- EXIF stripped in-memory when processing real-world photos
- No commercial-license dependencies

## CI Already Checks
- `ruff check` and `ruff format --check`
- `pyright` strict mode
- `pytest` with all tests passing
Do not flag issues that CI will catch (formatting, unused imports, type errors).

## What to Focus On
- Incorrect tensor shapes or normalization mismatches between models
- Missing or misleading docstrings, especially tensor shape documentation
- Circular imports or eager imports that break lazy loading
- Privacy violations (photo paths, EXIF data leaks)
- Missing error handling for file I/O and model loading
- Reproducibility issues (unseeded randomness, non-deterministic operations)
