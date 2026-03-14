---
applyTo: "src/**/*.py"
---

# Python ML Source Code Review

## Model Wrappers (`models.py`)
- Every `EmbeddingModel` subclass must implement `embed()`, `get_transforms()`, `input_size`, `embedding_dim`, `model_id`
- `embed()` must return L2-normalized tensors on CPU
- `get_transforms()` must return the correct normalization for that model (ImageNet vs CLIP)
- Use `torch.inference_mode()` inside `embed()`, not `torch.no_grad()`

## Gallery Pipeline (`gallery.py`)
- Embeddings saved as float16 `.npy` with companion `metadata.json`
- `GalleryImageDataset` uses model-specific transforms, not hardcoded ImageNet
- `embed_dataset()` logs progress and throughput

## Dataset (`dataset.py`)
- Splits at album level, not image level (prevents data leakage)
- `AlbumCoverDataset.__getitem__` must convert images to RGB

## General
- No `print()` — use `structlog`
- No `os.path` — use `pathlib.Path`
- All public functions need Google-style docstrings with tensor shapes
- `__all__` must be updated when adding public symbols
- Lazy imports in `__init__.py` — do not add eager imports that load torch
