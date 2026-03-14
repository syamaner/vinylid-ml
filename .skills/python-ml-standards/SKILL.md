# Python ML Standards — Coding Style & Conventions

Apply these standards when writing or modifying any Python code in this repository.

## Imports
- File-scoped imports only (no imports inside functions unless lazy-loading heavy libraries)
- Absolute imports only — no relative imports
- Define `__all__` exports in `__init__.py`

## Type Hints
- Type hints on all public functions and methods
- Use `Literal` for constrained string/int parameters
- Use `pathlib.Path` over `str` for all file path parameters and return types
- Use `torch.Tensor` (not `Any`) for tensor types

## Documentation
- Google-style docstrings on all public functions and classes
- Include `Args:`, `Returns:`, and `Raises:` sections
- Document tensor shapes in docstrings (e.g., `(B, C, H, W)`)

## Logging
- Use `structlog` for all structured logging
- No `print()` in library code (`src/vinylid_ml/`) — `print` is OK in scripts (`scripts/`)
- Log at appropriate levels: `debug` for verbose details, `info` for progress, `warning` for recoverable issues, `error` for failures

## Error Handling
- Custom exceptions inherit from `VinylIDError` base class
- Never use bare `except:` — always catch specific exception types
- Raise `ValueError` for invalid arguments, custom exceptions for domain errors

## File Paths
- Use `pathlib.Path` consistently — no `os.path` functions
- Accept `Path` parameters, return `Path` values

## Configuration
- Use dataclasses or Pydantic models for configuration — not raw dicts
- Load config from YAML files via `configs/` directory
- Constants in `UPPER_SNAKE_CASE` at module level

## Formatting
- `ruff` for linting and formatting
- Line length: 100 characters
- `pyright` strict mode for type checking
- Run `ruff check --fix` and `ruff format` before committing
