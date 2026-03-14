# Code Review — Pre-Commit Validation Checklist

Use this skill before every commit. All steps are mandatory — do not skip any.

## 1. Run All Tests
```bash
python -m pytest tests/ -v --tb=short
```
- All tests must pass (0 failures)
- Review any new warnings — address if related to new code

## 2. Run Linter
```bash
ruff check src/ tests/ scripts/
```
- Must report "All checks passed!"
- Fix any errors before proceeding (`ruff check --fix` for auto-fixable)
- Also run `ruff format --check src/ tests/ scripts/` to verify formatting

## 3. Run Type Checker
```bash
pyright src/
```
- Must report 0 errors and 0 warnings
- `type: ignore` comments are acceptable only for untyped third-party library calls (e.g., `torch.hub`, `open_clip`) — never for our own code

## 4. Code Review Checklist
Review every changed file against these criteria:

### Architecture & Design
- [ ] New public classes/functions have Google-style docstrings with Args/Returns/Raises
- [ ] Tensor shapes documented in docstrings (e.g., `(B, C, H, W)`)
- [ ] No circular imports introduced
- [ ] `__all__` updated in `__init__.py` if new public symbols added
- [ ] New dependencies declared in `pyproject.toml` before use
- [ ] No commercial-license dependencies added

### Coding Standards (from python-ml-standards)
- [ ] File-scoped imports (lazy-loading OK for heavy libraries like `open_clip`)
- [ ] Absolute imports only
- [ ] `pathlib.Path` over `str` for file paths
- [ ] `structlog` for logging — no `print()` in `src/`
- [ ] Custom exceptions inherit `VinylIDError`
- [ ] Constants in `UPPER_SNAKE_CASE`
- [ ] Configuration via dataclasses, not raw dicts

### ML Quality (from ml-quality)
- [ ] Random seeds set at script entry points
- [ ] Experiment configs logged (model, dataset, hyperparams, hardware)
- [ ] Metrics computed via `eval_metrics.py` — no ad-hoc calculations
- [ ] No real-world test photo paths in shared results

### Testing
- [ ] New code has corresponding tests (TDD: tests written first)
- [ ] Unit tests for pure functions, integration tests for model/IO code
- [ ] Tests cover both happy path and error cases
- [ ] No test implementation/fakes in production code

### Privacy & Security
- [ ] No test photo paths committed to git
- [ ] No secrets or API keys in code
- [ ] EXIF stripped in-memory when processing real-world photos

## 5. Commit Message Format
```
#<issue_number>: Brief description

Detailed explanation of what changed and why.

Ref: Phase 2 Plan §<section>

Co-Authored-By: Oz <oz-agent@warp.dev>
```

## Validation Failure Policy
If ANY validation step fails, do NOT commit. Fix the issue first, then re-run ALL validation steps from the beginning.
