# VinylID ML — Copilot Code Review Instructions

Python 3.11+ ML project for on-device album cover recognition. PyTorch, torchvision, ruff, pyright strict.

## Review Philosophy
- Only comment when you have **high confidence (>80%)** that a real defect exists
- **Maximum 5 comments per review** — prioritise the most important issues only
- Be concise: one sentence per comment when possible
- Focus on **actionable defects**, not observations or style preferences
- Do **not** re-flag an issue that has already been addressed, explained in a comment, or has a docstring justifying the decision
- Do **not** comment on code outside the changed lines (pre-existing code not in the diff)

## CI Already Covers — Do Not Flag
- Formatting (`ruff format`) — do not comment on style or line length
- Linting (`ruff check`) — do not flag unused imports, naming conventions, etc.
- Type errors (`pyright` strict mode) — do not flag missing type hints or type mismatches
- Test failures (`pytest`) — do not speculate about test correctness without evidence

## Known Project Decisions — Do Not Flag
These are intentional choices already documented in the codebase:
- **`torch.load(..., weights_only=False)`** — our checkpoints contain `optimizer_state_dict` and `loss_state_dict` produced exclusively by `scripts/train.py`. The comment in the code explains this. Do not flag.
- **`embed_finetuned.py --output-dir` vs `data_dir`** — `--output-dir` controls where embeddings are written; `data_dir` (manifest/splits) always comes from the config. This is intentional, not a bug.
- **`weights_only=True` suggestions** — our full checkpoint format (optimizer state, loss state) requires `False`. Do not suggest switching.
- **Pre-existing formatting issues** — any files not in the diff that have formatting differences are pre-existing and out of scope.

## Focus: High-Value Issues Only

### Always Flag
- Logic errors: off-by-one, incorrect condition, wrong variable used
- Tautological assertions in tests (e.g., `sorted(x) == sorted(x)` — always passes)
- Missing input validation that the CLI/API docs promise (e.g., method says raises ValueError but doesn’t)
- Privacy violations: real-world photo paths in code or results, EXIF not stripped
- Missing seeds for RNG in training/evaluation scripts (reproducibility requirement)
- Security: hardcoded secrets, path traversal in user-supplied strings, arbitrary code execution

### Flag with Justification Required
- Missing error handling for external I/O (only if the failure mode is unclear or silent)
- Docstring missing `Raises:` section (only for public functions that do raise)
- Tensor shape mismatches or wrong normalization applied to a model

### Never Flag
- "Consider refactoring" or "you could also" style suggestions
- Suggestions to add comments to self-documenting code
- Alternative implementations that are style preferences
- Anything already justified by an inline comment or docstring in the diff

## Project Architecture (Context)
- `src/vinylid_ml/` — library code. `scripts/` — CLI entry points. `tests/` — pytest unit + integration tests
- `EmbeddingModel` ABC: all wrappers implement `embed()`, `get_transforms()`, `input_size`
- `TrainingConfig` dataclass in `training.py` — all hyperparameters including `backbone_lr_mult`, `llrd_decay`, `unfreeze_blocks`, `patience`
- Checkpoints saved by `train.py` contain: `epoch`, `model_state_dict`, `loss_state_dict`, `optimizer_state_dict`, `val_recall_at_1`
- LLRD param groups for DINOv2: 14 groups (12 ViT blocks + norm/embed + head) with LR decaying 0.9× per block inward — this is correct and intentional
