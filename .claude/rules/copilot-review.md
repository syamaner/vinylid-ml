# Copilot / AI Code Review Instructions

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
- **`torch.load(..., weights_only=False)`** — our checkpoints contain `optimizer_state_dict` and `loss_state_dict` produced exclusively by `scripts/train.py`. Trusted internal file.
- **`embed_finetuned.py --output-dir` vs `data_dir`** — `--output-dir` controls where embeddings are written; `data_dir` (manifest/splits) always comes from the config. Intentional.

## Always Flag
- Logic errors: off-by-one, incorrect condition, wrong variable used
- Tautological assertions in tests (e.g., `sorted(x) == sorted(x)` — always passes)
- Missing input validation that the CLI/API docs promise
- Privacy violations: real-world photo paths in code or results, EXIF not stripped
- Missing seeds for RNG in training/evaluation scripts
- Security: hardcoded secrets, path traversal in user-supplied strings

## Never Flag
- "Consider refactoring" or "you could also" style suggestions
- Suggestions to add comments to self-documenting code
- Alternative implementations that are style preferences
- Anything already justified by an inline comment or docstring in the diff
