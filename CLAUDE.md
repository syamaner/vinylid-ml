@AGENTS.md

## Claude Code

### Epic State Management

This project tracks active work using state files in `docs/state/`.
Before beginning any task, read `docs/state/registry.md` to identify the
relevant epic. Then read the linked epic file to understand current status,
latest results, and the next experiment or step.

When you finish a task or are interrupted:
1. Update the epic state file — check off completed steps, update "Active Context"
   and "Latest Results", and log any decisions or blockers.
2. If the epic's overall status changes, update `docs/state/registry.md` too.

### Commands
- `python -m pytest tests/ -v --tb=short` — run unit tests (excludes @pytest.mark.integration)
- `ruff check src/ tests/ scripts/` — lint
- `ruff format --check src/ tests/ scripts/` — format check
- `pyright src/` — type check (strict mode, 0 errors required)
