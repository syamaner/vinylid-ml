# Epic #23: Hugging Face Hub Push

## Goal
Upload Sprint 3 evaluation artifacts to `syamaner/vinylid-eval` model repo.

## Status
- [x] `scripts/push_to_hub.py` created
- [ ] Execute after #22 report is generated

## What Gets Uploaded
- `results/comparison.csv`, `results/comparison.html`
- `results/multi_context_comparison.csv`, `results/multi_context_comparison.html`
- `results/summary.csv`, `results/phone_eval_summary.csv`,
  `results/phone_sample_eval_summary.csv`
- Per-run `metrics.json` and `config.json` for all model runs

## What Is Excluded (privacy guardrails)
- `per_query.csv` files (may contain phone photo filenames)
- Binary blobs: `*.pt`, `*.pth`, `*.npy`, `*.npz`, `*.safetensors`, etc.

## Run Command
```bash
# Dry run first to verify what will be uploaded:
python scripts/push_to_hub.py --dry-run

# Upload:
python scripts/push_to_hub.py \
    --commit-message "53: Sprint 3 — A-series phone-sample + D1 K-sweep + multi-context report"
```

## Active Context
Waiting for #22 to complete (remote results synced + compare_models.py run).
