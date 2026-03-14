# ML Quality — Experiment Reproducibility & Data Integrity

Apply these standards when writing or modifying experiment scripts, training code, or evaluation pipelines.

## Experiment Logging
- Every experiment must log: model config, dataset split hash, git SHA, all hyperparameters, start/end timestamps
- Results directory structure: `results/{model_name}/{timestamp}/` containing:
  - `metrics.json` — all evaluation metrics
  - `config.json` — full experiment configuration (model, dataset, augmentation, hardware)
  - Optional: plots, sample predictions, confusion matrices

## Reproducibility
- Set random seeds at script entry point:
  ```python
  torch.manual_seed(seed)
  numpy.random.seed(seed)
  random.seed(seed)
  ```
- Use `torch.use_deterministic_algorithms(True)` where possible (note: some operations don't support this on MPS)
- Evaluation scripts must be idempotent — re-running with same config + seed produces identical results

## Data Provenance
- Record dataset version (manifest hash) in every experiment config
- Record preprocessing steps and augmentation parameters
- Track which split was used (train/val/test) and the split hash

## Evaluation Integrity
- Compare models on identical test data with identical augmentation seeds — never mix evaluation sets between runs
- All metrics computations go through `src/vinylid_ml/eval_metrics.py` — no ad-hoc metric calculations in scripts
- Log hardware info (device, MPS/CPU, torch version) alongside results for reproducibility

## Privacy in Experiments
- Never log file paths to real-world test photos in results that will be shared publicly
- Results uploaded to HF Hub should contain only aggregate metrics, not per-image predictions with paths
- When saving sample predictions for debugging, use release_id (MusicBrainz UUID) not file paths
