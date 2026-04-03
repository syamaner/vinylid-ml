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

## Training & Performance Guardrails
- Validate every new CLI/config argument explicitly, including allowed ranges and invalid argument combinations. Fail with clear user-facing errors instead of relying on downstream framework exceptions.
- Do not assume training, validation, and test DataLoaders should share the same settings. Validation/test loaders should prefer conservative defaults unless there is evidence that more aggressive settings are safe.
- High-impact throughput knobs such as `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers` must be configurable. Prefer conservative defaults; aggressive settings should be opt-in or clearly justified.
- Only use `persistent_workers=True` on long-lived DataLoaders that are reused across epochs. Do not enable it on short-lived or repeatedly recreated loaders.
- If enabling optimizations that may reduce determinism (for example `torch.backends.cudnn.benchmark = True`), log or document that tradeoff explicitly.
- When reporting speed or hardware comparisons, clearly label whether the comparison is apples-to-apples (same workload/config) or best-stable practical (different configs chosen for stability/throughput). Do not present mixed-config comparisons as pure hardware benchmarks.

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

## LightGlue on MPS (Apple Silicon)
- **CrossBlock SDPA patch**: LightGlue's `CrossBlock` uses manual einsum on non-CUDA devices. Patch
  `layer.cross_attn.forward` with `types.MethodType` to call `F.scaled_dot_product_attention` — gives
  ~2.25x speedup (243ms → 108ms/match in same-domain micro-benchmark). See `local_features._patch_cross_attention_sdpa`.
- **`flash=True` does NOT help on MPS**: The half-precision SDPA path in `Attention.forward` is guarded
  by `q.device.type == "cuda"`. On MPS, the code always falls to the plain SDPA branch (no FP16).
- **MPS autocast is SLOWER for small tensors**: `torch.autocast(device_type="mps", dtype=torch.float16)`
  adds overhead that outweighs FP16 gains at LightGlue's attention matrix sizes (1×4×512×64).
  Metal FP16 benefit requires matrices ≥ 1024×1024 to be worthwhile.
- **Cross-domain early-exit never fires**: Phone photo → catalog pairs have low token confidence at every
  layer, so `depth_confidence=0.95` never triggers early stopping — all 9 layers always run (~350ms/match
  cross-domain vs ~108ms same-domain). Real-world latency is ~17–42s/query (thermal throttling) at 512 kp.
- **Pre-filter choice matters**: SuperPoint mean-descriptor pre-filter has ~15% recall@50 on phone×catalog
  pairs. A4-sscd (trained for copy detection) is the correct pre-filter for cross-domain retrieval.

## Epic State Management
- This project tracks active work in `docs/state/`.
- Before starting a story, read `docs/state/registry.md` to find the relevant epic file.
- Read the epic file for current status, latest results, and next steps.
- When finishing a task or pausing, update the epic file (check off steps, refresh Active Context and Latest Results, log decisions).
- If the epic's overall status changes, update `docs/state/registry.md` too.

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
