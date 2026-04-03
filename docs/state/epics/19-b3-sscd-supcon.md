# Epic #19 — B3: SSCD + SupCon

## Objective

Fine-tune the SSCD ResNet50 backbone with SupCon loss and evaluate whether it
can exceed the A4 zero-shot baseline (R@1=0.872) and/or the B2c DINOv2+SupCon
leader (R@1=0.8795).

## Status Map

- [x] Update GitHub issue #19 title/description to SupCon
- [x] Create branch `feature/19-b3-sscd-supcon`
- [x] Run 1 (baseline): `freeze_epochs=5`, `backbone_lr_mult=0.1`
- [x] Test-split evaluation for Run 1
- [x] Run 2 (retune): `freeze_epochs=10`, `backbone_lr_mult=0.01`
- [x] Decide on Run 3 hypothesis or close the story

## Active Context

**Current task:** None — story closed.

**Last action:** Follow-up remote CUDA throughput/stability work on
`feature/19-cuda-ssh-support` was completed and documented. A stable full SSCD
run on the RTX 4090 now exists, but it does not change the story outcome.

## Latest Results

### Run 1 — baseline (2026-04-02)
Config: `freeze_epochs=5`, `backbone_lr_mult=0.1`, `lr=1e-4`, `batch_size=32`
- Best val R@1: **0.8431** (epoch 5)
- Test: R@1=0.8604, R@5=0.8948, mAP@5=0.8751, MRR=0.8772
- Checkpoint: `results/B3-sscd-supcon/2026-04-02T22-17-48/best_checkpoint.pt`
- 16 epochs (early stopped); peaked right at unfreeze, then drifted down

### Run 2 — conservative retune (2026-04-02)
Config: `freeze_epochs=10`, `backbone_lr_mult=0.01`, `lr=1e-4`, `batch_size=32`
- Best val R@1: **0.8349** (epoch 1)
- No test eval — below Run 1 throughout
- Checkpoint: `results/B3-sscd-supcon-r2/2026-04-02T23-51-28/best_checkpoint.pt`
- 12 epochs (early stopped); never exceeded epoch-1 val; unfreeze had no effect

### Follow-up — remote CUDA readiness (2026-04-03)
Branch: `feature/19-cuda-ssh-support`
- Phase 1 CUDA support implemented and validated locally (`ruff`, `pyright`, targeted tests)
- Remote Linux copy of the gallery required **path rewriting** and **Unicode NFC normalization** before training
- Smoke test on Ubuntu + RTX 4090 succeeded: `results/B3-sscd-smoke/2026-04-03T13-28-01/`
- Smoke-test val metrics: R@1=0.8371, R@5=0.8686, mAP@5=0.8503, MRR=0.8535
- This confirms the remote CUDA environment is usable for future SSCD experiments, but does not change the story decision to close #19 without a Run 3

### Follow-up — remote CUDA stability study + full run (2026-04-03)
Branch: `feature/19-cuda-ssh-support`
- Initial aggressive RTX 4090 configs (`batch_size` 256, 192, 160, 144 with 24 workers) all failed at unfreeze with CUDA OOM; `batch_size=128` was the first stable full-run memory point on 24 GB VRAM
- A per-epoch validation DataLoader bug was found during the first `128/24` attempt: `_evaluate_val()` created a fresh loader each epoch with `persistent_workers=True`, which caused `OSError: [Errno 24] Too many open files`
- Fix applied locally and on the remote copy: keep persistent workers on the long-lived training loader, but remove them from the per-epoch validation loader
- Separate platform instability was also investigated: the desktop had repeated abrupt reboots with no machine-level shutdown markers, missing UPS telemetry (`COMMLOST`), and likely system instability under load
- After lowering RAM speed from **4800 → 4200** and restoring UPS USB monitoring (`apcaccess status` returned `STATUS: ONLINE`), a 4-epoch SSCD stability probe completed cleanly and the full 50-epoch run also completed cleanly
- Stable full-run artifact: `results/B3-sscd-supcon-cuda-bs128-w24-expandable/2026-04-03T18-44-36/`
- Best val epoch: **6**
  - R@1=**0.8412**
  - R@5=**0.8741**
  - mAP@5=**0.8545**
  - MRR=**0.8576**
- Timing summary from `training_log.json`:
  - frozen epochs (0–4): mean **52.3 s/epoch**, median **51.5 s/epoch**
  - unfrozen epochs (5–49): mean **61.3 s/epoch**, median **61.2 s/epoch**
- Comparison note:
  - quality is effectively in line with local B3 Run 1 val R@1 (**0.8431**) and better than local B3 Run 2 val R@1 (**0.8349**)
  - no remote test-split evaluation was run, so the best available B3 test result remains local Run 1 (`R@1=0.8604`)
  - exact local M3 Max B3 epoch timings were never recorded, so the device-speed note is limited to a **best-stable practical comparison**, not a pure apples-to-apples benchmark

### Reference — A4 zero-shot SSCD
- Test: R@1=0.872, R@5=0.898

### Reference — B2c DINOv2+SupCon (current leader)
- Test: R@1=0.8795, R@5=0.9053, mAP@5=0.8903, MRR=0.8925

## Blockers & Decisions

- **Decision (2026-04-02):** The initial SSCD recipe (0.1× backbone LR, 5-epoch
  freeze) peaked at unfreeze then degraded — same pattern as the B2 diagnostic.
- **Decision (2026-04-02):** Copying the B2c recipe (0.01× backbone LR, 10-epoch
  freeze) made things worse, not better. SSCD (TorchScript ResNet50) does not
  respond to the same fine-tuning knobs as DINOv2 ViT.
- **Decision (2026-04-03):** Closing #19 without a Run 3. SSCD's TorchScript
  JIT graph limits gradient flow compared to a native nn.Module. A third recipe
  would need a fundamentally different approach (distillation, non-JIT backbone)
  which exceeds this story's scope. B2c DINOv2+SupCon (R@1=0.8795) is confirmed
  as the Sprint 3 winner.
- **Follow-up note (2026-04-03):** Phase 1 CUDA support and a successful RTX 4090
  smoke test were completed on a separate branch. This reduces environment risk
  for any future experiments, but does not reopen the epic because no new B3
  training hypothesis was pursued.
- **Follow-up decision (2026-04-03):** The stable RTX 4090 full-run recipe for
  SSCD is `batch_size=128`, `num_workers=24`, `freeze_epochs=5`; larger batch
  sizes improved frozen-phase throughput but all failed at unfreeze on 24 GB VRAM.
- **Follow-up decision (2026-04-03):** `persistent_workers=True` must not be used
  on the per-epoch validation DataLoader. That optimization is safe only on the
  single long-lived training loader.
- **Follow-up decision (2026-04-03):** The remote full run improved throughput
  materially but did not produce a stronger B3 quality result than the local
  best-val baseline, and it still does not beat the Sprint 3 leader (B2c).
  Story #19 remains closed.
