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

**Last action:** Decided to close #19 without a Run 3. Neither recipe improved
over A4 zero-shot SSCD, and the TorchScript backbone limits fine-tuning
effectiveness. B2c DINOv2+SupCon remains the best fine-tuned model.

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
