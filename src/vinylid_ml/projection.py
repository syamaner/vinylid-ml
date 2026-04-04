"""Projection head and NT-Xent loss for frozen-backbone domain adaptation.

Phase 2 of the Sprint 4 accuracy push trains a small MLP projection head on
top of frozen A4-sscd (SSCD ResNet50) features using 203 labelled real phone
photo → album pairs.  The key design choice is to **pre-extract backbone
features once** and train only the head — this keeps training fast even with
a device-limited MPS backend and avoids back-propagating through the large
TorchScript SSCD model.

Architecture::

    SSCD backbone (frozen, 512-dim L2-normalized)
        │
        ▼
    Linear(512 → hidden_dim)  ←─ trained
        │
        ReLU
        │
    Dropout(p)
        │
        ▼
    Linear(hidden_dim → out_dim)  ←─ trained
        │
        L2-normalize
        │
        ▼
    128-dim domain-adapted embedding

Training uses NT-Xent (InfoNCE with symmetric cross-entropy) where each
phone-album pair in the batch is the positive and all cross-pairs are
in-batch negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

__all__ = [
    "NTXentLoss",
    "ProjectionHead",
]


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for domain-adapted retrieval.

    Maps frozen L2-normalised backbone features (512-dim from A4-sscd) to a
    lower-dimensional space trained to align phone-photo embeddings with their
    corresponding catalogue-image embeddings.

    Architecture: ``Linear(in_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → out_dim) → L2-normalize``

    Args:
        in_dim: Input feature dimensionality.  Default 512 (A4-sscd output).
        hidden_dim: Hidden layer width.  Default 256.
        out_dim: Output embedding dimensionality.  Default 128.
        dropout: Dropout probability applied after ReLU.  Default 0.2.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # Xavier init for stable early training on pre-normalized inputs
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project backbone features to the adapted embedding space.

        Args:
            x: L2-normalised backbone features of shape ``(B, in_dim)``.

        Returns:
            L2-normalised projected embeddings of shape ``(B, out_dim)``.
        """
        out = self.layers(x)
        return F.normalize(out, p=2, dim=-1)


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Computes symmetric InfoNCE over paired ``(anchor, positive)`` embeddings.
    In each batch of ``B`` pairs, the diagonal entries are treated as
    positives; all off-diagonal cross-pair similarities are negatives.

    Suitable for training a projection head on phone→album pairs where we have
    exactly one positive per anchor and rely on in-batch negatives for contrast.

    Reference: Chen et al., *A Simple Framework for Contrastive Learning of
    Visual Representations* (SimCLR, ICML 2020).

    Args:
        temperature: Softmax temperature for similarity scaling.  Default 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """Compute symmetric NT-Xent loss over a batch of paired embeddings.

        Args:
            z_anchor: L2-normalised anchor embeddings, shape ``(B, D)``
                (e.g. phone-photo projections).
            z_positive: L2-normalised positive embeddings, shape ``(B, D)``
                (e.g. corresponding album-image projections).

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If batch size is less than 2 (need at least one negative).
        """
        batch_size = z_anchor.shape[0]
        if batch_size < 2:
            raise ValueError(
                f"NT-Xent requires batch_size >= 2 for in-batch negatives, got {batch_size}"
            )

        device = z_anchor.device
        labels = torch.arange(batch_size, device=device)

        # Similarity matrices scaled by temperature  (B, B)
        sim_a2p = z_anchor @ z_positive.T / self.temperature
        sim_p2a = z_positive @ z_anchor.T / self.temperature

        # Symmetric cross-entropy: anchor→positive and positive→anchor
        loss_a2p = F.cross_entropy(sim_a2p, labels)
        loss_p2a = F.cross_entropy(sim_p2a, labels)

        return (loss_a2p + loss_p2a) / 2
