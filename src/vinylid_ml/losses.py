"""Metric learning loss functions for album cover fine-tuning.

Provides three loss strategies for training embedding models on the VinylID
dataset, where 67% of albums have only a single image:

- **ArcFaceLoss**: Classification-based angular margin loss.
- **ProxyAnchorLoss**: Proxy-based metric learning with soft assignment.
- **SupConLoss**: Supervised contrastive learning over augmented views.

All losses expect L2-normalized embeddings of shape ``(B, D)`` and integer
labels of shape ``(B,)``, and return a scalar loss tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

__all__ = [
    "ArcFaceLoss",
    "ProxyAnchorLoss",
    "SupConLoss",
]


class ArcFaceLoss(nn.Module):
    """ArcFace angular margin classification loss.

    Each class gets a learnable weight vector on the unit hypersphere.
    During forward, an angular margin ``m`` is added to the target-class
    angle before scaling and cross-entropy, encouraging larger inter-class
    angular separation.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition" (CVPR 2019).

    Args:
        embedding_dim: Dimensionality of input embeddings.
        num_classes: Number of classes (album IDs).
        margin: Angular margin in radians.
        scale: Logit scaling factor applied after margin.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        if not 0 < margin < math.pi:
            msg = f"ArcFace margin must be in (0, pi), got {margin}."
            raise ValueError(msg)
        if scale <= 0:
            msg = f"ArcFace scale must be > 0, got {scale}."
            raise ValueError(msg)
        self._num_classes = num_classes
        self._scale = scale
        self._margin = margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric constants
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        # Threshold for monotonicity: cos(pi - m)
        self._threshold = math.cos(math.pi - margin)
        # Fallback linear approximation coefficient
        self._mm = math.sin(math.pi - margin) * margin

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss.

        Args:
            embeddings: L2-normalized embeddings of shape ``(B, D)``.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        # Cosine similarity: (B, C)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # Add angular margin to target class: cos(theta + m)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))
        phi = cosine * self._cos_m - sine * self._sin_m

        # Numerical stability: when cos(theta) < cos(pi - m), use linear fallback
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        # Apply margin only to the target class
        one_hot = F.one_hot(labels, self._num_classes).to(embeddings.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self._scale

        return F.cross_entropy(logits, labels)


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor metric learning loss.

    Assigns a learnable proxy vector to each class and optimises a soft
    positive/negative assignment with margins.  Gradient signal is smoother
    than classification-based losses for classes with very few samples.

    Reference: Kim et al., "Proxy Anchor Loss for Deep Metric Learning"
    (CVPR 2020).

    Args:
        embedding_dim: Dimensionality of input embeddings.
        num_classes: Number of classes.
        margin: Margin for positive/negative separation.
        alpha: Scaling factor for the exponential terms.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.1,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        if margin < 0:
            msg = f"ProxyAnchor margin must be >= 0, got {margin}."
            raise ValueError(msg)
        if alpha <= 0:
            msg = f"ProxyAnchor alpha must be > 0, got {alpha}."
            raise ValueError(msg)
        self._num_classes = num_classes
        self._margin = margin
        self._alpha = alpha

        self.proxies = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.proxies)

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Proxy-Anchor loss.

        Args:
            embeddings: L2-normalized embeddings of shape ``(B, D)``.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        # Cosine similarity between embeddings and proxies: (B, C)
        cos = F.linear(F.normalize(embeddings), F.normalize(self.proxies))

        # Positive/negative masks: (C, B). Match embeddings' dtype to avoid
        # unintended upcasting in mixed-precision training.
        p_one_hot = F.one_hot(labels, self._num_classes).to(
            device=embeddings.device, dtype=embeddings.dtype
        ).T
        n_one_hot = 1.0 - p_one_hot

        # Which proxies have at least one positive sample in this batch?
        with_pos = p_one_hot.sum(dim=1) > 0
        num_with_pos = with_pos.sum().clamp(min=1).to(dtype=embeddings.dtype)

        # Positive term: softplus of margin-shifted similarities
        pos_exp = torch.exp(-self._alpha * (cos.T - self._margin))
        p_sim_sum = (p_one_hot * pos_exp).sum(dim=1)
        pos_term = torch.log(1.0 + p_sim_sum)
        pos_loss = pos_term[with_pos].sum() / num_with_pos

        # Negative term: softplus of margin-shifted similarities
        neg_exp = torch.exp(self._alpha * (cos.T + self._margin))
        n_sim_sum = (n_one_hot * neg_exp).sum(dim=1)
        neg_term = torch.log(1.0 + n_sim_sum)
        neg_loss = neg_term.sum() / self._num_classes

        return pos_loss + neg_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Pulls augmented views of the same class together and pushes different
    classes apart in embedding space.  No fixed classification head — may
    generalise better when most classes have very few samples.

    Reference: Khosla et al., "Supervised Contrastive Learning"
    (NeurIPS 2020).

    Args:
        temperature: Temperature scaling for the similarity logits.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            msg = f"SupConLoss temperature must be > 0, got {temperature}."
            raise ValueError(msg)
        self._temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: L2-normalized features.  Accepted shapes:

                * ``(B, N_views, D)`` — *N_views* augmented views per sample;
                  ``labels`` should be ``(B,)`` and will be repeated internally.
                * ``(N, D)`` — pre-concatenated views where ``labels`` has
                  shape ``(N,)`` with repeated labels for each view.

            labels: Integer class labels.  Shape ``(B,)`` when *features*
                is 3-D, or ``(N,)`` when *features* is 2-D.

        Returns:
            Scalar loss tensor.
        """
        if features.dim() == 3:
            batch_size, n_views, dim = features.shape
            features = features.reshape(batch_size * n_views, dim)
            labels = labels.repeat_interleave(n_views)

        device = features.device
        n = features.shape[0]

        # Pairwise cosine similarity scaled by temperature: (N, N)
        sim = features @ features.T / self._temperature

        # Mask out self-similarity
        self_mask = torch.eye(n, device=device, dtype=torch.bool)
        sim.masked_fill_(self_mask, float("-inf"))

        # Positive mask: same label, different index
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_eq & ~self_mask

        # If any sample has zero positives, clamp to avoid division by zero
        num_pos = pos_mask.sum(dim=1).clamp(min=1)

        # Log-softmax over the non-self dimension for numerical stability
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True
        )

        # Average log-probability over positive pairs.
        # Use masked_fill to avoid 0.0 * (-inf) = NaN on the diagonal.
        masked_log_prob = log_prob.masked_fill(~pos_mask, 0.0)
        mean_log_prob = masked_log_prob.sum(dim=1) / num_pos.float()

        return -mean_log_prob.mean()
