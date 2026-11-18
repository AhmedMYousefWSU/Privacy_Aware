"""Lightweight per-token privacy head  f_P : z_i (1024) -> p_{i,c} (C attributes).

Trained on frozen CLIP features with *soft* targets o_{i,c} ∈ [0,1] (patch overlap with class-c
polygons). At inference P_i = max_c p_{i,c} (no subjective per-class severity weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivacyHead(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, num_classes=24, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z):  # (..., in_dim) -> logits (..., C)
        return self.net(z)

    @torch.no_grad()
    def privacy_risk(self, z, class_weights=None):
        """P_i = max_c w_c p_{i,c}; w_c = 1 by default. Returns (P (...,), p (..., C))."""
        p = torch.sigmoid(self.forward(z.float()))
        pw = p if class_weights is None else p * class_weights.to(p)
        return pw.max(-1).values, p


def soft_bce_loss(logits, targets, pos_weight):
    """Weighted BCE with soft targets:  -[w+ y log p + (1-y) log(1-p)].

    pos_weight (C,) up-weights rare classes (passport, credit card, ...) relative to face/person.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def compute_pos_weight(overlap_sums, n_tokens, cap=100.0):
    """w+_c = (#negative mass / #positive mass), capped. overlap_sums: (C,) sum of o_{i,c} over train tokens."""
    pos = overlap_sums.clamp(min=1.0)
    return ((n_tokens - pos) / pos).clamp(max=cap)
