"""A small PyTorch 1D-CNN that predicts, for every one-second bin of a
session's aligned multimodal feature sequence, whether a protocol step
starts there. Trained on one split of sessions and evaluated on a held-out
split with boundary F1 (a predicted boundary within +/- BOUNDARY_TOLERANCE_S
of a true one counts as a match, each side matched at most once).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .alignment import N_FEATURES, SessionFeatures

BOUNDARY_TOLERANCE_S = 1


class BoundaryCNN(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, time) -> (batch, time)
        return self.net(x).squeeze(1)


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / np.where(std < 1e-6, 1.0, std)


def fit_normalizer(sessions: list[SessionFeatures]) -> tuple[np.ndarray, np.ndarray]:
    all_feats = np.concatenate([s.features for s in sessions], axis=0)
    return all_feats.mean(axis=0), all_feats.std(axis=0)


def train_boundary_model(
    train_sessions: list[SessionFeatures],
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> BoundaryCNN:
    torch.manual_seed(seed)
    model = BoundaryCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs = [torch.tensor(_normalize(s.features, mean, std).T, dtype=torch.float32) for s in train_sessions]
    ys = [torch.tensor(s.boundary_labels, dtype=torch.float32) for s in train_sessions]

    pos_weight = torch.tensor([8.0])  # boundaries are rare (one per step out of many seconds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for x, y in zip(xs, ys):
            logits = model(x.unsqueeze(0)).squeeze(0)
            loss = loss + criterion(logits, y)
        loss = loss / len(xs)
        loss.backward()
        optimizer.step()
    return model


def predict_boundaries(model: BoundaryCNN, session: SessionFeatures, mean: np.ndarray, std: np.ndarray, threshold: float = 0.5) -> list[int]:
    model.eval()
    x = torch.tensor(_normalize(session.features, mean, std).T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).numpy()
    return [i for i, p in enumerate(probs) if p >= threshold]


@dataclass
class BoundaryScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def score_boundaries(predicted_bins: list[int], true_bins: list[int], tolerance: int = BOUNDARY_TOLERANCE_S) -> BoundaryScore:
    """Optimal one-to-one matching (scipy linear_sum_assignment) between
    predicted and true boundaries within `tolerance`, maximizing the
    match count; not a greedy nearest-first heuristic, which can under-count
    true positives when two predictions compete for the same true boundary
    (see tests/test_boundary_oracle.py for the brute-force oracle this is
    diffed against)."""
    tp = _optimal_match_count(predicted_bins, true_bins, tolerance)
    fp = len(predicted_bins) - tp
    fn = len(true_bins) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return BoundaryScore(precision=precision, recall=recall, f1=f1, true_positives=tp, false_positives=fp, false_negatives=fn)


def _optimal_match_count(predicted_bins: list[int], true_bins: list[int], tolerance: int) -> int:
    if not predicted_bins or not true_bins:
        return 0
    big = 10 ** 6
    cost = np.full((len(predicted_bins), len(true_bins)), big, dtype=float)
    for i, p in enumerate(predicted_bins):
        for j, t in enumerate(true_bins):
            if abs(p - t) <= tolerance:
                cost[i, j] = 0.0
    row_ind, col_ind = linear_sum_assignment(cost)
    return int(sum(1 for r, c in zip(row_ind, col_ind) if cost[r, c] < big))
