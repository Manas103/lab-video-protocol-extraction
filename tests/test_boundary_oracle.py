"""score_boundaries uses scipy's optimal bipartite matching
(linear_sum_assignment) rather than a greedy nearest-first heuristic,
specifically because greedy nearest-first can under-count true positives
when two predictions compete for the same true boundary. This test diffs
it against a brute-force oracle (try every possible matching, exactly)
over random small cases, where brute force is obviously correct but too
slow to use as the real implementation.
"""
from __future__ import annotations

import itertools
import random

from src.boundary_model import _optimal_match_count


def _brute_force_match_count(predicted: list[int], true: list[int], tolerance: int) -> int:
    best = 0
    n = min(len(predicted), len(true))
    for k in range(n, -1, -1):
        for true_subset in itertools.permutations(true, k):
            for pred_subset in itertools.combinations(predicted, k):
                if all(abs(p - t) <= tolerance for p, t in zip(pred_subset, true_subset)):
                    best = max(best, k)
        if best == k:
            return best
    return best


def test_matches_brute_force_oracle_on_random_small_cases():
    rng = random.Random(7)
    for _ in range(40):
        n_pred = rng.randint(0, 5)
        n_true = rng.randint(0, 5)
        predicted = sorted(rng.sample(range(15), n_pred))
        true = sorted(rng.sample(range(15), n_true))
        tolerance = rng.choice([0, 1, 2])

        got = _optimal_match_count(predicted, true, tolerance)
        want = _brute_force_match_count(predicted, true, tolerance)
        assert got == want, (predicted, true, tolerance, got, want)


def test_contested_boundary_counted_once_not_twice():
    # Two predictions both within tolerance of the one true boundary at 10;
    # a naive greedy "every prediction grabs its nearest true boundary"
    # implementation would double-count this as 2 true positives.
    predicted = [9, 11]
    true = [10]
    assert _optimal_match_count(predicted, true, tolerance=1) == 1
