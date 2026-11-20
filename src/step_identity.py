"""Per-segment step identification: given one step-shaped segment of a
session (its frames, its own narration wav, and its telemetry rows), guess
which of the canonical step types it actually is. Two guessers:

- ``video_guess``: nearest match in RGB space against every known step's
  canonical color. This is literally everything the video modality can
  see in this synthetic dataset; it has no access to telemetry or
  narration, and a session's per-session lighting shift (an unmodeled
  global color cast, see session_builder.build_session) degrades it the
  same way a real camera's white balance would.
- ``fused_guess``: a weighted vote across color, narration word overlap,
  and telemetry-signature match, i.e. everything the aligned timeline
  provides that video alone does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import asr
from .protocols import ALL_PROTOCOLS, StepSpec
from .session_builder import _step_color

ALL_STEPS: dict[str, StepSpec] = {s.name: s for p in ALL_PROTOCOLS for s in p.steps}
CANONICAL_COLOR: dict[str, np.ndarray] = {name: np.array(_step_color(name), dtype=float) for name in ALL_STEPS}


_STOPWORDS = frozenset(
    "a an the to of in on at for and or with into onto from by over each its".split()
)


def _tokenize(text: str) -> set[str]:
    # Stopwords excluded: without this, two unrelated narrations sharing
    # only "the", "for", "at" measure nontrivial overlap, which let a
    # wrong-but-telemetry-matching candidate occasionally outscore the
    # right one in fused_guess's weighted vote (see
    # tests/test_step_identity.py and README Findings).
    return {w.strip(".,").lower() for w in text.split()} - _STOPWORDS


NARRATION_TOKENS: dict[str, set[str]] = {name: _tokenize(step.narration) for name, step in ALL_STEPS.items()}


@dataclass
class SegmentObservation:
    mean_color: np.ndarray
    narration_words: set[str]
    telemetry_channel: str
    telemetry_level: float


def video_guess(mean_color: np.ndarray) -> str:
    return min(ALL_STEPS, key=lambda name: float(np.sum((CANONICAL_COLOR[name] - mean_color) ** 2)))


def _narration_score(words: set[str], candidate: str) -> float:
    if not words:
        return 0.0
    overlap = len(words & NARRATION_TOKENS[candidate])
    return overlap / max(len(NARRATION_TOKENS[candidate]), 1)


def _telemetry_score(obs: SegmentObservation, candidate: str) -> float:
    step = ALL_STEPS[candidate]
    if obs.telemetry_channel != step.telemetry:
        return 0.0
    if step.telemetry_level == 0:
        return 1.0 if abs(obs.telemetry_level) < 0.5 else 0.0
    rel_err = abs(obs.telemetry_level - step.telemetry_level) / abs(step.telemetry_level)
    return 1.0 if rel_err < 0.25 else 0.0


def fused_guess(obs: SegmentObservation) -> tuple[str, bool]:
    """Returns (best_guess_name, telemetry_mismatch_flag). The telemetry
    mismatch flag fires when the best-scoring identity's own expected
    signature does not match observed telemetry, which is the one thing
    video_guess structurally cannot detect."""
    color_dists = {name: float(np.sum((CANONICAL_COLOR[name] - obs.mean_color) ** 2)) for name in ALL_STEPS}
    max_dist = max(color_dists.values()) or 1.0
    scores = {}
    for name in ALL_STEPS:
        color_score = 1.0 - color_dists[name] / max_dist
        scores[name] = 0.4 * color_score + 0.35 * _narration_score(obs.narration_words, name) + 0.25 * _telemetry_score(obs, name)
    best = max(scores, key=lambda n: scores[n])
    mismatch = _telemetry_score(obs, best) == 0.0
    return best, mismatch


def segment_observation(mean_color: np.ndarray, wav_path, model_dir, telemetry_channel: str, telemetry_level: float) -> SegmentObservation:
    words = {w.word for w in asr.transcribe(wav_path, model_dir)}
    return SegmentObservation(
        mean_color=mean_color,
        narration_words=words,
        telemetry_channel=telemetry_channel,
        telemetry_level=telemetry_level,
    )
