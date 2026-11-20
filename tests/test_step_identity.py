from __future__ import annotations

import numpy as np

from src.step_identity import ALL_STEPS, CANONICAL_COLOR, SegmentObservation, _tokenize, fused_guess, video_guess


def test_video_guess_is_exact_on_its_own_canonical_color():
    for name in ALL_STEPS:
        assert video_guess(CANONICAL_COLOR[name]) == name


def test_video_guess_is_robust_to_small_noise():
    rng = np.random.default_rng(0)
    correct = 0
    for name in ALL_STEPS:
        noisy = CANONICAL_COLOR[name] + rng.uniform(-3, 3, size=3)
        correct += int(video_guess(noisy) == name)
    assert correct == len(ALL_STEPS)


def test_fused_guess_flags_telemetry_mismatch():
    name = "incubate"  # heat, 56.0
    step = ALL_STEPS[name]
    obs_matching = SegmentObservation(
        mean_color=CANONICAL_COLOR[name],
        narration_words=_tokenize(step.narration),
        telemetry_channel=step.telemetry,
        telemetry_level=step.telemetry_level,
    )
    guess, mismatch = fused_guess(obs_matching)
    assert guess == name
    assert mismatch is False

    obs_wrong_telemetry = SegmentObservation(
        mean_color=CANONICAL_COLOR[name],
        narration_words=_tokenize(step.narration),
        telemetry_channel="idle",
        telemetry_level=0.0,
    )
    guess2, mismatch2 = fused_guess(obs_wrong_telemetry)
    assert guess2 == name  # color + narration still dominate the identity guess
    assert mismatch2 is True
