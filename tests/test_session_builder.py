from __future__ import annotations

import pytest

from src.protocols import PROTOCOLS_BY_ID
from src.session_builder import build_session


@pytest.fixture(scope="module")
def built_session(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("session") / "t1"
    gt = build_session("t1", PROTOCOLS_BY_ID["acid_base_titration"], seed=1, out_dir=out_dir)
    return gt, out_dir


def test_session_timeline_is_contiguous_and_monotonic(built_session):
    gt, _ = built_session
    assert len(gt.steps) == 6
    for i in range(len(gt.steps) - 1):
        assert gt.steps[i].end_s == gt.steps[i + 1].start_s
    assert gt.steps[0].start_s == 0.0
    assert abs(gt.total_duration_s - gt.steps[-1].end_s) < 1e-9
    for s in gt.steps:
        assert s.end_s > s.start_s


def test_every_step_has_a_narration_wav_and_at_least_one_frame(built_session):
    gt, out_dir = built_session
    for i in range(len(gt.steps)):
        assert (out_dir / f"step_{i:02d}_narration.wav").exists()
        frames = list((out_dir / "frames").glob(f"step_{i:02d}_frame_*.jpg"))
        assert len(frames) >= 1


def test_telemetry_rows_cover_the_session_duration(built_session):
    gt, out_dir = built_session
    with open(out_dir / "telemetry.csv") as f:
        lines = f.readlines()[1:]
    assert len(lines) >= len(gt.steps)
    last_t = float(lines[-1].split(",")[0])
    assert last_t <= gt.total_duration_s + 1.0


def test_deterministic_given_the_same_seed(tmp_path):
    gt1 = build_session("same1", PROTOCOLS_BY_ID["acid_base_titration"], seed=42, out_dir=tmp_path / "a")
    gt2 = build_session("same2", PROTOCOLS_BY_ID["acid_base_titration"], seed=42, out_dir=tmp_path / "b")
    assert [s.step_name for s in gt1.steps] == [s.step_name for s in gt2.steps]
    assert gt1.total_duration_s == gt2.total_duration_s
