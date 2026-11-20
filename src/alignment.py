"""Builds one time-aligned, per-second multimodal feature sequence for a
session: video activity, telemetry (level plus a one-hot of channel
type), and ASR word density, each binned onto the same integer-second
timeline the ground truth step boundaries live on. This is the "aligned
timeline" the fused deviation detector and the boundary model both
consume; the video-only detector only ever sees column 0.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import asr
from .video_features import load_activity_signal

VIDEO_FPS = 5
TELEMETRY_CHANNELS = ["idle", "heat", "spin", "weigh", "timer"]
FEATURE_NAMES = ["video_activity", "telemetry_level"] + [f"telemetry_{c}" for c in TELEMETRY_CHANNELS] + ["asr_word_density"]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class SessionFeatures:
    session_id: str
    n_seconds: int
    features: np.ndarray  # (n_seconds, N_FEATURES)
    boundary_labels: np.ndarray  # (n_seconds,) 1 where a step starts in that second
    step_names: list[str]
    step_start_bins: list[int]


def _load_ground_truth(session_dir: Path) -> dict:
    with open(session_dir / "ground_truth.json") as f:
        return json.load(f)


def build_session_features(session_dir: Path, model_dir: Path, use_asr: bool = True) -> SessionFeatures:
    gt = _load_ground_truth(session_dir)
    n_seconds = int(np.ceil(gt["total_duration_s"])) + 1
    features = np.zeros((n_seconds, N_FEATURES), dtype=np.float32)
    counts = np.zeros(n_seconds, dtype=np.int32)

    activity = load_activity_signal(session_dir / "frames")
    step_frame_offsets = _step_frame_offsets(session_dir)
    for i, a in enumerate(activity):
        step_idx, frame_idx = step_frame_offsets[i]
        step = gt["steps"][step_idx]
        t = step["start_s"] + frame_idx / VIDEO_FPS
        b = int(t)
        if b < n_seconds:
            features[b, 0] += a
            counts[b] += 1

    with open(session_dir / "telemetry.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t_s"])
            b = int(t)
            if b >= n_seconds:
                continue
            level = float(row["level"])
            channel = row["channel"]
            features[b, 1] = level
            if channel in TELEMETRY_CHANNELS:
                features[b, 2 + TELEMETRY_CHANNELS.index(channel)] = 1.0

    if use_asr:
        for i, step in enumerate(gt["steps"]):
            wav_path = session_dir / f"step_{i:02d}_narration.wav"
            words = asr.transcribe(wav_path, model_dir)
            for w in words:
                t = step["start_s"] + w.start_s
                b = int(t)
                if b < n_seconds:
                    features[b, -1] += 1.0

    nonzero = counts > 0
    features[nonzero, 0] = features[nonzero, 0] / counts[nonzero]

    boundary_labels = np.zeros(n_seconds, dtype=np.int32)
    step_start_bins = []
    for step in gt["steps"]:
        b = int(step["start_s"])
        if b < n_seconds:
            boundary_labels[b] = 1
            step_start_bins.append(b)

    return SessionFeatures(
        session_id=gt["session_id"],
        n_seconds=n_seconds,
        features=features,
        boundary_labels=boundary_labels,
        step_names=[s["step_name"] for s in gt["steps"]],
        step_start_bins=step_start_bins,
    )


def _step_frame_offsets(session_dir: Path) -> list[tuple[int, int]]:
    """Returns (step_index, frame_index_within_step) for every frame file,
    in the same sorted order load_activity_signal reads them in."""
    import re

    frame_paths = sorted((session_dir / "frames").glob("step_*_frame_*.jpg"))
    offsets = []
    for p in frame_paths:
        m = re.match(r"step_(\d+)_frame_(\d+)\.jpg", p.name)
        offsets.append((int(m.group(1)), int(m.group(2))))
    offsets.sort(key=lambda x: (x[0], x[1]))
    return offsets
