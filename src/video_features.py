"""Turns the per-frame synthetic video (a directory of JPEGs, read back
through OpenCV) into a single scalar "visual activity" signal per frame:
mean absolute pixel difference from the previous frame. This is the whole
video-only signal; it carries motion/scene-change information but nothing
about what a telemetry sensor would read, which is why the video-only
deviation detector (deviation_detectors.py) cannot see a telemetry-only
deviation by construction, not by omission.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def _frame_index_key(path: Path) -> tuple[int, int]:
    m = re.match(r"step_(\d+)_frame_(\d+)\.jpg", path.name)
    assert m, f"unexpected frame filename: {path.name}"
    return int(m.group(1)), int(m.group(2))


def load_mean_color_by_step(frames_dir: Path, n_steps: int) -> list[np.ndarray]:
    """Mean RGB (as a float array) over every frame belonging to each
    step index, in step order."""
    means = []
    for i in range(n_steps):
        frame_paths = sorted(frames_dir.glob(f"step_{i:02d}_frame_*.jpg"))
        imgs = [cv2.imread(str(p)) for p in frame_paths]
        imgs = [im for im in imgs if im is not None]
        if not imgs:
            means.append(np.zeros(3))
            continue
        stacked = np.stack(imgs, axis=0).astype(float)
        means.append(stacked.reshape(-1, 3).mean(axis=0))
    return means


def load_activity_signal(frames_dir: Path) -> list[float]:
    """Returns one activity value per frame, in session order, first
    frame given 0.0 (no previous frame to diff against)."""
    frame_paths = sorted(frames_dir.glob("step_*_frame_*.jpg"), key=_frame_index_key)
    activity: list[float] = []
    prev: np.ndarray | None = None
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"OpenCV failed to read frame: {p}")
        if prev is None:
            activity.append(0.0)
        else:
            activity.append(float(np.mean(np.abs(img.astype(int) - prev.astype(int)))))
        prev = img
    return activity
