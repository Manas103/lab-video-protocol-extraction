"""Builds one synthetic egocentric wet-lab session: real TTS narration
audio (Windows SAPI5 via pyttsx3), a real per-frame synthetic video
(written and read back through OpenCV, not a single video container; see
README "Honest framing" for why), and synthetic telemetry, all laid out on
one timeline whose segment boundaries come from the narration clips'
measured durations, not a hand-picked schedule.
"""
from __future__ import annotations

import json
import random
import wave
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pyttsx3

from .protocols import ProtocolSpec, StepSpec

VIDEO_FPS = 5
FRAME_SIZE = (96, 96)  # small deliberately; this is a synthetic proxy signal, not a realistic lab video
TELEMETRY_HZ = 1


@dataclass
class StepTiming:
    step_name: str
    start_s: float
    end_s: float
    narration: str
    telemetry: str
    telemetry_level: float


@dataclass
class SessionGroundTruth:
    session_id: str
    protocol_id: str
    seed: int
    steps: list[StepTiming]
    total_duration_s: float


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def synthesize_narration(text: str, out_path: Path, rate: int = 170) -> float:
    """Real SAPI5 text-to-speech via pyttsx3. Returns the measured clip
    duration in seconds (never assumed)."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return _wav_duration_seconds(out_path)


def _step_color(step_name: str) -> tuple[int, int, int]:
    # zlib.crc32, not Python's built-in hash(): str hashing is randomized
    # per process (PYTHONHASHSEED), which would make every step's color
    # change between runs for no reason connected to anything being
    # measured, and would break step_identity.py's CANONICAL_COLOR table
    # staying consistent with what was actually rendered.
    h = zlib.crc32(step_name.encode("utf-8")) % (2**24)
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def _render_frame(
    step: StepSpec,
    hand_xy: tuple[int, int],
    rng: random.Random,
    lighting_shift: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    color = _step_color(step.name)
    frame[:, :] = color
    # Deliberately noisy: a real egocentric camera's exposure and hand/arm
    # occlusion vary within a step at least this much; a boundary detector
    # that can only see a single frame's noise floor should not be able to
    # tell a within-step fluctuation from a real transition (see README
    # Findings for what happened before this was added).
    jitter = rng.randint(-28, 28)
    frame = frame.astype(int) + jitter + np.array(lighting_shift, dtype=int)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    cv2.circle(frame, hand_xy, 6, (255, 255, 255), -1)
    cv2.putText(frame, step.name[:10], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return frame


def build_session(
    session_id: str,
    protocol: ProtocolSpec,
    seed: int,
    out_dir: Path,
    step_overrides: dict[str, StepSpec] | None = None,
    step_order: list[str] | None = None,
    lighting_shift: tuple[int, int, int] = (0, 0, 0),
) -> SessionGroundTruth:
    """step_overrides / step_order let the deviation seeder substitute a
    modified step spec (e.g. a telemetry mismatch) or a reordered sequence
    of step names without duplicating this function."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    by_name = {s.name: s for s in protocol.steps}
    if step_overrides:
        by_name.update(step_overrides)
    names = step_order if step_order is not None else [s.name for s in protocol.steps]

    timings: list[StepTiming] = []
    telemetry_rows: list[tuple[float, str, float]] = []
    t = 0.0
    hand_xy = [FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2]

    for i, name in enumerate(names):
        step = by_name[name]
        wav_path = out_dir / f"step_{i:02d}_narration.wav"
        duration = synthesize_narration(step.narration, wav_path, rate=170 + rng.randint(-10, 10))

        n_frames = max(1, round(duration * VIDEO_FPS))
        for f in range(n_frames):
            hand_xy[0] = int(np.clip(hand_xy[0] + rng.randint(-4, 4), 4, FRAME_SIZE[0] - 4))
            hand_xy[1] = int(np.clip(hand_xy[1] + rng.randint(-4, 4), 4, FRAME_SIZE[1] - 4))
            frame = _render_frame(step, tuple(hand_xy), rng, lighting_shift)
            cv2.imwrite(str(frames_dir / f"step_{i:02d}_frame_{f:03d}.jpg"), frame)

        n_telemetry = max(1, round(duration * TELEMETRY_HZ))
        for k in range(n_telemetry):
            level = step.telemetry_level + rng.gauss(0, max(0.01, step.telemetry_level * 0.03))
            # Sensor latency: a real telemetry channel does not snap to the
            # new step's value the instant the step starts; up to half a
            # second of lag/lead is modeled directly on the timestamp.
            sample_t = t + k / TELEMETRY_HZ + rng.uniform(-0.4, 0.4)
            telemetry_rows.append((max(0.0, sample_t), step.telemetry, level))

        timings.append(
            StepTiming(
                step_name=name,
                start_s=t,
                end_s=t + duration,
                narration=step.narration,
                telemetry=step.telemetry,
                telemetry_level=step.telemetry_level,
            )
        )
        t += duration

    gt = SessionGroundTruth(session_id=session_id, protocol_id=protocol.protocol_id, seed=seed, steps=timings, total_duration_s=t)

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(asdict(gt), f, indent=2)
    with open(out_dir / "telemetry.csv", "w") as f:
        f.write("t_s,channel,level\n")
        for row in telemetry_rows:
            f.write(f"{row[0]:.3f},{row[1]},{row[2]:.4f}\n")

    return gt
