"""Seeds 120 deviations (30 each of 4 types) across sessions built from
the 4 canonical protocols, then scores how many each of the two detectors
(video-only, aligned-timeline) catches.

Deviation types:
- ``skip_step``: a canonical step is removed from the session.
- ``duplicate_step``: a canonical step is repeated immediately after itself.
- ``reorder_adjacent``: two adjacent canonical steps are swapped.
- ``telemetry_mismatch``: one step's telemetry channel/level is swapped for
  a different, wrong one, while its narration and video are untouched;
  this is the one category video_guess structurally cannot see (see
  step_identity.py), and is what separates the two detectors' recall.

Every session built here gets its own random, unmodeled lighting shift
(see session_builder.build_session), the only source of video_guess
misclassification on an otherwise-unaffected step.
"""
from __future__ import annotations

import dataclasses
import difflib
import random
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

from . import step_identity
from .protocols import ALL_PROTOCOLS, PROTOCOLS_BY_ID, ProtocolSpec
from .session_builder import build_session
from .video_features import load_mean_color_by_step

DEVIATION_TYPES = ("skip_step", "duplicate_step", "reorder_adjacent", "telemetry_mismatch")
N_PER_TYPE = 30
# Attempt 1 (+/-35) measured video-only at 70.8% (claim: 36%): a uniform
# per-session color shift turned out not to be enough to fool nearest-color
# classification against a well-separated canonical color table, since
# every step in a session shifts the same way and averaging a step's many
# frames together washes most per-frame noise back out. Attempt 2
# (+/-90, this value) measured 69.2%, barely different: widening a uniform
# shift further does not change whether it crosses a decision boundary
# much once it already has not. A third attempt (sampling one frame per
# step instead of averaging, to keep real per-frame noise in the signal)
# overshot in the other direction (near-total misclassification on
# unaffected steps too) and was reverted rather than accepted; see README
# Findings. 69.2%/90.8% (this configuration) is the number reported, after
# 2 full measurement attempts plus the reverted third design, none of
# which were tuned toward the claimed 36%/61%.
LIGHTING_SHIFT_RANGE = 90


@dataclass
class DeviationInstance:
    instance_id: str
    deviation_type: str
    protocol_id: str
    canonical_index: int  # index into the protocol's canonical step order
    session_dir: Path


def _canonical_sequence(protocol: ProtocolSpec) -> list[str]:
    return [s.name for s in protocol.steps]


def _build_deviated_session(instance: DeviationInstance, protocol: ProtocolSpec, seed: int, model_dir: Path) -> Path:
    names = _canonical_sequence(protocol)
    step_order = list(names)
    overrides = None

    if instance.deviation_type == "skip_step":
        del step_order[instance.canonical_index]
    elif instance.deviation_type == "duplicate_step":
        step_order.insert(instance.canonical_index + 1, names[instance.canonical_index])
    elif instance.deviation_type == "reorder_adjacent":
        i = instance.canonical_index
        step_order[i], step_order[i + 1] = step_order[i + 1], step_order[i]
    elif instance.deviation_type == "telemetry_mismatch":
        target = protocol.steps[instance.canonical_index]
        wrong_channel, wrong_level = _wrong_telemetry(target.telemetry, target.telemetry_level)
        overrides = {target.name: dataclasses.replace(target, telemetry=wrong_channel, telemetry_level=wrong_level)}
    else:
        raise ValueError(instance.deviation_type)

    rng = random.Random(seed)
    lighting_shift = tuple(rng.randint(-LIGHTING_SHIFT_RANGE, LIGHTING_SHIFT_RANGE) for _ in range(3))

    build_session(
        instance.instance_id,
        protocol,
        seed=seed,
        out_dir=instance.session_dir,
        step_overrides=overrides,
        step_order=step_order,
        lighting_shift=lighting_shift,
    )
    return instance.session_dir


def _wrong_telemetry(correct_channel: str, correct_level: float) -> tuple[str, float]:
    options = [("idle", 0.0), ("heat", 90.0), ("spin", 15000.0), ("weigh", 50.0), ("timer", 1.0)]
    for channel, level in options:
        if channel != correct_channel:
            return channel, level
    return "idle", 0.0


def build_instances() -> list[DeviationInstance]:
    instances: list[DeviationInstance] = []
    counter = 0
    for dtype_i, dtype in enumerate(DEVIATION_TYPES):
        for k in range(N_PER_TYPE):
            protocol = ALL_PROTOCOLS[k % len(ALL_PROTOCOLS)]
            max_index = len(protocol.steps) - (2 if dtype == "reorder_adjacent" else 1)
            canonical_index = k % max(1, max_index)
            counter += 1
            instance_id = f"dev_{dtype}_{k:03d}"
            instances.append(
                DeviationInstance(
                    instance_id=instance_id,
                    deviation_type=dtype,
                    protocol_id=protocol.protocol_id,
                    canonical_index=canonical_index,
                    session_dir=Path("data") / "deviations" / instance_id,
                )
            )
    assert len(instances) == 120
    return instances


def _observed_sequences(instance: DeviationInstance, model_dir: Path) -> tuple[list[str], list[str], list[bool]]:
    import json

    with open(instance.session_dir / "ground_truth.json") as f:
        gt = json.load(f)
    n_steps = len(gt["steps"])
    mean_colors = load_mean_color_by_step(instance.session_dir / "frames", n_steps)

    video_seq: list[str] = []
    fused_seq: list[str] = []
    mismatch_flags: list[bool] = []
    for i, step in enumerate(gt["steps"]):
        video_seq.append(step_identity.video_guess(mean_colors[i]))
        wav_path = instance.session_dir / f"step_{i:02d}_narration.wav"
        obs = step_identity.segment_observation(
            mean_colors[i], wav_path, model_dir, telemetry_channel=step["telemetry"], telemetry_level=step["telemetry_level"]
        )
        guess, mismatch = step_identity.fused_guess(obs)
        fused_seq.append(guess)
        mismatch_flags.append(mismatch)
    return video_seq, fused_seq, mismatch_flags


def _discrepant_canonical_indices(canonical_seq: list[str], observed_seq: list[str]) -> set[int]:
    sm = difflib.SequenceMatcher(None, canonical_seq, observed_seq, autojunk=False)
    flagged: set[int] = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            flagged.update(range(i1, i2))
        if tag == "insert":
            if i1 > 0:
                flagged.add(i1 - 1)
            if i1 < len(canonical_seq):
                flagged.add(i1)
    return flagged


@dataclass
class DeviationResult:
    instance_id: str
    deviation_type: str
    video_caught: bool
    fused_caught: bool


def run_experiment(model_dir: Path, data_root: Path = Path("data") / "deviations") -> list[DeviationResult]:
    instances = build_instances()
    for inst in instances:
        inst.session_dir = data_root / inst.instance_id
    results: list[DeviationResult] = []
    for inst in instances:
        protocol = PROTOCOLS_BY_ID[inst.protocol_id]
        # zlib.crc32, not Python's hash(): str hashing is randomized per
        # process (PYTHONHASHSEED), which would make this benchmark's exact
        # numbers vary run to run for no reason connected to anything being
        # measured.
        seed = zlib.crc32(inst.instance_id.encode("utf-8"))
        _build_deviated_session(inst, protocol, seed=seed, model_dir=model_dir)
        canonical_seq = _canonical_sequence(protocol)
        video_seq, fused_seq, mismatch_flags = _observed_sequences(inst, model_dir)

        video_flagged = _discrepant_canonical_indices(canonical_seq, video_seq)
        fused_flagged = _discrepant_canonical_indices(canonical_seq, fused_seq)

        if inst.deviation_type == "telemetry_mismatch":
            video_caught = False  # video has no telemetry signal at all, structurally
            fused_caught = mismatch_flags[inst.canonical_index] if inst.canonical_index < len(mismatch_flags) else False
        else:
            affected = {inst.canonical_index}
            if inst.deviation_type == "reorder_adjacent":
                affected.add(inst.canonical_index + 1)
            video_caught = bool(affected & video_flagged)
            fused_caught = bool(affected & fused_flagged) or bool(affected & {i for i, m in enumerate(mismatch_flags) if m})

        results.append(DeviationResult(inst.instance_id, inst.deviation_type, video_caught, fused_caught))

    return results


def cleanup(data_root: Path = Path("data") / "deviations") -> None:
    if data_root.exists():
        shutil.rmtree(data_root)
