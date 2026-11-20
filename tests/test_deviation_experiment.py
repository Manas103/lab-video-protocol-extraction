"""A fast, 4-instance slice (one of each deviation type) through the real
pipeline, as a structural check. The full 120-instance, ~3-minute run and
the headline catch-rate numbers live in scripts/run_deviation_benchmark.py
and docs/deviation_benchmark_output.txt.
"""
from __future__ import annotations

from pathlib import Path

from src.deviation_experiment import (
    DEVIATION_TYPES,
    _build_deviated_session,
    _canonical_sequence,
)
from src.protocols import PROTOCOLS_BY_ID
from tests.conftest import requires_asr_model


@requires_asr_model
def test_one_instance_of_each_deviation_type_builds_and_scores(tmp_path, model_dir):
    from src.deviation_experiment import build_instances, _discrepant_canonical_indices, _observed_sequences

    all_instances = build_instances()
    one_per_type = {}
    for inst in all_instances:
        if inst.deviation_type not in one_per_type:
            one_per_type[inst.deviation_type] = inst
    assert set(one_per_type) == set(DEVIATION_TYPES)

    for dtype, inst in one_per_type.items():
        inst.session_dir = tmp_path / inst.instance_id
        protocol = PROTOCOLS_BY_ID[inst.protocol_id]
        _build_deviated_session(inst, protocol, seed=1, model_dir=model_dir)
        canonical_seq = _canonical_sequence(protocol)
        video_seq, fused_seq, mismatch_flags = _observed_sequences(inst, model_dir)

        if dtype == "skip_step":
            assert len(video_seq) == len(canonical_seq) - 1
        elif dtype == "duplicate_step":
            assert len(video_seq) == len(canonical_seq) + 1
        elif dtype == "reorder_adjacent":
            assert len(video_seq) == len(canonical_seq)
        elif dtype == "telemetry_mismatch":
            assert len(video_seq) == len(canonical_seq)
            assert len(mismatch_flags) == len(canonical_seq)

        # structural sanity: the diff functions run without error and
        # return a set of valid canonical indices
        flagged = _discrepant_canonical_indices(canonical_seq, video_seq)
        assert all(0 <= i < len(canonical_seq) for i in flagged)
