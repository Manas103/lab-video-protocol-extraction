from __future__ import annotations

import numpy as np

from src.alignment import N_FEATURES, build_session_features
from src.protocols import PROTOCOLS_BY_ID
from src.session_builder import build_session
from tests.conftest import requires_asr_model


@requires_asr_model
def test_feature_matrix_shape_and_boundary_labels(tmp_path, model_dir):
    out_dir = tmp_path / "align1"
    gt = build_session("align1", PROTOCOLS_BY_ID["acid_base_titration"], seed=5, out_dir=out_dir)

    feats = build_session_features(out_dir, model_dir, use_asr=True)

    assert feats.features.shape == (feats.n_seconds, N_FEATURES)
    assert not np.isnan(feats.features).any()
    assert feats.boundary_labels.sum() == len(gt.steps)
    assert feats.step_start_bins == sorted(feats.step_start_bins)
    assert len(feats.step_start_bins) == len(gt.steps)


@requires_asr_model
def test_asr_word_density_is_nonzero_somewhere(tmp_path, model_dir):
    out_dir = tmp_path / "align2"
    build_session("align2", PROTOCOLS_BY_ID["gel_electrophoresis"], seed=6, out_dir=out_dir)
    feats = build_session_features(out_dir, model_dir, use_asr=True)
    assert feats.features[:, -1].sum() > 0, "ASR should have transcribed at least some words"
