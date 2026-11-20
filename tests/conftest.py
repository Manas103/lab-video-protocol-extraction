from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "vosk-model-small-en-us-0.15"

requires_asr_model = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="vosk-model-small-en-us-0.15 not downloaded (see README Building and running)",
)


@pytest.fixture()
def model_dir() -> Path:
    return MODEL_DIR
