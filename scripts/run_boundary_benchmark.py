#!/usr/bin/env python3
"""Claim: emitting structured protocol steps at 0.71 boundary F1.

Builds the per-second aligned feature sequence for all 23 sessions (real
ASR over each step's own narration clip, real OpenCV frame-difference
activity, synthetic telemetry), trains the boundary CNN on one split and
evaluates boundary F1 (+/- 1s tolerance) on a held-out split.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import build_session_features
from src.boundary_model import fit_normalizer, predict_boundaries, score_boundaries, train_boundary_model

MODEL_DIR = Path("models") / "vosk-model-small-en-us-0.15"
SESSIONS_DIR = Path("data") / "sessions"

# 5 sessions held out, one per protocol where possible, for a genuine
# train/test split rather than evaluating on training sessions.
TEST_SESSION_IDS = {
    "dna_extraction_05",
    "gel_electrophoresis_05",
    "cell_passaging_05",
    "acid_base_titration_04",
    "dna_extraction_04",
}


def main() -> int:
    t0 = time.time()
    session_dirs = sorted(SESSIONS_DIR.iterdir())
    assert len(session_dirs) == 23, f"expected 23 sessions, found {len(session_dirs)}"

    print("Building aligned multimodal features for all 23 sessions (real ASR pass per step)...")
    all_features = []
    for d in session_dirs:
        feats = build_session_features(d, MODEL_DIR, use_asr=True)
        all_features.append(feats)
        print(f"  {feats.session_id}: {feats.n_seconds}s, {len(feats.step_start_bins)} steps")

    train = [f for f in all_features if f.session_id not in TEST_SESSION_IDS]
    test = [f for f in all_features if f.session_id in TEST_SESSION_IDS]
    print(f"\ntrain sessions: {len(train)}, test sessions: {len(test)}")

    mean, std = fit_normalizer(train)
    model = train_boundary_model(train, mean, std, epochs=300, lr=0.01, seed=0)

    total_tp = total_fp = total_fn = 0
    per_session = []
    for feats in test:
        pred_bins = predict_boundaries(model, feats, mean, std)
        score = score_boundaries(pred_bins, feats.step_start_bins)
        total_tp += score.true_positives
        total_fp += score.false_positives
        total_fn += score.false_negatives
        per_session.append((feats.session_id, score))
        print(
            f"  {feats.session_id}: true_boundaries={feats.step_start_bins} predicted={pred_bins} "
            f"precision={score.precision:.2f} recall={score.recall:.2f} f1={score.f1:.2f}"
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print()
    print(f"MEASURED pooled boundary F1 over {len(test)} held-out sessions: {f1:.4f}")
    print(f"(precision={precision:.4f}, recall={recall:.4f}, tp={total_tp}, fp={total_fp}, fn={total_fn})")
    meets_claim = f1 >= 0.71
    print(f"claim = 0.71 boundary F1; meets_claim = {meets_claim}")
    print(f"total time: {time.time() - t0:.1f}s")

    Path("docs").mkdir(exist_ok=True)
    summary = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "test_sessions": sorted(TEST_SESSION_IDS),
        "meets_claim": meets_claim,
    }
    with open("docs/boundary_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0 if meets_claim else 1


if __name__ == "__main__":
    raise SystemExit(main())
