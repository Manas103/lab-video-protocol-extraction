#!/usr/bin/env python3
"""Claims: of 120 seeded deviations, video alone caught 36%, the aligned
timeline caught 61%. See src/deviation_experiment.py for how the 120
instances (30 each of skip/duplicate/reorder/telemetry-mismatch) are
built and scored.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deviation_experiment import DEVIATION_TYPES, cleanup, run_experiment

MODEL_DIR = Path("models") / "vosk-model-small-en-us-0.15"


def main() -> int:
    t0 = time.time()
    results = run_experiment(MODEL_DIR)
    assert len(results) == 120

    video_caught = sum(1 for r in results if r.video_caught)
    fused_caught = sum(1 for r in results if r.fused_caught)

    by_type: dict[str, dict[str, int]] = {t: {"video": 0, "fused": 0, "total": 0} for t in DEVIATION_TYPES}
    for r in results:
        by_type[r.deviation_type]["total"] += 1
        by_type[r.deviation_type]["video"] += int(r.video_caught)
        by_type[r.deviation_type]["fused"] += int(r.fused_caught)

    print("Per-type breakdown (video_caught/fused_caught out of total):")
    for t in DEVIATION_TYPES:
        b = by_type[t]
        print(f"  {t:20s} video {b['video']:2d}/{b['total']}   fused {b['fused']:2d}/{b['total']}")

    print()
    video_pct = 100.0 * video_caught / len(results)
    fused_pct = 100.0 * fused_caught / len(results)
    print(f"MEASURED: video alone caught {video_caught}/120 ({video_pct:.1f}%)")
    print(f"MEASURED: aligned timeline caught {fused_caught}/120 ({fused_pct:.1f}%)")
    print(f"claim: video 36%, aligned timeline 61%")
    video_meets = abs(video_pct - 36.0) <= 5.0
    fused_meets = abs(fused_pct - 61.0) <= 5.0
    print(f"meets_claim (video, +/-5pp) = {video_meets}")
    print(f"meets_claim (aligned timeline, +/-5pp) = {fused_meets}")
    print(f"total time: {time.time() - t0:.1f}s")

    Path("docs").mkdir(exist_ok=True)
    summary = {
        "video_caught": video_caught,
        "fused_caught": fused_caught,
        "total": len(results),
        "video_pct": video_pct,
        "fused_pct": fused_pct,
        "by_type": by_type,
        "meets_claim_video": video_meets,
        "meets_claim_fused": fused_meets,
    }
    with open("docs/deviation_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cleanup()
    return 0 if (video_meets and fused_meets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
