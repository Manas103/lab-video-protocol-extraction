#!/usr/bin/env python3
"""Builds the 23 clean sessions (claim 1: 23 public egocentric wet-lab
sessions is reconciled honestly to 23 synthetic ones, see README "Honest
framing") under data/sessions/, each with real TTS narration, real
per-frame OpenCV video, and synthetic telemetry, all time-aligned to the
narration's own measured duration.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.protocols import PROTOCOLS_BY_ID, SESSION_PLAN
from src.session_builder import build_session


def main() -> int:
    out_root = Path("data") / "sessions"
    out_root.mkdir(parents=True, exist_ok=True)

    session_index = 0
    t0 = time.time()
    for protocol_id, n_reps in SESSION_PLAN:
        protocol = PROTOCOLS_BY_ID[protocol_id]
        for rep in range(n_reps):
            session_id = f"{protocol_id}_{rep:02d}"
            session_dir = out_root / session_id
            print(f"[{session_index + 1:2d}/23] building {session_id} ...", flush=True)
            gt = build_session(session_id, protocol, seed=1000 + session_index, out_dir=session_dir)
            print(f"    {len(gt.steps)} steps, {gt.total_duration_s:.1f}s")
            session_index += 1

    print(f"built {session_index} sessions in {time.time() - t0:.1f}s")
    assert session_index == 23
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
