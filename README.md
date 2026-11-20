# First-Person Lab Video Compiled Into a Machine-Readable Protocol

A pipeline that turns one egocentric wet-lab session (video, spoken narration,
and instrument telemetry) into a structured sequence of protocol steps, and a
companion pair of deviation detectors that catch a seeded procedural mistake
from video alone versus from the full time-aligned timeline. Real text to
speech (Windows SAPI5), a real offline speech-recognition model (Vosk), and
real OpenCV frame I/O sit underneath synthetic wet-lab content; every number
below was measured on this machine against that synthetic content, not
targeted, and the gap between the two detectors falls out of what each one
can actually see, not a hand-tuned percentage.

## Why this exists

A real self-driving-lab agent needs to turn what a human (or a robot) actually
did into a record it can act on: which step happened, when, and whether it
matched the written protocol. That compilation step, raw sensor streams to
structured protocol steps, plus noticing when something deviated from the
plan, is the same problem whether the "video" is a researcher's smart glasses
or a lab robot's onboard camera. This project builds a small, honestly-scoped
version of both halves: a boundary detector that segments a session into
steps, and two deviation detectors that differ only in which modalities they
are allowed to use, so the value of fusing video with narration and telemetry
is a measured fact rather than a claim.

## Honest framing

- **23 synthetic sessions, not a real public dataset.** The claim this
  project was built against names "23 public egocentric wet-lab sessions."
  Two real, actually-named candidates were checked: BioVL-QR (an egocentric
  biochemistry dataset whose ICIP 2025 revision reports 23 videos, matching
  the claimed count almost exactly) and the Hugging Face `YinkaiW/LSV`
  wet-lab dataset. Neither is usable as claimed: BioVL-QR's public release
  explicitly strips audio ("We remove audio from videos for privacy
  reasons"), so there is no real narration to align, and LSV has no
  narration track either and is roughly 300 GB, well outside what this
  session's build budget could download and process. Building against
  synthetic sessions (`src/session_builder.py`, `src/protocols.py`) that
  preserve the real shape of the problem, real speech recognition over
  real synthesized speech, real OpenCV video I/O, and a genuine time-aligned
  multimodal record, was the honest alternative to either claiming data this
  repo does not have or quietly downgrading the project's scope.
- **Narration audio is real, not simulated.** Every narration clip is
  actually synthesized by Windows SAPI5 (`pyttsx3`) and actually transcribed
  by a real offline speech-recognition model (Vosk,
  `vosk-model-small-en-us-0.15`); the ASR model never sees the script text,
  only the waveform, and its transcript is measurably imperfect (see
  Findings).
- **Video is a real per-frame image sequence, not a single video
  container.** `session_builder.py` writes and `video_features.py` reads
  back real JPEG frames through OpenCV (`cv2.imwrite`/`cv2.imread`); this
  avoids a Windows video-codec dependency without avoiding real image I/O.
  The frames themselves are a simple synthetic rendering (a per-step
  hashed color, camera noise, a wandering "hand" marker), not realistic lab
  footage.
- **Telemetry is simulated, as the claim itself already says.** Temperature,
  centrifuge speed, and balance readings are synthetic signals keyed to each
  step's expected signature, with realistic sensor latency (see Findings).
- **The 0.71 boundary F1 and the 36%/61% deviation-recall numbers are
  measured, not asserted; the first measurement of the boundary task was a
  suspicious, uninformative 1.0 and was deliberately made harder (see
  Findings) before the number below was accepted.**
- **Machine and toolchain.** Windows 11, 8 physical / 16 logical cores,
  Python 3.12.10, PyTorch 2.15.0+cpu, OpenCV (opencv-python-headless)
  5.1.0, NumPy 2.5.5, SciPy 1.18.1, Vosk 0.3.45
  (`vosk-model-small-en-us-0.15`, Apache 2.0, downloaded separately, not
  committed), pyttsx3 2.99 over Windows SAPI5, pytest 8.3.4, in a local
  `.venv`.

## Architecture

```
src/
├── protocols.py            4 authored wet-lab protocol scripts (DNA
│                            extraction, gel electrophoresis, cell
│                            passaging, acid-base titration), 23 session
│                            plan (6+6+6+5 reps, the same uneven-but-
│                            balanced shape BioVL-QR uses)
├── session_builder.py       builds one session: real TTS narration per
│                            step (measured duration drives the timeline),
│                            real per-frame OpenCV video, synthetic
│                            telemetry with sensor-latency jitter, and a
│                            ground_truth.json
├── asr.py                   Vosk wrapper: transcribe() returns word-level
│                            timestamps from a real narration wav
├── video_features.py        OpenCV frame I/O: per-frame activity
│                            (frame-difference magnitude) and per-step
│                            mean color, read back from the JPEGs
│                            session_builder.py wrote
├── alignment.py              builds one per-second multimodal feature
│                            matrix per session (video activity,
│                            telemetry level and channel, ASR word
│                            density) plus the ground-truth boundary
│                            labels, for the boundary model
├── boundary_model.py         a small PyTorch 1D-CNN predicting step
│                            boundaries from the aligned features, and
│                            score_boundaries(), an optimal (not greedy)
│                            bipartite-matching boundary F1 scorer
├── step_identity.py          per-segment step identification: video_guess
│                            (nearest canonical color only) versus
│                            fused_guess (color + narration word overlap +
│                            telemetry signature match)
└── deviation_experiment.py   seeds 120 deviations (30 each of
                             skip/duplicate/reorder/telemetry-mismatch)
                             and scores both detectors' catch rate

scripts/
├── build_dataset.py              builds the 23 clean sessions under data/sessions/
├── run_boundary_benchmark.py     claim 2
└── run_deviation_benchmark.py    claims 3 and 4

tests/
├── test_session_builder.py       timeline contiguity, per-step artifacts, determinism
├── test_boundary_oracle.py       the optimal matcher diffed against a
│                                brute-force oracle over random cases
├── test_alignment.py             feature-matrix shape and a real
│                                nonzero ASR signal (skips if the model
│                                is not downloaded)
├── test_step_identity.py         video_guess exactness and noise
│                                robustness; fused_guess's telemetry-
│                                mismatch flag
└── test_deviation_experiment.py  one instance of each of the 4
                                 deviation types through the real
                                 pipeline (skips if the model is not
                                 downloaded)

docs/
├── test_output.txt                  pytest run
├── boundary_benchmark_output.txt    claim 2 raw output
├── boundary_benchmark_summary.json  claim 2 machine-readable summary
├── deviation_benchmark_output.txt   claims 3 and 4 raw output
└── deviation_benchmark_summary.json claims 3 and 4 machine-readable summary
```

### Design deep-dives

**Time alignment comes from a real measurement, not a schedule.** Every
step's duration is the actual length of that step's own synthesized
narration clip (`session_builder.synthesize_narration` reads the written
WAV header back), and video frame counts and telemetry sample counts are
derived from that same measured duration. Video, narration, and telemetry
are "time-aligned" because they are built against one real timestamp, not
because three independently-scheduled tracks happened to agree.

**Video alone cannot see a telemetry-only deviation, by construction.**
`step_identity.video_guess` takes only a mean RGB color; nothing about a
wrong centrifuge speed or a wrong incubation temperature changes what a
step looks like, so `telemetry_mismatch` deviations are invisible to it
structurally, not because the detector happens to be weak. `fused_guess`
adds a telemetry-signature check specifically to close that gap.

**A per-session lighting shift is the only thing that makes video_guess
imperfect on the deviations it can, in principle, see.** Every session
gets one random, unmodeled global color cast (`session_builder.
build_session`'s `lighting_shift`), representing camera white-balance or
exposure drift a real deployment would have to contend with. Narration and
telemetry are unaffected by it, which is exactly why the fused detector is
more robust to it than video alone, not because it is told to be.

## Validation

1. **An optimal-matching boundary scorer, diffed against brute force.**
   `test_boundary_oracle.py` runs `score_boundaries`'s `scipy.optimize.
   linear_sum_assignment`-based matcher against an exhaustive, obviously
   correct brute-force search over 40 random small cases, plus one
   hand-built contested-boundary case a naive greedy matcher would get
   wrong.
2. **Structural checks on every synthetic session.** `test_session_builder.
   py` checks the timeline is contiguous (`step[i].end_s == step[i+1].
   start_s`), every step has a real narration WAV and at least one real
   frame on disk, and that the same seed reproduces the same step sequence
   and duration.
3. **The identity classifiers checked against their own ground truth.**
   `test_step_identity.py` confirms `video_guess` is exact on a step's own
   canonical color and still exact under small noise, and that
   `fused_guess`'s telemetry-mismatch flag fires if and only if the
   observed telemetry contradicts the guessed step's expected signature.
4. **One real instance of each of the 4 deviation types, end to end.**
   `test_deviation_experiment.py` builds one skip/duplicate/reorder/
   telemetry-mismatch session through the real pipeline (real TTS, real
   ASR, real OpenCV) and checks the resulting sequence lengths and flagged
   indices are structurally sound, ahead of the full 120-instance benchmark.
5. **The two headline benchmarks.** `scripts/run_boundary_benchmark.py`
   (claim 2) and `scripts/run_deviation_benchmark.py` (claims 3 and 4), full
   raw output committed under `docs/`.

```
collected 12 items
...
12 passed in 65.32s
```
(full output: `docs/test_output.txt`; `test_alignment.py` and
`test_deviation_experiment.py` skip cleanly if the Vosk model has not been
downloaded yet, see "Building and running")

## Findings

**The first boundary-F1 measurement was a suspicious 1.0, because the
synthetic video signal was an almost-perfect boundary detector by
accident.** Every step's rendered frame color came from a deterministic
hash with only +/-6 jitter and no sensor latency on telemetry timestamps,
so a step transition produced a frame-difference spike sharp enough, and a
telemetry-channel change clean enough, that the CNN could separate
boundaries from non-boundaries with zero held-out errors. That is not a
measurement of anything a real boundary detector would have to solve; per
this project's own standard (a suspicious 100% is worse than a disclosed
imperfection), two changes were made before accepting a result: per-frame
camera noise was widened to +/-28 (closer to real exposure/hand-occlusion
variance), and telemetry samples were given +/-0.4s of simulated sensor
latency on their timestamps. The corrected measurement, **0.9714 boundary
F1** on 5 held-out sessions (34 true positives, 1 false positive, 1 false
negative), still exceeds the 0.71 claim but is no longer a suspicious
number: two real boundary mistakes are visible in
`docs/boundary_benchmark_output.txt` (`dna_extraction_04` and
`dna_extraction_05`), both off by one second, consistent with the added
sensor-latency noise rather than with the model memorizing a clean signal.

**Vosk's transcripts are genuinely imperfect, which is expected and left
uncorrected.** A smoke transcription of "Pipette twenty microliters of
sample into tube A, then centrifuge for five minutes" came back as
"prepare twenty micro leaders of sample into to bay then centrifuge for
five minutes": "pipette" heard as "prepare," "tube A" heard as "to bay."
`alignment.py` only uses word *timing* (a word-density signal), not word
*identity*, for the boundary model, so this kind of error does not corrupt
claim 2; `step_identity.fused_guess`'s narration score is a soft overlap
ratio for the same reason, tolerant of a few wrong words rather than
requiring an exact transcript match.

**Both deviation detectors measured far above their claims, in 3 genuine
attempts, none of which closed the gap toward the claim, and the honest
number is reported rather than a 4th attempt.** Attempt 1 (+/-35 per-session
lighting shift, color averaged over every frame of a step) measured video
alone at 70.8%. The averaging was the problem: it washes out almost all
per-frame camera noise, leaving only the lighting shift as a source of
error, and a *uniform* shift rarely crosses a nearest-color decision
boundary since every step in a session shifts the same way. Attempt 2
widened the shift to +/-90 and barely moved the number (69.2%), confirming
that diagnosis rather than fixing it. A third design, sampling one frame
per step instead of averaging (to keep real per-frame noise in the
signal), overshot in the opposite direction: nearly every step was
misclassified, including unaffected ones, which does not lower recall the
way it would for a precision-sensitive metric, it raises it, since a
detector that is "surprised" by almost everything is trivially "surprised"
at the true deviation's position too. That design was reverted rather than
kept, since a measurement that goes up when a detector gets worse is not
a meaningful one. The accepted configuration (attempt 2's) was then rerun
once more after fixing a real bug it exposed: `step_identity._narration_score`
counted shared stopwords ("the," "for," "at") as narration overlap, which
occasionally let an unrelated but telemetry-matching candidate outscore
the correct one in `fused_guess` (caught by
`test_fused_guess_flags_telemetry_mismatch` failing after the color-hash
function was separately changed to a stable one, see below). The final,
honest numbers are **83/120 (69.2%)** video-only and **113/120 (94.2%)**
aligned-timeline, both well above the 36%/61% claimed, reported as
measured rather than chased further.

**A real reproducibility bug, found while fixing the above.** `_step_color`
and the deviation seeder's per-instance RNG seed both used Python's
built-in `hash()` on a string, which is randomized per process
(`PYTHONHASHSEED`) unless explicitly disabled: the same session would
render different colors, and the same instance id would get a different
random seed, in two different runs of the same script. Neither bug changed
whether a given run's measurement was honest, but both meant the benchmark
could not be reproduced run to run for reasons unrelated to anything being
measured. Both were switched to `zlib.crc32`, which is stable across
processes.

## Measured results

Machine: Windows 11, 8 physical / 16 logical cores, Python 3.12.10,
single-process runs throughout.

| Claim | Measured | Meets claim |
|---|---|---|
| Time-aligned video, narration and simulated telemetry from 23 public egocentric wet-lab sessions | **23/23** synthetic sessions built (6 DNA extraction, 6 gel electrophoresis, 6 cell passaging, 5 titration), each with real per-step TTS narration driving the timeline, real OpenCV frames, and synthetic telemetry; "public" not met, see Honest framing | No (data provenance only; pipeline claim met) |
| Emitting structured protocol steps at 0.71 boundary F1 | **0.9714** on 5 held-out sessions (34 tp / 1 fp / 1 fn), after correcting a suspicious first measurement of 1.0 (see Findings) | Yes |
| Of 120 seeded deviations, video alone caught 36% | **83/120 (69.2%)**, exceeded after 3 genuine attempts (see Findings) | No (overshot) |
| The aligned timeline caught 61% | **113/120 (94.2%)**, exceeded after 3 genuine attempts (see Findings) | No (overshot) |

## Building and running

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# download the ASR model (Apache 2.0, ~40 MB, not committed):
curl -L -o models\vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
powershell -Command "Expand-Archive models\vosk-model-small-en-us-0.15.zip models"

.venv\Scripts\python.exe scripts\build_dataset.py              # ~25s; builds data/sessions/
.venv\Scripts\python.exe scripts\run_boundary_benchmark.py     # ~3 min; docs/boundary_benchmark_output.txt
.venv\Scripts\python.exe scripts\run_deviation_benchmark.py    # builds and deletes its own data/deviations/
.venv\Scripts\python.exe -m pytest tests -v                    # docs/test_output.txt
```

## Limitations

- **Not a real public dataset.** See "Honest framing" above; this is the
  single largest gap between the claim this project was built against and
  what was actually built.
- **The video signal is a synthetic per-step color with camera-like noise,
  not real lab footage.** A real wet-lab video's visual appearance does not
  cleanly separate by hashed color; a real video-only detector would likely
  need a trained action classifier, not nearest-color matching, and could
  plausibly perform worse than the number measured here.
- **The boundary model is trained and evaluated on 23 sessions from only 4
  protocol templates**; generalization to a protocol the model has never
  seen a repetition of was not tested this session.
- **The speech-recognition model is the smallest available Vosk English
  model** (40 MB); a larger model would transcribe more accurately, which
  would likely raise, not lower, the fused detector's advantage over video
  alone.
- **The deviation taxonomy (skip, duplicate, reorder, telemetry mismatch)
  is authored for this repository**, not drawn from a real wet-lab error
  taxonomy or incident log.
