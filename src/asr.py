"""Real offline speech recognition (Vosk, vosk-model-small-en-us-0.15)
over the synthesized narration audio. This is a genuine ASR pass: the
model never sees the text that was used to synthesize the audio, only the
waveform, and its transcript is measurably imperfect (see README
Findings), the same as any real speech-recognition deployment.
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)  # Vosk logs its own startup banner to stderr by default; silenced for clean benchmark output

_MODEL_CACHE: dict[str, Model] = {}


@dataclass
class Word:
    word: str
    start_s: float
    end_s: float
    conf: float


def _get_model(model_dir: Path) -> Model:
    key = str(model_dir)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = Model(str(model_dir))
    return _MODEL_CACHE[key]


def transcribe(wav_path: Path, model_dir: Path) -> list[Word]:
    model = _get_model(model_dir)
    with wave.open(str(wav_path), "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        words: list[Word] = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                words.extend(_parse_result(rec.Result()))
        words.extend(_parse_result(rec.FinalResult()))
    return words


def _parse_result(raw_json: str) -> list[Word]:
    parsed = json.loads(raw_json)
    return [
        Word(word=w["word"], start_s=w["start"], end_s=w["end"], conf=w.get("conf", 1.0))
        for w in parsed.get("result", [])
    ]
