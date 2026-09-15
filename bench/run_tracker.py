"""run_tracker.py - Run one beat/downbeat tracker over the benchmark tracks.

Writes bench/results/<tracker>/<track>.json in the shape beat_bench.py scores:
    {"beats": [s...], "downbeats": [s...], "bpm": float|null, "seconds": wall time}

    python bench/run_tracker.py beat_this [--cpu] [--dbn]
    python bench/run_tracker.py allin1
    python bench/run_tracker.py librosa          # raw librosa.beat_track, no camelot post-processing

Each tracker lives in whatever env can import it (env_mir for the neural ones);
the JSON on disk is the only thing the scorer reads.
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from beat_bench import audio_path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def tracks():
    return json.load(io.open(HERE / "tracks.json", encoding="utf-8"))


def save(tracker, name, beats, downbeats, bpm, secs):
    out = RESULTS / tracker
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"beats": [float(b) for b in beats], "downbeats": [float(d) for d in downbeats],
               "bpm": None if bpm is None else float(bpm), "seconds": secs},
              io.open(out / f"{name}.json", "w", encoding="utf-8"))
    print(f"  {name[:50]:52} beats={len(beats):5d} downbeats={len(downbeats):4d} bpm={bpm!s:>8} {secs:6.1f}s")


def run_beat_this(cpu=False, dbn=False):
    from beat_this.inference import File2Beats
    f2b = File2Beats(checkpoint_path="final0", device="cpu" if cpu else "cuda", dbn=dbn)
    tag = "beat_this" + ("_dbn" if dbn else "")
    for name, info in tracks().items():
        t0 = time.time()
        beats, downbeats = f2b(audio_path(info["audio"]))
        beats = np.asarray(beats); downbeats = np.asarray(downbeats)
        bpm = 60 / np.median(np.diff(beats)) if len(beats) > 2 else None
        save(tag, name, beats, downbeats, round(bpm, 3) if bpm else None, time.time() - t0)


def run_allin1():
    import allin1
    for name, info in tracks().items():
        t0 = time.time()
        r = allin1.analyze(audio_path(info["audio"]))
        save("allin1", name, r.beats, r.downbeats, r.bpm, time.time() - t0)
        segs = [{"start": s.start, "end": s.end, "label": s.label} for s in r.segments]
        json.dump(segs, io.open(RESULTS / "allin1" / f"{name}.segments.json", "w", encoding="utf-8"), indent=1)


def run_librosa():
    import librosa
    for name, info in tracks().items():
        t0 = time.time()
        y, sr = librosa.load(audio_path(info["audio"]), sr=44100, mono=True)
        tempo, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beats = librosa.frames_to_time(frames, sr=sr, hop_length=512)
        tempo = float(np.atleast_1d(tempo)[0])
        # librosa has no downbeat concept; emit every 4th beat from the first so the
        # downbeat score reflects that honestly (phase 0, no estimation)
        save("librosa", name, beats, beats[::4], round(tempo, 3), time.time() - t0)


if __name__ == "__main__":
    which = sys.argv[1]
    flags = set(sys.argv[2:])
    {"beat_this": lambda: run_beat_this(cpu="--cpu" in flags, dbn="--dbn" in flags),
     "allin1": run_allin1, "librosa": run_librosa}[which]()
