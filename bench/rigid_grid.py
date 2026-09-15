"""rigid_grid.py - Rekordbox's method, in camelot's stack: one tempo, phase from the audio.

The benchmark showed librosa's DP beat tracker drifts +-200 ms through a rigid
track even though librosa's *tempo estimate* matches Rekordbox to 0.01 BPM.
Fitting a grid to those drifting beats inherits the drift; taking the phase as
their circular mean is random. So: take the tempo, and find the phase by
scoring every candidate offset against the onset-strength envelope directly.
Downbeat phase then reuses camelot's own estimate_downbeat_phase() on the grid.

    python bench/rigid_grid.py                 # tempo from camelot's analysis_cache.json
    python bench/rigid_grid.py --tempo librosa # tempo from librosa.beat.tempo on the fly
    python bench/rigid_grid.py --comb          # refine the tempo by 2-D comb search (recommended:
                                               # the tempogram value is a bin centre, up to 0.2 BPM off)

Writes results/rigid_grid/<track>.json.
"""
import io
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

from beat_bench import audio_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from bpm_detect import estimate_downbeat_phase  # noqa: E402

RESULTS = HERE / "results"
SR, HOP = 44100, 512


def phase_from_onsets(onset_env, sr, hop, period, steps=64):
    """Best grid offset in [0, period): maximise summed onset strength on the grid.
    Coarse search at `steps` per period, then refine at 1 ms."""
    frame_t = np.arange(len(onset_env)) * hop / sr
    dur = frame_t[-1]

    def score(off):
        grid = off + period * np.arange(int((dur - off) / period) + 1)
        idx = np.clip(np.round(grid * sr / hop).astype(int), 0, len(onset_env) - 1)
        return onset_env[idx].sum()

    coarse = np.linspace(0, period, steps, endpoint=False)
    best = coarse[np.argmax([score(o) for o in coarse])]
    fine = best + np.arange(-period / steps, period / steps, 0.001)
    fine = fine[fine >= 0]
    return float(fine[np.argmax([score(o) for o in fine])])


def comb_refine(onset_env, sr, hop, bpm0, span=1.5, step=0.01):
    """2-D comb search: for each tempo in bpm0 +- span, the best-phase onset sum on a
    rigid grid; return the tempo with the highest peak. Immune to the half-beat phase
    ambiguity (it takes the max over phase) and to librosa's tempogram bin quantisation."""
    fps = sr / hop
    n = len(onset_env)
    best_bpm, best_score = bpm0, -1
    for bpm in np.arange(bpm0 - span, bpm0 + span + 1e-9, step):
        period_f = 60.0 / bpm * fps
        # phase resolution of one frame is enough to rank tempos
        score = 0
        for off in np.arange(0, period_f, 1.0):
            idx = np.arange(off, n, period_f).astype(int)
            sc = onset_env[idx].sum() / len(idx)
            if sc > score:
                score = sc
        if score > best_score:
            best_bpm, best_score = bpm, score
    return float(best_bpm)


def kick_onsets(y, sr, hop, fmax=150.0):
    """Onset strength restricted to the kick band, so hats/snares can't win the phase."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=16, fmax=fmax)
    return librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop)


def main(tempo_source="camelot"):
    out = RESULTS / ("rigid_grid_comb" if "--comb" in sys.argv else "rigid_grid")
    out.mkdir(parents=True, exist_ok=True)
    tracks = json.load(io.open(HERE / "tracks.json", encoding="utf-8"))
    for name, info in tracks.items():
        t0 = time.time()
        y, sr = librosa.load(audio_path(info["audio"]), sr=SR, mono=True)
        cache = ROOT / name / "analysis_cache.json"
        if tempo_source == "camelot" and cache.exists():
            bpm = json.load(io.open(cache, encoding="utf-8"))["tempo"]
        else:
            bpm = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr, hop_length=HOP))[0])
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        period = 60.0 / bpm
        if "--comb" in sys.argv:
            bpm = comb_refine(onset_env, sr, HOP, bpm); period = 60.0 / bpm
        off = phase_from_onsets(kick_onsets(y, sr, HOP), sr, HOP, period)
        grid = off + period * np.arange(int((len(y) / sr - off) / period) + 1)
        db = estimate_downbeat_phase(y, sr, grid)
        downbeats = grid[db["phase"]::4]
        json.dump({"beats": grid.tolist(), "downbeats": downbeats.tolist(), "bpm": bpm,
                   "phase_s": off, "downbeat_phase": db["phase"], "downbeat_conf": db["confidence"],
                   "seconds": time.time() - t0},
                  io.open(out / f"{name}.json", "w", encoding="utf-8"))
        print(f"  {name[:46]:48} bpm={bpm:8.3f} phase={off:6.3f}s dbphase={db['phase']} conf={db['confidence']:.2f} {time.time()-t0:5.1f}s")


if __name__ == "__main__":
    src = sys.argv[sys.argv.index("--tempo") + 1] if "--tempo" in sys.argv else "camelot"
    main(src)
