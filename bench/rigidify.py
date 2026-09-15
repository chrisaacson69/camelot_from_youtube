"""rigidify.py - Force a constant-tempo grid onto any tracker's beats.

Rekordbox's "simplistic" method: assume one tempo for the whole track and fit
phase to it. Machine-sequenced dance music IS rigid, so a tracker that follows
onsets frame by frame (librosa's DP tracker) trades away the one prior that
matters. This turns results/<tracker>/ into results/<tracker>_rigid/ so the
same scorer can say how much of a tracker's error was drift.

    python bench/rigidify.py librosa camelot beat_this

Method: robust line fit t_k = t0 + k * period over beat index k, iteratively
dropping beats whose residual exceeds 25% of a period (the tracker skipped or
doubled somewhere); then the downbeat phase = the residue class (mod 4) that
the tracker's own downbeats fall into most often.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _steps(d, period):
    """Beat-index increments: anything under 3/4 of a period is a double-time insert (0),
    not a rounding-up to 1."""
    r = d / period
    return np.where(r < 0.75, 0, np.round(r)).astype(int)


def fit_grid(beats, downbeats, beats_per_bar=4):
    beats = np.asarray(beats, float)
    # beat index by cumulative rounded intervals, so a skipped beat doesn't shift k
    d = np.diff(beats)
    med = np.median(d)
    # frame-quantised trackers alternate around the true period (480/500 ms at 50 fps
    # for 491.8 ms): average the single-beat intervals so a long gap is indexed right
    period0 = d[np.abs(d / med - 1) < 0.15].mean()
    steps = _steps(d, period0)
    # a tracker that went double-time emits beats half a period apart (step 0): drop them
    keep0 = np.concatenate([[True], steps > 0])
    beats = beats[keep0]
    k = np.concatenate([[0], np.cumsum(_steps(np.diff(beats), period0))]).astype(int)
    keep = np.ones(len(beats), bool)
    for _ in range(10):
        A = np.vstack([np.ones(keep.sum()), k[keep]]).T
        t0, period = np.linalg.lstsq(A, beats[keep], rcond=None)[0]
        resid = beats - (t0 + period * k)
        new_keep = np.abs(resid) < 0.25 * period
        if new_keep.sum() == keep.sum() and (new_keep == keep).all():
            break
        keep = new_keep
    n = int(np.round((beats[-1] - t0) / period)) + 1
    grid = t0 + period * np.arange(n)
    # downbeat phase: which residue class mod beats_per_bar the tracker's downbeats hit
    if len(downbeats):
        kd = np.round((np.asarray(downbeats, float) - t0) / period).astype(int)
        phase = np.bincount(kd % beats_per_bar, minlength=beats_per_bar).argmax()
    else:
        phase = 0
    return grid, grid[phase::beats_per_bar], 60.0 / period, float(np.std(resid[keep]) * 1000)


if __name__ == "__main__":
    for tracker in sys.argv[1:]:
        src = RESULTS / tracker
        out = RESULTS / f"{tracker}_rigid"
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.json")):
            if f.name.endswith(".segments.json"):
                continue
            d = json.load(io.open(f, encoding="utf-8"))
            grid, down, bpm, jitter_ms = fit_grid(d["beats"], d["downbeats"])
            json.dump({"beats": grid.tolist(), "downbeats": down.tolist(), "bpm": round(bpm, 3),
                       "jitter_ms": round(jitter_ms, 1)}, io.open(out / f.name, "w", encoding="utf-8"))
            print(f"  {tracker:10} {f.stem[:44]:46} bpm={bpm:8.3f} residual sd={jitter_ms:6.1f} ms")
