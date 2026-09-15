"""beat_bench.py - Score beat/downbeat trackers against the Rekordbox grid.

Reference: Rekordbox's own analysis (PQTZ tag in each track's ANLZ0000.DAT),
read via pyrekordbox. Candidates are stored as JSON under bench/results/
so each tracker can be run in its own environment:

    results/<tracker>/<track>.json  ->  {"beats": [...s], "downbeats": [...s]}

Usage:
    python bench/beat_bench.py reference          # dump Rekordbox grids -> results/rekordbox/
    python bench/beat_bench.py camelot            # dump camelot's cached grid -> results/camelot/
    python bench/beat_bench.py score              # score every results/<tracker>/ vs rekordbox

Metrics (mir_eval conventions, 70 ms window): F-measure for beats and for
downbeats, plus the median signed offset of matched beats (candidate minus
reference, ms) - the phase error a DJ would hear as "early" or "late".
Rekordbox stores beat times in integer milliseconds, so anything under
~1 ms of offset is quantisation, not disagreement.
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
WINDOW = 0.07  # seconds; mir_eval default


# ── Reference: Rekordbox ─────────────────────────────────────────────────

AUDIO_DIRS = [ROOT, ROOT.parent / "music"]   # sibling `music/` holds tracks not copied into this repo


def audio_path(name):
    for d in AUDIO_DIRS:
        if (d / name).exists():
            return str(d / name)
    raise FileNotFoundError(f"{name} not under {[str(d) for d in AUDIO_DIRS]}")


def dump_rekordbox():
    import warnings
    warnings.filterwarnings("ignore")
    from pyrekordbox import Rekordbox6Database
    from pyrekordbox.anlz import AnlzFile

    db = Rekordbox6Database()
    share = os.path.expandvars(r"%APPDATA%/Pioneer/rekordbox/share")
    out = RESULTS / "rekordbox"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for c in db.get_content():
        if not c.BPM or "/source/repos/music/" not in (c.FolderPath or ""):
            continue
        base = Path(c.FolderPath).stem
        tag = AnlzFile.parse_file(share + c.AnalysisDataPath).get_tag("PQTZ")
        t = tag.get_times().astype(float)
        if t.max() > 10000:          # stored in ms
            t = t / 1000.0
        bn = tag.get_beats()
        rec = {"beats": t.tolist(), "downbeats": t[bn == 1].tolist(),
               "bpm": c.BPM / 100, "src": c.FolderPath}
        json.dump(rec, io.open(out / f"{base}.json", "w", encoding="utf-8"))
        manifest[base] = {"audio": Path(c.FolderPath).name, "bpm": c.BPM / 100}
        print(f"  {base[:50]:52} beats={len(t):5d} downbeats={int((bn == 1).sum()):4d}")
    json.dump(manifest, io.open(HERE / "tracks.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(manifest)} reference grids -> {out}")


# ── Candidate: camelot's cached analysis ─────────────────────────────────

def dump_camelot():
    out = RESULTS / "camelot"
    out.mkdir(parents=True, exist_ok=True)
    for name in json.load(io.open(HERE / "tracks.json", encoding="utf-8")):
        proj = ROOT / name
        if not (proj / "analysis_cache.npz").exists():
            print(f"  skip (no camelot analysis): {name[:50]}")
            continue
        z = np.load(proj / "analysis_cache.npz", allow_pickle=True)
        cache = json.load(io.open(proj / "analysis_cache.json", encoding="utf-8"))
        beats = z["beat_times"].astype(float)
        downbeats = [m["start"] for m in cache["measures"]]
        json.dump({"beats": beats.tolist(), "downbeats": downbeats, "bpm": cache["tempo"]},
                  io.open(out / f"{name}.json", "w", encoding="utf-8"))
        print(f"  {name[:50]:52} beats={len(beats):5d} downbeats={len(downbeats):4d}")


# ── Scoring ──────────────────────────────────────────────────────────────

def f_measure(ref, est, window=WINDOW):
    """Greedy one-to-one matching within `window`; returns (F, offsets)."""
    ref = np.asarray(ref, float); est = np.asarray(est, float)
    if len(ref) == 0 or len(est) == 0:
        return 0.0, np.array([])
    used = np.zeros(len(est), bool)
    offsets = []
    idx = np.searchsorted(est, ref)
    for i, r in enumerate(ref):
        best, bestd = -1, window
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(est) and not used[j] and abs(est[j] - r) <= bestd:
                best, bestd = j, abs(est[j] - r)
        if best >= 0:
            used[best] = True
            offsets.append(est[best] - r)
    hits = len(offsets)
    p = hits / len(est); r_ = hits / len(ref)
    f = 0.0 if hits == 0 else 2 * p * r_ / (p + r_)
    return f, np.array(offsets)


def score():
    refdir = RESULTS / "rekordbox"
    trackers = sorted(d.name for d in RESULTS.iterdir() if d.is_dir() and d.name != "rekordbox")
    rows = []
    for tr in trackers:
        for f in sorted(refdir.glob("*.json")):
            cand = RESULTS / tr / f.name
            if not cand.exists():
                continue
            ref = json.load(io.open(f, encoding="utf-8"))
            est = json.load(io.open(cand, encoding="utf-8"))
            # score only over the span the reference covers (trackers may extend into silence)
            lo, hi = ref["beats"][0] - WINDOW, ref["beats"][-1] + WINDOW
            eb = [t for t in est["beats"] if lo <= t <= hi]
            ed = [t for t in est["downbeats"] if lo <= t <= hi]
            fb, ob = f_measure(ref["beats"], eb)
            fd, od = f_measure(ref["downbeats"], ed)
            rows.append((tr, f.stem, fb, fd, 1000 * np.median(ob) if len(ob) else np.nan,
                         est.get("bpm"), ref["bpm"], len(eb), len(ref["beats"])))
    print(f"{'tracker':10} {'track':44} {'beatF':>6} {'downF':>6} {'off ms':>7} {'bpm':>8} {'rb bpm':>7} {'n/ref':>9}")
    for r in rows:
        print(f"{r[0]:10} {r[1][:42]:44} {r[2]:6.3f} {r[3]:6.3f} {r[4]:7.1f} {r[5] if r[5] is None else round(r[5], 2)!s:>8} {r[6]:7.2f} {r[7]:4d}/{r[8]:<4d}")
    print()
    for tr in trackers:
        rs = [r for r in rows if r[0] == tr]
        if rs:
            print(f"{tr:10} mean beatF={np.mean([r[2] for r in rs]):.3f}  mean downF={np.mean([r[3] for r in rs]):.3f}  "
                  f"tracks={len(rs)}  downF>=0.9 on {sum(r[3] >= 0.9 for r in rs)}")
    json.dump([dict(zip(["tracker", "track", "beat_f", "down_f", "offset_ms", "bpm", "rb_bpm", "n_est", "n_ref"], r)) for r in rows],
              io.open(RESULTS / "scores.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"reference": dump_rekordbox, "camelot": dump_camelot, "score": score}[cmd]()
