"""run.py - allin1 over a folder of audio; writes the benchmark's JSON shape + segments.

    python run.py /workspace/tracks /workspace/results

Output per track (basename without extension):
    results/allin1/<track>.json           {"beats", "downbeats", "bpm", "seconds"}
    results/allin1/<track>.segments.json  [{"start", "end", "label"}, ...]
Copy results/ back next to camelot_from_youtube/bench/results/ and run beat_bench.py score.

allin1's README warns MP3 decoders differ by a few ms of offset - so the same files are
decoded to WAV with ffmpeg first, and that WAV is what allin1 sees. The benchmark's other
trackers ran on the MP3s via librosa/audioread; a constant few-ms offset is well inside
the 70 ms scoring window, but it is why 'offset_ms' in the scores is reported at all.
"""
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import allin1

src = Path(sys.argv[1])
out = Path(sys.argv[2]) / "allin1"
out.mkdir(parents=True, exist_ok=True)
wav_dir = src / "_wav"
wav_dir.mkdir(exist_ok=True)

files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac"})
print(f"{len(files)} tracks")
for p in files:
    t0 = time.time()
    wav = wav_dir / (p.stem + ".wav")
    if p.suffix.lower() != ".wav" and not wav.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-ar", "44100", str(wav)], check=True)
    r = allin1.analyze(str(wav if p.suffix.lower() != ".wav" else p), keep_byproducts=False)
    json.dump({"beats": [float(b) for b in r.beats], "downbeats": [float(d) for d in r.downbeats],
               "bpm": float(r.bpm), "seconds": time.time() - t0},
              io.open(out / f"{p.stem}.json", "w", encoding="utf-8"))
    json.dump([{"start": float(s.start), "end": float(s.end), "label": s.label} for s in r.segments],
              io.open(out / f"{p.stem}.segments.json", "w", encoding="utf-8"), indent=1)
    print(f"  {p.stem[:50]:52} bpm={r.bpm:7.2f} beats={len(r.beats):5d} downbeats={len(r.downbeats):4d} "
          f"segments={len(r.segments):3d} {time.time()-t0:6.1f}s")
print("done ->", out)
