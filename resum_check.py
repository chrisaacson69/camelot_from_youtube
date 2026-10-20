#!/usr/bin/env python3
"""
Stem Re-summation Check
=======================
Oracle check 1 of the G7 pilot (Vault: research/eyes-and-ears-for-llms.md): do the separated
stems add back up to the mix? Reads the stem folders written by separate_stems.py and reports,
per track, the reconstruction residual and how much clipping the saved stems carry.

Usage:
    python resum_check.py STEMS_DIR [STEMS_DIR ...]
    python resum_check.py --root ./pilot_stems          # every */separation_info.json under root
    python resum_check.py --root ./pilot_stems --csv resum.csv

Metrics (per track):
    residual_db   10*log10(|mix - sum(stems)|^2 / |mix|^2). Lower is better; -40 dB or below
                  means the stems re-sum almost exactly. Demucs does not enforce re-summation,
                  so this is a real measurement, not a tautology.
    clip_<stem>   fraction of samples at full scale in the saved stem (clamp on save adds residual)
    share_<stem>  stem energy / sum of stem energies; a large 'other' share on electronic tracks
                  is the "other is a catch-all" hazard in numbers (the ears still decide mush)

Environment:
    env_demucs (uses demucs.audio.AudioFile -> ffmpeg to decode the mix at the model rate)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mix(path, samplerate, channels):
    from demucs.audio import AudioFile
    wav = AudioFile(path).read(streams=0, samplerate=samplerate, channels=channels)
    return wav.numpy().astype(np.float64)


def check_track(stems_dir):
    stems_dir = Path(stems_dir)
    info = json.loads((stems_dir / "separation_info.json").read_text())
    sr = info["sample_rate"]

    stems = {}
    for name, p in info["stems"].items():
        data, file_sr = sf.read(p, dtype="float64", always_2d=True)
        if file_sr != sr:
            raise ValueError(f"{p}: {file_sr} Hz, expected {sr} Hz")
        stems[name] = data.T  # (channels, samples)

    channels = next(iter(stems.values())).shape[0]
    mix = load_mix(info["input"], sr, channels)

    n = min(mix.shape[1], *(s.shape[1] for s in stems.values()))
    mix = mix[:, :n]
    total = sum(s[:, :n] for s in stems.values())

    mix_energy = np.sum(mix ** 2)
    resid_energy = np.sum((mix - total) ** 2)
    stem_energy = {k: np.sum(s[:, :n] ** 2) for k, s in stems.items()}
    stem_sum = sum(stem_energy.values())

    row = {
        "track": Path(info["input"]).stem,
        "model": info["model"],
        "seconds": round(n / sr, 1),
        "residual_db": round(10 * np.log10(resid_energy / mix_energy), 2),
    }
    for k, s in stems.items():
        row[f"clip_{k}"] = round(float(np.mean(np.abs(s[:, :n]) >= 0.9999)), 6)
    for k, e in stem_energy.items():
        row[f"share_{k}"] = round(float(e / stem_sum), 3)
    return row


def main():
    ap = argparse.ArgumentParser(description="Check that separated stems re-sum to the mix")
    ap.add_argument("dirs", nargs="*", help="stem folders containing separation_info.json")
    ap.add_argument("--root", help="scan this folder for */separation_info.json")
    ap.add_argument("--csv", help="also write results to this CSV")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    if args.root:
        dirs += sorted(p.parent for p in Path(args.root).rglob("separation_info.json"))
    if not dirs:
        ap.error("no stem folders given")

    rows = []
    for d in dirs:
        try:
            rows.append(check_track(d))
        except Exception as e:
            print(f"FAIL {d}: {e}", file=sys.stderr)

    if not rows:
        sys.exit(1)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))

    res = np.array([r["residual_db"] for r in rows])
    print(f"\n{len(rows)} tracks  residual_db  median {np.median(res):.2f}  worst {res.max():.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
