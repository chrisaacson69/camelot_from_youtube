#!/usr/bin/env python3
"""
key_detect.py - Measure-aligned key and Camelot detection from audio.

Uses the beat grid from bpm_detect.py to create measure-aligned analysis
windows, then estimates key per measure using CQT chroma features correlated
against Krumhansl key profiles.  Groups measures into key segments and
produces a timeline of key changes across the track.

The beat-grid alignment means every chroma window corresponds to an actual
musical bar — no arbitrary time slices that might straddle key changes.

Usage:
    python key_detect.py "audio.wav"
    python key_detect.py "audio.wav" --phrase-bars 4
    python key_detect.py "audio.wav" --save-plot key_chart.png
    python key_detect.py "audio.wav" --json-out key_results.json
    python key_detect.py "audio.wav" --no-plot
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Reuse beat detection from bpm_detect
from bpm_detect import detect_beats, group_beats_into_measures, format_time


# ---------------------------------------------------------------------------
# Krumhansl key profiles and Camelot mapping
# ---------------------------------------------------------------------------

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PC_TO_NOTE = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

CAMELOT_MAJOR = {
    "B": "1B", "Gb": "2B", "Db": "3B", "Ab": "4B", "Eb": "5B", "Bb": "6B",
    "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}

CAMELOT_MINOR = {
    "Ab": "1A", "Eb": "2A", "Bb": "3A", "F": "4A", "C": "5A", "G": "6A",
    "D": "7A", "A": "8A", "E": "9A", "B": "10A", "Gb": "11A", "Db": "12A",
}

# Map Camelot codes to colors for plotting
CAMELOT_COLORS = {}
for i, code in enumerate(["1A","2A","3A","4A","5A","6A","7A","8A","9A","10A","11A","12A",
                           "1B","2B","3B","4B","5B","6B","7B","8B","9B","10B","11B","12B"]):
    CAMELOT_COLORS[code] = plt.cm.tab20(i / 24)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Key estimation
# ---------------------------------------------------------------------------

def estimate_key(chroma_mean):
    """Estimate key from a mean chroma vector (12-dim, already normalized).

    Returns dict with tonic_pc, mode, key_name, camelot, score, confidence,
    and top_candidates.
    """
    maj_prof = _normalize(KRUMHANSL_MAJOR)
    min_prof = _normalize(KRUMHANSL_MINOR)

    candidates = []
    for tonic in range(12):
        maj_score = float(np.dot(chroma_mean, np.roll(maj_prof, tonic)))
        min_score = float(np.dot(chroma_mean, np.roll(min_prof, tonic)))
        candidates.append((maj_score, tonic, "major"))
        candidates.append((min_score, tonic, "minor"))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_tonic, best_mode = candidates[0]
    second_score = candidates[1][0]

    gap = best_score - second_score
    confidence = float(np.clip(gap / 0.05, 0.0, 1.0))

    note = PC_TO_NOTE[best_tonic]
    key_name = f"{note} {best_mode}"
    camelot_map = CAMELOT_MAJOR if best_mode == "major" else CAMELOT_MINOR
    camelot = camelot_map.get(note, "?")

    top = []
    for score, tonic, mode in candidates[:6]:
        n = PC_TO_NOTE[tonic]
        cm = CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR
        top.append({
            "score": round(float(score), 4),
            "key": f"{n} {mode}",
            "camelot": cm.get(n, "?"),
        })

    return {
        "tonic_pc": int(best_tonic),
        "mode": best_mode,
        "key_name": key_name,
        "camelot": camelot,
        "score": round(float(best_score), 4),
        "confidence": round(float(confidence), 3),
        "top_candidates": top,
    }


# ---------------------------------------------------------------------------
# Measure-aligned chroma and key analysis
# ---------------------------------------------------------------------------

def compute_measure_chroma(y, sr, measures, use_harmonic=True):
    """Compute mean chroma vector for each measure, aligned to beat grid.

    Args:
        y: Audio time series.
        sr: Sample rate.
        measures: List of measure dicts from group_beats_into_measures().
        use_harmonic: Apply HPSS harmonic separation first.

    Returns list of dicts, one per measure, each with chroma_mean and key estimate.
    """
    if use_harmonic:
        y_analysis = librosa.effects.harmonic(y)
    else:
        y_analysis = y

    results = []
    for m in measures:
        start_sample = int(m["start"] * sr)
        end_sample = int(m["end"] * sr)
        segment = y_analysis[start_sample:end_sample]

        if len(segment) < sr * 0.2:  # skip very short segments
            results.append(None)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_mean = _normalize(chroma.mean(axis=1))

        key_est = estimate_key(chroma_mean)
        key_est["measure_num"] = m["measure_num"]
        key_est["start"] = m["start"]
        key_est["end"] = m["end"]
        key_est["bpm"] = m["bpm"]
        results.append(key_est)

    return results


def compute_phrase_chroma(y, sr, measures, phrase_bars=4, use_harmonic=True):
    """Compute key estimate for multi-bar phrases (e.g., 4-bar or 8-bar groups).

    Longer windows give more stable chroma features, similar to how span-based
    BPM averages out frame quantization.  This is the primary analysis unit.
    """
    if use_harmonic:
        y_analysis = librosa.effects.harmonic(y)
    else:
        y_analysis = y

    phrases = []
    for i in range(0, len(measures), phrase_bars):
        group = measures[i:i + phrase_bars]
        if not group:
            continue

        start_time = group[0]["start"]
        end_time = group[-1]["end"]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y_analysis[start_sample:end_sample]

        if len(segment) < sr * 0.5:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_mean = _normalize(chroma.mean(axis=1))

        key_est = estimate_key(chroma_mean)
        key_est["phrase_num"] = len(phrases) + 1
        key_est["start_measure"] = group[0]["measure_num"]
        key_est["end_measure"] = group[-1]["measure_num"]
        key_est["start"] = start_time
        key_est["end"] = end_time
        key_est["measure_count"] = len(group)
        phrases.append(key_est)

    return phrases


# ---------------------------------------------------------------------------
# Key segmentation (group phrases by key)
# ---------------------------------------------------------------------------

def find_key_segments(phrases, min_segment_phrases=2):
    """Group consecutive phrases with the same Camelot code into segments.

    Similar philosophy to BPM segmentation: short blips get absorbed.
    """
    if not phrases:
        return []

    # Pass 1: group consecutive same-key phrases
    raw = []
    current = [phrases[0]]
    for p in phrases[1:]:
        if p["camelot"] == current[-1]["camelot"]:
            current.append(p)
        else:
            raw.append(current)
            current = [p]
    raw.append(current)

    # Pass 2: absorb short segments into neighbors
    if len(raw) <= 1:
        return [_finalize_key_segment(g) for g in raw]

    groups = list(raw)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            if len(groups[i]) < min_segment_phrases and len(groups) > 1:
                # Absorb into neighbor with higher total confidence
                best_idx = None
                best_conf = -1
                for ni in [i - 1, i + 1]:
                    if 0 <= ni < len(groups):
                        nc = sum(p["confidence"] for p in groups[ni])
                        if nc > best_conf:
                            best_conf = nc
                            best_idx = ni
                if best_idx is not None:
                    if best_idx < i:
                        groups[best_idx].extend(groups[i])
                    else:
                        groups[best_idx] = groups[i] + groups[best_idx]
                    groups.pop(i)
                    changed = True
                    continue
            i += 1

    # Pass 3: merge adjacent same-key after absorption
    merged = [groups[0]]
    for g in groups[1:]:
        # Use most-voted key in each group
        prev_key = _majority_camelot(merged[-1])
        curr_key = _majority_camelot(g)
        if prev_key == curr_key:
            merged[-1].extend(g)
        else:
            merged.append(g)

    return [_finalize_key_segment(g) for g in merged]


def _majority_camelot(phrases):
    """Most common Camelot code in a group of phrases."""
    from collections import Counter
    counts = Counter(p["camelot"] for p in phrases)
    return counts.most_common(1)[0][0]


def _finalize_key_segment(phrases):
    """Build segment summary from a group of phrases."""
    from collections import Counter
    camelot_counts = Counter(p["camelot"] for p in phrases)
    dominant_camelot = camelot_counts.most_common(1)[0][0]

    # Find representative phrase (highest confidence with dominant key)
    rep = max((p for p in phrases if p["camelot"] == dominant_camelot),
              key=lambda p: p["confidence"])

    confidences = [p["confidence"] for p in phrases]
    scores = [p["score"] for p in phrases]

    return {
        "start_measure": phrases[0]["start_measure"],
        "end_measure": phrases[-1]["end_measure"],
        "start_time": phrases[0]["start"],
        "end_time": phrases[-1]["end"],
        "camelot": dominant_camelot,
        "key_name": rep["key_name"],
        "confidence_mean": round(float(np.mean(confidences)), 3),
        "confidence_max": round(float(max(confidences)), 3),
        "score_mean": round(float(np.mean(scores)), 4),
        "phrase_count": len(phrases),
        "camelot_counts": dict(camelot_counts),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(segments, phrases, duration):
    """Compute overall summary: dominant key, key changes, coverage."""
    if not segments:
        return {}

    # Dominant by time coverage
    coverage = {}
    for seg in segments:
        dur = seg["end_time"] - seg["start_time"]
        coverage[seg["camelot"]] = coverage.get(seg["camelot"], 0) + dur

    dominant_code = max(coverage, key=coverage.get)
    dominant_seg = next(s for s in segments if s["camelot"] == dominant_code)

    # Key changes
    changes = []
    for i in range(1, len(segments)):
        if segments[i]["camelot"] != segments[i-1]["camelot"]:
            changes.append({
                "time": round(segments[i]["start_time"], 2),
                "bar": segments[i]["start_measure"],
                "from_camelot": segments[i-1]["camelot"],
                "from_key": segments[i-1]["key_name"],
                "to_camelot": segments[i]["camelot"],
                "to_key": segments[i]["key_name"],
            })

    return {
        "dominant": {
            "key_name": dominant_seg["key_name"],
            "camelot": dominant_code,
            "coverage": round(coverage[dominant_code] / duration, 3),
        },
        "key_changes": changes,
        "segment_count": len(segments),
        "phrase_count": len(phrases),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_key_analysis(y, sr, beat_times, measures, measure_keys,
                      phrases, segments, beats_per_bar, title="",
                      zoom_start=None, zoom_end=None, phrase_bars=4):
    """
    Three-panel chart:
      1. Waveform with key segment colors + phrase boundaries (measures only when zoomed)
      2. Key Timeline - solid colored blocks per segment with key labels
      3. Confidence per phrase
    """
    duration = len(y) / sr
    times = np.arange(len(y)) / sr

    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.6, 0.5],
                           hspace=0.3, figure=fig)

    # --- Collect unique Camelot codes for legend ---
    all_codes = sorted(set(p["camelot"] for p in phrases))
    code_colors = {}
    cmap = plt.cm.Set3 if len(all_codes) <= 12 else plt.cm.tab20
    for i, code in enumerate(all_codes):
        code_colors[code] = cmap(i / max(len(all_codes), 1))

    # --- Build phrase boundary times (every phrase_bars measures) ---
    phrase_boundaries = set()
    for i in range(0, len(measures), phrase_bars):
        phrase_boundaries.add(measures[i]["start"])
    # Include the very last measure end
    if measures:
        phrase_boundaries.add(measures[-1]["end"])

    # --- Panel 1: Waveform with key segment colors ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888",
             linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.6)

    for seg in segments:
        c = code_colors.get(seg["camelot"], "#cccccc")
        ax1.axvspan(seg["start_time"], seg["end_time"], alpha=0.25, color=c)
        mid = (seg["start_time"] + seg["end_time"]) / 2
        if view_start <= mid <= view_end:
            ax1.text(mid, 0, f'{seg["camelot"]}\n{seg["key_name"]}',
                     fontsize=8, ha="center", va="bottom", color="#222222",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", alpha=0.85))

    if zoomed:
        # Zoomed: show beat ticks and measure boundaries
        for bt in beat_times:
            if view_start <= bt <= view_end:
                ax1.axvline(x=bt, color="#cc4444", linewidth=0.6, alpha=0.4)
        for m in measures:
            if view_start <= m["start"] <= view_end:
                ax1.axvline(x=m["start"], color="#2244aa",
                            linewidth=1.2, alpha=0.6)
    else:
        # Full view: show phrase boundaries only (less noisy)
        for pb in phrase_boundaries:
            if view_start <= pb <= view_end:
                ax1.axvline(x=pb, color="#2244aa", linewidth=0.5, alpha=0.35)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Key Segments ({beats_per_bar}/4)  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    # Legend
    patches = [mpatches.Patch(color=code_colors[c], label=c) for c in all_codes]
    ax1.legend(handles=patches, fontsize=7, loc="upper right", ncol=min(len(all_codes), 6))

    # --- Panel 2: Key Timeline (segment-level colored blocks) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for seg in segments:
        c = code_colors.get(seg["camelot"], "#cccccc")
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]
        seg_dur = seg_end - seg_start

        # Solid colored block for the segment
        ax2.barh(0, seg_dur, left=seg_start,
                 height=0.8, color=c, edgecolor="#333333", linewidth=0.8)

        # Label: Camelot code + key name centered in the block
        mid = (seg_start + seg_end) / 2
        if view_start <= mid <= view_end:
            view_width = view_end - view_start
            if seg_dur / view_width > 0.06:
                # Wide enough for 2-line label
                ax2.text(mid, 0, f'{seg["camelot"]}\n{seg["key_name"]}',
                         fontsize=8, ha="center", va="center",
                         fontweight="bold", color="#111111")
            elif seg_dur / view_width > 0.025:
                # Narrow: just Camelot code
                ax2.text(mid, 0, seg["camelot"],
                         fontsize=7, ha="center", va="center",
                         fontweight="bold", color="#111111")

    ax2.set_yticks([])
    ax2.set_ylabel("Key")
    ax2.set_title("Key Timeline")
    ax2.set_ylim(-0.5, 0.5)

    # --- Panel 3: Confidence ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for p in phrases:
        c = code_colors.get(p["camelot"], "#cccccc")
        ax3.bar(p["start"], p["confidence"], width=p["end"] - p["start"],
                align="edge", color=c, edgecolor="#667799",
                linewidth=0.3, alpha=0.7)
    ax3.set_ylabel("Conf")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("Key Detection Confidence per Phrase")
    ax3.set_ylim(0, 1.1)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Key and Camelot detection using beat-grid-aligned chroma analysis")
    parser.add_argument("audio", help="Path to audio file (WAV/MP3/FLAC)")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (default: 44100)")
    parser.add_argument("--hop-length", type=int, default=512,
                        help="Hop length for beat tracker (default: 512)")
    parser.add_argument("--beats-per-bar", type=int, default=4,
                        help="Beats per bar (default: 4)")
    parser.add_argument("--phrase-bars", type=int, default=4,
                        help="Bars per analysis phrase (default: 4)")
    parser.add_argument("--min-segment-phrases", type=int, default=2,
                        help="Minimum phrases for a key segment (default: 2)")
    parser.add_argument("--full-mix", action="store_true",
                        help="Use full mix instead of harmonic-only")
    parser.add_argument("--zoom-start", type=float, default=None,
                        help="Zoom chart start time in seconds")
    parser.add_argument("--zoom-end", type=float, default=None,
                        help="Zoom chart end time in seconds")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart")
    parser.add_argument("--save-plot", help="Save chart to file instead of showing")
    parser.add_argument("--json-out", help="Write JSON results to file")
    args = parser.parse_args()

    path = Path(args.audio)
    if not path.exists():
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # --- 1. Beat detection (reuse from bpm_detect) ---
    print(f"Loading {path.name} (sr={args.sr}, hop={args.hop_length})...",
          file=sys.stderr)
    data = detect_beats(str(path), sr=args.sr, hop_length=args.hop_length)
    beat_times = data["beat_times"]
    y, sr = data["y"], data["sr"]
    print(f"Detected {len(beat_times)} beats (librosa tempo: {data['tempo']:.1f} BPM)",
          file=sys.stderr)

    if len(beat_times) < 2:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Group beats into measures ---
    measures = group_beats_into_measures(beat_times, args.beats_per_bar)
    print(f"Grouped into {len(measures)} measures", file=sys.stderr)

    # --- 3. Per-measure key analysis ---
    print(f"Analyzing key per measure (harmonic={'off' if args.full_mix else 'on'})...",
          file=sys.stderr)
    measure_keys = compute_measure_chroma(y, sr, measures,
                                          use_harmonic=not args.full_mix)

    # --- 4. Phrase-level key analysis ---
    print(f"Analyzing key per {args.phrase_bars}-bar phrase...", file=sys.stderr)
    phrases = compute_phrase_chroma(y, sr, measures, phrase_bars=args.phrase_bars,
                                    use_harmonic=not args.full_mix)

    # --- 5. Key segmentation ---
    segments = find_key_segments(phrases,
                                 min_segment_phrases=args.min_segment_phrases)

    # --- 6. Summary ---
    duration = len(y) / sr
    summary = compute_summary(segments, phrases, duration)

    # --- Print results ---
    print()
    print(f"=== Key Analysis: {path.name} ===")
    print()

    if summary.get("dominant"):
        d = summary["dominant"]
        print(f"Dominant Key: {d['key_name']}  ({d['camelot']})  "
              f"coverage: {d['coverage']*100:.1f}%")
    print(f"Tempo: {data['tempo']:.1f} BPM (librosa)")
    print(f"Phrases: {len(phrases)} ({args.phrase_bars}-bar),  "
          f"Segments: {len(segments)}")
    print()

    # Key changes
    if summary.get("key_changes"):
        print(f"Key Changes:")
        for kc in summary["key_changes"]:
            print(f"  Bar {kc['bar']:>4}  ({format_time(kc['time'])})  "
                  f"{kc['from_camelot']} ({kc['from_key']}) -> "
                  f"{kc['to_camelot']} ({kc['to_key']})")
        print()
    else:
        print(f"No key changes detected.")
        print()

    # Segment detail
    print(f"Key Segments:")
    print(f"  {'#':<3} {'Bars':<12} {'Time':<14} {'Key':<12} {'Camelot':<8} "
          f"{'Conf':>5}  {'Phrases':>7}")
    print(f"  {'-'*68}")
    for i, seg in enumerate(segments, 1):
        t = f"{format_time(seg['start_time'])}-{format_time(seg['end_time'])}"
        b = f"{seg['start_measure']}-{seg['end_measure']}"
        print(f"  {i:<3} {b:<12} {t:<14} {seg['key_name']:<12} "
              f"{seg['camelot']:<8} {seg['confidence_mean']:>5.2f}  "
              f"{seg['phrase_count']:>7}")
    print()

    # Per-phrase compact table
    print(f"Per-Phrase Key ({len(phrases)} phrases, {args.phrase_bars}-bar):")
    row_size = 8
    for row_start in range(0, min(len(phrases), 48), row_size):
        row = phrases[row_start:row_start + row_size]
        bars = "  ".join(f"{p['start_measure']:>4}" for p in row)
        keys = "  ".join(f"{p['camelot']:>4}" for p in row)
        conf = "  ".join(f"{p['confidence']:>4.2f}" for p in row)
        print(f"  Bar:  {bars}")
        print(f"  Key:  {keys}")
        print(f"  Conf: {conf}")
        print()
    if len(phrases) > 48:
        print(f"  ... ({len(phrases) - 48} more phrases)")
        print()

    # --- JSON output ---
    if args.json_out:
        result = {
            "file": path.name,
            "duration": round(duration, 2),
            "bpm": round(data["tempo"], 2),
            "sr": sr,
            "beats_per_bar": args.beats_per_bar,
            "phrase_bars": args.phrase_bars,
            "summary": summary,
            "segments": segments,
            "phrases": phrases,
        }
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"JSON written to {args.json_out}")

    # --- Plot ---
    if not args.no_plot:
        fig = plot_key_analysis(
            y, sr, beat_times, measures, measure_keys,
            phrases, segments, args.beats_per_bar,
            title=f"- {path.name}",
            zoom_start=args.zoom_start, zoom_end=args.zoom_end,
            phrase_bars=args.phrase_bars
        )
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {args.save_plot}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
