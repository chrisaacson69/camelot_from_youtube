#!/usr/bin/env python3
"""
bpm_detect.py - Precise BPM detection from audio.

Uses librosa's beat_track() dynamic programming beat tracker, then analyzes
beat placement at multiple resolutions to separate real tempo changes from
frame quantization artifacts.

Key insight: beat_track() places beats at frame boundaries (hop_length/sr
seconds per frame). At sr=44100, hop=512 this is ~11.6ms resolution.
Adjacent intervals can only differ by multiples of that step, so single-beat
BPM values are quantized to a handful of discrete values. We compute BPM
over multi-beat spans to average out this quantization and reveal the true
tempo, while preserving measure-level granularity for structural analysis.

Defaults: sr=44100, hop_length=512 (11.6ms frame resolution).
Higher sr with the same hop gives finer frame resolution without destabilizing
the beat tracker (smaller hops actually make beat_track() less stable).

Usage:
    python bpm_detect.py "audio.wav"
    python bpm_detect.py "audio.wav" --beats-per-bar 3
    python bpm_detect.py "audio.wav" --save-plot bpm_chart.png
    python bpm_detect.py "audio.wav" --no-plot
    python bpm_detect.py "audio.wav" --sr 22050   # lower resolution, faster
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Downbeat (beat-1) phase detection
# ---------------------------------------------------------------------------

def estimate_downbeat_phase(y, sr, beat_times, beats_per_bar=4):
    """Score each phase offset (0 to beats_per_bar-1) to find beat 1.

    Uses four audio cues (ported from camelot_from_youtube.py):
      1. Onset strength — downbeats typically have louder attacks
      2. Bass energy — kicks often hit on beat 1
      3. Harmonic change — chord changes at bar boundaries
      4. Interval consistency — uniform beat spacing when aligned

    Returns dict:
        phase: int — beat index offset that is beat 1 (0 = already aligned)
        confidence: float (0-1) — margin between best and second-best
        scores: list — combined score per offset
    """
    if len(beat_times) < beats_per_bar * 2:
        return {"phase": 0, "confidence": 0.0, "scores": []}

    beat_times = np.asarray(beat_times, dtype=float)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    # --- 1. Onset strength ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames_clipped = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_strengths = onset_env[beat_frames_clipped]
    onset_scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        idxs = list(range(offset, len(beat_strengths), beats_per_bar))
        if idxs:
            onset_scores[offset] = np.sum(beat_strengths[idxs])

    # --- 2. Bass energy (<150 Hz) ---
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_energy = np.sum(S[freqs < 150, :], axis=0)
    beat_frames_bass = np.clip(beat_frames, 0, len(bass_energy) - 1)
    beat_bass = bass_energy[beat_frames_bass]
    bass_scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        idxs = list(range(offset, len(beat_bass), beats_per_bar))
        if idxs:
            bass_scores[offset] = np.sum(beat_bass[idxs])

    # --- 3. Harmonic change (chroma flux) ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_flux = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
    beat_frames_chr = np.clip(beat_frames, 0, len(chroma_flux) - 1)
    beat_flux = chroma_flux[beat_frames_chr]
    harmony_scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        idxs = list(range(offset, len(beat_flux), beats_per_bar))
        if idxs:
            harmony_scores[offset] = np.sum(beat_flux[idxs])

    # --- 4. Interval consistency ---
    beat_intervals = np.diff(beat_times)
    interval_scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        bar_consistencies = []
        for bar_start in range(offset, len(beat_times) - beats_per_bar,
                               beats_per_bar):
            if bar_start + beats_per_bar - 1 < len(beat_intervals):
                bar_ivl = beat_intervals[bar_start:bar_start + beats_per_bar]
                if len(bar_ivl) == beats_per_bar:
                    bar_consistencies.append(1.0 / (1.0 + np.std(bar_ivl)))
        if bar_consistencies:
            interval_scores[offset] = np.mean(bar_consistencies)

    # --- Normalize each to 0-1 ---
    def _norm(scores):
        mn, mx = scores.min(), scores.max()
        if mx - mn > 1e-12:
            return (scores - mn) / (mx - mn)
        return np.ones_like(scores) * 0.25

    combined = (0.30 * _norm(onset_scores)
                + 0.30 * _norm(bass_scores)
                + 0.20 * _norm(harmony_scores)
                + 0.20 * _norm(interval_scores))

    best = int(np.argmax(combined))
    sorted_scores = np.sort(combined)[::-1]
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    else:
        confidence = 1.0

    return {
        "phase": best,
        "confidence": round(float(confidence), 4),
        "scores": [round(float(s), 4) for s in combined],
    }


# ---------------------------------------------------------------------------
# Beat grid origin trimming
# ---------------------------------------------------------------------------

def trim_silent_beats(y, sr, beat_times, hop_length=512, beats_per_bar=4,
                      energy_ratio=0.1):
    """Remove leading beats that fall before real musical content.

    The librosa beat tracker can lock onto faint transients during an intro
    (synth pads, atmospheric textures, etc.), producing a beat grid that
    starts one or more bars before the actual rhythmic content begins.

    Uses bass energy (<150 Hz) as the discriminator rather than overall
    onset strength, because intros often have harmonic content (pads, synths)
    with real onset energy but no bass/kick presence.

    Strategy:
      1. Compute bass energy (< 150 Hz) at each beat position via STFT.
      2. Compute median bass energy across ALL beats as a baseline.
      3. Walk forward from beat 0: while a beat's bass energy is below
         ``energy_ratio * median``, mark it for removal.
      4. Stop as soon as we hit a beat with real bass energy.
      5. Round down to the nearest full bar boundary so the grid
         stays aligned to complete measures.

    Args:
        y: Audio signal.
        sr: Sample rate.
        beat_times: Array of beat times from beat_track().
        hop_length: Hop length used for beat detection.
        beats_per_bar: Beats per measure (for bar-aligned trimming).
        energy_ratio: A beat is "low bass" if its bass energy is below
            this fraction of the median bass energy across all beats.

    Returns:
        dict: trimmed_beat_times, beats_trimmed (int), bars_trimmed (int)
    """
    if len(beat_times) < beats_per_bar * 2:
        return {"trimmed_beat_times": beat_times,
                "beats_trimmed": 0, "bars_trimmed": 0}

    beat_times = np.asarray(beat_times, dtype=float)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr,
                                         hop_length=hop_length)

    # Bass energy (< 150 Hz) via STFT
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_energy = np.sum(S[freqs < 150, :], axis=0)

    beat_frames_clipped = np.clip(beat_frames, 0, len(bass_energy) - 1)
    beat_bass = bass_energy[beat_frames_clipped]

    # Median bass energy across all beats
    median_bass = float(np.median(beat_bass))
    if median_bass < 1e-6:
        # No bass anywhere — nothing to trim
        return {"trimmed_beat_times": beat_times,
                "beats_trimmed": 0, "bars_trimmed": 0}

    threshold = energy_ratio * median_bass

    # Walk forward: count consecutive low-bass beats from the start
    low_bass_count = 0
    for i in range(len(beat_bass)):
        if beat_bass[i] < threshold:
            low_bass_count += 1
        else:
            break

    if low_bass_count == 0:
        return {"trimmed_beat_times": beat_times,
                "beats_trimmed": 0, "bars_trimmed": 0}

    # Round down to full bars so we don't create orphan beats
    bars_to_trim = low_bass_count // beats_per_bar
    if bars_to_trim == 0:
        return {"trimmed_beat_times": beat_times,
                "beats_trimmed": 0, "bars_trimmed": 0}

    beats_to_trim = bars_to_trim * beats_per_bar
    trimmed = beat_times[beats_to_trim:]

    return {"trimmed_beat_times": trimmed,
            "beats_trimmed": int(beats_to_trim),
            "bars_trimmed": int(bars_to_trim)}


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

def detect_beats(path: str, sr: int = 44100, hop_length: int = 512) -> dict:
    """Load audio and run librosa beat_track(), then detect downbeat phase.

    Args:
        path: Audio file path.
        sr: Sample rate. 44100 gives finer frame resolution than 22050
            without destabilizing the beat tracker.
        hop_length: Hop length for beat_track(). 512 is the sweet spot;
            smaller values (256, 128) actually reduce beat tracker stability.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )

    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else float(tempo)
    else:
        tempo = float(tempo)

    frame_duration = hop_length / sr

    # Trim silent intro beats (ghost beats before real musical content)
    trim = trim_silent_beats(y, sr, beat_times, hop_length=hop_length)
    beat_times = trim["trimmed_beat_times"]
    # Update beat_frames to match trimmed beat_times
    beat_frames = librosa.time_to_frames(beat_times, sr=sr,
                                         hop_length=hop_length)

    # Downbeat detection (on trimmed grid)
    db = estimate_downbeat_phase(y, sr, beat_times)

    return {
        "y": y,
        "sr": sr,
        "beat_times": beat_times,
        "beat_frames": beat_frames,
        "tempo": tempo,
        "hop_length": hop_length,
        "frame_duration": frame_duration,
        "beats_trimmed": trim["beats_trimmed"],
        "bars_trimmed": trim["bars_trimmed"],
        "downbeat_phase": db["phase"],
        "downbeat_confidence": db["confidence"],
        "downbeat_scores": db["scores"],
    }


# ---------------------------------------------------------------------------
# Interval and BPM computation
# ---------------------------------------------------------------------------

def compute_beat_intervals(beat_times: np.ndarray) -> dict:
    """Compute per-beat intervals and instantaneous BPM."""
    intervals = np.diff(beat_times)
    bpms = 60.0 / intervals
    return {"intervals": intervals, "bpms": bpms}


def compute_span_bpm(beat_times: np.ndarray, span: int) -> dict:
    """
    Compute BPM over N-beat spans to reduce frame quantization noise.

    For a span of N beats, BPM = 60 * N / (beat_times[i+N] - beat_times[i]).
    Each value is centered at the midpoint of the span.

    Larger spans = less noise, but less sensitivity to real tempo changes.
    """
    if span >= len(beat_times):
        return {"times": np.array([]), "bpms": np.array([])}

    span_durations = beat_times[span:] - beat_times[:-span]
    span_bpms = 60.0 * span / span_durations
    # Center each measurement at the midpoint of the span
    midpoints = (beat_times[span:] + beat_times[:-span]) / 2

    return {
        "times": midpoints,
        "bpms": span_bpms,
        "span": span,
        "median": float(np.median(span_bpms)),
        "std": float(np.std(span_bpms)),
    }


# ---------------------------------------------------------------------------
# Measure grouping
# ---------------------------------------------------------------------------

def group_beats_into_measures(beat_times: np.ndarray, beats_per_bar: int = 4,
                              phase: int = 0) -> list:
    """
    Group beats into measures, optionally starting from a downbeat phase.

    Args:
        beat_times: Array of beat positions in seconds (physical time).
        beats_per_bar: Beats per measure (e.g. 4 for 4/4 time).
        phase: Beat index of the first downbeat (0 = first beat is beat 1).
            If phase > 0, beats 0..phase-1 form a partial pickup measure
            (measure_num=0) and full bars start from beat index ``phase``
            with measure_num=1.

    BPM per measure is computed from the total span (first beat to first beat
    of next measure) rather than median of internal intervals, to avoid
    quantization bias.
    """
    measures = []
    total_beats = len(beat_times)

    # Pickup measure (partial bar before the first downbeat)
    if phase > 0 and phase < total_beats:
        pickup_beats = beat_times[0:phase]
        if len(pickup_beats) >= 2:
            next_time = float(beat_times[phase])
            span_time = next_time - float(pickup_beats[0])
            beat_count = len(pickup_beats)
            bpm = 60.0 * beat_count / span_time if span_time > 0 else 0.0
            measures.append({
                "measure_num": 0,
                "beat_start_idx": 0,
                "beat_count": beat_count,
                "start": float(pickup_beats[0]),
                "end": next_time,
                "span_time": float(span_time),
                "bpm": round(bpm, 2),
            })

    # Full measures starting from the downbeat
    start = phase if phase > 0 else 0
    bar_num = 1

    for i in range(start, total_beats, beats_per_bar):
        end_idx = min(i + beats_per_bar, total_beats)
        measure_beats = beat_times[i:end_idx]

        if len(measure_beats) < 2:
            continue

        # BPM from span: time from first beat to first beat of next measure
        if end_idx < total_beats:
            span_time = beat_times[end_idx] - measure_beats[0]
            end_time = float(beat_times[end_idx])
        else:
            # Last (possibly partial) measure: use internal intervals
            span_time = measure_beats[-1] - measure_beats[0]
            end_time = float(measure_beats[-1] + np.median(np.diff(measure_beats)))

        beat_count = end_idx - i
        bpm = 60.0 * beat_count / span_time if span_time > 0 else 0.0

        measures.append({
            "measure_num": bar_num,
            "beat_start_idx": i,
            "beat_count": beat_count,
            "start": float(measure_beats[0]),
            "end": end_time,
            "span_time": float(span_time),
            "bpm": round(bpm, 2),
        })
        bar_num += 1

    return measures


# ---------------------------------------------------------------------------
# Segment detection (BPM change regions)
# ---------------------------------------------------------------------------

def find_bpm_segments(measures: list, tolerance: float = 2.0,
                      min_segment_bars: int = 4) -> list:
    """
    Group consecutive measures with similar BPM into segments.

    Two-pass approach:
      1. Group measures by BPM similarity (tolerance-based).
      2. Absorb short segments (< min_segment_bars) into their neighbors.
         Short segments are almost always frame quantization artifacts —
         a single measure where beats land on unlucky frame boundaries.
         Real tempo changes persist across multiple measures.

    Args:
        measures: List of measure dicts from group_beats_into_measures().
        tolerance: Max BPM difference to consider "same tempo" (default: 2.0).
        min_segment_bars: Minimum measures for a segment to stand on its own.
            Shorter segments get absorbed into the nearest neighbor by BPM.
    """
    if not measures:
        return []

    # --- Pass 1: initial grouping by tolerance ---
    raw_segments = []
    current = [measures[0]]

    for m in measures[1:]:
        seg_bpms = [sm["bpm"] for sm in current]
        seg_median = float(np.median(seg_bpms))

        if abs(m["bpm"] - seg_median) <= tolerance:
            current.append(m)
        else:
            raw_segments.append(current)
            current = [m]
    raw_segments.append(current)

    # --- Pass 2: absorb short segments into neighbors ---
    if len(raw_segments) <= 1:
        segments = []
        for group in raw_segments:
            _finalize_segment(segments, group)
        return segments

    merged = _absorb_short_segments(raw_segments, min_segment_bars)

    # --- Pass 3: merge adjacent segments whose medians are within tolerance ---
    # After absorbing short segments, adjacent segments may now have similar
    # BPMs that only split because of running-median drift during pass 1.
    final = _merge_similar_adjacent(merged, tolerance)

    segments = []
    for group in final:
        _finalize_segment(segments, group)
    return segments


def _absorb_short_segments(raw_segments: list, min_bars: int) -> list:
    """Absorb segments shorter than min_bars into their best neighbor.

    Strategy: iterate until no more short segments can be absorbed.
    A short segment is merged into whichever adjacent segment has the
    closest median BPM — this preserves real transitions while cleaning
    up quantization blips.
    """
    groups = list(raw_segments)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            if len(groups[i]) < min_bars and len(groups) > 1:
                # Find best neighbor to absorb into
                best_idx = None
                best_diff = float('inf')
                short_median = float(np.median([m["bpm"] for m in groups[i]]))

                for neighbor_idx in [i - 1, i + 1]:
                    if 0 <= neighbor_idx < len(groups):
                        n_median = float(np.median(
                            [m["bpm"] for m in groups[neighbor_idx]]))
                        diff = abs(short_median - n_median)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = neighbor_idx

                if best_idx is not None:
                    if best_idx < i:
                        groups[best_idx].extend(groups[i])
                        groups.pop(i)
                    else:
                        groups[best_idx] = groups[i] + groups[best_idx]
                        groups.pop(i)
                    changed = True
                    continue  # re-check from same index
            i += 1

    return groups


def _merge_similar_adjacent(groups: list, tolerance: float) -> list:
    """Merge adjacent groups whose median BPMs are within tolerance.

    This catches cases where pass 1 over-split due to running-median drift:
    e.g., a long 133.4 segment followed by a 132.5 segment that are really
    the same tempo, just quantized differently.
    """
    if len(groups) <= 1:
        return groups

    merged = [groups[0]]
    for group in groups[1:]:
        prev_median = float(np.median([m["bpm"] for m in merged[-1]]))
        curr_median = float(np.median([m["bpm"] for m in group]))
        if abs(prev_median - curr_median) <= tolerance:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def _finalize_segment(segments: list, measures: list):
    bpms = [m["bpm"] for m in measures]
    segments.append({
        "start_measure": measures[0]["measure_num"],
        "end_measure": measures[-1]["measure_num"],
        "start_time": measures[0]["start"],
        "end_time": measures[-1]["end"],
        "bpm": round(float(np.median(bpms)), 2),
        "bpm_std": round(float(np.std(bpms)), 3),
        "measure_count": len(measures),
    })


# ---------------------------------------------------------------------------
# Frame quantization diagnostics
# ---------------------------------------------------------------------------

def quantization_report(intervals: np.ndarray, frame_duration: float) -> dict:
    """
    Analyze how frame quantization affects beat intervals.
    Reports how many discrete interval values exist and the quantization step.
    """
    unique_intervals = np.unique(np.round(intervals, 6))
    diffs = np.diff(unique_intervals)

    return {
        "frame_duration_ms": round(frame_duration * 1000, 2),
        "unique_interval_count": len(unique_intervals),
        "unique_intervals": unique_intervals,
        "interval_step_ms": round(float(np.median(diffs)) * 1000, 2) if len(diffs) > 0 else 0,
        "matches_frame_hop": len(diffs) > 0 and abs(float(np.median(diffs)) - frame_duration) < 0.001,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def plot_bpm_analysis(y, sr, beat_times, measures, segments, span_data,
                      beats_per_bar, title="", zoom_start=None, zoom_end=None):
    """
    Multi-panel chart:
      1. Waveform with beat markers and measure boundaries
      2. BPM: per-measure bars + smoothed span curve + segment regions

    zoom_start/zoom_end: optional time range in seconds to zoom into.
    """
    duration = len(y) / sr
    times = np.arange(len(y)) / sr

    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)

    # --- Panel 1: Waveform ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888",
             linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.6)

    # Segment background colors
    seg_colors = plt.cm.Pastel1(np.linspace(0, 0.8, max(len(segments), 1)))
    for i, seg in enumerate(segments):
        ax1.axvspan(seg["start_time"], seg["end_time"],
                     alpha=0.2, color=seg_colors[i % len(seg_colors)])
        mid = (seg["start_time"] + seg["end_time"]) / 2
        if view_start <= mid <= view_end:
            ax1.text(mid, 0, f'{seg["bpm"]:.1f}',
                     fontsize=8, ha="center", va="bottom", color="#333333",
                     fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                     facecolor="white", alpha=0.8))

    # Beat ticks — thicker when zoomed so individual beats are visible
    for bt in beat_times:
        if view_start <= bt <= view_end:
            ax1.axvline(x=bt, color="#cc4444",
                        linewidth=0.8 if zoomed else 0.2,
                        alpha=0.5 if zoomed else 0.3)

    # Measure boundaries — labeled with bar number when zoomed
    for m in measures:
        if view_start <= m["start"] <= view_end:
            ax1.axvline(x=m["start"], color="#2244aa",
                        linewidth=1.5 if zoomed else 0.6,
                        alpha=0.7 if zoomed else 0.5)
            if zoomed:
                ax1.text(m["start"], 0.98, f'{m["measure_num"]}',
                         fontsize=7, ha="center", va="top", color="#2244aa",
                         fontweight="bold", transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Beat Grid ({beats_per_bar}/4)  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    # --- Panel 2: BPM ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Segment regions
    for i, seg in enumerate(segments):
        ax2.axvspan(seg["start_time"], seg["end_time"],
                     alpha=0.15, color=seg_colors[i % len(seg_colors)])

    # Per-measure BPM as thin bars
    for m in measures:
        ax2.bar(m["start"], m["bpm"], width=m["end"] - m["start"],
                align="edge", color="#aabbdd", edgecolor="#667799",
                linewidth=0.3, alpha=0.6)

    # Smoothed span BPM curve
    if span_data and len(span_data["times"]) > 0:
        ax2.plot(span_data["times"], span_data["bpms"],
                 color="#cc3333", linewidth=1.5, alpha=0.9,
                 label=f'{span_data["span"]}-beat span (median: {span_data["median"]:.1f})')

    # Segment median lines
    for seg in segments:
        ax2.hlines(seg["bpm"], seg["start_time"], seg["end_time"],
                    colors="#224488", linewidth=2, alpha=0.7)

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("BPM")
    ax2.set_title("BPM per Measure (bars) with Smoothed Span Curve")
    ax2.legend(fontsize=9, loc="upper right")

    # Y range
    all_bpms = [m["bpm"] for m in measures]
    if all_bpms:
        bpm_min, bpm_max = min(all_bpms), max(all_bpms)
        margin = max(2, (bpm_max - bpm_min) * 0.3)
        ax2.set_ylim(bpm_min - margin, bpm_max + margin)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BPM detection with per-beat precision and frame quantization awareness")
    parser.add_argument("audio", help="Path to audio file (WAV/MP3/FLAC)")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (default: 44100 for 11.6ms frame resolution)")
    parser.add_argument("--hop-length", type=int, default=512,
                        help="Hop length for beat tracker (default: 512)")
    parser.add_argument("--beats-per-bar", type=int, default=4,
                        help="Beats per bar (default: 4)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="BPM tolerance for segment grouping (default: 2.0)")
    parser.add_argument("--min-segment-bars", type=int, default=4,
                        help="Minimum bars for a segment to stand alone (default: 4)")
    parser.add_argument("--span", type=int, default=8,
                        help="Beat span for smoothed BPM curve (default: 8)")
    parser.add_argument("--zoom-start", type=float, default=None,
                        help="Zoom chart start time in seconds")
    parser.add_argument("--zoom-end", type=float, default=None,
                        help="Zoom chart end time in seconds")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart")
    parser.add_argument("--save-plot", help="Save chart to file instead of showing")
    args = parser.parse_args()

    path = Path(args.audio)
    if not path.exists():
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # --- 1. Detect beats ---
    print(f"Loading {path.name} (sr={args.sr}, hop={args.hop_length})...",
          file=sys.stderr)
    data = detect_beats(str(path), sr=args.sr, hop_length=args.hop_length)
    beat_times = data["beat_times"]
    print(f"Detected {len(beat_times)} beats (librosa tempo: {data['tempo']:.1f} BPM)",
          file=sys.stderr)
    if data["bars_trimmed"] > 0:
        print(f"Trimmed {data['bars_trimmed']} silent intro bar(s) "
              f"({data['beats_trimmed']} beats) from beat grid",
              file=sys.stderr)

    if len(beat_times) < 2:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Per-beat intervals ---
    beat_data = compute_beat_intervals(beat_times)

    # --- 3. Frame quantization diagnostic ---
    quant = quantization_report(beat_data["intervals"], data["frame_duration"])

    # --- 4. Span-based BPM (smoothed) ---
    span_data = compute_span_bpm(beat_times, args.span)

    # --- 5. Downbeat phase + group into measures ---
    phase = data["downbeat_phase"]
    phase_conf = data["downbeat_confidence"]
    print(f"Downbeat phase: {phase} (confidence: {phase_conf:.3f})",
          file=sys.stderr)
    if data["downbeat_scores"]:
        print(f"  Phase scores: {data['downbeat_scores']}", file=sys.stderr)
    measures = group_beats_into_measures(beat_times, args.beats_per_bar,
                                          phase=phase)

    # --- 6. Find segments ---
    segments = find_bpm_segments(measures, tolerance=args.tolerance,
                                  min_segment_bars=args.min_segment_bars)

    # --- Print results ---
    print()
    print(f"=== BPM Analysis: {path.name} ===")
    print()

    # Frame quantization info
    print(f"Frame Quantization:")
    print(f"  Hop frame duration: {quant['frame_duration_ms']:.2f}ms")
    print(f"  Unique beat intervals: {quant['unique_interval_count']} "
          f"(step: {quant['interval_step_ms']:.2f}ms)")
    print(f"  Matches hop size: {quant['matches_frame_hop']}")
    print()

    # Multi-resolution BPM
    print(f"BPM at different resolutions:")
    for s in [1, 2, 4, 8, 16]:
        sd = compute_span_bpm(beat_times, s)
        if len(sd["bpms"]) > 0:
            print(f"  {s:2d}-beat span: {sd['median']:7.2f} BPM  (std: {sd['std']:.3f})")
    print()

    # Segment summary
    if len(segments) <= 10:
        print(f"BPM Segments ({len(segments)}, tolerance={args.tolerance} BPM):")
        print(f"  {'#':<4} {'Bars':<12} {'Time':<15} {'BPM':>7}  {'Measures':>8}")
        print(f"  {'-'*55}")
        for i, seg in enumerate(segments, 1):
            t = f"{format_time(seg['start_time'])}-{format_time(seg['end_time'])}"
            b = f"{seg['start_measure']}-{seg['end_measure']}"
            print(f"  {i:<4} {b:<12} {t:<15} {seg['bpm']:>7.1f}  {seg['measure_count']:>8}")
    else:
        print(f"BPM Segments ({len(segments)}, tolerance={args.tolerance} BPM):")
        print(f"  Many segments detected - likely frame quantization noise.")
        print(f"  Try increasing --tolerance (current: {args.tolerance})")
        print(f"  Showing first 10:")
        print(f"  {'#':<4} {'Bars':<12} {'Time':<15} {'BPM':>7}  {'Measures':>8}")
        print(f"  {'-'*55}")
        for i, seg in enumerate(segments[:10], 1):
            t = f"{format_time(seg['start_time'])}-{format_time(seg['end_time'])}"
            b = f"{seg['start_measure']}-{seg['end_measure']}"
            print(f"  {i:<4} {b:<12} {t:<15} {seg['bpm']:>7.1f}  {seg['measure_count']:>8}")
        print(f"  ... ({len(segments) - 10} more)")
    print()

    # Per-measure BPM compact table
    print(f"Per-Measure BPM ({len(measures)} measures, {args.beats_per_bar}/4 time):")
    row_size = 8
    for row_start in range(0, min(len(measures), 64), row_size):
        row = measures[row_start:row_start + row_size]
        nums = "  ".join(f"{m['measure_num']:>3}" for m in row)
        bpms = "  ".join(f"{m['bpm']:>6.1f}" for m in row)
        print(f"  Bar: {nums}")
        print(f"  BPM: {bpms}")
        print()
    if len(measures) > 64:
        print(f"  ... ({len(measures) - 64} more measures)")
        print()

    # --- Plot ---
    if not args.no_plot:
        fig = plot_bpm_analysis(
            data["y"], data["sr"], beat_times, measures, segments,
            span_data, args.beats_per_bar, title=f"- {path.name}",
            zoom_start=args.zoom_start, zoom_end=args.zoom_end
        )
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {args.save_plot}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
