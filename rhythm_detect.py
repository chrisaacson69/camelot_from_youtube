"""rhythm_detect.py - Per-bar rhythmic pattern classification.

Classifies each bar's kick pattern as one of:
  4/4, 8th-note, half-time, no-kick, complex

Uses bandpass filtering (30-120 Hz) to isolate kick drum, then onset
detection to measure inter-onset intervals relative to beat positions.

Usage (standalone):
    python rhythm_detect.py "track.wav" --bpm 122 --first-downbeat 0.174
"""

import numpy as np
from scipy.signal import butter, sosfilt
import librosa


def kick_envelope(audio, sr=44100, low_hz=30, high_hz=120, smooth_ms=3.0):
    """Bandpass filter + rectify + smooth -> kick onset envelope.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    low_hz, high_hz : float
        Bandpass range for kick detection.
    smooth_ms : float
        Smoothing window in milliseconds.

    Returns
    -------
    np.ndarray
        Kick onset envelope (same length as audio).
    """
    sos = butter(4, [low_hz, high_hz], btype='band', fs=sr, output='sos')
    filtered = sosfilt(sos, audio)
    env = np.abs(filtered)
    win = max(int(smooth_ms / 1000 * sr), 1)
    return np.convolve(env, np.ones(win) / win, mode='same')


def classify_bar_rhythm(audio_segment, sr, beat_interval, beats_per_bar=4,
                        onset_threshold=None):
    """Classify the rhythmic pattern of a single bar.

    Parameters
    ----------
    audio_segment : np.ndarray
        Audio samples for one bar.
    sr : int
        Sample rate.
    beat_interval : float
        Duration of one beat in seconds.
    beats_per_bar : int
        Beats per bar (default 4).
    onset_threshold : float or None
        Onset detection threshold. None = auto.

    Returns
    -------
    dict with keys:
        pattern : str
            One of "4/4", "8th-note", "half-time", "no-kick", "complex"
        kick_density : float
            Onsets per beat (0 = silent, 1 = quarter-note, 2 = eighth-note)
        onsets : int
            Raw onset count in this bar
        confidence : float
            Classification confidence (0-1)
    """
    if len(audio_segment) < sr * 0.3:
        return {"pattern": "no-kick", "kick_density": 0.0, "onsets": 0,
                "confidence": 0.5}

    # Get kick envelope with heavier smoothing to eliminate resonance peaks
    # Kick resonance oscillates at 50-100ms; smooth with 30ms window
    env = kick_envelope(audio_segment, sr, smooth_ms=30.0)

    # Check if there's meaningful kick energy
    env_peak = np.max(env)
    env_rms = np.sqrt(np.mean(env ** 2))

    if env_peak < 1e-5 or env_rms < 1e-6:
        return {"pattern": "no-kick", "kick_density": 0.0, "onsets": 0,
                "confidence": 0.9}

    # Detect onsets on the kick envelope using peak picking
    bar_duration = len(audio_segment) / sr

    # Onset detection via peak-picking on the envelope
    # Higher threshold (40%) to avoid resonance secondary peaks
    if onset_threshold is None:
        onset_threshold = env_peak * 0.4

    # Find peaks: local maxima above threshold with minimum spacing
    # Min spacing = 60% of an 8th-note interval (smallest expected gap)
    eighth_note_samples = int(beat_interval * 0.5 * sr)
    min_spacing_samples = max(int(eighth_note_samples * 0.6), int(beat_interval * 0.3 * sr))
    onset_positions = []

    i = 0
    while i < len(env):
        if env[i] >= onset_threshold:
            # Find the peak in this region
            peak_start = i
            while i < len(env) and env[i] >= onset_threshold * 0.5:
                i += 1
            peak_end = i
            peak_idx = peak_start + np.argmax(env[peak_start:peak_end])
            onset_positions.append(peak_idx)
            i = peak_idx + min_spacing_samples
        else:
            i += 1

    n_onsets = len(onset_positions)
    kick_density = n_onsets / beats_per_bar if beats_per_bar > 0 else 0

    # No onsets detected
    if n_onsets == 0:
        return {"pattern": "no-kick", "kick_density": 0.0, "onsets": 0,
                "confidence": 0.7}

    # Classify based on onset count and regularity
    # Expected counts: half-time=2, 4/4=4, 8th-note=8
    eighth_note_interval = beat_interval / 2
    quarter_note_interval = beat_interval
    half_note_interval = beat_interval * 2

    # Compute inter-onset intervals
    if n_onsets >= 2:
        onset_times = np.array(onset_positions) / sr
        iois = np.diff(onset_times)
        median_ioi = float(np.median(iois))
        ioi_std = float(np.std(iois))
        regularity = 1.0 - min(ioi_std / median_ioi, 1.0) if median_ioi > 0 else 0.0
    else:
        median_ioi = bar_duration
        regularity = 0.5

    # Classification logic
    pattern = "complex"
    confidence = 0.5

    if n_onsets <= 1:
        pattern = "half-time"
        confidence = 0.6

    elif 1.5 <= n_onsets <= 2.5 or abs(n_onsets - 2) <= 0.5:
        # ~2 onsets: half-time
        if regularity > 0.5:
            pattern = "half-time"
            confidence = min(0.5 + regularity * 0.4, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif 3 <= n_onsets <= 5:
        # ~4 onsets: 4/4
        ioi_ratio = median_ioi / quarter_note_interval if quarter_note_interval > 0 else 0
        if 0.6 < ioi_ratio < 1.5 and regularity > 0.25:
            pattern = "4/4"
            confidence = min(0.5 + regularity * 0.3 +
                           (1.0 - abs(ioi_ratio - 1.0)) * 0.2, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif 6 <= n_onsets <= 10:
        # ~8 onsets: 8th-note
        ioi_ratio = median_ioi / eighth_note_interval if eighth_note_interval > 0 else 0
        if 0.6 < ioi_ratio < 1.5 and regularity > 0.2:
            pattern = "8th-note"
            confidence = min(0.5 + regularity * 0.3 +
                           (1.0 - abs(ioi_ratio - 1.0)) * 0.2, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif n_onsets > 10:
        # Very dense — likely 16th-note or noise
        pattern = "complex"
        confidence = 0.3

    return {
        "pattern": pattern,
        "kick_density": round(kick_density, 2),
        "onsets": n_onsets,
        "confidence": round(confidence, 3),
    }


def classify_rhythm(y, sr, measures, beat_interval, beats_per_bar=4):
    """Classify kick pattern for all bars in the track.

    Parameters
    ----------
    y : np.ndarray
        Full mono audio signal.
    sr : int
        Sample rate.
    measures : list[dict]
        Each dict has "start" and "end" in seconds, "measure_num".
    beat_interval : float
        Median beat interval in seconds.
    beats_per_bar : int
        Beats per bar (default 4).

    Returns
    -------
    list[dict]
        One classification dict per measure, with added "bar" field.
    """
    results = []
    for m in measures:
        start_sample = int(m["start"] * sr)
        end_sample = int(m["end"] * sr)

        # Bounds check
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        if end_sample <= start_sample:
            results.append({
                "bar": m["measure_num"],
                "pattern": "no-kick",
                "kick_density": 0.0,
                "onsets": 0,
                "confidence": 0.5,
            })
            continue

        segment = y[start_sample:end_sample]
        classification = classify_bar_rhythm(
            segment, sr, beat_interval, beats_per_bar
        )
        classification["bar"] = m["measure_num"]
        results.append(classification)

    return results


def summarize_rhythm(bar_rhythms):
    """Compute summary statistics and identify pattern changes.

    Parameters
    ----------
    bar_rhythms : list[dict]
        Output of classify_rhythm().

    Returns
    -------
    dict with keys:
        dominant_pattern : str
        pattern_coverage : dict mapping pattern -> fraction
        pattern_changes : list of dicts with bar, from_pattern, to_pattern
        total_bars : int
    """
    if not bar_rhythms:
        return {
            "dominant_pattern": "unknown",
            "pattern_coverage": {},
            "pattern_changes": [],
            "total_bars": 0,
        }

    # Count patterns
    from collections import Counter
    pattern_counts = Counter(r["pattern"] for r in bar_rhythms)
    total = len(bar_rhythms)

    coverage = {p: round(c / total, 3) for p, c in pattern_counts.most_common()}
    dominant = pattern_counts.most_common(1)[0][0]

    # Find pattern changes (transitions between different patterns)
    changes = []
    prev_pattern = bar_rhythms[0]["pattern"]
    for r in bar_rhythms[1:]:
        if r["pattern"] != prev_pattern:
            changes.append({
                "bar": r["bar"],
                "from": prev_pattern,
                "to": r["pattern"],
            })
            prev_pattern = r["pattern"]

    return {
        "dominant_pattern": dominant,
        "pattern_coverage": coverage,
        "pattern_changes": changes,
        "total_bars": total,
    }


def summarize_rhythm_sections(bar_rhythms, measures=None):
    """Group consecutive bars with the same pattern into sections.

    Parameters
    ----------
    bar_rhythms : list[dict]
        Output of classify_rhythm().
    measures : list[dict] or None
        Measure data for time info.

    Returns
    -------
    list[dict]
        Each dict has: start_bar, end_bar, pattern, bar_count,
        avg_kick_density, start_time, end_time
    """
    if not bar_rhythms:
        return []

    # Build a time lookup if measures provided
    time_lookup = {}
    if measures:
        for m in measures:
            time_lookup[m["measure_num"]] = (m["start"], m["end"])

    sections = []
    current_pattern = bar_rhythms[0]["pattern"]
    section_start = bar_rhythms[0]["bar"]
    section_densities = [bar_rhythms[0]["kick_density"]]

    for i in range(1, len(bar_rhythms)):
        r = bar_rhythms[i]
        if r["pattern"] != current_pattern:
            # Close current section
            prev_bar = bar_rhythms[i - 1]["bar"]
            section = {
                "start_bar": section_start,
                "end_bar": prev_bar,
                "pattern": current_pattern,
                "bar_count": len(section_densities),
                "avg_kick_density": round(
                    sum(section_densities) / len(section_densities), 2
                ),
            }
            if section_start in time_lookup:
                section["start_time"] = round(time_lookup[section_start][0], 2)
            if prev_bar in time_lookup:
                section["end_time"] = round(time_lookup[prev_bar][1], 2)
            sections.append(section)

            # Start new section
            current_pattern = r["pattern"]
            section_start = r["bar"]
            section_densities = [r["kick_density"]]
        else:
            section_densities.append(r["kick_density"])

    # Close final section
    last_bar = bar_rhythms[-1]["bar"]
    section = {
        "start_bar": section_start,
        "end_bar": last_bar,
        "pattern": current_pattern,
        "bar_count": len(section_densities),
        "avg_kick_density": round(
            sum(section_densities) / len(section_densities), 2
        ),
    }
    if section_start in time_lookup:
        section["start_time"] = round(time_lookup[section_start][0], 2)
    if last_bar in time_lookup:
        section["end_time"] = round(time_lookup[last_bar][1], 2)
    sections.append(section)

    return sections


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Classify kick rhythm per bar.")
    ap.add_argument("audio", help="Path to audio file.")
    ap.add_argument("--bpm", type=float, required=True, help="Track BPM.")
    ap.add_argument("--first-downbeat", type=float, default=0.0,
                    help="First downbeat time in seconds.")
    ap.add_argument("--sr", type=int, default=44100, help="Sample rate.")
    ap.add_argument("--json-out", help="Write JSON output to file.")
    args = ap.parse_args()

    y, sr = librosa.load(args.audio, sr=args.sr, mono=True)
    beat_interval = 60.0 / args.bpm

    # Build measure list
    bar_duration = beat_interval * 4
    measures = []
    t = args.first_downbeat
    bar_num = 0
    while t + bar_duration <= len(y) / sr + bar_duration * 0.5:
        measures.append({
            "measure_num": bar_num,
            "start": t,
            "end": min(t + bar_duration, len(y) / sr),
        })
        t += bar_duration
        bar_num += 1

    bar_rhythms = classify_rhythm(y, sr, measures, beat_interval)
    summary = summarize_rhythm(bar_rhythms)
    sections = summarize_rhythm_sections(bar_rhythms, measures)

    result = {
        "summary": summary,
        "sections": sections,
        "per_bar": bar_rhythms,
    }

    out = json.dumps(result, indent=2)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(out, encoding="utf-8")
        print(f"Saved to {args.json_out}")
    else:
        print(out)
