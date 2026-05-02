"""alignment.py - Cross-correlate kick envelopes for phase alignment.

Find the optimal sample offset to align two tracks' kick drums,
producing an rb_offset value for CyborgDJ specs.

Ported from cyborgdj/reverse-engineer/dinka-giga-mix/align_to_master.py
with additions:
- Multi-probe consistency checking
- JSON output
- CyborgDJ spec integration recommendations

Usage (standalone):
    python alignment.py \
        --track-a "track_a.mp3" --track-b "track_b.mp3" \
        --bpm-a 122 --bpm-b 122 \
        --first-beat-a 0.174 --first-beat-b 0.511 \
        --bars-a 216-240 --bars-b 0-24
"""

import numpy as np
from scipy.signal import butter, sosfilt, correlate
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


def fine_alignment(seg_a, seg_b, sr, max_offset_ms=100):
    """Cross-correlate kick envelopes within a narrow window.

    Uses parabolic interpolation for sub-sample accuracy.

    Parameters
    ----------
    seg_a, seg_b : np.ndarray
        Audio segments to align.
    sr : int
        Sample rate.
    max_offset_ms : float
        Maximum search window in milliseconds.

    Returns
    -------
    tuple (int, float)
        (integer_lag, refined_lag) in samples.
        Positive = B should shift later relative to A.
    """
    env_a = kick_envelope(seg_a, sr)
    env_b = kick_envelope(seg_b, sr)

    min_len = min(len(env_a), len(env_b))
    env_a = env_a[:min_len]
    env_b = env_b[:min_len]

    max_off = int(max_offset_ms / 1000 * sr)
    xcorr = correlate(env_a, env_b, mode='full')
    center = min_len - 1
    region = xcorr[center - max_off:center + max_off + 1]
    lags = np.arange(-max_off, max_off + 1)
    peak_idx = np.argmax(region)
    best_lag = int(lags[peak_idx])

    # Parabolic interpolation for sub-sample accuracy
    if 0 < peak_idx < len(region) - 1:
        a, b, c = float(region[peak_idx - 1]), float(region[peak_idx]), float(region[peak_idx + 1])
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            refined = peak_idx + 0.5 * (a - c) / d
        else:
            refined = float(peak_idx)
        refined_lag = refined - max_off
    else:
        refined_lag = float(best_lag)

    return best_lag, refined_lag


def find_kick_offset(audio_a, audio_b, sr, max_offset_ms=100):
    """Find optimal sample offset to align two tracks' kicks.

    Parameters
    ----------
    audio_a, audio_b : np.ndarray
        Audio segments from the transition zone of each track.
    sr : int
        Sample rate.
    max_offset_ms : float
        Maximum search window in milliseconds.

    Returns
    -------
    dict with keys:
        offset_samples : int
        offset_samples_refined : float
        offset_ms : float
        correlation_peak : float (0-1, higher = more confident)
    """
    lag_int, lag_refined = fine_alignment(audio_a, audio_b, sr, max_offset_ms)

    # Compute correlation peak for confidence
    env_a = kick_envelope(audio_a, sr)
    env_b = kick_envelope(audio_b, sr)
    min_len = min(len(env_a), len(env_b))
    env_a = env_a[:min_len]
    env_b = env_b[:min_len]

    # Normalize correlation to 0-1
    norm_a = np.sqrt(np.sum(env_a ** 2))
    norm_b = np.sqrt(np.sum(env_b ** 2))
    if norm_a > 0 and norm_b > 0:
        max_off = int(max_offset_ms / 1000 * sr)
        xcorr = correlate(env_a, env_b, mode='full')
        center = min_len - 1
        region = xcorr[center - max_off:center + max_off + 1]
        peak_val = float(np.max(region)) / (norm_a * norm_b)
    else:
        peak_val = 0.0

    return {
        "offset_samples": lag_int,
        "offset_samples_refined": round(lag_refined, 2),
        "offset_ms": round(lag_int / sr * 1000, 2),
        "correlation_peak": round(min(peak_val, 1.0), 4),
    }


def extract_bar_audio(audio, sr, bpm, first_beat_sample, bar_start, bar_end):
    """Extract audio for a range of bars.

    Parameters
    ----------
    audio : np.ndarray
        Full mono audio.
    sr : int
        Sample rate.
    bpm : float
        Track BPM.
    first_beat_sample : int
        Sample position of first beat.
    bar_start, bar_end : int
        Bar range (0-based, end exclusive).

    Returns
    -------
    np.ndarray
        Audio segment for the specified bars.
    """
    samples_per_bar = int(60.0 / bpm * 4 * sr)
    start_sample = first_beat_sample + bar_start * samples_per_bar
    end_sample = first_beat_sample + bar_end * samples_per_bar

    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    return audio[start_sample:end_sample]


def multi_probe_alignment(audio_a, audio_b, sr, bpm,
                          first_beat_a, first_beat_b,
                          bar_start_a, bar_start_b,
                          n_probes=4, probe_length_bars=4,
                          max_offset_ms=100):
    """Run alignment at multiple points and check consistency.

    More robust than single-point alignment. Takes multiple short
    segments from the overlap zone and verifies consistent offset.

    Parameters
    ----------
    audio_a, audio_b : np.ndarray
        Full audio for each track.
    sr : int
        Sample rate.
    bpm : float
        Shared BPM (after time-stretching).
    first_beat_a, first_beat_b : int
        First beat sample for each track.
    bar_start_a, bar_start_b : int
        Starting bar in each track's transition zone.
    n_probes : int
        Number of probe points.
    probe_length_bars : int
        Length of each probe in bars.
    max_offset_ms : float
        Maximum search window.

    Returns
    -------
    dict with keys:
        offset_samples : int (median of probes)
        offset_ms : float
        consistency : float (0-1, fraction of probes agreeing within 10ms)
        probes : list[dict] — per-probe results
        recommendation : dict — CyborgDJ spec recommendation
    """
    probes = []
    samples_per_bar = int(60.0 / bpm * 4 * sr)

    for i in range(n_probes):
        bar_off = i * probe_length_bars
        seg_a = extract_bar_audio(
            audio_a, sr, bpm, first_beat_a,
            bar_start_a + bar_off,
            bar_start_a + bar_off + probe_length_bars
        )
        seg_b = extract_bar_audio(
            audio_b, sr, bpm, first_beat_b,
            bar_start_b + bar_off,
            bar_start_b + bar_off + probe_length_bars
        )

        if len(seg_a) < sr * 0.5 or len(seg_b) < sr * 0.5:
            continue

        result = find_kick_offset(seg_a, seg_b, sr, max_offset_ms)
        result["probe_bars_a"] = f"{bar_start_a + bar_off}-{bar_start_a + bar_off + probe_length_bars}"
        result["probe_bars_b"] = f"{bar_start_b + bar_off}-{bar_start_b + bar_off + probe_length_bars}"
        probes.append(result)

    if not probes:
        return {
            "offset_samples": 0,
            "offset_ms": 0.0,
            "consistency": 0.0,
            "probes": [],
            "recommendation": {
                "suggested_rb_offset": 0,
                "rationale": "No valid probes — insufficient audio in overlap region.",
            },
        }

    # Compute median offset and consistency
    offsets = [p["offset_samples"] for p in probes]
    median_offset = int(np.median(offsets))
    median_ms = round(median_offset / sr * 1000, 2)

    # Consistency: fraction of probes within 10ms of median
    tolerance_samples = int(0.010 * sr)  # 10ms
    agreeing = sum(1 for o in offsets if abs(o - median_offset) <= tolerance_samples)
    consistency = round(agreeing / len(probes), 2)

    recommendation = {
        "suggested_rb_offset": median_offset,
        "apply_to": "track_b",
        "offset_ms": median_ms,
        "rationale": (
            f"Track B kicks {'lag' if median_offset > 0 else 'lead'} Track A "
            f"by {abs(median_ms):.1f}ms ({abs(median_offset)} samples at {sr}Hz). "
            f"{'Add' if median_offset > 0 else 'Subtract'} {abs(median_offset)} "
            f"{'to' if median_offset > 0 else 'from'} Track B's first_beat_sample."
        ),
    }

    return {
        "offset_samples": median_offset,
        "offset_ms": median_ms,
        "consistency": consistency,
        "probes": probes,
        "recommendation": recommendation,
    }


def alignment_to_json(result, name_a, name_b, first_beat_a=None,
                      first_beat_b=None):
    """Export alignment result as JSON-serializable dict.

    Returns
    -------
    dict suitable for json.dumps()
    """
    output = {
        "track_a": {"name": name_a},
        "track_b": {"name": name_b},
        "alignment": {
            "offset_samples": result["offset_samples"],
            "offset_ms": result["offset_ms"],
            "consistency": result.get("consistency", 1.0),
        },
        "recommendation": result["recommendation"],
    }

    if first_beat_a is not None:
        output["track_a"]["first_beat_sample"] = first_beat_a
    if first_beat_b is not None:
        output["track_b"]["first_beat_sample"] = first_beat_b
        output["recommendation"]["new_first_beat_sample"] = (
            first_beat_b + result["offset_samples"]
        )

    if "probes" in result:
        output["probes"] = result["probes"]

    return output


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Phase alignment between two tracks via kick cross-correlation")
    ap.add_argument("--track-a", required=True, help="Track A audio file")
    ap.add_argument("--track-b", required=True, help="Track B audio file")
    ap.add_argument("--bpm", type=float, required=True, help="Shared BPM")
    ap.add_argument("--first-beat-a", type=int, required=True,
                    help="Track A first_beat_sample")
    ap.add_argument("--first-beat-b", type=int, required=True,
                    help="Track B first_beat_sample")
    ap.add_argument("--bars-a", default="0-16",
                    help="Bar range in Track A (e.g. '216-232')")
    ap.add_argument("--bars-b", default="0-16",
                    help="Bar range in Track B (e.g. '0-16')")
    ap.add_argument("--sr", type=int, default=44100, help="Sample rate")
    ap.add_argument("--max-offset-ms", type=float, default=100.0,
                    help="Maximum phase offset search window (ms)")
    ap.add_argument("--json-out", help="Write JSON output to file")
    args = ap.parse_args()

    print(f"Loading Track A: {args.track_a}")
    y_a, sr = librosa.load(args.track_a, sr=args.sr, mono=True)
    print(f"Loading Track B: {args.track_b}")
    y_b, _ = librosa.load(args.track_b, sr=args.sr, mono=True)

    # Parse bar ranges
    bars_a = args.bars_a.split("-")
    bar_start_a, bar_end_a = int(bars_a[0]), int(bars_a[1])
    bars_b = args.bars_b.split("-")
    bar_start_b, bar_end_b = int(bars_b[0]), int(bars_b[1])

    n_bars = min(bar_end_a - bar_start_a, bar_end_b - bar_start_b)

    result = multi_probe_alignment(
        y_a, y_b, sr, args.bpm,
        args.first_beat_a, args.first_beat_b,
        bar_start_a, bar_start_b,
        n_probes=max(1, n_bars // 4),
        probe_length_bars=4,
        max_offset_ms=args.max_offset_ms,
    )

    name_a = args.track_a.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name_b = args.track_b.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    output = alignment_to_json(
        result, name_a, name_b,
        first_beat_a=args.first_beat_a,
        first_beat_b=args.first_beat_b,
    )

    out_str = json.dumps(output, indent=2)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(out_str, encoding="utf-8")
        print(f"\nJSON saved to: {args.json_out}")
    else:
        print(out_str)

    # Summary
    rec = result["recommendation"]
    print(f"\nPhase alignment: {result['offset_ms']}ms "
          f"({result['offset_samples']} samples)")
    print(f"Consistency: {result.get('consistency', 'N/A')}")
    print(f"Recommendation: {rec['rationale']}")
