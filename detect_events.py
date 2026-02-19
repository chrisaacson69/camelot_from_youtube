#!/usr/bin/env python3
"""
Event detector for audio files using onset strength drops.

Based on analysis of known events:
- Fills/new elements: ~50-70% onset drop, ~80-100% recovery
- Vocals in: ~60-65% onset drop, ~80% recovery
- Drums out/breakdown: ~80% onset drop, <30% recovery (stays low)

Usage:
    python detect_events.py --audio "file.wav" --bpm 129.2
    python detect_events.py --audio "file.wav" --bpm auto
"""
import argparse
import json
import numpy as np
import librosa

# Detection parameters (tuned from analyze_onset_drops.py results)
MIN_DROP_PCT = 40.0          # Minimum % drop to consider as event
FILL_DROP_RANGE = (40, 75)   # Fills typically 50-70% drop
BREAKDOWN_DROP_MIN = 75.0    # Breakdowns typically >75% drop
RECOVERY_THRESHOLD = 50.0    # <50% recovery = breakdown/drums out, >50% = fill

# How many beats to look at for "before" and "after" windows
CONTEXT_BEATS = 8  # 2 bars

def detect_bpm_and_beats(y, sr):
    """Detect BPM and beat times from audio."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        precise_bpm = 60.0 / median_interval
    else:
        precise_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    return precise_bpm, beat_times

def extract_onset_per_beat(y, sr, beat_times):
    """Extract onset strength values aligned to beats."""
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Interpolate to beat positions
    onset_at_beats = np.interp(beat_times, onset_times, onset_env)

    return onset_at_beats

def analyze_drop_at_beat(onset_values, beat_idx, context_beats=CONTEXT_BEATS):
    """
    Analyze onset drop at a specific beat position.
    Returns dict with drop stats or None if not enough context.
    """
    if beat_idx < context_beats or beat_idx + context_beats * 2 >= len(onset_values):
        return None

    # Before window: context_beats before the event
    before_vals = onset_values[beat_idx - context_beats:beat_idx]

    # Event window: 2 bars starting at beat_idx
    event_vals = onset_values[beat_idx:beat_idx + context_beats]

    # After window: 2 bars after event
    after_vals = onset_values[beat_idx + context_beats:beat_idx + context_beats * 2]

    if len(before_vals) == 0 or len(event_vals) == 0:
        return None

    before_mean = np.mean(before_vals)
    event_min = np.min(event_vals)
    event_mean = np.mean(event_vals)
    after_mean = np.mean(after_vals) if len(after_vals) > 0 else event_mean

    # Calculate drop
    abs_drop = before_mean - event_min
    pct_drop = (abs_drop / before_mean * 100) if before_mean > 0 else 0

    # Calculate recovery
    recovery = after_mean - event_min
    recovery_pct = (recovery / abs_drop * 100) if abs_drop > 0 else 0

    return {
        "before_mean": before_mean,
        "event_min": event_min,
        "event_mean": event_mean,
        "after_mean": after_mean,
        "abs_drop": abs_drop,
        "pct_drop": pct_drop,
        "recovery": recovery,
        "recovery_pct": recovery_pct,
    }

def classify_event(drop_stats):
    """
    Classify an event based on drop and recovery characteristics.

    Returns: event_type, confidence
    """
    pct_drop = drop_stats["pct_drop"]
    recovery_pct = drop_stats["recovery_pct"]

    # Not a significant event
    if pct_drop < MIN_DROP_PCT:
        return None, 0.0

    # High drop + low recovery = breakdown/drums out
    if pct_drop >= BREAKDOWN_DROP_MIN and recovery_pct < RECOVERY_THRESHOLD:
        confidence = min(1.0, (pct_drop - 60) / 40)  # Scale 60-100% to 0-1
        return "breakdown", confidence

    # Medium drop + high recovery = fill or new element
    if FILL_DROP_RANGE[0] <= pct_drop <= FILL_DROP_RANGE[1] and recovery_pct > RECOVERY_THRESHOLD:
        confidence = min(1.0, recovery_pct / 100)
        return "fill", confidence

    # High drop but with recovery = major transition (vocals, etc)
    if pct_drop > FILL_DROP_RANGE[1] and recovery_pct > RECOVERY_THRESHOLD:
        confidence = min(1.0, pct_drop / 100)
        return "transition", confidence

    # Moderate drop, unclear recovery
    if pct_drop >= MIN_DROP_PCT:
        return "event", 0.5

    return None, 0.0

def detect_events(y, sr, bpm=None, beats_per_bar=4):
    """
    Detect events in audio using onset strength drops.

    Returns list of detected events with bar numbers and classifications.
    """
    # Get BPM and beats
    if bpm is None or bpm == "auto":
        detected_bpm, beat_times = detect_bpm_and_beats(y, sr)
    else:
        detected_bpm = float(bpm)
        # Still detect beats for alignment
        _, beat_times = detect_bpm_and_beats(y, sr)

    print(f"Using BPM: {detected_bpm:.2f}")
    print(f"Total beats detected: {len(beat_times)}")

    # Extract onset values per beat
    onset_values = extract_onset_per_beat(y, sr, beat_times)

    events = []
    checked_bars = set()

    # Check at phrase boundaries (every 8 bars) and every bar within phrases
    total_beats = len(onset_values)
    total_bars = total_beats // beats_per_bar

    # First pass: check every bar at the downbeat
    for bar in range(total_bars):
        beat_idx = bar * beats_per_bar

        drop_stats = analyze_drop_at_beat(onset_values, beat_idx, CONTEXT_BEATS)
        if drop_stats is None:
            continue

        event_type, confidence = classify_event(drop_stats)

        if event_type is not None:
            bar_time = beat_times[beat_idx] if beat_idx < len(beat_times) else 0
            events.append({
                "bar": bar,
                "beat": beat_idx,
                "time": round(bar_time, 2),
                "time_fmt": f"{int(bar_time // 60)}:{bar_time % 60:05.2f}",
                "type": event_type,
                "confidence": round(confidence, 2),
                "pct_drop": round(drop_stats["pct_drop"], 1),
                "recovery_pct": round(drop_stats["recovery_pct"], 1),
            })
            checked_bars.add(bar)

    # Remove duplicate events that are within 2 bars of each other (keep highest confidence)
    events = merge_nearby_events(events)

    return {
        "bpm": round(detected_bpm, 2),
        "total_bars": total_bars,
        "total_beats": total_beats,
        "events": events,
    }

def merge_nearby_events(events, min_gap_bars=2):
    """Merge events that are too close together, keeping the highest confidence one."""
    if not events:
        return events

    # Sort by bar number
    sorted_events = sorted(events, key=lambda e: e["bar"])

    merged = []
    current_group = [sorted_events[0]]

    for event in sorted_events[1:]:
        # If this event is close to the current group
        if event["bar"] - current_group[-1]["bar"] <= min_gap_bars:
            current_group.append(event)
        else:
            # Pick the best from current group
            best = max(current_group, key=lambda e: e["confidence"])
            merged.append(best)
            current_group = [event]

    # Don't forget the last group
    if current_group:
        best = max(current_group, key=lambda e: e["confidence"])
        merged.append(best)

    return merged

def format_output(results):
    """Format results for display."""
    lines = []
    lines.append("=" * 70)
    lines.append("EVENT DETECTION RESULTS")
    lines.append("=" * 70)
    lines.append(f"BPM: {results['bpm']}")
    lines.append(f"Total bars: {results['total_bars']}")
    lines.append(f"Events detected: {len(results['events'])}")
    lines.append("")

    if results['events']:
        lines.append(f"{'Bar':>5} {'Time':>8} {'Type':<12} {'Conf':>5} {'Drop%':>6} {'Recov%':>7}")
        lines.append("-" * 50)

        for event in results['events']:
            lines.append(
                f"{event['bar']:>5} {event['time_fmt']:>8} {event['type']:<12} "
                f"{event['confidence']:>5.2f} {event['pct_drop']:>6.1f} {event['recovery_pct']:>7.1f}"
            )
    else:
        lines.append("No significant events detected.")

    lines.append("")
    return "\n".join(lines)

def main():
    global MIN_DROP_PCT  # Declare at start of function

    parser = argparse.ArgumentParser(description="Detect events using onset drops")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--bpm", default="auto", help="BPM (number or 'auto')")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--json-out", help="Output JSON file")
    parser.add_argument("--min-drop", type=float, default=MIN_DROP_PCT,
                       help=f"Minimum drop %% to detect (default: {MIN_DROP_PCT})")

    args = parser.parse_args()

    # Update global threshold if specified
    MIN_DROP_PCT = args.min_drop

    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=args.sr)

    bpm = None if args.bpm == "auto" else float(args.bpm)

    print("Detecting events...")
    results = detect_events(y, sr, bpm=bpm)

    print(format_output(results))

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json_out}")

if __name__ == "__main__":
    main()
