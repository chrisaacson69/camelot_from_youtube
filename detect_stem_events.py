#!/usr/bin/env python3
"""
Stem-based Event Detector
=========================
Detects musical events by analyzing energy changes in separated stems.
Identifies WHAT changed (drums, bass, vocals/melodic) not just THAT something changed.

Event Types Detected:
- drums_in / drums_out: Drum energy changes (breakdowns, drops)
- bass_in / bass_out: Bass energy changes (build-ups, drops)
- melodic_in / melodic_out: Vocal/synth lead changes
- full_drop: Multiple stems change together (big moment)
- breakdown: Energy drops across multiple stems

Usage:
    python detect_stem_events.py --stems-dir "NothingLeftBreeder_stems"
    python detect_stem_events.py --stems-dir "stems" --bpm 128
    python detect_stem_events.py --stems-dir "stems" --original "track.wav"  # include original for BPM
"""

import argparse
import json
import numpy as np
import librosa
from pathlib import Path


# Detection thresholds
ENERGY_DROP_THRESHOLD = 0.4    # 40% drop = significant
ENERGY_RISE_THRESHOLD = 0.4    # 40% rise = significant
MIN_ENERGY_LEVEL = 0.01        # Below this = "silent"
CONTEXT_BARS = 2               # Look 2 bars before/after for comparison
BEATS_PER_BAR = 4


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


def compute_bar_energy(y, sr, beat_times, beats_per_bar=BEATS_PER_BAR):
    """
    Compute RMS energy per bar, aligned to detected beats.

    Returns array of energy values, one per bar.
    """
    n_beats = len(beat_times)
    n_bars = n_beats // beats_per_bar

    bar_energies = []

    for bar_idx in range(n_bars):
        start_beat = bar_idx * beats_per_bar
        end_beat = start_beat + beats_per_bar

        if end_beat >= n_beats:
            break

        start_time = beat_times[start_beat]
        end_time = beat_times[end_beat] if end_beat < n_beats else beat_times[-1]

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= end_sample:
            bar_energies.append(0.0)
            continue

        segment = y[start_sample:end_sample]
        rms = np.sqrt(np.mean(segment**2))
        bar_energies.append(rms)

    return np.array(bar_energies)


def normalize_energy(energy_array):
    """Normalize energy to 0-1 range."""
    if len(energy_array) == 0:
        return energy_array
    max_e = np.max(energy_array)
    if max_e > 0:
        return energy_array / max_e
    return energy_array


def compute_onset_per_bar(y, sr, beat_times, beats_per_bar=BEATS_PER_BAR):
    """
    Compute onset strength per bar (from original detect_events.py approach).
    Returns normalized onset strength array, one value per bar.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Interpolate to beat positions
    onset_at_beats = np.interp(beat_times, onset_times, onset_env)

    # Average per bar
    n_beats = len(beat_times)
    n_bars = n_beats // beats_per_bar

    bar_onset = []
    for bar_idx in range(n_bars):
        start_beat = bar_idx * beats_per_bar
        end_beat = min(start_beat + beats_per_bar, n_beats)
        bar_onset.append(np.mean(onset_at_beats[start_beat:end_beat]))

    return np.array(bar_onset)


def detect_onset_drops(onset_array, context_bars=CONTEXT_BARS, drop_threshold=0.4):
    """
    Detect significant drops in onset strength (from original approach).
    Returns list of (bar_idx, drop_pct, recovery_pct) tuples.
    """
    drops = []
    n_bars = len(onset_array)

    for bar_idx in range(context_bars, n_bars - context_bars * 2):
        before = np.mean(onset_array[bar_idx - context_bars:bar_idx])
        current_min = np.min(onset_array[bar_idx:bar_idx + context_bars])
        after = np.mean(onset_array[bar_idx + context_bars:bar_idx + context_bars * 2])

        if before > 0.01:  # Avoid division by near-zero
            drop_pct = (before - current_min) / before

            if drop_pct > drop_threshold:
                # Recovery
                abs_drop = before - current_min
                recovery = after - current_min
                recovery_pct = (recovery / abs_drop) if abs_drop > 0 else 0

                drops.append({
                    'bar': bar_idx,
                    'drop_pct': float(drop_pct),
                    'recovery_pct': float(recovery_pct),
                    'before': float(before),
                    'current_min': float(current_min),
                    'after': float(after),
                })

    return drops


def detect_energy_changes(energy_array, context_bars=CONTEXT_BARS):
    """
    Detect significant energy changes in a stem.

    Returns list of (bar_idx, change_type, magnitude) tuples.
    change_type: 'rise' or 'drop'
    magnitude: 0-1 scale of how significant
    """
    changes = []
    n_bars = len(energy_array)

    for bar_idx in range(context_bars, n_bars - context_bars):
        # Energy before this bar
        before = np.mean(energy_array[bar_idx - context_bars:bar_idx])
        # Energy at this bar
        current = energy_array[bar_idx]
        # Energy after this bar
        after = np.mean(energy_array[bar_idx + 1:bar_idx + context_bars + 1])

        # Check for drop (before was high, current/after is low)
        if before > MIN_ENERGY_LEVEL:
            drop_pct = (before - current) / before
            if drop_pct > ENERGY_DROP_THRESHOLD:
                # Verify it stays low
                sustained = after < before * 0.7
                changes.append({
                    'bar': bar_idx,
                    'type': 'drop',
                    'magnitude': float(drop_pct),
                    'before': float(before),
                    'current': float(current),
                    'after': float(after),
                    'sustained': bool(sustained)
                })

        # Check for rise (before was low, current/after is high)
        if current > MIN_ENERGY_LEVEL:
            if before < MIN_ENERGY_LEVEL:
                rise_pct = 1.0  # From silence
            else:
                rise_pct = (current - before) / before if before > 0 else 1.0

            if rise_pct > ENERGY_RISE_THRESHOLD:
                # Verify it stays high
                sustained = after > before * 1.3 if before > MIN_ENERGY_LEVEL else after > MIN_ENERGY_LEVEL
                changes.append({
                    'bar': bar_idx,
                    'type': 'rise',
                    'magnitude': float(min(rise_pct, 2.0) / 2.0),  # Cap at 1.0
                    'before': float(before),
                    'current': float(current),
                    'after': float(after),
                    'sustained': bool(sustained)
                })

    return changes


def load_stems(stems_dir):
    """Load all stem audio files from directory."""
    stems_dir = Path(stems_dir)
    stems = {}

    # Standard Demucs stem names
    stem_names = ['drums', 'bass', 'vocals', 'other']

    for name in stem_names:
        stem_path = stems_dir / f"{name}.wav"
        if stem_path.exists():
            print(f"  Loading {name}...")
            y, sr = librosa.load(stem_path, sr=22050, mono=True)
            stems[name] = {'audio': y, 'sr': sr}
        else:
            print(f"  Warning: {name}.wav not found")

    return stems


def analyze_stems(stems_dir, bpm=None, original_audio=None):
    """
    Main analysis function - detect events in each stem.

    Args:
        stems_dir: Path to directory containing stem wav files
        bpm: Optional BPM override (otherwise auto-detect)
        original_audio: Optional path to original audio for BPM detection
    """
    print(f"Loading stems from: {stems_dir}")
    stems = load_stems(stems_dir)

    if not stems:
        raise ValueError("No stems found!")

    # Use first available stem for beat detection, or original if provided
    if original_audio:
        print(f"Loading original for beat detection: {original_audio}")
        y_ref, sr_ref = librosa.load(original_audio, sr=22050, mono=True)
    else:
        # Use drums stem for beat detection (most reliable)
        if 'drums' in stems:
            y_ref = stems['drums']['audio']
            sr_ref = stems['drums']['sr']
        else:
            first_stem = list(stems.keys())[0]
            y_ref = stems[first_stem]['audio']
            sr_ref = stems[first_stem]['sr']

    # Detect BPM and beats
    if bpm is None or bpm == "auto":
        detected_bpm, beat_times = detect_bpm_and_beats(y_ref, sr_ref)
    else:
        detected_bpm = float(bpm)
        _, beat_times = detect_bpm_and_beats(y_ref, sr_ref)

    print(f"BPM: {detected_bpm:.2f}")
    print(f"Total beats: {len(beat_times)}")
    print(f"Total bars: {len(beat_times) // BEATS_PER_BAR}")

    # Analyze each stem
    stem_analysis = {}

    for stem_name, stem_data in stems.items():
        print(f"\nAnalyzing {stem_name}...")
        y = stem_data['audio']
        sr = stem_data['sr']

        # Compute bar-aligned energy
        bar_energy = compute_bar_energy(y, sr, beat_times)
        bar_energy_norm = normalize_energy(bar_energy)

        # Detect changes
        changes = detect_energy_changes(bar_energy_norm)

        stem_analysis[stem_name] = {
            'bar_energy': [float(e) for e in bar_energy_norm],
            'changes': changes,
            'mean_energy': float(np.mean(bar_energy_norm)),
            'max_energy': float(np.max(bar_energy_norm)) if len(bar_energy_norm) > 0 else 0,
        }

        print(f"  Changes detected: {len(changes)}")
        for c in changes[:5]:  # Show first 5
            print(f"    Bar {c['bar']}: {c['type']} ({c['magnitude']:.1%})")

    # Compute onset strength on original/reference audio
    print("\nAnalyzing onset strength...")
    onset_per_bar = compute_onset_per_bar(y_ref, sr_ref, beat_times)
    onset_norm = normalize_energy(onset_per_bar)
    onset_drops = detect_onset_drops(onset_norm)
    print(f"  Onset drops detected: {len(onset_drops)}")

    # Combine stem changes into musical events
    events = classify_combined_events(stem_analysis, beat_times, detected_bpm, onset_drops)

    return {
        'stems_dir': str(stems_dir),
        'bpm': round(detected_bpm, 2),
        'total_bars': len(beat_times) // BEATS_PER_BAR,
        'stem_analysis': stem_analysis,
        'onset_analysis': {
            'drops': onset_drops,
            'mean': float(np.mean(onset_norm)),
        },
        'events': events,
    }


def classify_combined_events(stem_analysis, beat_times, bpm, onset_drops=None):
    """
    Combine changes across stems into musical events.

    Looks for:
    - Single stem changes (drums_in, bass_out, etc.)
    - Combined changes (full_drop, breakdown, build)
    - Onset strength drops (corroborating evidence)
    """
    events = []

    # Create lookup for onset drops by bar
    onset_drop_bars = {}
    if onset_drops:
        for drop in onset_drops:
            onset_drop_bars[drop['bar']] = drop

    # Collect all changes with their stem source
    all_changes = []
    for stem_name, analysis in stem_analysis.items():
        for change in analysis['changes']:
            all_changes.append({
                'stem': stem_name,
                **change
            })

    # Sort by bar
    all_changes.sort(key=lambda x: x['bar'])

    # Group changes that happen at the same bar (within 1 bar tolerance)
    bar_groups = {}
    for change in all_changes:
        bar = change['bar']
        # Find nearby bar group
        found_group = None
        for group_bar in bar_groups:
            if abs(group_bar - bar) <= 1:
                found_group = group_bar
                break

        if found_group is not None:
            bar_groups[found_group].append(change)
        else:
            bar_groups[bar] = [change]

    # Classify each group
    for bar, changes in bar_groups.items():
        # Check if there's a corroborating onset drop nearby
        onset_drop = None
        for offset in [0, -1, 1]:  # Check bar and adjacent bars
            if bar + offset in onset_drop_bars:
                onset_drop = onset_drop_bars[bar + offset]
                break

        event = classify_event_group(bar, changes, beat_times, bpm, onset_drop)
        if event:
            events.append(event)

    # Sort by bar
    events.sort(key=lambda x: x['bar'])

    return events


def classify_event_group(bar, changes, beat_times, bpm, onset_drop=None):
    """Classify a group of stem changes into a musical event."""

    # Get stems involved and their change types
    stems_dropping = [c for c in changes if c['type'] == 'drop']
    stems_rising = [c for c in changes if c['type'] == 'rise']

    drop_stems = set(c['stem'] for c in stems_dropping)
    rise_stems = set(c['stem'] for c in stems_rising)

    # Calculate time
    beat_idx = bar * BEATS_PER_BAR
    if beat_idx < len(beat_times):
        time_sec = beat_times[beat_idx]
    else:
        time_sec = bar * (BEATS_PER_BAR * 60 / bpm)

    time_fmt = f"{int(time_sec // 60)}:{time_sec % 60:05.2f}"

    # Calculate overall magnitude
    magnitudes = [c['magnitude'] for c in changes]
    avg_magnitude = np.mean(magnitudes) if magnitudes else 0

    # Classify based on what's happening
    event_type = None
    description = []

    # Multiple stems dropping = breakdown
    if len(drop_stems) >= 2:
        if 'drums' in drop_stems:
            event_type = 'breakdown'
            description.append('drums_out')
        if 'bass' in drop_stems:
            description.append('bass_out')
        if 'vocals' in drop_stems or 'other' in drop_stems:
            description.append('melodic_out')

    # Multiple stems rising = drop/build
    elif len(rise_stems) >= 2:
        if 'drums' in rise_stems:
            event_type = 'drop' if 'bass' in rise_stems else 'drums_in'
            description.append('drums_in')
        if 'bass' in rise_stems:
            description.append('bass_in')
        if 'vocals' in rise_stems or 'other' in rise_stems:
            description.append('melodic_in')

    # Single stem changes
    elif len(changes) == 1:
        change = changes[0]
        stem = change['stem']
        ctype = change['type']

        # Rename 'vocals' to 'melodic' for clarity (since it includes synths)
        stem_label = 'melodic' if stem in ['vocals', 'other'] else stem

        event_type = f"{stem_label}_{ctype == 'rise' and 'in' or 'out'}"
        description.append(event_type)

    # Mixed (some dropping, some rising) = transition
    elif drop_stems and rise_stems:
        event_type = 'transition'
        for stem in drop_stems:
            stem_label = 'melodic' if stem in ['vocals', 'other'] else stem
            description.append(f"{stem_label}_out")
        for stem in rise_stems:
            stem_label = 'melodic' if stem in ['vocals', 'other'] else stem
            description.append(f"{stem_label}_in")

    if not event_type:
        return None

    # Determine phrase alignment (8, 16, 32 bar boundaries)
    phrase_alignment = None
    if bar % 32 == 0:
        phrase_alignment = 32
    elif bar % 16 == 0:
        phrase_alignment = 16
    elif bar % 8 == 0:
        phrase_alignment = 8
    elif bar % 4 == 0:
        phrase_alignment = 4

    # Boost confidence if corroborated by onset drop
    onset_corroborated = onset_drop is not None
    if onset_corroborated:
        # Blend onset info with stem confidence
        onset_confidence = onset_drop['drop_pct']
        avg_magnitude = (avg_magnitude + onset_confidence) / 2

    return {
        'bar': bar,
        'beat': bar * BEATS_PER_BAR,
        'time': round(time_sec, 2),
        'time_fmt': time_fmt,
        'type': event_type,
        'details': description,
        'confidence': round(avg_magnitude, 2),
        'stems_involved': list(drop_stems | rise_stems),
        'on_phrase': phrase_alignment is not None,
        'phrase_alignment': phrase_alignment,
        'onset_corroborated': onset_corroborated,
    }


def format_output(results):
    """Format results for display."""
    lines = []
    lines.append("=" * 80)
    lines.append("STEM-BASED EVENT DETECTION")
    lines.append("=" * 80)
    lines.append(f"BPM: {results['bpm']}")
    lines.append(f"Total bars: {results['total_bars']}")
    lines.append(f"Events detected: {len(results['events'])}")
    lines.append("")

    # Per-stem summary
    lines.append("Stem Summary:")
    lines.append("-" * 40)
    for stem_name, analysis in results['stem_analysis'].items():
        n_changes = len(analysis['changes'])
        mean_e = analysis['mean_energy']
        lines.append(f"  {stem_name:8s}: {n_changes:2d} changes, avg energy: {mean_e:.2f}")
    lines.append("")

    # Events
    if results['events']:
        lines.append(f"{'Bar':>5} {'Time':>8} {'Type':<14} {'Conf':>5} {'Phr':>4} {'Ons':>3} {'Details'}")
        lines.append("-" * 80)

        for event in results['events']:
            details = ', '.join(event['details'])
            phrase = f"{event.get('phrase_alignment', '-'):>3}" if event.get('on_phrase') else "  -"
            onset = "Y" if event.get('onset_corroborated') else " "
            lines.append(
                f"{event['bar']:>5} {event['time_fmt']:>8} {event['type']:<14} "
                f"{event['confidence']:>5.2f} {phrase} {onset:>3} {details}"
            )
    else:
        lines.append("No significant events detected.")

    lines.append("")
    return "\n".join(lines)


def main():
    global ENERGY_DROP_THRESHOLD, ENERGY_RISE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Detect events using stem analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python detect_stem_events.py --stems-dir "NothingLeftBreeder_stems"

  # With specific BPM
  python detect_stem_events.py --stems-dir "stems" --bpm 128

  # Use original track for beat detection
  python detect_stem_events.py --stems-dir "stems" --original "track.wav"
        """
    )

    parser.add_argument("--stems-dir", "-d", required=True,
                        help="Directory containing stem .wav files")
    parser.add_argument("--bpm", default="auto",
                        help="BPM (number or 'auto')")
    parser.add_argument("--original", "-o",
                        help="Original audio file for BPM detection")
    parser.add_argument("--json-out", "-j",
                        help="Output JSON file")
    parser.add_argument("--drop-threshold", type=float, default=ENERGY_DROP_THRESHOLD,
                        help=f"Energy drop threshold (default: {ENERGY_DROP_THRESHOLD})")
    parser.add_argument("--rise-threshold", type=float, default=ENERGY_RISE_THRESHOLD,
                        help=f"Energy rise threshold (default: {ENERGY_RISE_THRESHOLD})")

    args = parser.parse_args()

    # Update thresholds
    ENERGY_DROP_THRESHOLD = args.drop_threshold
    ENERGY_RISE_THRESHOLD = args.rise_threshold

    bpm = None if args.bpm == "auto" else float(args.bpm)

    results = analyze_stems(
        args.stems_dir,
        bpm=bpm,
        original_audio=args.original
    )

    print(format_output(results))

    if args.json_out:
        # Remove large arrays for cleaner JSON
        output = {k: v for k, v in results.items() if k != 'stem_analysis'}
        output['stem_summary'] = {
            name: {
                'n_changes': len(data['changes']),
                'mean_energy': data['mean_energy'],
                'changes': data['changes']
            }
            for name, data in results['stem_analysis'].items()
        }

        with open(args.json_out, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
