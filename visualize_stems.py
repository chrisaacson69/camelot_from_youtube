#!/usr/bin/env python3
"""
Stem Energy Visualization
=========================
Creates a multi-panel visualization showing energy over time for each stem,
with detected events marked. Helps understand what the stem detector is seeing.

Usage:
    python visualize_stems.py --stems-dir "NothingLeftBreeder_stems"
    python visualize_stems.py --stems-dir "stems" --original "track.wav" --output "stems_viz.png"
"""

import argparse
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path


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
    """Compute RMS energy per bar."""
    n_beats = len(beat_times)
    n_bars = n_beats // beats_per_bar

    bar_energies = []
    bar_times = []

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
            bar_times.append(start_time)
            continue

        segment = y[start_sample:end_sample]
        rms = np.sqrt(np.mean(segment**2))
        bar_energies.append(rms)
        bar_times.append(start_time)

    return np.array(bar_energies), np.array(bar_times)


def compute_onset_strength_per_bar(y, sr, beat_times, beats_per_bar=BEATS_PER_BAR):
    """Compute onset strength per bar (for comparison with original detector)."""
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


def normalize(arr):
    """Normalize array to 0-1 range."""
    if len(arr) == 0:
        return arr
    max_val = np.max(arr)
    if max_val > 0:
        return arr / max_val
    return arr


def load_events(stems_dir):
    """Load events from stem_events.json if it exists."""
    events_file = Path(stems_dir) / "stem_events.json"
    if events_file.exists():
        with open(events_file) as f:
            data = json.load(f)
            return data.get('events', [])
    return []


def visualize_stems(stems_dir, original_audio=None, output_path=None, show_onset=True):
    """
    Create visualization of stem energies over time.

    Args:
        stems_dir: Path to directory with stem wav files
        original_audio: Optional path to original for BPM/onset analysis
        output_path: Optional path to save PNG
        show_onset: Include onset strength panel
    """
    stems_dir = Path(stems_dir)
    stem_names = ['drums', 'bass', 'vocals', 'other']

    # Load stems
    stems = {}
    for name in stem_names:
        stem_path = stems_dir / f"{name}.wav"
        if stem_path.exists():
            print(f"Loading {name}...")
            y, sr = librosa.load(stem_path, sr=22050, mono=True)
            stems[name] = {'audio': y, 'sr': sr}

    if not stems:
        print("No stems found!")
        return

    # Get beat times from drums or original
    if original_audio:
        print(f"Loading original for beat detection: {original_audio}")
        y_ref, sr_ref = librosa.load(original_audio, sr=22050, mono=True)
    elif 'drums' in stems:
        y_ref = stems['drums']['audio']
        sr_ref = stems['drums']['sr']
    else:
        first_stem = list(stems.keys())[0]
        y_ref = stems[first_stem]['audio']
        sr_ref = stems[first_stem]['sr']

    bpm, beat_times = detect_bpm_and_beats(y_ref, sr_ref)
    print(f"BPM: {bpm:.2f}, Bars: {len(beat_times) // BEATS_PER_BAR}")

    # Compute energy per bar for each stem
    stem_energies = {}
    bar_times = None

    for name, data in stems.items():
        energy, times = compute_bar_energy(data['audio'], data['sr'], beat_times)
        stem_energies[name] = normalize(energy)
        if bar_times is None:
            bar_times = times

    # Compute onset strength if requested
    onset_strength = None
    if show_onset and original_audio:
        print("Computing onset strength...")
        onset_strength = compute_onset_strength_per_bar(y_ref, sr_ref, beat_times)
        onset_strength = normalize(onset_strength)

    # Load events
    events = load_events(stems_dir)

    # Create visualization
    n_panels = len(stems) + (1 if onset_strength is not None else 0) + 1  # +1 for combined
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.5 * n_panels), sharex=True)

    colors = {
        'drums': '#e74c3c',    # Red
        'bass': '#3498db',     # Blue
        'vocals': '#2ecc71',   # Green
        'other': '#9b59b6',    # Purple
        'onset': '#f39c12',    # Orange
    }

    bar_numbers = np.arange(len(bar_times))

    # Event markers
    event_bars = {e['bar']: e for e in events}

    # Panel index
    panel = 0

    # Individual stem panels
    for name in stem_names:
        if name not in stem_energies:
            continue

        ax = axes[panel]
        energy = stem_energies[name]

        # Plot energy
        ax.fill_between(bar_numbers[:len(energy)], energy, alpha=0.3, color=colors.get(name, 'gray'))
        ax.plot(bar_numbers[:len(energy)], energy, color=colors.get(name, 'gray'), linewidth=1)

        # Mark events involving this stem
        for bar, event in event_bars.items():
            if name in event.get('stems_involved', []) or \
               (name == 'vocals' and 'melodic' in str(event.get('details', []))) or \
               (name == 'other' and 'melodic' in str(event.get('details', []))):
                if bar < len(energy):
                    marker = 'v' if 'out' in event['type'] or 'drop' in str(event.get('details', [])) else '^'
                    ax.axvline(bar, color='black', alpha=0.3, linestyle='--')
                    ax.scatter([bar], [energy[bar]], color='black', marker=marker, s=50, zorder=5)

        ax.set_ylabel(name.capitalize(), fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Add phrase markers (every 8 bars)
        for b in range(0, len(bar_numbers), 8):
            ax.axvline(b, color='gray', alpha=0.2, linestyle=':')

        panel += 1

    # Onset strength panel
    if onset_strength is not None:
        ax = axes[panel]
        # Ensure same length
        min_len = min(len(bar_numbers), len(onset_strength))
        ax.fill_between(bar_numbers[:min_len], onset_strength[:min_len], alpha=0.3, color=colors['onset'])
        ax.plot(bar_numbers[:min_len], onset_strength[:min_len], color=colors['onset'], linewidth=1)
        ax.set_ylabel('Onset\nStrength', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Phrase markers
        for b in range(0, len(bar_numbers), 8):
            ax.axvline(b, color='gray', alpha=0.2, linestyle=':')

        panel += 1

    # Combined panel
    ax = axes[panel]
    for name in stem_names:
        if name in stem_energies:
            energy = stem_energies[name]
            ax.plot(bar_numbers[:len(energy)], energy, color=colors.get(name, 'gray'),
                   linewidth=1.5, alpha=0.7, label=name.capitalize())

    # Mark all events
    for bar, event in event_bars.items():
        if bar < len(bar_numbers):
            event_type = event['type']
            if event_type == 'breakdown':
                ax.axvspan(bar, bar + 2, color='red', alpha=0.2)
            elif event_type == 'drop':
                ax.axvspan(bar, bar + 2, color='green', alpha=0.2)
            elif 'drums' in event_type:
                ax.axvline(bar, color='red', alpha=0.4, linestyle='-')
            elif 'bass' in event_type:
                ax.axvline(bar, color='blue', alpha=0.4, linestyle='-')
            elif 'melodic' in event_type:
                ax.axvline(bar, color='green', alpha=0.4, linestyle='-')

    ax.set_ylabel('Combined', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Bar Number', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phrase markers
    for b in range(0, len(bar_numbers), 8):
        ax.axvline(b, color='gray', alpha=0.2, linestyle=':')

    # Title
    fig.suptitle(f'Stem Energy Analysis - {stems_dir.name} (BPM: {bpm:.1f})', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stem energies over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--stems-dir", "-d", required=True,
                        help="Directory containing stem .wav files")
    parser.add_argument("--original", "-o",
                        help="Original audio file for BPM/onset detection")
    parser.add_argument("--output", "-out",
                        help="Output PNG file (default: show interactively)")
    parser.add_argument("--no-onset", action="store_true",
                        help="Don't show onset strength panel")

    args = parser.parse_args()

    visualize_stems(
        args.stems_dir,
        original_audio=args.original,
        output_path=args.output,
        show_onset=not args.no_onset
    )


if __name__ == "__main__":
    main()
