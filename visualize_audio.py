#!/usr/bin/env python3
"""
visualize_audio.py

Audio visualization tool for analyzing librosa feature data aligned to beats.
Helps identify patterns for event detection (fills, vocals, drops, etc.)

Usage:
    python visualize_audio.py --audio "file.wav" --start-bar 0 --bars 32
    python visualize_audio.py --audio "file.wav" --start-bar 48 --bars 16 --bpm 128
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import librosa

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)


def detect_beats_and_downbeat(y: np.ndarray, sr: int,
                               beats_per_bar: int = 4,
                               manual_bpm: float = None) -> Dict[str, Any]:
    """
    Detect beats and estimate first downbeat.
    Returns beat times, BPM, and downbeat info.
    """
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Handle librosa version differences
    if hasattr(tempo, '__len__'):
        bpm = float(tempo[0]) if len(tempo) > 0 else float(tempo)
    else:
        bpm = float(tempo)

    if manual_bpm:
        bpm = manual_bpm

    # Convert to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        return {"error": "Too few beats detected", "bpm": bpm}

    # Compute beat intervals
    beat_intervals = np.diff(beat_times)
    median_interval = float(np.median(beat_intervals))

    # Simple downbeat estimation using onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strengths = onset_env[np.clip(beat_frames, 0, len(onset_env)-1)]

    # Score each phase
    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        indices = list(range(offset, len(beat_strengths), beats_per_bar))
        if indices:
            scores[offset] = np.sum(beat_strengths[indices])

    best_phase = int(np.argmax(scores))

    # Extrapolate backward if first beat is late
    first_beat_time = beat_times[0]
    prepended_beats = []

    if first_beat_time > median_interval * 1.5:
        t = first_beat_time - median_interval
        while t >= 0:
            prepended_beats.insert(0, t)
            t -= median_interval

    if prepended_beats:
        beat_times = np.concatenate([prepended_beats, beat_times])
        adjusted_phase = (best_phase + len(prepended_beats)) % beats_per_bar
    else:
        adjusted_phase = best_phase

    first_downbeat_idx = adjusted_phase

    return {
        "bpm": round(bpm, 2),
        "beat_times": beat_times,
        "beat_count": len(beat_times),
        "median_interval": round(median_interval, 4),
        "first_downbeat_idx": first_downbeat_idx,
        "first_downbeat_time": round(float(beat_times[first_downbeat_idx]), 4),
        "phase": adjusted_phase,
        "beats_prepended": len(prepended_beats)
    }


def extract_features_beat_aligned(y: np.ndarray, sr: int,
                                   beat_times: np.ndarray,
                                   start_bar: int = 0,
                                   num_bars: int = 32,
                                   beats_per_bar: int = 4,
                                   first_downbeat_idx: int = 0) -> Dict[str, Any]:
    """
    Extract audio features aligned to beat grid.

    Returns feature arrays indexed by beat number within the specified bar range.
    """
    # Calculate beat range
    start_beat = start_bar * beats_per_bar + first_downbeat_idx
    end_beat = start_beat + num_bars * beats_per_bar

    if start_beat >= len(beat_times):
        return {"error": f"Start bar {start_bar} is beyond track length"}

    end_beat = min(end_beat, len(beat_times) - 1)
    actual_beats = end_beat - start_beat

    # Get time range
    start_time = beat_times[start_beat]
    end_time = beat_times[end_beat] if end_beat < len(beat_times) else beat_times[-1]

    # Extract the audio segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_segment = y[start_sample:end_sample]

    # Compute features for the full segment first
    # These will be frame-based, then we map to beats

    # 1. Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr) + start_time

    # 2. RMS energy
    rms = librosa.feature.rms(y=y_segment)[0]
    rms_times = librosa.times_like(rms, sr=sr) + start_time

    # 3. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)[0]
    centroid_times = librosa.times_like(centroid, sr=sr) + start_time

    # 4. Spectral flux (change in spectrum)
    S = np.abs(librosa.stft(y_segment))
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    flux_times = librosa.times_like(spectral_flux, sr=sr) + start_time

    # 5. Frequency band energies
    freqs = librosa.fft_frequencies(sr=sr)

    # Bass (20-250 Hz)
    bass_mask = (freqs >= 20) & (freqs < 250)
    bass_energy = np.sum(S[bass_mask, :]**2, axis=0)

    # Mids (250-2000 Hz) - vocal range
    mid_mask = (freqs >= 250) & (freqs < 2000)
    mid_energy = np.sum(S[mid_mask, :]**2, axis=0)

    # High-mids (2000-6000 Hz) - presence/clarity
    high_mid_mask = (freqs >= 2000) & (freqs < 6000)
    high_mid_energy = np.sum(S[high_mid_mask, :]**2, axis=0)

    # Highs (6000+ Hz)
    high_mask = freqs >= 6000
    high_energy = np.sum(S[high_mask, :]**2, axis=0)

    band_times = librosa.times_like(bass_energy, sr=sr) + start_time

    # 6. Chroma features (for harmonic content)
    chroma = librosa.feature.chroma_cqt(y=y_segment, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr) + start_time

    # 7. Harmonic/Percussive separation (HPSS)
    # This helps distinguish melodic content from drums/percussion
    y_harmonic, y_percussive = librosa.effects.hpss(y_segment)

    # RMS of harmonic and percussive components
    rms_harmonic = librosa.feature.rms(y=y_harmonic)[0]
    rms_percussive = librosa.feature.rms(y=y_percussive)[0]
    hpss_times = librosa.times_like(rms_harmonic, sr=sr) + start_time

    # Now map frame-based features to beat grid
    selected_beat_times = beat_times[start_beat:end_beat+1]
    beat_numbers = np.arange(actual_beats + 1)

    def map_to_beats(feature, feature_times, beat_times):
        """Map frame-based feature to beat times using interpolation."""
        return np.interp(beat_times, feature_times, feature)

    # Map all features to beats
    onset_at_beats = map_to_beats(onset_env, onset_times, selected_beat_times)
    rms_at_beats = map_to_beats(rms, rms_times, selected_beat_times)
    centroid_at_beats = map_to_beats(centroid, centroid_times, selected_beat_times)
    flux_at_beats = map_to_beats(spectral_flux, flux_times, selected_beat_times[:-1])  # flux has one less frame

    bass_at_beats = map_to_beats(bass_energy, band_times, selected_beat_times)
    mid_at_beats = map_to_beats(mid_energy, band_times, selected_beat_times)
    high_mid_at_beats = map_to_beats(high_mid_energy, band_times, selected_beat_times)
    high_at_beats = map_to_beats(high_energy, band_times, selected_beat_times)

    # HPSS at beats
    harmonic_at_beats = map_to_beats(rms_harmonic, hpss_times, selected_beat_times)
    percussive_at_beats = map_to_beats(rms_percussive, hpss_times, selected_beat_times)

    return {
        "start_bar": start_bar,
        "num_bars": num_bars,
        "actual_beats": actual_beats,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "beat_times": selected_beat_times.tolist(),
        "beat_numbers": beat_numbers.tolist(),

        # Raw waveform segment for display
        "waveform": y_segment,
        "waveform_sr": sr,
        "waveform_start_time": start_time,

        # Beat-aligned features
        "onset_strength": onset_at_beats.tolist(),
        "rms_energy": rms_at_beats.tolist(),
        "spectral_centroid": centroid_at_beats.tolist(),
        "spectral_flux": flux_at_beats.tolist(),

        # Frequency bands (beat-aligned)
        "bass_energy": bass_at_beats.tolist(),
        "mid_energy": mid_at_beats.tolist(),
        "high_mid_energy": high_mid_at_beats.tolist(),
        "high_energy": high_at_beats.tolist(),

        # HPSS (beat-aligned)
        "harmonic_energy": harmonic_at_beats.tolist(),
        "percussive_energy": percussive_at_beats.tolist(),

        # Frame-based data for detailed view
        "onset_env_frames": onset_env.tolist(),
        "onset_env_times": onset_times.tolist(),
        "rms_frames": rms.tolist(),
        "rms_times": rms_times.tolist(),
    }


def plot_beat_aligned_features(features: Dict[str, Any],
                                beat_info: Dict[str, Any],
                                output_path: str = None,
                                show: bool = True) -> None:
    """
    Create a multi-panel plot showing audio features aligned to beats.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for plotting", file=sys.stderr)
        return

    start_bar = features["start_bar"]
    num_bars = features["num_bars"]
    beats_per_bar = 4
    bpm = beat_info["bpm"]

    beat_times = np.array(features["beat_times"])
    beat_numbers = np.arange(len(beat_times))

    # Create figure with subplots (7 panels now)
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(7, 1, height_ratios=[2, 1, 1, 1, 1, 1, 1], hspace=0.3, figure=fig)

    # Custom x-axis formatter for bar.beat notation
    def bar_beat_formatter(x, pos):
        beat_in_range = int(x)
        bar = start_bar + beat_in_range // beats_per_bar
        beat = (beat_in_range % beats_per_bar) + 1
        return f"{bar}.{beat}"

    # Panel 1: Waveform
    ax1 = fig.add_subplot(gs[0])
    waveform = features["waveform"]
    waveform_sr = features["waveform_sr"]
    waveform_start = features["waveform_start_time"]
    waveform_times = np.arange(len(waveform)) / waveform_sr + waveform_start

    # Convert waveform times to beat numbers for x-axis alignment
    waveform_beats = np.interp(waveform_times, beat_times, beat_numbers)

    ax1.plot(waveform_beats, waveform, color='steelblue', linewidth=0.3, alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio Waveform - Bars {start_bar}-{start_bar + num_bars} @ {bpm} BPM')
    ax1.set_xlim(0, len(beat_times)-1)

    # Add bar lines
    for bar in range(num_bars + 1):
        beat_pos = bar * beats_per_bar
        if beat_pos < len(beat_times):
            ax1.axvline(x=beat_pos, color='red', linestyle='-', alpha=0.5, linewidth=1)

    # Add beat lines (lighter)
    for beat in range(len(beat_times)):
        if beat % beats_per_bar != 0:  # Skip downbeats (already drawn)
            ax1.axvline(x=beat, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    ax1.xaxis.set_major_formatter(FuncFormatter(bar_beat_formatter))
    ax1.xaxis.set_major_locator(MultipleLocator(beats_per_bar))

    # Panel 2: Onset Strength
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    onset = np.array(features["onset_strength"])
    ax2.fill_between(beat_numbers[:len(onset)], onset, alpha=0.7, color='orange')
    ax2.plot(beat_numbers[:len(onset)], onset, color='darkorange', linewidth=1)
    ax2.set_ylabel('Onset\nStrength')
    ax2.set_xlim(0, len(beat_times)-1)

    # Panel 3: RMS Energy
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    rms = np.array(features["rms_energy"])
    ax3.fill_between(beat_numbers[:len(rms)], rms, alpha=0.7, color='green')
    ax3.plot(beat_numbers[:len(rms)], rms, color='darkgreen', linewidth=1)
    ax3.set_ylabel('RMS\nEnergy')

    # Panel 4: Spectral Centroid
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    centroid = np.array(features["spectral_centroid"])
    ax4.plot(beat_numbers[:len(centroid)], centroid, color='purple', linewidth=1.5)
    ax4.set_ylabel('Spectral\nCentroid (Hz)')

    # Panel 5: Frequency Bands
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    bass = np.array(features["bass_energy"])
    mids = np.array(features["mid_energy"])
    high_mids = np.array(features["high_mid_energy"])

    # Normalize for comparison
    max_val = max(np.max(bass), np.max(mids), np.max(high_mids), 1e-10)
    ax5.plot(beat_numbers[:len(bass)], bass/max_val, label='Bass (20-250Hz)', color='brown', linewidth=1)
    ax5.plot(beat_numbers[:len(mids)], mids/max_val, label='Mids (250-2kHz)', color='blue', linewidth=1)
    ax5.plot(beat_numbers[:len(high_mids)], high_mids/max_val, label='Hi-Mids (2-6kHz)', color='cyan', linewidth=1)
    ax5.set_ylabel('Band\nEnergy')
    ax5.legend(loc='upper right', fontsize=8)

    # Panel 6: Spectral Flux (change detection)
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    flux = np.array(features["spectral_flux"])
    ax6.fill_between(beat_numbers[:len(flux)], flux, alpha=0.7, color='red')
    ax6.plot(beat_numbers[:len(flux)], flux, color='darkred', linewidth=1)
    ax6.set_ylabel('Spectral\nFlux')

    # Panel 7: Harmonic vs Percussive (HPSS)
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    harmonic = np.array(features.get("harmonic_energy", []))
    percussive = np.array(features.get("percussive_energy", []))

    if len(harmonic) > 0 and len(percussive) > 0:
        # Normalize for comparison
        h_max = max(np.max(harmonic), 1e-10)
        p_max = max(np.max(percussive), 1e-10)
        ax7.plot(beat_numbers[:len(harmonic)], harmonic/h_max, label='Harmonic', color='blue', linewidth=1.2)
        ax7.plot(beat_numbers[:len(percussive)], percussive/p_max, label='Percussive', color='orange', linewidth=1.2)
        ax7.set_ylabel('HPSS\n(normalized)')
        ax7.legend(loc='upper right', fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'HPSS data not available', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_ylabel('HPSS')

    ax7.set_xlabel(f'Bar.Beat (starting from bar {start_bar})')

    # Add bar lines to all panels
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7]:
        for bar in range(num_bars + 1):
            beat_pos = bar * beats_per_bar
            if beat_pos < len(beat_times):
                ax.axvline(x=beat_pos, color='red', linestyle='-', alpha=0.3, linewidth=1)


    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}", file=sys.stderr)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Visualize audio features aligned to beats for event detection analysis."
    )

    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--start-bar", type=int, default=0, help="Starting bar number (default: 0)")
    ap.add_argument("--bars", type=int, default=32, help="Number of bars to display (default: 32, max recommended: 32)")
    ap.add_argument("--bpm", type=float, default=None, help="Manual BPM override (auto-detects if not provided)")
    ap.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar (default: 4)")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    ap.add_argument("--output", "-o", help="Output image file (e.g., output.png)")
    ap.add_argument("--no-show", action="store_true", help="Don't display the plot (useful for batch processing)")
    ap.add_argument("--json-out", help="Export feature data to JSON file")

    args = ap.parse_args()

    # Validate
    if args.bars > 64:
        print(f"Warning: {args.bars} bars is very large. Recommend 32 or fewer for readability.", file=sys.stderr)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading audio: {audio_path}", file=sys.stderr)
    y, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)", file=sys.stderr)

    # Detect beats
    print("Detecting beats...", file=sys.stderr)
    beat_info = detect_beats_and_downbeat(y, sr, args.beats_per_bar, args.bpm)

    if "error" in beat_info:
        print(f"Error: {beat_info['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"BPM: {beat_info['bpm']}", file=sys.stderr)
    print(f"Total beats: {beat_info['beat_count']}", file=sys.stderr)
    print(f"First downbeat: {beat_info['first_downbeat_time']:.3f}s (index {beat_info['first_downbeat_idx']})", file=sys.stderr)

    # Check if requested bars are valid
    total_bars = beat_info['beat_count'] // args.beats_per_bar
    if args.start_bar >= total_bars:
        print(f"Error: Start bar {args.start_bar} exceeds track length ({total_bars} bars)", file=sys.stderr)
        sys.exit(1)

    # Extract features
    print(f"Extracting features for bars {args.start_bar}-{args.start_bar + args.bars}...", file=sys.stderr)
    features = extract_features_beat_aligned(
        y, sr,
        beat_times=np.array(beat_info["beat_times"]),
        start_bar=args.start_bar,
        num_bars=args.bars,
        beats_per_bar=args.beats_per_bar,
        first_downbeat_idx=beat_info["first_downbeat_idx"]
    )

    if "error" in features:
        print(f"Error: {features['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {features['actual_beats']} beats ({features['start_time']:.2f}s - {features['end_time']:.2f}s)", file=sys.stderr)

    # Export JSON if requested
    if args.json_out:
        # Remove waveform from JSON (too large)
        json_features = {k: v for k, v in features.items() if k not in ['waveform']}
        json_features["beat_info"] = beat_info
        json_features["beat_info"]["beat_times"] = beat_info["beat_times"].tolist() if hasattr(beat_info["beat_times"], 'tolist') else beat_info["beat_times"]

        with open(args.json_out, 'w') as f:
            json.dump(json_features, f, indent=2)
        print(f"Exported features to: {args.json_out}", file=sys.stderr)

    # Plot
    if HAS_MATPLOTLIB:
        plot_beat_aligned_features(
            features,
            beat_info,
            output_path=args.output,
            show=not args.no_show
        )
    else:
        print("Matplotlib not available - skipping plot", file=sys.stderr)
        if not args.json_out:
            print("Use --json-out to export feature data instead", file=sys.stderr)


if __name__ == "__main__":
    main()
