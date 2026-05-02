"""spectral_analysis.py - Per-bar third-octave spectral energy analysis.

Computes energy in 31 ISO 266 third-octave bands for each bar (measure),
producing a (n_bars, 31) matrix of dBFS values suitable for heatmap
visualization and downstream collision analysis.

Note: At sr=44100/n_fft=2048, FFT resolution is ~21.5 Hz.  Bands below
40 Hz have only 1-2 bins.  This is acceptable since sub-bass energy is
broadband and the analysis is used for comparative (not absolute) purposes.

Usage:
    python spectral_analysis.py "track.wav" [--save-plot out.png] [--json-out out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa

# ── ISO 266 third-octave center frequencies (31 bands, 20 Hz – 20 kHz) ──

THIRD_OCTAVE_CENTERS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
    20000,
])

_THIRD_OCTAVE_FACTOR = 2 ** (1.0 / 6.0)  # band edge multiplier

NOISE_FLOOR_DB = -80.0


# ── Core functions ──────────────────────────────────────────────────────

def compute_band_edges(centers):
    """Compute lower and upper edges for each third-octave band.

    Returns np.ndarray of shape (n_bands, 2) with [low_hz, high_hz] rows.
    """
    f = _THIRD_OCTAVE_FACTOR
    return np.column_stack([centers / f, centers * f])


def compute_bar_spectra(y, sr, measures, centers=THIRD_OCTAVE_CENTERS,
                        hop_length=512):
    """Compute per-bar energy in third-octave bands via STFT.

    Follows the same STFT → frame-slice pattern as
    event_detect.compute_bar_features.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    measures : list[dict]
        Each dict has "start", "end" (seconds) and "measure_num".
    centers : np.ndarray
        Band center frequencies (default: 31 ISO third-octave bands).
    hop_length : int
        STFT hop length (default 512, matching event_detect).

    Returns
    -------
    dict with keys:
        bar_spectra : np.ndarray (n_bars, n_bands) — dBFS values
        band_centers : np.ndarray (n_bands,) — Hz
        band_edges : np.ndarray (n_bands, 2) — [low_hz, high_hz]
        bar_times : list of (start, end) in seconds
        bar_nums : list of measure_num ints
    """
    # Full-track STFT (same as event_detect line 301)
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    edges = compute_band_edges(centers)
    n_bands = len(centers)

    # Precompute boolean masks: which FFT bins fall in each band
    band_masks = []
    for low, high in edges:
        mask = (freqs >= low) & (freqs < high)
        band_masks.append(mask)

    bar_spectra = []
    bar_times = []
    bar_nums = []

    for m in measures:
        bar_times.append((m["start"], m["end"]))
        bar_nums.append(m["measure_num"])

        # Short-bar guard (same as event_detect line 310: < 0.1s)
        duration = m["end"] - m["start"]
        if duration < 0.1:
            bar_spectra.append(np.full(n_bands, NOISE_FLOOR_DB))
            continue

        # Frame boundaries (same math as event_detect lines 326-328)
        start_frame = int(m["start"] * sr / hop_length)
        end_frame = min(int(m["end"] * sr / hop_length), S.shape[1])

        if end_frame <= start_frame:
            bar_spectra.append(np.full(n_bands, NOISE_FLOOR_DB))
            continue

        bar_S = S[:, start_frame:end_frame]
        bar_power = bar_S ** 2

        # Mean power across frames, then sum within each band
        mean_power_per_bin = bar_power.mean(axis=1)  # (n_fft_bins,)

        band_db = np.full(n_bands, NOISE_FLOOR_DB)
        for i, mask in enumerate(band_masks):
            if mask.any():
                band_power = mean_power_per_bin[mask].mean()
                db = 10.0 * np.log10(band_power + 1e-12)
                band_db[i] = max(db, NOISE_FLOOR_DB)

        bar_spectra.append(band_db)

    return {
        "bar_spectra": np.array(bar_spectra),
        "band_centers": centers.copy(),
        "band_edges": edges,
        "bar_times": bar_times,
        "bar_nums": bar_nums,
    }


def compute_spectral_summary(bar_spectra_result):
    """Compute summary statistics from bar spectra.

    Parameters
    ----------
    bar_spectra_result : dict
        Output of compute_bar_spectra.

    Returns
    -------
    dict with keys:
        mean_spectrum, peak_spectrum, dominant_band_idx, dominant_band_hz,
        spectral_centroid_hz, bandwidth_hz, energy_profile
    """
    spectra = bar_spectra_result["bar_spectra"]  # (n_bars, n_bands)
    centers = bar_spectra_result["band_centers"]

    mean_spectrum = spectra.mean(axis=0)
    peak_spectrum = spectra.max(axis=0)

    dominant_idx = int(np.argmax(mean_spectrum))
    dominant_hz = float(centers[dominant_idx])

    # Spectral centroid: energy-weighted mean frequency
    # Convert dB back to linear for weighting
    linear = 10.0 ** (mean_spectrum / 10.0)
    total = linear.sum()
    if total > 0:
        centroid = float(np.sum(centers * linear) / total)
    else:
        centroid = 0.0

    # Bandwidth: frequency range within 10 dB of peak
    peak_db = peak_spectrum.max()
    above_threshold = peak_spectrum >= (peak_db - 10.0)
    active_indices = np.where(above_threshold)[0]
    if len(active_indices) > 0:
        bw_low = float(centers[active_indices[0]])
        bw_high = float(centers[active_indices[-1]])
    else:
        bw_low, bw_high = 0.0, 0.0

    # Energy profile description
    if dominant_hz < 200:
        char = "Bass-heavy"
    elif dominant_hz < 1000:
        char = "Mid-focused"
    else:
        char = "Bright"
    profile = f"{char} (dominant {dominant_hz:.0f} Hz), active {bw_low:.0f}\u2013{bw_high:.0f} Hz"

    return {
        "mean_spectrum": [round(float(x), 1) for x in mean_spectrum],
        "peak_spectrum": [round(float(x), 1) for x in peak_spectrum],
        "dominant_band_idx": dominant_idx,
        "dominant_band_hz": dominant_hz,
        "spectral_centroid_hz": round(centroid, 1),
        "bandwidth_hz": [bw_low, bw_high],
        "energy_profile": profile,
    }


# ── Visualization ───────────────────────────────────────────────────────

def plot_spectrogram(bar_spectra_result, title="Spectral Energy",
                     save_path=None):
    """Plot a heatmap of per-bar spectral energy.

    X-axis = time (bar boundaries), Y-axis = frequency (log scale),
    color = dB.
    """
    import matplotlib.pyplot as plt

    spectra = bar_spectra_result["bar_spectra"]  # (n_bars, n_bands)
    edges = bar_spectra_result["band_edges"]     # (n_bands, 2)
    bar_times = bar_spectra_result["bar_times"]  # list of (start, end)

    # Build time edges from bar boundaries
    t_edges = [bar_times[0][0]] + [t[1] for t in bar_times]

    # Build frequency edges from band edges
    f_edges = [edges[0, 0]] + [edges[i, 1] for i in range(len(edges))]

    T, F = np.meshgrid(t_edges, f_edges)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18)

    pcm = ax.pcolormesh(T, F, spectra.T, shading="flat", cmap="inferno",
                        vmin=NOISE_FLOOR_DB, vmax=0)
    ax.set_yscale("log")
    ax.set_ylim(20, 20000)
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_yticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k"])

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("dB", fontsize=9)

    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    return fig


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-bar third-octave spectral analysis")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--save-plot", default=None,
                        help="Save spectrogram to PNG")
    parser.add_argument("--json-out", default=None,
                        help="Save spectral summary to JSON")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load audio + BPM detection (reuse bpm_detect)
    sys.path.insert(0, str(Path(__file__).parent))
    from bpm_detect import detect_beats, group_beats_into_measures

    print(f"Loading: {audio_path.name}")
    sr = 44100
    bpm_data = detect_beats(str(audio_path), sr=sr, hop_length=512)
    y = bpm_data["y"]
    beat_times = bpm_data["beat_times"]
    phase = bpm_data["downbeat_phase"]
    measures = group_beats_into_measures(beat_times, 4, phase=phase)
    print(f"  {len(measures)} bars, ~{bpm_data['tempo']:.1f} BPM")

    # Spectral analysis
    print("Computing bar spectra...")
    result = compute_bar_spectra(y, sr, measures)
    summary = compute_spectral_summary(result)

    print(f"\nSpectral Summary:")
    print(f"  Profile: {summary['energy_profile']}")
    print(f"  Centroid: {summary['spectral_centroid_hz']} Hz")
    print(f"  Dominant band: {summary['dominant_band_hz']} Hz")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON: {args.json_out}")

    # Plot
    import matplotlib.pyplot as plt
    fig = plot_spectrogram(result, title=f"Spectral Energy — {audio_path.stem}",
                           save_path=args.save_plot)
    if not args.save_plot:
        plt.show()


if __name__ == "__main__":
    main()
