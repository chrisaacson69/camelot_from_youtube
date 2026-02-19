#!/usr/bin/env python3
"""
visualize_timeline.py

Visualization tools for camelot_from_youtube timeline analysis.

Two modes:
1. Summary view: Key timeline with colored regions + energy event markers
2. Detailed view: Per-segment histograms of all metrics (RMS, onset, centroid, etc.)

Usage:
    python visualize_timeline.py output.json --mode summary
    python visualize_timeline.py output.json --mode detailed
    python visualize_timeline.py output.json --mode both
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Run: pip install matplotlib")


# Color scheme for Camelot codes
CAMELOT_COLORS = {
    # A codes (minor) - cooler colors
    "1A": "#1e3a5f",   # Ab minor - dark blue
    "2A": "#2e5984",   # Eb minor
    "3A": "#3d78a8",   # Bb minor
    "4A": "#4a97cc",   # F minor
    "5A": "#5bb5e0",   # C minor
    "6A": "#6bcfef",   # G minor - light blue
    "7A": "#5fa89e",   # D minor - teal
    "8A": "#52827d",   # A minor
    "9A": "#456b5c",   # E minor
    "10A": "#38543b",  # B minor - dark green
    "11A": "#4a6b3a",  # F# minor
    "12A": "#5c8239",  # C# minor

    # B codes (major) - warmer colors
    "1B": "#8b4513",   # B major - brown
    "2B": "#a0522d",   # F# major
    "3B": "#cd853f",   # Db major
    "4B": "#daa520",   # Ab major - gold
    "5B": "#ffd700",   # Eb major - yellow
    "6B": "#ffb347",   # Bb major - orange
    "7B": "#ff8c00",   # F major
    "8B": "#ff6347",   # C major - tomato
    "9B": "#dc143c",   # G major - crimson
    "10B": "#c71585",  # D major - violet red
    "11B": "#9932cc",  # A major - purple
    "12B": "#6a0dad",  # E major - dark purple
}

# Energy event colors
EVENT_COLORS = {
    "DROP": "#ff0000",
    "bass drop": "#ff4500",
    "build": "#ffa500",
    "drums in": "#32cd32",
    "drums out": "#228b22",
    "breakdown": "#4169e1",
    "energy drop": "#1e90ff",
    "bass in": "#ff6347",
    "bass out": "#8b0000",
    "new element": "#9400d3",
    "texture shift": "#da70d6",
    "change": "#808080",
}


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def get_camelot_color(camelot: str) -> str:
    """Get color for a Camelot code."""
    return CAMELOT_COLORS.get(camelot, "#cccccc")


def classify_energy_event(event_types: List[str]) -> str:
    """Convert list of event types to a human-readable label."""
    if "energy_increase" in event_types and "drums_increase" in event_types:
        return "DROP"
    elif "energy_increase" in event_types and "bass_increase" in event_types:
        return "bass drop"
    elif "energy_increase" in event_types:
        return "build"
    elif "drums_increase" in event_types:
        return "drums in"
    elif "energy_decrease" in event_types and "drums_decrease" in event_types:
        return "breakdown"
    elif "energy_decrease" in event_types:
        return "energy drop"
    elif "drums_decrease" in event_types:
        return "drums out"
    elif "bass_increase" in event_types:
        return "bass in"
    elif "bass_decrease" in event_types:
        return "bass out"
    elif "timbre_change" in event_types:
        return "new element"
    elif "texture_change" in event_types:
        return "texture shift"
    else:
        return "change"


def plot_summary_view(data: Dict[str, Any], save_path: Optional[str] = None):
    """
    Summary view: Key timeline with colored regions + energy event markers.

    Shows:
    - Top: Key regions as colored horizontal bars
    - Middle: Energy events as vertical markers
    - Bottom: Bar numbers / time axis
    """
    estimate = data.get("estimate", {})
    timeline = estimate.get("timeline", [])
    energy_events = estimate.get("energy_events", [])
    duration = estimate.get("duration", 0)
    bpm = estimate.get("bpm", 120)
    summary = estimate.get("summary", {})

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), height_ratios=[2, 1, 2],
                              gridspec_kw={'hspace': 0.3}, constrained_layout=True)

    # --- Top: Key Timeline ---
    ax_key = axes[0]
    ax_key.set_xlim(0, duration)
    ax_key.set_ylim(0, 1)
    ax_key.set_ylabel("Key")
    ax_key.set_title(f"Key Timeline - Dominant: {summary.get('dominant', {}).get('camelot', '?')} "
                     f"({summary.get('dominant', {}).get('key', '?')})")

    # Draw key regions as colored rectangles
    legend_entries = {}
    for entry in timeline:
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        camelot = entry.get("camelot", "?")
        seg_type = entry.get("type", "unknown")

        color = get_camelot_color(camelot)
        alpha = 0.9 if seg_type == "stable" else 0.5

        rect = Rectangle((start, 0), end - start, 1, facecolor=color, alpha=alpha,
                         edgecolor='white', linewidth=0.5)
        ax_key.add_patch(rect)

        # Add label in center of region
        if end - start > 20:  # Only label if wide enough
            ax_key.text((start + end) / 2, 0.5, camelot, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

        # Track for legend
        if camelot not in legend_entries:
            legend_entries[camelot] = color

    ax_key.set_yticks([])

    # --- Middle: Energy Events ---
    ax_energy = axes[1]
    ax_energy.set_xlim(0, duration)
    ax_energy.set_ylim(0, 1)
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy Events (Structural Changes)")

    event_legend = {}
    for event in energy_events:
        time = event.get("time", 0)
        event_types = event.get("types", [])
        label = classify_energy_event(event_types)
        color = EVENT_COLORS.get(label, "#808080")

        # Draw vertical line
        ax_energy.axvline(x=time, color=color, alpha=0.8, linewidth=2)

        # Add small label
        ax_energy.text(time, 0.9, label[:6], ha='center', va='top',
                      fontsize=7, rotation=45, color=color)

        if label not in event_legend:
            event_legend[label] = color

    ax_energy.set_yticks([])

    # --- Bottom: Metrics Overview ---
    ax_metrics = axes[2]
    raw_segments = estimate.get("raw_segments", [])

    if raw_segments:
        times = [s.get("start", 0) + (s.get("end", s.get("start", 0) + s.get("duration", 0)) - s.get("start", 0)) / 2
                 for s in raw_segments]
        rms = [s.get("energy", {}).get("rms_mean", 0) for s in raw_segments]
        onset = [s.get("energy", {}).get("onset_density", 0) for s in raw_segments]

        # Normalize for plotting
        rms_norm = np.array(rms) / max(rms) if max(rms) > 0 else rms
        onset_norm = np.array(onset) / max(onset) if max(onset) > 0 else onset

        ax_metrics.fill_between(times, 0, rms_norm, alpha=0.5, label='Energy (RMS)', color='blue')
        ax_metrics.plot(times, onset_norm, 'g-', alpha=0.7, label='Onset Density', linewidth=1.5)
        ax_metrics.legend(loc='upper right')

    ax_metrics.set_xlim(0, duration)
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.set_xlabel("Time (seconds)")
    ax_metrics.set_ylabel("Normalized")
    ax_metrics.set_title("Energy & Rhythm Overview")

    # Add bar grid lines (use bars_per_segment from analysis)
    bars_per_segment = estimate.get("bars_per_segment", 8)
    seconds_per_bar = (4 * 60.0) / bpm
    bar_times = np.arange(0, duration, seconds_per_bar * bars_per_segment)
    for ax in axes:
        for t in bar_times:
            ax.axvline(x=t, color='gray', alpha=0.2, linewidth=0.5)

    # Add time labels on x-axis
    for ax in axes[:-1]:
        ax.set_xticks([])
    time_ticks = np.arange(0, duration, 60)  # Every minute
    axes[-1].set_xticks(time_ticks)
    axes[-1].set_xticklabels([format_time(t) for t in time_ticks])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary view saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_detailed_view(data: Dict[str, Any], save_path: Optional[str] = None):
    """
    Detailed view: Per-segment histograms of all metrics.

    Shows stacked plots of:
    - Key confidence
    - RMS energy
    - Onset density
    - Spectral centroid
    - Low frequency ratio
    """
    estimate = data.get("estimate", {})
    raw_segments = estimate.get("raw_segments", [])
    duration = estimate.get("duration", 0)
    bpm = estimate.get("bpm", 120)

    if not raw_segments:
        print("No raw segment data available for detailed view")
        return

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={'hspace': 0.1}, constrained_layout=True)

    # Extract data
    times = [s.get("start", 0) for s in raw_segments]
    widths = [s.get("end", s.get("start", 0) + s.get("duration", 0)) - s.get("start", 0)
              for s in raw_segments]

    confidence = [s.get("confidence", 0) for s in raw_segments]
    rms = [s.get("energy", {}).get("rms_mean", 0) for s in raw_segments]
    rms_std = [s.get("energy", {}).get("rms_std", 0) for s in raw_segments]
    onset = [s.get("energy", {}).get("onset_density", 0) for s in raw_segments]
    centroid = [s.get("energy", {}).get("spectral_centroid", 0) for s in raw_segments]
    low_freq = [s.get("energy", {}).get("low_freq_ratio", 0) for s in raw_segments]

    # Get colors based on Camelot code
    colors = [get_camelot_color(s.get("camelot", "?")) for s in raw_segments]

    # --- Plot 0: Key with Confidence ---
    ax = axes[0]
    bars = ax.bar(times, confidence, width=widths, align='edge', color=colors, alpha=0.8)
    ax.set_ylabel("Key Conf.")
    ax.set_ylim(0, max(confidence) * 1.1 if confidence else 1)
    ax.set_title("Key Detection Confidence (color = Camelot code)")

    # Add Camelot labels
    for i, s in enumerate(raw_segments):
        if confidence[i] > 0.05:
            ax.text(times[i] + widths[i]/2, confidence[i], s.get("camelot", ""),
                   ha='center', va='bottom', fontsize=7, rotation=45)

    # --- Plot 1: RMS Energy ---
    ax = axes[1]
    ax.bar(times, rms, width=widths, align='edge', color='steelblue', alpha=0.7)
    ax.errorbar([t + w/2 for t, w in zip(times, widths)], rms, yerr=rms_std,
                fmt='none', color='darkblue', alpha=0.5, capsize=2)
    ax.set_ylabel("RMS Energy")
    ax.set_title("RMS Energy (with std dev)")

    # --- Plot 2: Onset Density ---
    ax = axes[2]
    ax.bar(times, onset, width=widths, align='edge', color='forestgreen', alpha=0.7)
    ax.set_ylabel("Onsets/sec")
    ax.set_title("Onset Density (rhythmic activity)")

    # --- Plot 3: Spectral Centroid ---
    ax = axes[3]
    ax.bar(times, centroid, width=widths, align='edge', color='darkorange', alpha=0.7)
    ax.set_ylabel("Centroid (Hz)")
    ax.set_title("Spectral Centroid (brightness/timbre)")

    # --- Plot 4: Low Frequency Ratio ---
    ax = axes[4]
    ax.bar(times, low_freq, width=widths, align='edge', color='darkred', alpha=0.7)
    ax.set_ylabel("Low Freq %")
    ax.set_title("Low Frequency Ratio (bass presence)")

    # --- Plot 5: Combined normalized view ---
    ax = axes[5]

    # Normalize all metrics to 0-1
    def norm(arr):
        arr = np.array(arr)
        if arr.max() > 0:
            return arr / arr.max()
        return arr

    x_centers = [t + w/2 for t, w in zip(times, widths)]
    ax.plot(x_centers, norm(rms), 'b-', label='RMS', alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(onset), 'g-', label='Onset', alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(centroid), 'orange', label='Centroid', alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(low_freq), 'r-', label='LowFreq', alpha=0.7, linewidth=1.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel("Normalized")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("All Metrics Normalized (for comparison)")

    # Add grid lines at segment boundaries (bars_per_segment from analysis)
    bars_per_segment = estimate.get("bars_per_segment", 8)
    seconds_per_bar = (4 * 60.0) / bpm
    bar_times = np.arange(0, duration, seconds_per_bar * bars_per_segment)
    for ax in axes:
        for t in bar_times:
            ax.axvline(x=t, color='gray', alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, duration)

    # Time axis labels
    time_ticks = np.arange(0, duration, 30)  # Every 30 seconds
    axes[-1].set_xticks(time_ticks)
    axes[-1].set_xticklabels([format_time(t) for t in time_ticks], rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detailed view saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return

    parser = argparse.ArgumentParser(description="Visualize timeline analysis results")
    parser.add_argument("json_file", help="Path to timeline JSON output file")
    parser.add_argument("--mode", choices=["summary", "detailed", "both"], default="both",
                        help="Visualization mode (default: both)")
    parser.add_argument("--save", help="Save plots to files (prefix, e.g., 'track1' -> track1_summary.png)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (default if --save not specified)")

    args = parser.parse_args()

    # Load JSON data
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if this is timeline data
    if data.get("analysis", {}).get("type") != "timeline":
        print("Warning: This doesn't appear to be timeline analysis output.")
        print("Run with --timeline flag: python camelot_from_youtube.py --audio file.wav --timeline")

    # Determine output mode
    show_plots = args.show or not args.save

    if args.mode in ["summary", "both"]:
        save_path = f"{args.save}_summary.png" if args.save else None
        plot_summary_view(data, save_path if args.save else None)
        if show_plots and not args.save:
            plot_summary_view(data, None)

    if args.mode in ["detailed", "both"]:
        save_path = f"{args.save}_detailed.png" if args.save else None
        plot_detailed_view(data, save_path if args.save else None)
        if show_plots and not args.save:
            plot_detailed_view(data, None)


if __name__ == "__main__":
    main()
