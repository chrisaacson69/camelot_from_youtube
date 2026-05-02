"""collision.py - Two-track spectral collision analysis.

Compare two tracks' spectral overlap in transition zones, identify
collision hotspots, and suggest EQ crossover frequencies.

Ported from cyborgdj/scripts/spectral_collision.py with additions:
- JSON output
- EQ recommendations for CyborgDJ specs
- Standalone CLI

Usage (standalone):
    python collision.py \
        --track-a "path/to/Track A project/" --bars-a 200-260 \
        --track-b "path/to/Track B project/" --bars-b 1-61 \
        [--json-out collision.json] [--save-plot collision.png]
"""

import json
import sys
from pathlib import Path

import numpy as np

from spectral_analysis import NOISE_FLOOR_DB


# ── Data loading ────────────────────────────────────────────────────────

def load_track_spectra(project_dir):
    """Load spectral data from a track's analysis cache.

    Parameters
    ----------
    project_dir : str or Path
        Path to the track's project directory containing
        analysis_cache.npz and analysis_cache.json.

    Returns
    -------
    dict with keys: bar_spectra, band_centers, band_edges, measures,
                    loudness_dbfs, lufs, name
    """
    d = Path(project_dir)
    npz_path = d / "analysis_cache.npz"
    json_path = d / "analysis_cache.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"No analysis_cache.npz in {d}")
    if not json_path.exists():
        raise FileNotFoundError(f"No analysis_cache.json in {d}")

    npz = np.load(str(npz_path), allow_pickle=False)

    if "bar_spectra" not in npz:
        raise ValueError(
            f"No spectral data in {npz_path}. "
            "Re-run analysis with spectral analysis enabled.")

    with open(json_path, "r") as f:
        data = json.load(f)

    result = {
        "bar_spectra": npz["bar_spectra"],
        "band_centers": npz["band_centers"],
        "band_edges": npz["band_edges"],
        "measures": data["measures"],
        "loudness_dbfs": data.get("loudness_dbfs"),
        "lufs": data.get("lufs"),
        "name": data.get("_audio_name", d.name),
    }
    npz.close()
    return result


def parse_bar_range(bar_spec, n_bars):
    """Parse '200-260' into start, end indices (0-based, end exclusive).

    Bar numbers in the spec are 1-based measure_nums.
    """
    parts = bar_spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bar range must be 'start-end', got '{bar_spec}'")
    start = int(parts[0]) - 1  # convert to 0-based index
    end = int(parts[1])        # end is exclusive in slice
    if start < 0 or end > n_bars or start >= end:
        raise ValueError(
            f"Bar range {bar_spec} out of bounds (track has {n_bars} bars)")
    return start, end


# ── Loudness normalization ──────────────────────────────────────────────

def _normalize_spectra(spectra, loudness_dbfs, lufs):
    """Normalize spectra to 0 dBFS peak reference."""
    ref = lufs if lufs is not None else loudness_dbfs
    if ref is None:
        return spectra
    offset = -ref
    normalized = spectra + offset
    return np.clip(normalized, NOISE_FLOOR_DB, None)


# ── Collision analysis ──────────────────────────────────────────────────

def compute_collision(spectra_a, spectra_b, bars_a, bars_b,
                      track_a_data=None, track_b_data=None):
    """Compute spectral collision between two track regions.

    Parameters
    ----------
    spectra_a, spectra_b : dict
        Output of load_track_spectra.
    bars_a, bars_b : str
        Bar range specs like '200-260'.
    track_a_data, track_b_data : dict, optional
        If provided, used for loudness normalization.

    Returns
    -------
    dict with keys:
        collision : np.ndarray (n_bars, n_bands)
        total_per_bar : np.ndarray
        total_per_band : np.ndarray
        hotspot_bars : list[int]
        hotspot_bands : list[int]
        bar_spectra_a, bar_spectra_b : np.ndarray
        band_centers, band_edges : np.ndarray
    """
    sa = spectra_a["bar_spectra"]
    sb = spectra_b["bar_spectra"]

    start_a, end_a = parse_bar_range(bars_a, sa.shape[0])
    start_b, end_b = parse_bar_range(bars_b, sb.shape[0])

    region_a = sa[start_a:end_a].copy()
    region_b = sb[start_b:end_b].copy()

    if region_a.shape[0] != region_b.shape[0]:
        raise ValueError(
            f"Bar ranges must be equal length: A has {region_a.shape[0]} bars, "
            f"B has {region_b.shape[0]} bars")

    if not np.allclose(spectra_a["band_centers"], spectra_b["band_centers"]):
        raise ValueError("Tracks have different band_centers")

    # Loudness normalization
    if track_a_data is not None:
        region_a = _normalize_spectra(
            region_a, track_a_data.get("loudness_dbfs"),
            track_a_data.get("lufs"))
    if track_b_data is not None:
        region_b = _normalize_spectra(
            region_b, track_b_data.get("loudness_dbfs"),
            track_b_data.get("lufs"))

    # Collision: min(dB_A, dB_B) - NOISE_FLOOR_DB
    collision = np.maximum(
        np.minimum(region_a, region_b) - NOISE_FLOOR_DB, 0.0)

    total_per_bar = collision.sum(axis=1)
    total_per_band = collision.sum(axis=0)

    bar_threshold = np.percentile(total_per_bar, 75) if len(total_per_bar) > 4 else 0
    band_threshold = np.percentile(total_per_band, 75) if len(total_per_band) > 4 else 0

    hotspot_bars = [int(i) for i in np.where(total_per_bar > bar_threshold)[0]]
    hotspot_bands = [int(i) for i in np.where(total_per_band > band_threshold)[0]]

    return {
        "collision": collision,
        "total_per_bar": total_per_bar,
        "total_per_band": total_per_band,
        "hotspot_bars": hotspot_bars,
        "hotspot_bands": hotspot_bands,
        "bar_spectra_a": region_a,
        "bar_spectra_b": region_b,
        "band_centers": spectra_a["band_centers"],
        "band_edges": spectra_a["band_edges"],
    }


def find_spectral_gaps(spectra_a, spectra_b, bars_a, bars_b,
                       track_a_data=None, track_b_data=None,
                       min_gap_db=10.0):
    """Find frequency bands where one track dominates by >= min_gap_db.

    Returns
    -------
    dict with keys:
        gaps : list[dict] — frequency_hz, a_dominates, mean_gap_db, consistency
        suggested_crossovers : list[dict] — frequency_hz, below_dominant,
                               above_dominant, confidence
    """
    sa = spectra_a["bar_spectra"]
    sb = spectra_b["bar_spectra"]

    start_a, end_a = parse_bar_range(bars_a, sa.shape[0])
    start_b, end_b = parse_bar_range(bars_b, sb.shape[0])

    region_a = sa[start_a:end_a].copy()
    region_b = sb[start_b:end_b].copy()

    if region_a.shape[0] != region_b.shape[0]:
        raise ValueError("Bar ranges must be equal length")

    if track_a_data is not None:
        region_a = _normalize_spectra(
            region_a, track_a_data.get("loudness_dbfs"),
            track_a_data.get("lufs"))
    if track_b_data is not None:
        region_b = _normalize_spectra(
            region_b, track_b_data.get("loudness_dbfs"),
            track_b_data.get("lufs"))

    centers = spectra_a["band_centers"]
    n_bands = region_a.shape[1]

    gaps = []
    for i in range(n_bands):
        diff = region_a[:, i] - region_b[:, i]
        mean_diff = float(np.mean(diff))
        abs_gap = abs(mean_diff)

        if abs_gap >= min_gap_db:
            a_dom = mean_diff > 0
            if a_dom:
                consistency = float(np.mean(diff > 0))
            else:
                consistency = float(np.mean(diff < 0))

            gaps.append({
                "frequency_hz": float(centers[i]),
                "band_idx": i,
                "a_dominates": a_dom,
                "mean_gap_db": round(abs_gap, 1),
                "consistency": round(consistency, 2),
            })

    # Find crossover suggestions
    suggested_crossovers = []
    for j in range(len(gaps) - 1):
        g1 = gaps[j]
        g2 = gaps[j + 1]
        if g1["a_dominates"] != g2["a_dominates"]:
            xover_hz = float(np.sqrt(g1["frequency_hz"] * g2["frequency_hz"]))
            confidence = min(g1["consistency"], g2["consistency"])
            below = "A" if g1["a_dominates"] else "B"
            above = "A" if g2["a_dominates"] else "B"
            suggested_crossovers.append({
                "frequency_hz": round(xover_hz, 0),
                "below_dominant": below,
                "above_dominant": above,
                "confidence": round(confidence, 2),
                "gap_below_db": g1["mean_gap_db"],
                "gap_above_db": g2["mean_gap_db"],
            })

    return {
        "gaps": gaps,
        "suggested_crossovers": suggested_crossovers,
    }


# ── EQ Recommendations ─────────────────────────────────────────────────

def generate_eq_recommendations(gap_result, collision_result,
                                name_a, name_b):
    """Generate actionable EQ recommendations for a transition.

    Returns
    -------
    dict with keys:
        strategy : str — "bass_swap", "full_eq", "simple_fade"
        bass_handoff_hz : float or None
        incoming_eq : dict — CyborgDJ eq_3band format
        rationale : str
    """
    xovers = gap_result["suggested_crossovers"]
    gaps = gap_result["gaps"]

    # Find bass-range crossover (below 400 Hz)
    bass_xover = None
    for x in xovers:
        if x["frequency_hz"] <= 400 and x["confidence"] >= 0.5:
            bass_xover = x
            break

    # Check if there's significant bass collision
    centers = collision_result["band_centers"]
    band_totals = collision_result["total_per_band"]
    bass_mask = centers < 200
    bass_collision = float(band_totals[bass_mask].sum()) if bass_mask.any() else 0
    total_collision = float(band_totals.sum())
    bass_fraction = bass_collision / total_collision if total_collision > 0 else 0

    if bass_xover and bass_fraction > 0.15:
        # Bass swap strategy
        hz = bass_xover["frequency_hz"]
        strategy = "bass_swap"
        incoming_eq = {
            "low": [0.0, 1.0],
            "mid": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low_freq": int(hz),
        }
        rationale = (
            f"Outgoing track ({name_a}) dominates below {hz:.0f} Hz. "
            f"Bass swap recommended: kill incoming bass initially, ramp to full. "
            f"Bass accounts for {bass_fraction:.0%} of total collision."
        )
    elif total_collision > 0 and bass_fraction > 0.3:
        # High bass collision but no clear crossover
        strategy = "bass_swap"
        hz = 200  # Default
        incoming_eq = {
            "low": [0.0, 1.0],
            "mid": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low_freq": int(hz),
        }
        rationale = (
            f"Significant bass collision ({bass_fraction:.0%} of total) "
            f"but no clear crossover point. Defaulting to 200 Hz bass swap."
        )
    else:
        # Low collision — simple fade may suffice
        strategy = "simple_fade"
        hz = None
        incoming_eq = None
        rationale = (
            f"Low bass collision ({bass_fraction:.0%} of total). "
            f"Simple equal-power crossfade should work."
        )

    return {
        "strategy": strategy,
        "bass_handoff_hz": hz,
        "incoming_eq": incoming_eq,
        "rationale": rationale,
    }


# ── Reporting ───────────────────────────────────────────────────────────

def collision_report(collision_result, gap_result, name_a, name_b,
                     eq_recs=None):
    """Generate a human-readable collision report.

    Returns
    -------
    str
    """
    lines = []
    lines.append(f"Spectral Collision Report: {name_a} vs {name_b}")
    lines.append("=" * 60)

    col = collision_result
    centers = col["band_centers"]

    total = float(col["collision"].sum())
    lines.append(f"\nTotal collision score: {total:.0f}")
    lines.append(f"Hotspot bars (top 25%): {len(col['hotspot_bars'])} of "
                 f"{col['collision'].shape[0]}")

    band_totals = col["total_per_band"]
    top_band_idx = np.argsort(band_totals)[::-1][:5]
    lines.append("\nTop 5 colliding frequency bands:")
    for idx in top_band_idx:
        lines.append(f"  {centers[idx]:>8.0f} Hz  score={band_totals[idx]:.0f}")

    bar_totals = col["total_per_bar"]
    top_bar_idx = np.argsort(bar_totals)[::-1][:5]
    lines.append(f"\nTop 5 colliding bars (0-indexed within region):")
    for idx in top_bar_idx:
        lines.append(f"  Bar {idx:>3d}  score={bar_totals[idx]:.0f}")

    gaps = gap_result["gaps"]
    lines.append(f"\nSpectral gaps (>= threshold): {len(gaps)} bands")
    for g in gaps:
        dom = name_a if g["a_dominates"] else name_b
        lines.append(
            f"  {g['frequency_hz']:>8.0f} Hz  {dom} dominates by "
            f"{g['mean_gap_db']:.1f} dB ({g['consistency']:.0%} consistent)")

    xovers = gap_result["suggested_crossovers"]
    if xovers:
        lines.append(f"\nSuggested crossover frequencies:")
        for x in xovers:
            below_name = name_a if x["below_dominant"] == "A" else name_b
            above_name = name_a if x["above_dominant"] == "A" else name_b
            lines.append(
                f"  {x['frequency_hz']:.0f} Hz  "
                f"(below: {below_name}, above: {above_name}, "
                f"confidence: {x['confidence']:.0%})")
    else:
        lines.append("\nNo clear crossover suggestions found.")

    if eq_recs:
        lines.append(f"\nEQ Recommendation:")
        lines.append(f"  Strategy: {eq_recs['strategy']}")
        if eq_recs["bass_handoff_hz"]:
            lines.append(f"  Bass handoff: {eq_recs['bass_handoff_hz']} Hz")
        lines.append(f"  {eq_recs['rationale']}")

    return "\n".join(lines)


def collision_to_json(collision_result, gap_result, eq_recs,
                      name_a, name_b, bars_a, bars_b):
    """Export collision analysis as a JSON-serializable dict.

    Returns
    -------
    dict suitable for json.dumps()
    """
    col = collision_result
    centers = col["band_centers"]

    # Top colliding bands
    band_totals = col["total_per_band"]
    top_band_idx = np.argsort(band_totals)[::-1][:5]
    top_bands = [
        {"frequency_hz": float(centers[i]), "score": round(float(band_totals[i]), 1)}
        for i in top_band_idx
    ]

    return {
        "track_a": {"name": name_a, "bars": bars_a},
        "track_b": {"name": name_b, "bars": bars_b},
        "collision_score": round(float(col["collision"].sum()), 1),
        "n_bars": int(col["collision"].shape[0]),
        "hotspot_bars": col["hotspot_bars"],
        "hotspot_bands_hz": [float(centers[i]) for i in col["hotspot_bands"]],
        "top_colliding_bands": top_bands,
        "per_bar_collision": [round(float(v), 1) for v in col["total_per_bar"]],
        "spectral_gaps": gap_result["gaps"],
        "suggested_crossovers": gap_result["suggested_crossovers"],
        "eq_recommendation": eq_recs,
    }


# ── Visualization ───────────────────────────────────────────────────────

def plot_collision(collision_result, gap_result, name_a, name_b,
                   save_path=None):
    """Three-panel collision figure."""
    import matplotlib.pyplot as plt

    col = collision_result
    centers = col["band_centers"]
    edges = col["band_edges"]
    n_bars = col["collision"].shape[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={"height_ratios": [4, 2, 2]})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.06,
                        hspace=0.35)

    t_edges = np.arange(n_bars + 1)
    f_edges = [edges[0, 0]] + [edges[i, 1] for i in range(len(edges))]
    T, F = np.meshgrid(t_edges, f_edges)

    pcm = ax1.pcolormesh(T, F, col["collision"].T, shading="flat",
                         cmap="YlOrRd", vmin=0,
                         vmax=np.percentile(col["collision"], 95))
    ax1.set_yscale("log")
    ax1.set_ylim(20, 20000)
    ax1.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax1.set_yticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k"])

    for x in gap_result["suggested_crossovers"]:
        ax1.axhline(y=x["frequency_hz"], color="cyan", linewidth=1.5,
                     linestyle="--", alpha=0.8)
        ax1.text(n_bars * 0.02, x["frequency_hz"] * 1.1,
                 f'{x["frequency_hz"]:.0f} Hz',
                 color="cyan", fontsize=8, fontweight="bold")

    cb = fig.colorbar(pcm, ax=ax1, pad=0.02)
    cb.set_label("Collision Score", fontsize=8)
    ax1.set_xlabel("Bar (within overlap region)", fontsize=9)
    ax1.set_ylabel("Frequency (Hz)", fontsize=9)
    ax1.set_title(f"Spectral Collision: {name_a} vs {name_b}", fontsize=10)
    ax1.tick_params(labelsize=8)

    band_labels = [f"{c:.0f}" if c < 1000 else f"{c/1000:.1f}k"
                   for c in centers]
    y_pos = np.arange(len(centers))
    ax2.barh(y_pos, col["total_per_band"], color="#cc6633", alpha=0.8)
    ax2.set_yticks(y_pos[::3])
    ax2.set_yticklabels([band_labels[i] for i in range(0, len(centers), 3)],
                         fontsize=7)
    ax2.set_xlabel("Total Collision", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    ax2.set_title("Collision by Frequency Band", fontsize=10)
    ax2.tick_params(labelsize=8)

    ax3.plot(np.arange(n_bars), col["total_per_bar"],
             color="#336699", linewidth=1.2)
    ax3.fill_between(np.arange(n_bars), col["total_per_bar"],
                     alpha=0.3, color="#336699")
    ax3.set_xlabel("Bar (within overlap region)", fontsize=9)
    ax3.set_ylabel("Total Collision", fontsize=9)
    ax3.set_title("Collision Over Time", fontsize=10)
    ax3.tick_params(labelsize=8)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    return fig


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Spectral collision analysis between two tracks")
    parser.add_argument("--track-a", required=True,
                        help="Path to Track A project directory")
    parser.add_argument("--bars-a", required=True,
                        help="Bar range for Track A (e.g. '200-260')")
    parser.add_argument("--track-b", required=True,
                        help="Path to Track B project directory")
    parser.add_argument("--bars-b", required=True,
                        help="Bar range for Track B (e.g. '1-61')")
    parser.add_argument("--save-plot", default=None,
                        help="Save collision plot to PNG")
    parser.add_argument("--min-gap-db", type=float, default=10.0,
                        help="Minimum dB gap to consider a spectral gap")
    parser.add_argument("--json-out", default=None,
                        help="Write JSON output to file")
    args = parser.parse_args()

    print(f"Loading Track A: {args.track_a}")
    data_a = load_track_spectra(args.track_a)
    print(f"  {data_a['name']} -- {data_a['bar_spectra'].shape[0]} bars")

    print(f"Loading Track B: {args.track_b}")
    data_b = load_track_spectra(args.track_b)
    print(f"  {data_b['name']} -- {data_b['bar_spectra'].shape[0]} bars")

    print(f"\nComputing collision: bars {args.bars_a} vs {args.bars_b}")
    col = compute_collision(data_a, data_b, args.bars_a, args.bars_b,
                            track_a_data=data_a, track_b_data=data_b)

    print("Finding spectral gaps...")
    gaps = find_spectral_gaps(data_a, data_b, args.bars_a, args.bars_b,
                              track_a_data=data_a, track_b_data=data_b,
                              min_gap_db=args.min_gap_db)

    eq_recs = generate_eq_recommendations(gaps, col,
                                          data_a["name"], data_b["name"])

    report = collision_report(col, gaps, data_a["name"], data_b["name"],
                              eq_recs=eq_recs)
    print(f"\n{report}")

    if args.json_out:
        result = collision_to_json(col, gaps, eq_recs,
                                   data_a["name"], data_b["name"],
                                   args.bars_a, args.bars_b)
        Path(args.json_out).write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nJSON saved to: {args.json_out}")

    if args.save_plot:
        plot_collision(col, gaps, data_a["name"], data_b["name"],
                       save_path=args.save_plot)


if __name__ == "__main__":
    main()
