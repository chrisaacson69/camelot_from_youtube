#!/usr/bin/env python3
"""
event_detect.py - Structural event detection from audio.

Two analysis modes:
  1. Stem mode (--stems-dir or auto-separate): Computes phrase-level presence
     for each stem (drums, bass, vocals, other) and detects where stems
     turn on/off.  Gives labeled events like "drums in", "bass out".
  2. Feature mode (no stems): Uses onset strength, RMS, spectral centroid,
     and low-frequency ratio to find structural boundaries.

Both modes work at phrase granularity (default 4 bars) to avoid bar-by-bar
chatter.  Events are the large discrete moments — a small number of high-
confidence markers suitable for DJ cue points.

Usage:
    python event_detect.py "audio.wav"
    python event_detect.py "audio.wav" --stems-dir "audio_stems"
    python event_detect.py "audio.wav" --auto-separate
    python event_detect.py "audio.wav" --save-plot events.png
    python event_detect.py "audio.wav" --json-out events.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bpm_detect import detect_beats, group_beats_into_measures, format_time


# ---------------------------------------------------------------------------
# Stem loading and auto-separation
# ---------------------------------------------------------------------------

STEM_NAMES = ["drums", "bass", "vocals", "other"]


def find_or_create_stems(audio_path, stems_dir=None, auto_separate=False):
    """Find existing stems or run Demucs to create them.

    Returns (stems_dir_path, stems_dict) or (None, None) if stems unavailable.
    stems_dict maps stem name -> audio path.
    """
    audio_path = Path(audio_path)

    # Check explicit stems dir
    if stems_dir:
        stems_dir = Path(stems_dir)
        if stems_dir.exists():
            stems = _load_stem_paths(stems_dir)
            if stems:
                return stems_dir, stems
            print(f"Warning: --stems-dir {stems_dir} exists but no stem .wav files found",
                  file=sys.stderr)
        else:
            print(f"Warning: --stems-dir {stems_dir} not found", file=sys.stderr)

    # Check conventional location: <audio_stem>_stems/
    default_dir = audio_path.parent / f"{audio_path.stem}_stems"
    if default_dir.exists():
        stems = _load_stem_paths(default_dir)
        if stems:
            print(f"Found existing stems in {default_dir}", file=sys.stderr)
            return default_dir, stems

    # Auto-separate if requested
    if auto_separate:
        return _run_demucs(audio_path, default_dir)

    return None, None


def _load_stem_paths(stems_dir):
    """Check for stem wav files in directory, return dict of found stems."""
    stems = {}
    for name in STEM_NAMES:
        p = stems_dir / f"{name}.wav"
        if p.exists():
            stems[name] = str(p)
    return stems if stems else None


def _run_demucs(audio_path, output_dir):
    """Shell out to separate_stems.py using env_demucs."""
    # Find the demucs environment
    repo_dir = Path(__file__).parent
    demucs_python = repo_dir / "env_demucs" / "Scripts" / "python.exe"
    sep_script = repo_dir / "separate_stems.py"

    if not demucs_python.exists():
        print(f"Warning: env_demucs not found at {demucs_python}", file=sys.stderr)
        print("  Run stem separation manually or install env_demucs.", file=sys.stderr)
        print("  Falling back to feature-only mode.", file=sys.stderr)
        return None, None

    if not sep_script.exists():
        print(f"Warning: separate_stems.py not found", file=sys.stderr)
        return None, None

    print(f"Auto-separating stems with Demucs...", file=sys.stderr)
    print(f"  This may take a few minutes.", file=sys.stderr)

    try:
        result = subprocess.run(
            [str(demucs_python), str(sep_script),
             "--audio", str(audio_path),
             "--output-dir", str(output_dir)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"Demucs failed: {result.stderr[:500]}", file=sys.stderr)
            return None, None

        stems = _load_stem_paths(output_dir)
        if stems:
            print(f"Stems created in {output_dir}", file=sys.stderr)
            return output_dir, stems
        else:
            print(f"Demucs ran but no stems found in {output_dir}", file=sys.stderr)
            return None, None

    except subprocess.TimeoutExpired:
        print("Demucs timed out (>10 min)", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error running Demucs: {e}", file=sys.stderr)
        return None, None


# ---------------------------------------------------------------------------
# Stem analysis: phrase-level presence
# ---------------------------------------------------------------------------

def compute_stem_phrase_energy(stem_audio, sr, measures, phrase_bars=4):
    """Compute RMS energy per phrase for a single stem.

    Returns list of floats, one per phrase.
    """
    energies = []
    for i in range(0, len(measures), phrase_bars):
        group = measures[i:i + phrase_bars]
        if not group:
            continue
        start_sample = int(group[0]["start"] * sr)
        end_sample = int(group[-1]["end"] * sr)
        segment = stem_audio[start_sample:end_sample]
        if len(segment) > 0:
            energies.append(float(np.sqrt(np.mean(segment ** 2))))
        else:
            energies.append(0.0)
    return energies


def compute_stem_presence(stem_energies_by_name, presence_threshold=0.15):
    """Classify each stem as present/absent per phrase.

    Uses per-stem adaptive threshold: a stem is "present" when its energy
    is above `presence_threshold` fraction of its own max energy.

    Returns dict of stem_name -> list of booleans (one per phrase).
    """
    presence = {}
    for stem_name, energies in stem_energies_by_name.items():
        arr = np.array(energies)
        max_e = arr.max()
        if max_e < 1e-8:
            # Stem is always silent
            presence[stem_name] = [False] * len(energies)
        else:
            threshold = max_e * presence_threshold
            presence[stem_name] = [bool(e > threshold) for e in arr]
    return presence


def detect_stem_events(stem_presence, phrases, min_gap_phrases=2):
    """Detect events where stems turn on or off.

    An event occurs at phrase i when any stem's presence differs from
    phrase i-1.  Multiple simultaneous stem changes at the same phrase
    are grouped into one event.

    Returns list of event dicts.
    """
    n_phrases = len(phrases)
    stem_names = list(stem_presence.keys())

    raw_events = []
    for i in range(1, n_phrases):
        changes = []
        for stem in stem_names:
            prev = stem_presence[stem][i - 1]
            curr = stem_presence[stem][i]
            if prev != curr:
                direction = "in" if curr else "out"
                label = "melodic" if stem in ("vocals", "other") else stem
                changes.append({"stem": stem, "label": label, "direction": direction})

        if changes:
            raw_events.append({
                "phrase_idx": i,
                "bar": phrases[i]["start_bar"],
                "time": phrases[i]["start"],
                "changes": changes,
            })

    if not raw_events:
        return []

    # Merge events that are too close (keep the one with more stem changes)
    merged = []
    for evt in raw_events:
        if merged and (evt["phrase_idx"] - merged[-1]["phrase_idx"]) < min_gap_phrases:
            # Merge into previous if this has more changes
            if len(evt["changes"]) > len(merged[-1]["changes"]):
                merged[-1] = evt
            else:
                # Add the extra changes to existing event
                merged[-1]["changes"].extend(evt["changes"])
        else:
            merged.append(evt)

    # Build final events
    events = []
    for evt in merged:
        changes = evt["changes"]

        # Classify the event type
        ins = [c for c in changes if c["direction"] == "in"]
        outs = [c for c in changes if c["direction"] == "out"]

        if len(outs) >= 2 and any(c["stem"] == "drums" for c in outs):
            event_type = "breakdown"
        elif len(ins) >= 2 and any(c["stem"] == "drums" for c in ins):
            event_type = "drop"
        elif len(ins) >= 2:
            event_type = "build"
        elif len(outs) >= 2:
            event_type = "breakdown"
        else:
            # Single stem change
            c = changes[0]
            event_type = f"{c['label']}_{c['direction']}"

        # Build description from individual changes
        desc_parts = []
        for c in changes:
            desc_parts.append(f"{c['label']} {c['direction']}")
        description = ", ".join(desc_parts)

        # Score: more stems changing = higher score
        score = min(1.0, len(changes) * 0.3 + 0.1)

        events.append({
            "bar": evt["bar"],
            "time": round(evt["time"], 2),
            "time_fmt": format_time(evt["time"]),
            "type": event_type,
            "score": round(score, 3),
            "n_stems": len(changes),
            "description": description,
            "stems_in": [c["label"] for c in ins],
            "stems_out": [c["label"] for c in outs],
        })

    return events


# ---------------------------------------------------------------------------
# Feature-only analysis (fallback when no stems)
# ---------------------------------------------------------------------------

def compute_bar_features(y, sr, measures):
    """Compute audio features per bar, including HPSS-split onsets.

    Features per bar:
      - onset_strength: full-mix onset (overall rhythmic activity)
      - onset_percussive: percussive-only onset (drums/hits)
      - onset_harmonic: harmonic-only onset (melodic/tonal changes)
      - rms: full-mix RMS energy
      - centroid: spectral centroid (brightness)
      - low_freq_ratio: energy below 250Hz / total
    """
    # HPSS separation for split onset envelopes
    print("  HPSS separation...", file=sys.stderr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Onset envelopes: full, percussive, harmonic
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_perc = librosa.onset.onset_strength(y=y_percussive, sr=sr)
    onset_harm = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # STFT for spectral features (full mix)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    results = []
    for m in measures:
        start_sample = int(m["start"] * sr)
        end_sample = int(m["end"] * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.1:
            results.append(None)
            continue

        rms = float(np.sqrt(np.mean(segment ** 2)))

        bar_mask = (onset_times >= m["start"]) & (onset_times < m["end"])
        onset_vals = onset_env[bar_mask]
        onset_mean = float(np.mean(onset_vals)) if len(onset_vals) > 0 else 0.0

        perc_vals = onset_perc[bar_mask]
        perc_mean = float(np.mean(perc_vals)) if len(perc_vals) > 0 else 0.0

        harm_vals = onset_harm[bar_mask]
        harm_mean = float(np.mean(harm_vals)) if len(harm_vals) > 0 else 0.0

        hop = 512
        start_frame = int(m["start"] * sr / hop)
        end_frame = min(int(m["end"] * sr / hop), S.shape[1])

        if end_frame > start_frame:
            bar_S = S[:, start_frame:end_frame]
            bar_power = bar_S ** 2
            total_power = bar_power.sum()
            if total_power > 1e-12:
                centroid = float(np.sum(freqs[:, None] * bar_power) / total_power)
                low_mask = freqs < 250
                low_ratio = float(bar_power[low_mask].sum() / total_power)
            else:
                centroid, low_ratio = 0.0, 0.0
        else:
            centroid, low_ratio = 0.0, 0.0

        results.append({
            "bar_num": m["measure_num"],
            "start": m["start"],
            "end": m["end"],
            "onset_strength": onset_mean,
            "onset_percussive": perc_mean,
            "onset_harmonic": harm_mean,
            "rms": rms,
            "centroid": centroid,
            "low_freq_ratio": low_ratio,
        })
    return results


def compute_phrase_features(bar_features, phrase_bars=4):
    """Average bar features into phrase-level features."""
    phrases = []
    valid_bars = [b for b in bar_features if b is not None]

    for i in range(0, len(valid_bars), phrase_bars):
        group = valid_bars[i:i + phrase_bars]
        if not group:
            continue
        phrases.append({
            "phrase_num": len(phrases) + 1,
            "start_bar": group[0]["bar_num"],
            "end_bar": group[-1]["bar_num"],
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "bar_count": len(group),
            "onset_strength": float(np.mean([b["onset_strength"] for b in group])),
            "onset_percussive": float(np.mean([b["onset_percussive"] for b in group])),
            "onset_harmonic": float(np.mean([b["onset_harmonic"] for b in group])),
            "rms": float(np.mean([b["rms"] for b in group])),
            "centroid": float(np.mean([b["centroid"] for b in group])),
            "low_freq_ratio": float(np.mean([b["low_freq_ratio"] for b in group])),
        })
    return phrases


def normalize_feature(values):
    """Normalize to 0-1 range."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def detect_feature_events(phrases, context=2, min_score=0.15,
                           min_features=2, min_gap_phrases=2,
                           bar_features=None, min_onset_drop=0.40,
                           bar_context=2):
    """Feature-only event detection (no stems) — 2-pass approach.

    Pass 1 (phrase-level): Multi-feature structural boundaries where several
        audio features change simultaneously.  Catches major transitions.
    Pass 2 (bar-level): Onset strength drops at bar granularity.  Catches
        fills, element entrances, and short events that get smoothed away
        at phrase scale.  Only runs if bar_features are provided.

    Events from pass 2 are only kept if they are not too close to an
    existing pass 1 event (i.e., they add new information).
    """
    events = []

    # ---------------------------------------------------------------
    # Pass 1: Phrase-level multi-feature change detection
    # ---------------------------------------------------------------
    # Use HPSS-split onsets (percussive/harmonic) instead of combined onset,
    # giving better separation of "drums changed" vs "melody changed".
    if len(phrases) >= context + 1:
        feature_names = ["onset_percussive", "onset_harmonic", "rms",
                         "centroid", "low_freq_ratio"]
        raw = {f: np.array([p[f] for p in phrases]) for f in feature_names}
        normed = {f: normalize_feature(raw[f]) for f in feature_names}
        weights = {
            "onset_percussive": 0.25,  # drums/hits
            "onset_harmonic": 0.20,    # melodic/tonal
            "rms": 0.25,               # overall energy
            "centroid": 0.15,          # brightness/timbre
            "low_freq_ratio": 0.15,    # bass presence
        }

        changes = []
        for i in range(context, len(phrases)):
            feature_deltas = {}
            weighted_change = 0.0
            for f in feature_names:
                before = np.mean(normed[f][max(0, i - context):i])
                current = normed[f][i]
                delta = current - before
                feature_deltas[f] = float(delta)
                weighted_change += abs(delta) * weights[f]

            n_changed = sum(1 for f in feature_names
                            if abs(feature_deltas[f]) > 0.15)

            changes.append({
                "phrase_idx": i,
                "bar": phrases[i]["start_bar"],
                "time": phrases[i]["start"],
                "score": float(weighted_change),
                "n_features": n_changed,
                "deltas": feature_deltas,
            })

        # Threshold
        candidates = [c for c in changes
                      if c["score"] >= min_score and c["n_features"] >= min_features]
        if not candidates:
            candidates = [c for c in changes
                          if c["score"] >= min_score and c["n_features"] >= 1]

        # Keep strongest, enforce gap
        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            kept = []
            used = set()
            for c in candidates:
                if not any(abs(c["phrase_idx"] - u) < min_gap_phrases for u in used):
                    kept.append(c)
                    used.add(c["phrase_idx"])
            kept.sort(key=lambda c: c["time"])

            for c in kept:
                d = c["deltas"]
                perc_dir = d["onset_percussive"]
                harm_dir = d["onset_harmonic"]
                rms_dir = d["rms"]

                # Direction from dominant signals
                avg_onset = (perc_dir + harm_dir) / 2
                if avg_onset < -0.1 and rms_dir < -0.1:
                    direction = "decrease"
                elif avg_onset > 0.1 and rms_dir > 0.1:
                    direction = "increase"
                elif abs(avg_onset) > abs(rms_dir):
                    direction = "decrease" if avg_onset < 0 else "increase"
                elif abs(rms_dir) > 0.1:
                    direction = "decrease" if rms_dir < 0 else "increase"
                else:
                    direction = "change"

                desc = []
                labels = {
                    "onset_percussive": "perc",
                    "onset_harmonic": "melodic",
                    "rms": "energy",
                    "centroid": "brightness",
                    "low_freq_ratio": "bass",
                }
                for f, delta in d.items():
                    if abs(delta) > 0.15:
                        desc.append(f"{labels[f]}{'+'if delta>0 else '-'}")

                events.append({
                    "bar": c["bar"],
                    "time": round(c["time"], 2),
                    "time_fmt": format_time(c["time"]),
                    "type": direction,
                    "score": round(c["score"], 3),
                    "n_features": c["n_features"],
                    "pass": 1,
                    "description": ", ".join(desc) if desc else "subtle change",
                })

    # ---------------------------------------------------------------
    # Pass 2: Bar-level onset drop scan (percussive + harmonic)
    # ---------------------------------------------------------------
    if bar_features is not None:
        valid_bars = [b for b in bar_features if b is not None]
        if len(valid_bars) > bar_context * 3:
            # Scan both percussive and harmonic onsets at bar level
            perc_vals = np.array([b["onset_percussive"] for b in valid_bars])
            harm_vals = np.array([b["onset_harmonic"] for b in valid_bars])
            perc_norm = normalize_feature(perc_vals)
            harm_norm = normalize_feature(harm_vals)

            # Bar numbers from pass 1 events (to enforce gap)
            p1_bars = {e["bar"] for e in events}

            bar_events = []

            for signal_name, onset_norm in [("perc", perc_norm),
                                             ("harmonic", harm_norm)]:
                for i in range(bar_context, len(valid_bars) - bar_context):
                    before = np.mean(onset_norm[i - bar_context:i])
                    current = onset_norm[i]
                    after = np.mean(onset_norm[i + 1:i + bar_context + 1])

                    if before < 0.05:
                        continue  # Skip if already low

                    drop_pct = (before - current) / before

                    if drop_pct >= min_onset_drop:
                        abs_drop = before - current
                        recovery = (after - current) / abs_drop if abs_drop > 0.01 else 0
                        recovery = float(np.clip(recovery, 0, 2))

                        # Classify based on which signal and drop/recovery
                        if signal_name == "perc":
                            if drop_pct >= 0.75 and recovery < 0.50:
                                etype = "breakdown"
                                edesc = "perc drop (stays low)"
                            elif drop_pct >= 0.50 and recovery > 0.50:
                                etype = "fill"
                                edesc = f"perc drop {drop_pct:.0%}, recovery {recovery:.0%}"
                            else:
                                etype = "perc_event"
                                edesc = f"perc drop {drop_pct:.0%}"
                        else:  # harmonic
                            if drop_pct >= 0.75 and recovery < 0.50:
                                etype = "melodic_out"
                                edesc = "melodic drop (stays low)"
                            elif drop_pct >= 0.50 and recovery > 0.50:
                                etype = "melodic_change"
                                edesc = f"melodic drop {drop_pct:.0%}, recovery {recovery:.0%}"
                            else:
                                etype = "melodic_event"
                                edesc = f"melodic drop {drop_pct:.0%}"

                        bar_num = valid_bars[i]["bar_num"]
                        bar_events.append({
                            "bar": bar_num,
                            "time": round(valid_bars[i]["start"], 2),
                            "time_fmt": format_time(valid_bars[i]["start"]),
                            "type": etype,
                            "score": round(float(drop_pct), 3),
                            "n_features": 1,
                            "pass": 2,
                            "description": edesc,
                            "_bar_idx": i,
                            "_signal": signal_name,
                        })

            # Deduplicate bar-level events (keep strongest per 3-bar window)
            bar_events.sort(key=lambda e: e["score"], reverse=True)
            kept_bar = []
            used_bar_idx = set()
            for be in bar_events:
                idx = be["_bar_idx"]
                if any(abs(idx - u) < 3 for u in used_bar_idx):
                    continue
                used_bar_idx.add(idx)
                kept_bar.append(be)

            # Only keep bar events that are far enough from pass 1 events
            # "Far enough" = more than phrase_bars / 2 bars away
            phrase_bars = (phrases[0]["end_bar"] - phrases[0]["start_bar"] + 1
                           if phrases else 4)
            gap_bars = max(phrase_bars // 2, 2)

            for be in kept_bar:
                too_close = any(abs(be["bar"] - p1bar) <= gap_bars
                                for p1bar in p1_bars)
                if not too_close:
                    # Remove internal keys before appending
                    be.pop("_bar_idx", None)
                    be.pop("_signal", None)
                    events.append(be)

    # Sort all events by time
    events.sort(key=lambda e: e["time"])
    return events


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stem_analysis(y, sr, measures, phrases, events, stem_energies,
                       stem_presence, beats_per_bar, title="",
                       phrase_bars=4, zoom_start=None, zoom_end=None):
    """Chart for stem-based analysis: waveform + per-stem energy + events."""
    duration = len(y) / sr
    times = np.arange(len(y)) / sr
    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    stem_names = [s for s in STEM_NAMES if s in stem_energies]
    n_stems = len(stem_names)

    fig = plt.figure(figsize=(16, 4 + n_stems * 1.8), constrained_layout=True)
    gs = gridspec.GridSpec(1 + n_stems, 1,
                           height_ratios=[1.5] + [0.7] * n_stems,
                           hspace=0.3, figure=fig)

    # Phrase boundary times
    phrase_boundaries = set()
    for i in range(0, len(measures), phrase_bars):
        phrase_boundaries.add(measures[i]["start"])
    if measures:
        phrase_boundaries.add(measures[-1]["end"])

    event_times_set = {e["time"] for e in events}

    # Color map for stems
    stem_colors = {
        "drums": "#cc4444",
        "bass": "#4488cc",
        "vocals": "#44aa66",
        "other": "#aa8844",
    }

    # --- Panel 1: Waveform + event markers ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888",
             linewidth=0.4 if zoomed else 0.2,
             alpha=0.7 if zoomed else 0.5)

    for e in events:
        if view_start <= e["time"] <= view_end:
            ax1.axvline(x=e["time"], color="#cc2222",
                        linewidth=2.0, alpha=0.7)
            ax1.text(e["time"], 0.95,
                     f'Bar {e["bar"]}\n{e["type"]}',
                     transform=ax1.get_xaxis_transform(),
                     fontsize=6, ha="center", va="top", color="#cc2222",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", alpha=0.85))

    if zoomed:
        for m in measures:
            if view_start <= m["start"] <= view_end:
                ax1.axvline(x=m["start"], color="#2244aa",
                            linewidth=0.8, alpha=0.4)
    else:
        for pb in phrase_boundaries:
            if view_start <= pb <= view_end:
                ax1.axvline(x=pb, color="#2244aa", linewidth=0.4, alpha=0.25)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Events  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    # --- Stem energy panels ---
    for panel_idx, stem_name in enumerate(stem_names):
        ax = fig.add_subplot(gs[1 + panel_idx], sharex=ax1)
        energies = stem_energies[stem_name]
        presence = stem_presence[stem_name]
        color = stem_colors.get(stem_name, "#888888")

        for pi, p in enumerate(phrases):
            if pi < len(energies):
                e_val = energies[pi]
                is_present = presence[pi] if pi < len(presence) else False
                alpha = 0.8 if is_present else 0.25
                ax.bar(p["start"], e_val, width=p["end"] - p["start"],
                       align="edge", color=color, alpha=alpha,
                       edgecolor="#333333", linewidth=0.3)

        # Mark events on this panel
        for e in events:
            if view_start <= e["time"] <= view_end:
                ax.axvline(x=e["time"], color="#cc2222",
                           linewidth=1.5, alpha=0.5, linestyle="--")

        ax.set_ylabel(stem_name.capitalize())
        ax.set_title(f"{stem_name.capitalize()} Energy (shaded = present)")

    # X label on bottom panel
    fig.axes[-1].set_xlabel("Time (seconds)")

    return fig


def plot_feature_analysis(y, sr, measures, phrases, events,
                          beats_per_bar, title="", phrase_bars=4,
                          zoom_start=None, zoom_end=None):
    """Chart for feature-only analysis with HPSS split.

    5 panels: waveform, percussive onset, harmonic onset, RMS, change score.
    """
    duration = len(y) / sr
    times = np.arange(len(y)) / sr
    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.2, 0.5, 0.5, 0.5, 0.5],
                           hspace=0.3, figure=fig)

    phrase_boundaries = set()
    for i in range(0, len(measures), phrase_bars):
        phrase_boundaries.add(measures[i]["start"])
    if measures:
        phrase_boundaries.add(measures[-1]["end"])

    event_bars = {e["bar"] for e in events}

    # Panel 1: Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888",
             linewidth=0.4 if zoomed else 0.2,
             alpha=0.7 if zoomed else 0.5)

    for e in events:
        if view_start <= e["time"] <= view_end:
            color = "#cc2222" if e.get("type") == "decrease" else (
                "#2266cc" if e.get("type") == "increase" else "#888800")
            ax1.axvline(x=e["time"], color=color, linewidth=2.0, alpha=0.7)
            ax1.text(e["time"], 0.95, f'Bar {e["bar"]}\n{e.get("type","")}',
                     transform=ax1.get_xaxis_transform(),
                     fontsize=5, ha="center", va="top", color=color,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15",
                               facecolor="white", alpha=0.85))

    if zoomed:
        for m in measures:
            if view_start <= m["start"] <= view_end:
                ax1.axvline(x=m["start"], color="#2244aa",
                            linewidth=0.8, alpha=0.4)
    else:
        for pb in phrase_boundaries:
            if view_start <= pb <= view_end:
                ax1.axvline(x=pb, color="#2244aa", linewidth=0.4, alpha=0.25)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Structural Events  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    # Panel 2: Percussive onset
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for p in phrases:
        is_event = p["start_bar"] in event_bars
        ax2.bar(p["start"], p["onset_percussive"], width=p["end"] - p["start"],
                align="edge", color="#cc4444" if is_event else "#cc8866",
                edgecolor="#445566", linewidth=0.3, alpha=0.7)
    ax2.set_ylabel("Perc")
    ax2.set_title("Percussive Onset per Phrase (drums/hits)")

    # Panel 3: Harmonic onset
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for p in phrases:
        is_event = p["start_bar"] in event_bars
        ax3.bar(p["start"], p["onset_harmonic"], width=p["end"] - p["start"],
                align="edge", color="#cc4444" if is_event else "#6688aa",
                edgecolor="#445566", linewidth=0.3, alpha=0.7)
    ax3.set_ylabel("Harm")
    ax3.set_title("Harmonic Onset per Phrase (melodic/tonal)")

    # Panel 4: RMS
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for p in phrases:
        is_event = p["start_bar"] in event_bars
        ax4.bar(p["start"], p["rms"], width=p["end"] - p["start"],
                align="edge", color="#cc4444" if is_event else "#66aa88",
                edgecolor="#445566", linewidth=0.3, alpha=0.7)
    ax4.set_ylabel("RMS")
    ax4.set_title("RMS Energy per Phrase")

    # Panel 5: Change score
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    feature_names = ["onset_percussive", "onset_harmonic", "rms",
                     "centroid", "low_freq_ratio"]
    raw = {f: np.array([p[f] for p in phrases]) for f in feature_names}
    normed = {f: normalize_feature(raw[f]) for f in feature_names}
    weights = {"onset_percussive": 0.25, "onset_harmonic": 0.20,
               "rms": 0.25, "centroid": 0.15, "low_freq_ratio": 0.15}
    context = 2
    scores = np.zeros(len(phrases))
    for i in range(context, len(phrases)):
        for f in feature_names:
            before = np.mean(normed[f][max(0, i - context):i])
            scores[i] += abs(normed[f][i] - before) * weights[f]

    for i, p in enumerate(phrases):
        is_event = p["start_bar"] in event_bars
        ax5.bar(p["start"], scores[i], width=p["end"] - p["start"],
                align="edge", color="#cc4444" if is_event else "#aa8866",
                edgecolor="#445566", linewidth=0.3, alpha=0.7)

    ax5.axhline(y=0.15, color="#cc2222", linewidth=1.0, linestyle="--",
                alpha=0.5, label="min_score threshold")
    ax5.legend(fontsize=7, loc="upper right")
    ax5.set_ylabel("Score")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_title("Combined Feature Change Score per Phrase")

    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(events, phrases, duration, mode="feature"):
    """Summary statistics."""
    if not events:
        return {"event_count": 0, "mode": mode}

    minutes = duration / 60.0
    scores = [e["score"] for e in events]

    return {
        "mode": mode,
        "event_count": len(events),
        "events_per_minute": round(len(events) / minutes, 2) if minutes > 0 else 0,
        "avg_score": round(float(np.mean(scores)), 3),
        "max_score": round(float(max(scores)), 3),
        "phrase_count": len(phrases),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structural event detection (stem-aware or feature-based)")
    parser.add_argument("audio", help="Path to audio file (WAV/MP3/FLAC)")
    parser.add_argument("--stems-dir", default=None,
                        help="Directory containing stem .wav files")
    parser.add_argument("--auto-separate", action="store_true",
                        help="Run Demucs stem separation if stems not found")
    parser.add_argument("--presence-threshold", type=float, default=0.15,
                        help="Stem presence threshold as fraction of max (default: 0.15)")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (default: 44100)")
    parser.add_argument("--hop-length", type=int, default=512,
                        help="Hop length for beat tracker (default: 512)")
    parser.add_argument("--beats-per-bar", type=int, default=4,
                        help="Beats per bar (default: 4)")
    parser.add_argument("--phrase-bars", type=int, default=4,
                        help="Bars per analysis phrase (default: 4)")
    parser.add_argument("--context", type=int, default=2,
                        help="Context phrases for feature change detection (default: 2)")
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Min score for feature events (default: 0.15)")
    parser.add_argument("--min-features", type=int, default=2,
                        help="Min features for feature events (default: 2)")
    parser.add_argument("--min-gap", type=int, default=2,
                        help="Min phrases between events (default: 2)")
    parser.add_argument("--zoom-start", type=float, default=None)
    parser.add_argument("--zoom-end", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip chart")
    parser.add_argument("--save-plot", help="Save chart to file")
    parser.add_argument("--json-out", help="Write JSON results to file")
    args = parser.parse_args()

    path = Path(args.audio)
    if not path.exists():
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # --- 1. Beat detection ---
    print(f"Loading {path.name} (sr={args.sr}, hop={args.hop_length})...",
          file=sys.stderr)
    data = detect_beats(str(path), sr=args.sr, hop_length=args.hop_length)
    beat_times = data["beat_times"]
    y, sr_val = data["y"], data["sr"]
    print(f"Detected {len(beat_times)} beats (librosa tempo: {data['tempo']:.1f} BPM)",
          file=sys.stderr)

    if len(beat_times) < 8:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Measures ---
    measures = group_beats_into_measures(beat_times, args.beats_per_bar)
    print(f"Grouped into {len(measures)} measures", file=sys.stderr)

    # --- 3. Try to get stems ---
    stems_dir, stem_paths = find_or_create_stems(
        str(path), stems_dir=args.stems_dir, auto_separate=args.auto_separate)

    use_stems = stem_paths is not None
    stem_energies = {}
    stem_presence = {}

    if use_stems:
        print(f"Stem mode: loading from {stems_dir}", file=sys.stderr)
        for stem_name, stem_path in stem_paths.items():
            print(f"  Loading {stem_name}...", file=sys.stderr)
            y_stem, _ = librosa.load(stem_path, sr=args.sr, mono=True)
            energies = compute_stem_phrase_energy(y_stem, args.sr, measures,
                                                  phrase_bars=args.phrase_bars)
            stem_energies[stem_name] = energies

        stem_presence = compute_stem_presence(stem_energies,
                                               presence_threshold=args.presence_threshold)

        # Build phrase list (for timing info)
        phrases = []
        for i in range(0, len(measures), args.phrase_bars):
            group = measures[i:i + args.phrase_bars]
            if group:
                phrases.append({
                    "phrase_num": len(phrases) + 1,
                    "start_bar": group[0]["measure_num"],
                    "end_bar": group[-1]["measure_num"],
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                })
        print(f"  {len(phrases)} phrases, {len(stem_energies)} stems", file=sys.stderr)

        events = detect_stem_events(stem_presence, phrases,
                                     min_gap_phrases=args.min_gap)
        mode = "stems"
    else:
        print("Feature mode: no stems available", file=sys.stderr)
        print("Computing per-bar features...", file=sys.stderr)
        bar_features = compute_bar_features(y, sr_val, measures)
        phrases = compute_phrase_features(bar_features, phrase_bars=args.phrase_bars)
        print(f"  {len(phrases)} phrases", file=sys.stderr)

        events = detect_feature_events(phrases, context=args.context,
                                        min_score=args.min_score,
                                        min_features=args.min_features,
                                        min_gap_phrases=args.min_gap,
                                        bar_features=bar_features)
        mode = "features"

    # --- Summary ---
    duration = len(y) / sr_val
    summary = compute_summary(events, phrases, duration, mode=mode)

    # --- Print ---
    print()
    print(f"=== Event Detection: {path.name} ===")
    print(f"Mode: {mode}")
    print()
    print(f"Tempo: {data['tempo']:.1f} BPM")
    print(f"Duration: {format_time(duration)}")
    print(f"Phrases: {len(phrases)} ({args.phrase_bars}-bar)")
    print(f"Events detected: {summary['event_count']}")
    print()

    if events:
        if use_stems:
            print(f"{'#':>3} {'Bar':>5} {'Time':>8} {'Score':>6} "
                  f"{'Type':<14} {'Description'}")
            print(f"  {'-'*65}")
            for i, e in enumerate(events, 1):
                print(f"{i:>3} {e['bar']:>5} {e['time_fmt']:>8} "
                      f"{e['score']:>6.3f} {e['type']:<14} {e['description']}")
        else:
            print(f"{'#':>3} {'Bar':>5} {'Time':>8} {'Score':>6} {'Pass':>4} "
                  f"{'Type':<12} {'Description'}")
            print(f"  {'-'*75}")
            for i, e in enumerate(events, 1):
                p = e.get('pass', '?')
                print(f"{i:>3} {e['bar']:>5} {e['time_fmt']:>8} "
                      f"{e['score']:>6.3f} {p:>4} "
                      f"{e.get('type', ''):<12} {e['description']}")
        print()
    else:
        print("No significant structural events detected.")
        print()

    # --- JSON ---
    if args.json_out:
        result = {
            "file": path.name,
            "duration": round(duration, 2),
            "bpm": round(data["tempo"], 2),
            "sr": sr_val,
            "beats_per_bar": args.beats_per_bar,
            "phrase_bars": args.phrase_bars,
            "summary": summary,
            "events": events,
        }
        if use_stems:
            result["stem_presence_summary"] = {
                stem: sum(presence)
                for stem, presence in stem_presence.items()
            }
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"JSON written to {args.json_out}")

    # --- Plot ---
    if not args.no_plot:
        if use_stems:
            fig = plot_stem_analysis(
                y, sr_val, measures, phrases, events,
                stem_energies, stem_presence, args.beats_per_bar,
                title=f"- {path.name}", phrase_bars=args.phrase_bars,
                zoom_start=args.zoom_start, zoom_end=args.zoom_end
            )
        else:
            fig = plot_feature_analysis(
                y, sr_val, measures, phrases, events,
                args.beats_per_bar, title=f"- {path.name}",
                phrase_bars=args.phrase_bars,
                zoom_start=args.zoom_start, zoom_end=args.zoom_end
            )
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {args.save_plot}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
