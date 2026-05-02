
#!/usr/bin/env python3
"""
camelot_from_youtube.py (v3)

Features:
- Single-window key + Camelot estimation from local audio file.
- Consensus (multi-window) analysis with STRICT voting:
    * Ignore windows below --min-confidence.
    * If none qualify, fall back to the single best-confidence window.
- Sample-rate switch: --sr
- Window generation from BPM + bars:
    * --bpm --bar-start --bar-step --num-windows --beats-per-bar
  OR explicit starts:
    * --starts "90,148,206"
- Harmonic-only analysis by default (good for EDM); disable with --full-mix.
- Optional YouTube metadata via YouTube Data API (title/channel) if you supply:
    * --url + --yt-api-key
  (Metadata only; key detection comes from the audio file.)

Notes:
- In locked-down environments, avoid .weba/.webm/.opus unless you can decode them.
  WAV/MP3/FLAC are recommended.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

import numpy as np
import librosa

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; environment variables still work


# -----------------------------
# Camelot mapping (flat spellings)
# -----------------------------
PC_TO_NOTE_FLAT = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}

CAMELOT_MAJOR = {
    "B": "1B",
    "Gb": "2B",
    "Db": "3B",
    "Ab": "4B",
    "Eb": "5B",
    "Bb": "6B",
    "F": "7B",
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
}

CAMELOT_MINOR = {
    "Ab": "1A",
    "Eb": "2A",
    "Bb": "3A",
    "F": "4A",
    "C": "5A",
    "G": "6A",
    "D": "7A",
    "A": "8A",
    "E": "9A",
    "B": "10A",
    "Gb": "11A",
    "Db": "12A",
}

UNSUPPORTED_HINT_EXTS = {".weba", ".webm", ".opus"}


# -----------------------------
# Krumhansl key profiles
# -----------------------------
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def rotate_profile(profile: np.ndarray, n: int) -> np.ndarray:
    return np.roll(profile, n)


@dataclass
class KeyEstimate:
    tonic_pc: int
    mode: str
    score: float
    confidence: float
    top_candidates: List[Dict[str, Any]]


def estimate_key_from_audio(y: np.ndarray, sr: int) -> KeyEstimate:
    """
    Estimate key by correlating mean chroma (CQT) against Krumhansl profiles.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = normalize(chroma.mean(axis=1))

    maj_prof = normalize(KRUMHANSL_MAJOR)
    min_prof = normalize(KRUMHANSL_MINOR)

    candidates = []
    for tonic in range(12):
        maj_score = float(np.dot(chroma_mean, rotate_profile(maj_prof, tonic)))
        min_score = float(np.dot(chroma_mean, rotate_profile(min_prof, tonic)))
        candidates.append((maj_score, tonic, "major"))
        candidates.append((min_score, tonic, "minor"))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_tonic, best_mode = candidates[0]
    second_score = candidates[1][0]

    # Confidence heuristic:
    # gap between best and second, scaled.
    gap = best_score - second_score
    confidence = float(np.clip(gap / 0.05, 0.0, 1.0))

    top = []
    for score, tonic, mode in candidates[:6]:
        top.append({
            "score": round(float(score), 4),
            "tonic": PC_TO_NOTE_FLAT[int(tonic)],
            "mode": mode
        })

    return KeyEstimate(
        tonic_pc=int(best_tonic),
        mode=str(best_mode),
        score=round(float(best_score), 4),
        confidence=round(float(confidence), 3),
        top_candidates=top
    )


def to_camelot(tonic_pc: int, mode: str) -> Dict[str, Optional[str]]:
    note = PC_TO_NOTE_FLAT[int(tonic_pc)]
    if mode == "major":
        return {"key": f"{note} major", "camelot": CAMELOT_MAJOR.get(note)}
    else:
        return {"key": f"{note} minor", "camelot": CAMELOT_MINOR.get(note)}


def camelot_number(c: Optional[str]) -> Optional[int]:
    if not c:
        return None
    m = re.fullmatch(r"(\d+)[AB]", c.strip())
    return int(m.group(1)) if m else None


# -----------------------------
# YouTube helpers (optional metadata)
# -----------------------------
def extract_youtube_id(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
    except Exception:
        return None

    if u.netloc in ("youtu.be",):
        return u.path.strip("/")

    if "youtube.com" in u.netloc:
        if u.path == "/watch":
            qs = parse_qs(u.query)
            return qs.get("v", [None])[0]
        m = re.match(r"^/(shorts|embed)/([^/?]+)", u.path)
        if m:
            return m.group(2)

    return None


def fetch_youtube_metadata(video_id: str, api_key: str) -> Dict[str, Any]:
    if requests is None:
        return {"error": "requests not installed. pip install requests"}

    endpoint = "https://www.googleapis.com/youtube/v3/videos"
    params = {"key": api_key, "id": video_id, "part": "snippet"}
    r = requests.get(endpoint, params=params, timeout=20)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:500]}

    data = r.json()
    if not data.get("items"):
        return {"error": "No items returned", "videoId": video_id}

    snip = data["items"][0].get("snippet", {})
    return {
        "videoId": video_id,
        "title": snip.get("title"),
        "channelTitle": snip.get("channelTitle"),
        "channelId": snip.get("channelId"),
        "publishedAt": snip.get("publishedAt"),
    }


# -----------------------------
# Audio helpers
# -----------------------------
def harmonic_only(y: np.ndarray) -> np.ndarray:
    # Reduce percussion influence
    return librosa.effects.harmonic(y)


def load_audio_segment(path: str, sr: int, mono: bool, start: float, duration: float) -> (np.ndarray, int):
    return librosa.load(path, sr=sr, mono=mono, offset=float(start), duration=float(duration))


# -----------------------------
# BPM Detection
# -----------------------------
def detect_bpm(path: str, sr: int, start: float = 30.0, duration: float = 60.0) -> Dict[str, Any]:
    """
    Detect BPM from audio using librosa's beat tracking.
    Analyzes a segment of the audio (default: 30-90 seconds) to estimate tempo.
    Returns dict with bpm and confidence info.
    """
    result = detect_bpm_enhanced(path, sr, start, duration)
    # Return simplified result for backward compatibility
    return {
        "bpm": result.get("bpm"),
        "detected_from": result.get("detected_from"),
        "beat_count": result.get("beat_count", 0),
        "error": result.get("error")
    }


def detect_bpm_enhanced(path: str, sr: int, start: float = 30.0, duration: float = 60.0,
                        full_track: bool = False,
                        use_hybrid_downbeat: bool = True,
                        manual_first_downbeat: float = None,
                        beats_per_bar: int = 4,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Enhanced BPM detection with beat grid analysis and hybrid downbeat detection.

    Args:
        path: Path to audio file
        sr: Sample rate
        start: Start time for analysis window (ignored if full_track=True)
        duration: Duration of analysis window (ignored if full_track=True)
        full_track: If True, analyze entire track for beats (slower but more accurate)
        use_hybrid_downbeat: If True, use hybrid downbeat detection (onset+bass+harmony+interval)
        manual_first_downbeat: Manual override for first downbeat time in seconds
        beats_per_bar: Beats per bar for downbeat detection (default 4)
        verbose: Print progress messages

    Returns:
        bpm: High-precision BPM (computed from median beat intervals)
        bpm_librosa: Original librosa tempo estimate (for comparison)
        bpm_confidence: How consistent the beat intervals are (0-1, higher = more stable)
        beat_times: Array of beat positions in seconds
        first_downbeat: Estimated time of first downbeat (beat 1 of a bar)
        first_downbeat_idx: Index into beat_times array for first downbeat
        downbeat_confidence: Confidence in downbeat detection (0-1)
        downbeat_method: Method used for downbeat detection
        tempo_stable: Boolean - is tempo consistent throughout?
        tempo_changes: List of tempo change points if detected
    """
    try:
        # Load audio segment (or full track if requested)
        if full_track:
            if verbose:
                print("Loading full track for beat analysis...", file=sys.stderr)
            y, _sr = librosa.load(path, sr=sr, mono=True)
            actual_start = 0.0
            actual_duration = len(y) / sr
            if verbose:
                print(f"Track duration: {actual_duration:.2f}s ({actual_duration/60:.1f} minutes)", file=sys.stderr)
        else:
            y, _sr = librosa.load(path, sr=sr, mono=True, offset=start, duration=duration)
            actual_start = start
            actual_duration = duration

        if y is None or len(y) < sr * 5:
            return {"error": "Audio segment too short for BPM detection", "bpm": None}

        # Get tempo estimate and beat frames from librosa
        if verbose:
            print("Running beat detection...", file=sys.stderr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Handle librosa version differences for tempo
        if hasattr(tempo, '__len__'):
            bpm_librosa = float(tempo[0]) if len(tempo) > 0 else float(tempo)
        else:
            bpm_librosa = float(tempo)

        # Convert beat frames to times (in seconds, relative to segment start)
        beat_times_relative = librosa.frames_to_time(beat_frames, sr=sr)

        if verbose:
            print(f"Detected {len(beat_times_relative)} beats at ~{bpm_librosa:.1f} BPM", file=sys.stderr)

        # Need at least 2 beats to compute intervals
        if len(beat_times_relative) < 2:
            return {
                "bpm": round(bpm_librosa, 3),
                "bpm_librosa": bpm_librosa,
                "detected_from": f"{actual_start}-{actual_start + actual_duration}s",
                "beat_count": len(beat_times_relative),
                "bpm_confidence": 0.0,
                "tempo_stable": False,
                "error": "Too few beats detected for interval analysis"
            }

        # Compute beat intervals (time between consecutive beats)
        beat_intervals = np.diff(beat_times_relative)

        # Compute BPM from median interval (more robust than mean)
        median_interval = float(np.median(beat_intervals))
        bpm_from_intervals = 60.0 / median_interval if median_interval > 0 else bpm_librosa

        # Compute confidence based on interval consistency
        interval_std = float(np.std(beat_intervals))
        interval_cv = interval_std / median_interval if median_interval > 0 else 1.0
        bpm_confidence = max(0.0, min(1.0, 1.0 - (interval_cv * 5)))

        # Detect tempo changes
        tempo_changes = []
        tempo_stable = True
        if len(beat_intervals) > 8:
            window_size = 4
            for i in range(window_size, len(beat_intervals) - window_size):
                before_avg = np.mean(beat_intervals[i-window_size:i])
                after_avg = np.mean(beat_intervals[i:i+window_size])
                change_ratio = after_avg / before_avg if before_avg > 0 else 1.0

                if abs(change_ratio - 1.0) > 0.03:
                    tempo_stable = False
                    change_time = beat_times_relative[i] + actual_start
                    old_bpm = 60.0 / before_avg
                    new_bpm = 60.0 / after_avg
                    tempo_changes.append({
                        "time": round(change_time, 3),
                        "old_bpm": round(old_bpm, 2),
                        "new_bpm": round(new_bpm, 2),
                        "change_percent": round((change_ratio - 1.0) * 100, 1)
                    })

        # Deduplicate tempo changes
        if tempo_changes:
            filtered_changes = [tempo_changes[0]]
            for change in tempo_changes[1:]:
                if change["time"] - filtered_changes[-1]["time"] > 2.0:
                    filtered_changes.append(change)
            tempo_changes = filtered_changes

        # Convert beat times to absolute (relative to track start)
        beat_times_absolute = [float(t) + actual_start for t in beat_times_relative]

        # Determine first downbeat
        downbeat_method = "simple"
        downbeat_confidence = 0.0
        downbeat_details = None
        first_downbeat_idx = 0

        if manual_first_downbeat is not None:
            # Manual override
            if verbose:
                print(f"Using manual first downbeat: {manual_first_downbeat:.3f}s", file=sys.stderr)
            first_downbeat_idx, first_downbeat = find_downbeat_from_time(
                beat_times_absolute, manual_first_downbeat
            )
            downbeat_confidence = 1.0
            downbeat_method = "manual"
            if verbose:
                print(f"Snapped to beat {first_downbeat_idx} at {first_downbeat:.3f}s", file=sys.stderr)

        elif use_hybrid_downbeat and full_track and len(beat_times_relative) >= 8:
            # Use hybrid detection on full track
            if verbose:
                print("Estimating first downbeat (hybrid method)...", file=sys.stderr)

            phase_offset, downbeat_confidence, downbeat_details = estimate_first_downbeat_hybrid(
                y, sr, beat_times_relative, beat_intervals, beats_per_bar, verbose=verbose
            )
            downbeat_method = "hybrid"

            if verbose and downbeat_details:
                print(f"\n  Per-method winners:", file=sys.stderr)
                for method, winner in downbeat_details.get("winner_by_method", {}).items():
                    marker = " <--" if winner == phase_offset else ""
                    print(f"    {method}: beat {winner}{marker}", file=sys.stderr)

            # =====================================================
            # BACKWARD EXTRAPOLATION: Prepend missing early beats
            # =====================================================
            # librosa often misses the first few beats. We use the detected phase
            # to extrapolate backward and find where the actual first beat 1 would be.
            #
            # Example: First detected beat at 3.5s, median_interval=0.5s, phase=1
            # - This means beat index 1 is "beat 1" of a measure
            # - Beat index 0 is "beat 4" of the previous measure
            # - We need to prepend beats going back toward time 0

            first_detected_time = beat_times_absolute[0]

            if first_detected_time > median_interval * 1.5:
                # There are likely missing beats before the first detected one
                # Calculate how many beats we can fit before the first detected beat
                beats_to_prepend = []
                extrapolated_time = first_detected_time - median_interval

                while extrapolated_time >= -median_interval * 0.25:  # Allow slight negative for rounding
                    if extrapolated_time >= 0:
                        beats_to_prepend.insert(0, extrapolated_time)
                    extrapolated_time -= median_interval

                num_prepended = len(beats_to_prepend)

                if num_prepended > 0:
                    if verbose:
                        print(f"\n  Backward extrapolation:", file=sys.stderr)
                        print(f"    First detected beat was at {first_detected_time:.3f}s", file=sys.stderr)
                        print(f"    Prepending {num_prepended} extrapolated beats", file=sys.stderr)
                        print(f"    New first beat at {beats_to_prepend[0]:.3f}s", file=sys.stderr)

                    # Prepend the extrapolated beats
                    beat_times_absolute = beats_to_prepend + beat_times_absolute

                    # Adjust the phase offset since we added beats at the front
                    # The phase tells us which position (0-3) is "beat 1"
                    # After prepending N beats, the new phase offset = (phase + N) % beats_per_bar
                    # But we want to find the FIRST downbeat in the new array
                    adjusted_phase = (phase_offset + num_prepended) % beats_per_bar

                    # The first downbeat index is the first beat where (index % beats_per_bar) == adjusted_phase
                    # Since we want the earliest downbeat, it's just adjusted_phase (if < len) else we search
                    first_downbeat_idx = adjusted_phase

                    if verbose:
                        print(f"    Original phase: {phase_offset}, Adjusted phase: {adjusted_phase}", file=sys.stderr)
                        print(f"    First downbeat now at index {first_downbeat_idx} ({beat_times_absolute[first_downbeat_idx]:.3f}s)", file=sys.stderr)

                    # Store extrapolation info
                    if downbeat_details is None:
                        downbeat_details = {}
                    downbeat_details["extrapolation"] = {
                        "original_first_beat": round(first_detected_time, 4),
                        "beats_prepended": num_prepended,
                        "new_first_beat": round(beats_to_prepend[0], 4),
                        "original_phase": phase_offset,
                        "adjusted_phase": adjusted_phase
                    }
                    downbeat_method = "hybrid+extrapolation"
                else:
                    first_downbeat_idx = phase_offset
            else:
                # First detected beat is close enough to start, no extrapolation needed
                first_downbeat_idx = phase_offset

            first_downbeat = beat_times_absolute[first_downbeat_idx]
        else:
            # Simple extrapolation method (original behavior)
            first_beat_absolute = beat_times_absolute[0] if beat_times_absolute else 0.0
            if median_interval > 0 and first_beat_absolute > 0:
                beats_before = int(first_beat_absolute / median_interval)
                first_downbeat = first_beat_absolute - (beats_before * median_interval)
                # Find closest beat in the array
                first_downbeat_idx, first_downbeat = find_downbeat_from_time(
                    beat_times_absolute, first_downbeat
                )
            else:
                first_downbeat = 0.0
                first_downbeat_idx = 0
            downbeat_method = "extrapolation"

        if verbose:
            print(f"\nFirst downbeat: beat {first_downbeat_idx} at {first_downbeat:.3f}s", file=sys.stderr)
            print(f"Detection method: {downbeat_method}", file=sys.stderr)
            print(f"Confidence: {downbeat_confidence:.3f}", file=sys.stderr)

        result = {
            "bpm": round(bpm_from_intervals, 3),
            "bpm_librosa": round(bpm_librosa, 3),
            "bpm_confidence": round(bpm_confidence, 3),
            "detected_from": f"{actual_start}-{actual_start + actual_duration}s",
            "beat_count": len(beat_times_absolute),
            "beat_times": [round(t, 4) for t in beat_times_absolute],
            "first_downbeat": round(first_downbeat, 4),
            "first_downbeat_idx": first_downbeat_idx,
            "downbeat_confidence": round(downbeat_confidence, 4),
            "downbeat_method": downbeat_method,
            "median_beat_interval": round(median_interval, 4),
            "tempo_stable": tempo_stable,
            "tempo_changes": tempo_changes if tempo_changes else None,
            "beat_intervals": {
                "mean": round(float(np.mean(beat_intervals)), 4),
                "std": round(interval_std, 4),
                "min": round(float(np.min(beat_intervals)), 4),
                "max": round(float(np.max(beat_intervals)), 4)
            }
        }

        if downbeat_details:
            result["downbeat_analysis"] = downbeat_details

        return result

    except Exception as e:
        return {"error": str(e), "bpm": None}


# -----------------------------
# Hybrid Downbeat Detection
# -----------------------------
def score_downbeat_by_onset_strength(y: np.ndarray, sr: int,
                                     beat_times: np.ndarray,
                                     beats_per_bar: int = 4) -> np.ndarray:
    """
    Score each phase offset (0-3) by onset strength at downbeat positions.
    Downbeats typically have stronger onsets (louder attacks).
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_strengths = onset_env[beat_frames]

    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        downbeat_indices = list(range(offset, len(beat_strengths), beats_per_bar))
        if downbeat_indices:
            scores[offset] = np.sum(beat_strengths[downbeat_indices])
    return scores


def score_downbeat_by_bass_energy(y: np.ndarray, sr: int,
                                  beat_times: np.ndarray,
                                  beats_per_bar: int = 4) -> np.ndarray:
    """
    Score each phase offset by bass energy at downbeat positions.
    In EDM/electronic music, kicks often hit on beats 1 and 3.
    """
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 150
    bass_energy = np.sum(S[bass_mask, :], axis=0)

    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.clip(beat_frames, 0, len(bass_energy) - 1)
    beat_bass = bass_energy[beat_frames]

    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        downbeat_indices = list(range(offset, len(beat_bass), beats_per_bar))
        if downbeat_indices:
            scores[offset] = np.sum(beat_bass[downbeat_indices])
    return scores


def score_downbeat_by_harmonic_change(y: np.ndarray, sr: int,
                                      beat_times: np.ndarray,
                                      beats_per_bar: int = 4) -> np.ndarray:
    """
    Score each phase offset by harmonic change at downbeat positions.
    Chord changes often occur on beat 1 of a bar.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_flux = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)

    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.clip(beat_frames, 0, len(chroma_flux) - 1)
    beat_flux = chroma_flux[beat_frames]

    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        downbeat_indices = list(range(offset, len(beat_flux), beats_per_bar))
        if downbeat_indices:
            scores[offset] = np.sum(beat_flux[downbeat_indices])
    return scores


def score_downbeat_by_interval_consistency(beat_times: np.ndarray,
                                           beat_intervals: np.ndarray,
                                           beats_per_bar: int = 4) -> np.ndarray:
    """
    Score each phase offset by how consistent the beat intervals are
    when grouped into bars starting from that offset.
    """
    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        bar_scores = []
        for bar_start in range(offset, len(beat_times) - beats_per_bar, beats_per_bar):
            if bar_start + beats_per_bar - 1 < len(beat_intervals):
                bar_intervals = beat_intervals[bar_start:bar_start + beats_per_bar]
                if len(bar_intervals) == beats_per_bar:
                    consistency = 1.0 / (1.0 + np.std(bar_intervals))
                    bar_scores.append(consistency)
        if bar_scores:
            scores[offset] = np.mean(bar_scores)
    return scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val > 0:
        return (scores - min_val) / (max_val - min_val)
    return np.ones_like(scores) * 0.25


def estimate_first_downbeat_hybrid(y: np.ndarray, sr: int,
                                   beat_times: List[float],
                                   beat_intervals: np.ndarray = None,
                                   beats_per_bar: int = 4,
                                   weights: Dict[str, float] = None,
                                   verbose: bool = False) -> tuple:
    """
    Estimate the downbeat phase using multiple audio cues (hybrid approach).

    This function determines which beat position (0 to beats_per_bar-1) in the
    detected beat array represents "beat 1" of a measure. The calling code then
    uses this phase information to extrapolate backward and find any missing
    early beats.

    Combines four analysis methods:
    1. Onset strength (louder attacks on downbeats)
    2. Bass energy (kicks typically on beats 1 and 3)
    3. Harmonic change (chord changes on bar boundaries)
    4. Interval consistency (consistent beat groupings)

    Returns:
        Tuple of (phase_offset, confidence, detailed_scores)
        - phase_offset: Which position (0-3) in the beat array is "beat 1"
        - confidence: How confident we are (based on score margin)
        - detailed_scores: Breakdown by analysis method
    """
    if len(beat_times) < 8:
        return 0, 0.0, {}

    beat_times = np.array(beat_times)
    if beat_intervals is None:
        beat_intervals = np.diff(beat_times)

    if weights is None:
        weights = {
            "onset_strength": 0.30,
            "bass_energy": 0.30,
            "harmonic_change": 0.20,
            "interval_consistency": 0.20
        }

    if verbose:
        print("  Analyzing onset strength...", file=sys.stderr)
    onset_scores = score_downbeat_by_onset_strength(y, sr, beat_times, beats_per_bar)

    if verbose:
        print("  Analyzing bass energy...", file=sys.stderr)
    bass_scores = score_downbeat_by_bass_energy(y, sr, beat_times, beats_per_bar)

    if verbose:
        print("  Analyzing harmonic changes...", file=sys.stderr)
    harmony_scores = score_downbeat_by_harmonic_change(y, sr, beat_times, beats_per_bar)

    if verbose:
        print("  Analyzing interval consistency...", file=sys.stderr)
    interval_scores = score_downbeat_by_interval_consistency(beat_times, beat_intervals, beats_per_bar)

    # Normalize and combine
    onset_norm = normalize_scores(onset_scores)
    bass_norm = normalize_scores(bass_scores)
    harmony_norm = normalize_scores(harmony_scores)
    interval_norm = normalize_scores(interval_scores)

    combined_scores = (
        weights["onset_strength"] * onset_norm +
        weights["bass_energy"] * bass_norm +
        weights["harmonic_change"] * harmony_norm +
        weights["interval_consistency"] * interval_norm
    )

    # The phase tells us which offset (0-3) represents "beat 1"
    best_phase = int(np.argmax(combined_scores))

    # Confidence: how much better is best vs second best
    sorted_scores = np.sort(combined_scores)[::-1]
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    else:
        confidence = 1.0

    if verbose:
        print(f"  Phase scores: {[round(s, 3) for s in combined_scores.tolist()]}", file=sys.stderr)
        print(f"  Best phase: {best_phase} (confidence: {confidence:.3f})", file=sys.stderr)

    detailed = {
        "combined_scores": combined_scores.tolist(),
        "weights": weights,
        "phase": best_phase,
        "winner_by_method": {
            "onset_strength": int(np.argmax(onset_scores)),
            "bass_energy": int(np.argmax(bass_scores)),
            "harmonic_change": int(np.argmax(harmony_scores)),
            "interval_consistency": int(np.argmax(interval_scores))
        }
    }

    return best_phase, confidence, detailed


def find_downbeat_from_time(beat_times: List[float], target_time: float) -> tuple:
    """
    Find the beat index closest to a given time.
    Useful when user provides a manual first downbeat time.
    """
    beat_times = np.array(beat_times)
    closest_idx = int(np.argmin(np.abs(beat_times - target_time)))
    return closest_idx, float(beat_times[closest_idx])


# -----------------------------
# Energy / Structural Analysis
# -----------------------------
def analyze_energy_segment(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze energy characteristics of an audio segment.
    Returns RMS energy, onset density, and spectral centroid.
    """
    # RMS energy (loudness)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Onset detection (rhythmic activity / percussive events)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration_sec = len(y) / sr
    onset_density = len(onsets) / duration_sec if duration_sec > 0 else 0

    # Spectral centroid (brightness / timbre)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Low frequency energy ratio (bass presence)
    # Compare energy below 250Hz to total
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_mask = freqs < 250
    low_energy = np.sum(S[low_freq_mask, :] ** 2)
    total_energy = np.sum(S ** 2)
    low_freq_ratio = float(low_energy / total_energy) if total_energy > 0 else 0

    return {
        "rms_mean": round(rms_mean, 6),
        "rms_std": round(rms_std, 6),
        "onset_density": round(onset_density, 2),
        "spectral_centroid": round(centroid_mean, 1),
        "low_freq_ratio": round(low_freq_ratio, 3)
    }


def analyze_micro_segment(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Enhanced analysis for short segments (1-2 bars) to detect fills and vocals.
    Includes frequency band analysis for vocal detection.
    """
    duration_sec = len(y) / sr
    if duration_sec < 0.5:
        return {}

    # Basic energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))

    # Onset detection - higher resolution for fills
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_density = len(onsets) / duration_sec if duration_sec > 0 else 0

    # Onset burstiness - are onsets clustered (fill-like) or evenly spaced?
    if len(onsets) >= 3:
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        onset_intervals = np.diff(onset_times)
        onset_variance = float(np.std(onset_intervals)) if len(onset_intervals) > 0 else 0
    else:
        onset_variance = 0

    # Frequency band analysis for vocal detection
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Bass (20-250 Hz)
    bass_mask = (freqs >= 20) & (freqs < 250)
    bass_energy = np.sum(S[bass_mask, :] ** 2)

    # Low-mids (250-500 Hz) - male vocals, warmth
    low_mid_mask = (freqs >= 250) & (freqs < 500)
    low_mid_energy = np.sum(S[low_mid_mask, :] ** 2)

    # Mids (500-2000 Hz) - vocal presence, most intelligibility
    mid_mask = (freqs >= 500) & (freqs < 2000)
    mid_energy = np.sum(S[mid_mask, :] ** 2)

    # High-mids (2000-4000 Hz) - vocal clarity, brightness
    high_mid_mask = (freqs >= 2000) & (freqs < 4000)
    high_mid_energy = np.sum(S[high_mid_mask, :] ** 2)

    # Highs (4000+ Hz) - air, sibilance
    high_mask = freqs >= 4000
    high_energy = np.sum(S[high_mask, :] ** 2)

    total_energy = np.sum(S ** 2)
    if total_energy > 0:
        bass_ratio = float(bass_energy / total_energy)
        low_mid_ratio = float(low_mid_energy / total_energy)
        mid_ratio = float(mid_energy / total_energy)
        high_mid_ratio = float(high_mid_energy / total_energy)
        high_ratio = float(high_energy / total_energy)
        # Vocal band ratio (250Hz - 4kHz captures most vocal energy)
        vocal_band_ratio = float((low_mid_energy + mid_energy + high_mid_energy) / total_energy)
    else:
        bass_ratio = low_mid_ratio = mid_ratio = high_mid_ratio = high_ratio = vocal_band_ratio = 0

    # Spectral flatness - vocals tend to be less flat than noise/synths
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness))

    # Spectral contrast - vocals have distinct formants (peaks)
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = float(np.mean(contrast))
    except Exception:
        contrast_mean = 0

    return {
        "rms_mean": round(rms_mean, 6),
        "rms_max": round(rms_max, 6),
        "onset_density": round(onset_density, 2),
        "onset_variance": round(onset_variance, 4),
        "bass_ratio": round(bass_ratio, 4),
        "low_mid_ratio": round(low_mid_ratio, 4),
        "mid_ratio": round(mid_ratio, 4),
        "high_mid_ratio": round(high_mid_ratio, 4),
        "high_ratio": round(high_ratio, 4),
        "vocal_band_ratio": round(vocal_band_ratio, 4),
        "spectral_flatness": round(flatness_mean, 4),
        "spectral_contrast": round(contrast_mean, 4)
    }


def detect_micro_events(path: str, sr: int, beat_times: List[float],
                        first_downbeat_idx: int, beats_per_bar: int = 4,
                        bars_per_micro: int = 2,
                        duration_limit: float = None) -> List[Dict[str, Any]]:
    """
    Detect fine-grained events (fills, vocals) at 2-bar resolution.

    This runs separately from the 8-bar structural analysis to catch
    events that happen within phrases.

    Args:
        path: Audio file path
        sr: Sample rate
        beat_times: Array of beat positions
        first_downbeat_idx: Index of first downbeat (beat 1)
        beats_per_bar: Beats per bar (default 4)
        bars_per_micro: Bars per micro-segment (default 2)
        duration_limit: Only analyze up to this many seconds (for testing)

    Returns:
        List of micro-events with time, bar, and type
    """
    if not beat_times or len(beat_times) < 8:
        return []

    beat_times = np.array(beat_times)
    beats_per_segment = bars_per_micro * beats_per_bar

    # Generate micro-segment boundaries
    micro_segments = []
    current_beat_idx = first_downbeat_idx
    bar_num = 0

    while current_beat_idx < len(beat_times):
        end_beat_idx = current_beat_idx + beats_per_segment

        start_time = beat_times[current_beat_idx]

        # Check duration limit
        if duration_limit and start_time > duration_limit:
            break

        if end_beat_idx < len(beat_times):
            end_time = beat_times[end_beat_idx]
        else:
            # Use last beat + estimated duration
            if len(beat_times) > 1:
                avg_interval = np.median(np.diff(beat_times[-10:]))
                remaining_beats = beats_per_segment - (len(beat_times) - current_beat_idx)
                end_time = beat_times[-1] + remaining_beats * avg_interval
            else:
                break

        micro_segments.append({
            "start": float(start_time),
            "end": float(end_time),
            "bar_start": bar_num,
            "bar_end": bar_num + bars_per_micro
        })

        current_beat_idx = end_beat_idx
        bar_num += bars_per_micro

    if len(micro_segments) < 3:
        return []

    # Analyze each micro-segment
    micro_data = []
    for seg in micro_segments:
        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            continue

        try:
            y, _ = librosa.load(path, sr=sr, mono=True, offset=seg["start"], duration=duration)
            if y is not None and len(y) > sr * 0.3:
                analysis = analyze_micro_segment(y, sr)
                analysis["start"] = seg["start"]
                analysis["end"] = seg["end"]
                analysis["bar_start"] = seg["bar_start"]
                analysis["bar_end"] = seg["bar_end"]
                micro_data.append(analysis)
        except Exception:
            continue

    if len(micro_data) < 3:
        return []

    # Calculate baseline statistics from all segments
    onset_densities = [m["onset_density"] for m in micro_data]
    onset_variances = [m["onset_variance"] for m in micro_data]
    vocal_ratios = [m["vocal_band_ratio"] for m in micro_data]
    mid_ratios = [m["mid_ratio"] for m in micro_data]
    contrasts = [m["spectral_contrast"] for m in micro_data]

    onset_mean = np.mean(onset_densities)
    onset_std = np.std(onset_densities) if len(onset_densities) > 1 else 1
    vocal_mean = np.mean(vocal_ratios)
    vocal_std = np.std(vocal_ratios) if len(vocal_ratios) > 1 else 0.01
    mid_mean = np.mean(mid_ratios)
    mid_std = np.std(mid_ratios) if len(mid_ratios) > 1 else 0.01
    contrast_mean = np.mean(contrasts)
    contrast_std = np.std(contrasts) if len(contrasts) > 1 else 0.1

    events = []

    for i, seg in enumerate(micro_data):
        event_types = []

        # --- FILL DETECTION ---
        # Fills are characterized by:
        # 1. Sudden increase in onset density compared to LOCAL context (not just global)
        # 2. Occur at phrase boundaries (bars 6-8 of 8-bar phrase, or 14-16 of 16-bar)
        # 3. Often have high onset variance (irregular timing)

        onset_z = (seg["onset_density"] - onset_mean) / onset_std if onset_std > 0 else 0

        # Calculate LOCAL onset context (4 segments before and after)
        local_start = max(0, i - 4)
        local_end = min(len(micro_data), i + 4)
        local_onsets = [micro_data[j]["onset_density"] for j in range(local_start, local_end) if j != i]
        local_onset_mean = np.mean(local_onsets) if local_onsets else onset_mean
        local_onset_max = np.max(local_onsets) if local_onsets else onset_mean

        # Compare to immediate previous segment
        prev_onset = micro_data[i-1]["onset_density"] if i > 0 else seg["onset_density"]
        onset_increase = seg["onset_density"] - prev_onset
        onset_local_z = (seg["onset_density"] - local_onset_mean) / (np.std(local_onsets) + 0.1) if local_onsets else 0

        # Position check: fills often at phrase boundaries
        bar_in_8 = seg["bar_start"] % 8   # Position in 8-bar phrase
        bar_in_16 = seg["bar_start"] % 16  # Position in 16-bar phrase
        is_phrase_end_8 = bar_in_8 >= 6    # Last 2 bars of 8-bar phrase
        is_phrase_end_16 = bar_in_16 >= 14  # Last 2 bars of 16-bar phrase
        is_phrase_boundary = is_phrase_end_8 or is_phrase_end_16

        # Fill detection criteria
        # Fills are typically at bars 6-7 of 8-bar phrases (positions where bar % 8 == 6 or 7)
        # Key characteristics:
        # 1. Elevated onset variance (irregular timing - the defining feature of fills)
        # 2. At phrase boundary positions
        # 3. Sometimes with increased onset density
        is_fill = False
        fill_confidence = 0

        # Path 1: High onset variance at phrase boundary (main fill indicator)
        # Fills have irregular timing (variance > 0.04) vs steady beat (variance < 0.02)
        if is_phrase_boundary and seg["onset_variance"] > 0.045:
            is_fill = True
            fill_confidence = seg["onset_variance"] * 10

        # Path 2: Strong local spike in onset density at phrase boundary
        if onset_local_z > 1.0 and onset_increase > 0.4 and is_phrase_boundary:
            is_fill = True
            fill_confidence = max(fill_confidence, onset_local_z)

        # Path 3: Very strong onset increase (likely a fill regardless of position)
        if onset_increase > 1.2:
            is_fill = True
            fill_confidence = max(fill_confidence, onset_increase)

        # Path 4: Combined moderate signals at phrase boundary
        if is_phrase_boundary and seg["onset_variance"] > 0.035 and onset_local_z > 0.3:
            is_fill = True
            fill_confidence = max(fill_confidence, seg["onset_variance"] * 8 + onset_local_z * 0.3)

        if is_fill:
            event_types.append("fill")

        # --- VOCAL DETECTION ---
        # Vocals cause: increased mid/vocal band ratio, increased spectral contrast
        # Need SIGNIFICANT changes to avoid noise

        vocal_z = (seg["vocal_band_ratio"] - vocal_mean) / vocal_std if vocal_std > 0 else 0
        mid_z = (seg["mid_ratio"] - mid_mean) / mid_std if mid_std > 0 else 0
        contrast_z = (seg["spectral_contrast"] - contrast_mean) / contrast_std if contrast_std > 0 else 0

        # Check for vocal entry (compare to previous segment)
        if i > 0 and "fill" not in event_types:
            prev = micro_data[i - 1]
            vocal_increase = seg["vocal_band_ratio"] - prev["vocal_band_ratio"]
            mid_increase = seg["mid_ratio"] - prev["mid_ratio"]
            contrast_increase = seg["spectral_contrast"] - prev["spectral_contrast"]

            # Stricter thresholds - need BOTH significant increase AND elevated level
            # to avoid false positives
            vocals_in_detected = False
            vocals_out_detected = False

            # Vocals in: large increase in vocal band + elevated contrast
            if vocal_increase > 0.08 and contrast_increase > 0.5 and vocal_z > 0.8:
                vocals_in_detected = True
            # Alternative: very large increase in vocal band alone
            elif vocal_increase > 0.15:
                vocals_in_detected = True

            # Vocals out: significant decrease
            if vocal_increase < -0.08 and contrast_increase < -0.3:
                vocals_out_detected = True

            if vocals_in_detected:
                event_types.append("vocals_in")
            elif vocals_out_detected:
                event_types.append("vocals_out")

        # --- NEW ELEMENT DETECTION ---
        # Look for spectral changes that aren't fills or vocals
        if i > 0 and not event_types:
            prev = micro_data[i - 1]
            # Significant high-mid change (new synth, effects)
            high_mid_change = abs(seg["high_mid_ratio"] - prev["high_mid_ratio"])
            bass_change = abs(seg["bass_ratio"] - prev["bass_ratio"])

            # Stricter threshold
            if high_mid_change > 0.05 or bass_change > 0.08:
                event_types.append("new_element")

        if event_types:
            events.append({
                "time": round(seg["start"], 3),
                "bar": seg["bar_start"],
                "types": event_types,
                "onset_density": seg["onset_density"],
                "onset_increase": round(onset_increase, 2),
                "onset_local_z": round(onset_local_z, 2),
                "onset_z": round(onset_z, 2),
                "onset_variance": seg["onset_variance"],
                "vocal_band_ratio": seg["vocal_band_ratio"],
                "vocal_z": round(vocal_z, 2),
                "bar_in_phrase": bar_in_8
            })

    return events


def classify_micro_event(event_types: List[str]) -> str:
    """Convert micro-event types to human-readable label."""
    if "fill" in event_types:
        return "fill"
    elif "vocals_in" in event_types:
        return "vocals in"
    elif "vocals_out" in event_types:
        return "vocals out"
    elif "new_element" in event_types:
        return "new element"
    else:
        return "change"


def detect_energy_events(segments: List[Dict[str, Any]],
                         rms_threshold: float = 1.0,
                         onset_threshold: float = 0.8,
                         local_window: int = 4) -> List[Dict[str, Any]]:
    """
    Detect significant energy changes between segments.

    Uses both global and LOCAL comparisons to catch gradual builds
    that might be missed when comparing to whole-track statistics.

    Returns list of events like:
    - "energy_increase" (drums come in, drop)
    - "energy_decrease" (breakdown, outro)
    - "texture_change" (new instrument, filter sweep)
    """
    if len(segments) < 2:
        return []

    events = []

    # Get arrays of metrics
    rms_values = [s.get("energy", {}).get("rms_mean", 0) for s in segments]
    onset_values = [s.get("energy", {}).get("onset_density", 0) for s in segments]
    centroid_values = [s.get("energy", {}).get("spectral_centroid", 0) for s in segments]
    low_freq_values = [s.get("energy", {}).get("low_freq_ratio", 0) for s in segments]
    rms_std_values = [s.get("energy", {}).get("rms_std", 0) for s in segments]

    # Calculate global stats for relative comparisons
    rms_std = np.std(rms_values) if len(rms_values) > 1 else 1
    onset_std = np.std(onset_values) if len(onset_values) > 1 else 1
    centroid_std = np.std(centroid_values) if len(centroid_values) > 1 else 1

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_seg = segments[i - 1]

        prev_rms = prev_seg.get("energy", {}).get("rms_mean", 0)
        curr_rms = seg.get("energy", {}).get("rms_mean", 0)
        prev_onset = prev_seg.get("energy", {}).get("onset_density", 0)
        curr_onset = seg.get("energy", {}).get("onset_density", 0)
        prev_centroid = prev_seg.get("energy", {}).get("spectral_centroid", 0)
        curr_centroid = seg.get("energy", {}).get("spectral_centroid", 0)
        prev_low = prev_seg.get("energy", {}).get("low_freq_ratio", 0)
        curr_low = seg.get("energy", {}).get("low_freq_ratio", 0)
        prev_rms_std = prev_seg.get("energy", {}).get("rms_std", 0)
        curr_rms_std = seg.get("energy", {}).get("rms_std", 0)

        event_types = []

        # Calculate LOCAL stats (nearby segments) for more sensitive detection
        local_start = max(0, i - local_window)
        local_end = min(len(segments), i + local_window)
        local_rms = rms_values[local_start:local_end]
        local_onset = onset_values[local_start:local_end]
        local_centroid = centroid_values[local_start:local_end]

        local_rms_std = np.std(local_rms) if len(local_rms) > 1 else rms_std
        local_onset_std = np.std(local_onset) if len(local_onset) > 1 else onset_std
        local_centroid_std = np.std(local_centroid) if len(local_centroid) > 1 else centroid_std

        # Use the MORE SENSITIVE of global or local comparison
        use_rms_std = min(rms_std, local_rms_std) if local_rms_std > 0.001 else rms_std
        use_onset_std = min(onset_std, local_onset_std) if local_onset_std > 0.1 else onset_std
        use_centroid_std = min(centroid_std, local_centroid_std) if local_centroid_std > 10 else centroid_std

        # RMS change (energy/loudness)
        if use_rms_std > 0.0001:
            rms_change_norm = (curr_rms - prev_rms) / use_rms_std
            # Also check absolute change (>15% relative change is significant)
            rms_change_pct = (curr_rms - prev_rms) / prev_rms if prev_rms > 0.01 else 0
            if rms_change_norm > rms_threshold or rms_change_pct > 0.15:
                event_types.append("energy_increase")
            elif rms_change_norm < -rms_threshold or rms_change_pct < -0.15:
                event_types.append("energy_decrease")

        # Onset density change (drums/percussion)
        if use_onset_std > 0.1:
            onset_change_norm = (curr_onset - prev_onset) / use_onset_std
            # Also check absolute change (>0.5 onsets/sec is noticeable)
            onset_change_abs = curr_onset - prev_onset
            if onset_change_norm > onset_threshold or onset_change_abs > 0.8:
                event_types.append("drums_increase")
            elif onset_change_norm < -onset_threshold or onset_change_abs < -0.8:
                event_types.append("drums_decrease")

        # Spectral centroid change (brightness/new instruments)
        if use_centroid_std > 10:
            centroid_change_norm = (curr_centroid - prev_centroid) / use_centroid_std
            # Also check absolute change (>200 Hz shift is noticeable)
            centroid_change_abs = abs(curr_centroid - prev_centroid)
            if abs(centroid_change_norm) > 1.0 or centroid_change_abs > 250:
                event_types.append("timbre_change")

        # Low frequency change (bass drop/removal)
        low_change = curr_low - prev_low
        if low_change > 0.08:  # Lowered from 0.1
            event_types.append("bass_increase")
        elif low_change < -0.08:
            event_types.append("bass_decrease")

        # NEW: Detect texture changes via RMS variance
        # If the segment becomes more "dynamic" (higher variance), something changed
        rms_var_change = curr_rms_std - prev_rms_std
        if abs(rms_var_change) > 0.02 and "timbre_change" not in event_types:
            event_types.append("texture_change")

        if event_types:
            events.append({
                "time": seg.get("start", 0),
                "bar": seg.get("bar_start"),
                "types": event_types,
                "rms_change": round(curr_rms - prev_rms, 6),
                "onset_change": round(curr_onset - prev_onset, 2),
                "centroid_change": round(curr_centroid - prev_centroid, 1)
            })

    return events


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


# -----------------------------
# Starts generation: explicit or BPM/bars
# -----------------------------
def parse_starts_arg(starts_str: Optional[str]) -> Optional[List[float]]:
    if not starts_str:
        return None
    parts = [p.strip() for p in starts_str.split(",") if p.strip()]
    return [float(p) for p in parts]


def bars_to_seconds(bars: float, bpm: float, beats_per_bar: float = 4.0) -> float:
    return (bars * beats_per_bar * 60.0) / bpm


def make_bar_starts(bpm: float, bar_start: int, bar_step: int, num_windows: int, beats_per_bar: float) -> List[float]:
    starts = []
    for i in range(num_windows):
        bars = bar_start + i * bar_step
        starts.append(round(bars_to_seconds(bars, bpm, beats_per_bar), 3))
    return starts


# -----------------------------
# Analysis functions
# -----------------------------
def analyze_window(path: str, sr: int, mono: bool, start: float, duration: float, use_harmonic: bool) -> Dict[str, Any]:
    y, _sr = load_audio_segment(path, sr=sr, mono=mono, start=start, duration=duration)
    if y is None or len(y) < sr * 5:
        return {"start": float(start), "duration": float(duration), "error": "Window too short or failed to load."}

    if use_harmonic:
        y = harmonic_only(y)

    est = estimate_key_from_audio(y, sr)
    cam = to_camelot(est.tonic_pc, est.mode)

    return {
        "start": float(start),
        "duration": float(duration),
        "key": cam["key"],
        "camelot": cam["camelot"],
        "confidence": est.confidence,
        "score": est.score,
        "top_candidates": est.top_candidates
    }


def strict_consensus(windows: List[Dict[str, Any]], min_conf: float) -> Dict[str, Any]:
    """
    STRICT consensus:
    - Only windows with confidence >= min_conf count.
    - If none qualify, fall back to the single best-confidence window (and warn).
    Returns both exact camelot and family (number) consensus.
    """
    valid = [w for w in windows if w.get("camelot") and w.get("confidence") is not None]
    if not valid:
        return {"error": "No valid windows produced a Camelot result.", "windows_used": 0}

    strong = [w for w in valid if float(w["confidence"]) >= float(min_conf)]

    used = strong
    fallback_used = False
    if not strong:
        # fallback to best-confidence window so you always get something actionable
        used = [max(valid, key=lambda w: (float(w.get("confidence", 0.0)), float(w.get("score", 0.0))))]
        fallback_used = True

    # Exact camelot vote
    camelot_counts = Counter(w["camelot"] for w in used)
    best_camelot, best_count = camelot_counts.most_common(1)[0]
    stability_exact = best_count / len(used)

    # Family (number) vote
    nums = [camelot_number(w["camelot"]) for w in used]
    nums = [n for n in nums if n is not None]
    num_counts = Counter(nums)
    best_num, best_num_count = num_counts.most_common(1)[0]
    stability_family = best_num_count / len(used)

    # Representative windows
    best_exact_window = max([w for w in used if w["camelot"] == best_camelot],
                            key=lambda w: (float(w["confidence"]), float(w["score"])))
    best_family_window = max([w for w in used if camelot_number(w["camelot"]) == best_num],
                             key=lambda w: (float(w["confidence"]), float(w["score"])))

    return {
        "min_confidence": float(min_conf),
        "fallback_used": fallback_used,
        "windows_used": len(used),
        "exact": {
            "camelot": best_camelot,
            "key": best_exact_window["key"],
            "stability": round(stability_exact, 3),
            "counts": dict(camelot_counts),
        },
        "family": {
            "camelot_number": best_num,
            "representative": {
                "camelot": best_family_window["camelot"],
                "key": best_family_window["key"],
            },
            "stability": round(stability_family, 3),
            "counts": {str(k): v for k, v in num_counts.items()},
        }
    }


def consensus_analysis(path: str,
                       sr: int,
                       mono: bool,
                       window: float,
                       starts: List[float],
                       use_harmonic: bool,
                       min_conf: float) -> Dict[str, Any]:
    windows = [analyze_window(path, sr, mono, st, window, use_harmonic) for st in starts]
    consensus = strict_consensus(windows, min_conf=min_conf)
    return {"consensus": consensus, "windows": windows}


# -----------------------------
# Timeline Analysis
# -----------------------------
@dataclass
class TimelineSegment:
    """A segment of the timeline with key information."""
    start: float
    end: float
    key: str
    camelot: str
    confidence: float
    score: float
    segment_type: str = "stable"  # "stable", "transition", "uncertain"
    bar_start: Optional[int] = None
    bar_end: Optional[int] = None


def get_audio_duration(path: str, sr: int) -> float:
    """Get the total duration of an audio file in seconds."""
    return librosa.get_duration(path=path, sr=sr)


def generate_segment_times_dynamic(duration: float, beat_times: List[float],
                                    bars_per_segment: int = 8,
                                    beats_per_bar: int = 4,
                                    first_downbeat_idx: int = 0,
                                    bpm: float = None) -> List[Dict[str, Any]]:
    """
    Generate segment boundaries by counting through the actual beat array.
    This is the dynamic beat-based approach that naturally handles tempo variations.

    Instead of calculating times from BPM, we directly use beat positions:
    - Segment N starts at beat (N * bars_per_segment * beats_per_bar) from first downbeat
    - This naturally handles tempo variations and avoids drift

    Args:
        duration: Total track duration in seconds
        beat_times: List of actual beat positions in seconds
        bars_per_segment: Number of bars per segment (default 8)
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        first_downbeat_idx: Index of first downbeat in beat_times array
        bpm: BPM for local tempo calculation (optional)

    Returns:
        List of segment dictionaries with start, end, bar info, and local BPM
    """
    if not beat_times or len(beat_times) < 2:
        # Fall back to simple duration-based segments if no beats
        return [{
            "start": 0.0,
            "end": round(duration, 4),
            "bar_start": 0,
            "bar_end": bars_per_segment,
            "beats_in_segment": 0,
            "segment_type": "fallback"
        }]

    beats_per_segment = bars_per_segment * beats_per_bar
    beat_times = np.array(beat_times)

    segments = []
    bar_num = 0

    # Start from the first downbeat
    current_beat_idx = first_downbeat_idx

    # Handle any content before the first downbeat
    if first_downbeat_idx > 0 and beat_times[0] > 0.1:
        end_time = float(beat_times[first_downbeat_idx])
        segments.append({
            "start": 0.0,
            "end": round(end_time, 4),
            "bar_start": -1,  # Pre-downbeat content
            "bar_end": 0,
            "beat_idx_start": 0,
            "beat_idx_end": first_downbeat_idx,
            "beats_in_segment": first_downbeat_idx,
            "segment_type": "intro_pickup",
            "local_bpm": None
        })

    while current_beat_idx < len(beat_times):
        # Find the end beat for this segment
        end_beat_idx = current_beat_idx + beats_per_segment

        # Get start time
        start_time = beat_times[current_beat_idx]

        # Get end time (or track end if we run out of beats)
        if end_beat_idx < len(beat_times):
            end_time = beat_times[end_beat_idx]
            actual_beats = beats_per_segment
        else:
            end_time = duration
            actual_beats = len(beat_times) - current_beat_idx

        # Calculate actual BPM for this segment based on beat intervals
        segment_beat_times = beat_times[current_beat_idx:min(end_beat_idx, len(beat_times))]
        if len(segment_beat_times) > 1:
            segment_intervals = np.diff(segment_beat_times)
            local_bpm = 60.0 / float(np.median(segment_intervals))
        else:
            local_bpm = bpm

        segment = {
            "start": round(float(start_time), 4),
            "end": round(float(end_time), 4),
            "bar_start": bar_num,
            "bar_end": bar_num + bars_per_segment,
            "beat_idx_start": int(current_beat_idx),
            "beat_idx_end": int(min(end_beat_idx, len(beat_times))),
            "beats_in_segment": int(actual_beats),
            "local_bpm": round(float(local_bpm), 2) if local_bpm else None,
            "segment_type": "full" if actual_beats == beats_per_segment else "partial"
        }

        segments.append(segment)

        # Move to next segment
        current_beat_idx = end_beat_idx
        bar_num += bars_per_segment

    return segments


def generate_segment_times_synthetic(duration: float, bpm: float, bars_per_segment: int = 8,
                                     beats_per_bar: float = 4.0,
                                     first_downbeat: float = 0.0,
                                     median_beat_interval: float = None) -> List[Dict[str, Any]]:
    """
    Generate segment boundaries using synthetic BPM-based calculation.
    This is the legacy approach - kept for comparison and fallback.

    Args:
        duration: Total track duration in seconds
        bpm: Beats per minute
        bars_per_segment: Number of bars per segment (default 8)
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        first_downbeat: Time of first downbeat (beat 1) for phase alignment
        median_beat_interval: Beat interval for beat grid generation
    """
    seconds_per_bar = (beats_per_bar * 60.0) / bpm
    segment_duration = bars_per_segment * seconds_per_bar

    segments = []

    # Generate a synthetic beat grid
    beat_interval = median_beat_interval if median_beat_interval else (60.0 / bpm)

    synthetic_beats = []
    t = first_downbeat
    while t < duration + beat_interval:
        if t >= 0:
            synthetic_beats.append(t)
        t += beat_interval

    t = first_downbeat - beat_interval
    while t >= 0:
        synthetic_beats.insert(0, t)
        t -= beat_interval

    use_beat_snapping = len(synthetic_beats) > 0

    def snap_to_beat(target_time: float) -> float:
        if not use_beat_snapping:
            return target_time
        closest_idx = np.argmin(np.abs(np.array(synthetic_beats) - target_time))
        return synthetic_beats[closest_idx]

    current_time = 0.0
    bar_num = 0

    while current_time < duration:
        theoretical_end = current_time + segment_duration

        if use_beat_snapping and theoretical_end < duration:
            end_time = snap_to_beat(theoretical_end)
        else:
            end_time = min(theoretical_end, duration)

        end_bar = bar_num + bars_per_segment

        if end_time - current_time < segment_duration * 0.5 and segments:
            segments[-1]["end"] = round(end_time, 4)
            segments[-1]["bar_end"] = end_bar
            break

        segments.append({
            "start": round(current_time, 4),
            "end": round(end_time, 4),
            "bar_start": bar_num,
            "bar_end": end_bar,
            "segment_type": "synthetic"
        })

        current_time = end_time
        bar_num = end_bar

    return segments


def generate_segment_times(duration: float, bpm: float, bars_per_segment: int = 8,
                           beats_per_bar: float = 4.0,
                           first_downbeat: float = 0.0,
                           beat_times: List[float] = None,
                           first_downbeat_idx: int = 0,
                           median_beat_interval: float = None,
                           use_dynamic: bool = True) -> List[Dict[str, Any]]:
    """
    Generate segment boundaries aligned to bars.

    Uses dynamic beat-based approach by default (counts through actual beat array).
    Falls back to synthetic BPM-based approach if no beat times provided.

    Args:
        duration: Total track duration in seconds
        bpm: Beats per minute
        bars_per_segment: Number of bars per segment (default 8)
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        first_downbeat: Time of first downbeat (beat 1) for phase alignment
        beat_times: List of actual beat times (for dynamic approach)
        first_downbeat_idx: Index of first downbeat in beat_times array
        median_beat_interval: Beat interval for synthetic grid
        use_dynamic: If True, use dynamic beat-based approach (default)

    Returns:
        List of {start, end, bar_start, bar_end, ...} dicts
    """
    if use_dynamic and beat_times and len(beat_times) >= 8:
        # Use dynamic beat-based approach (recommended)
        return generate_segment_times_dynamic(
            duration=duration,
            beat_times=beat_times,
            bars_per_segment=bars_per_segment,
            beats_per_bar=int(beats_per_bar),
            first_downbeat_idx=first_downbeat_idx,
            bpm=bpm
        )
    else:
        # Fall back to synthetic approach
        return generate_segment_times_synthetic(
            duration=duration,
            bpm=bpm,
            bars_per_segment=bars_per_segment,
            beats_per_bar=beats_per_bar,
            first_downbeat=first_downbeat,
            median_beat_interval=median_beat_interval
        )


def analyze_segments(path: str, sr: int, segments: List[Dict[str, Any]],
                     use_harmonic: bool, include_energy: bool = True) -> List[Dict[str, Any]]:
    """Analyze key and energy for each segment."""
    results = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        analysis = analyze_window(path, sr, True, seg["start"], duration, use_harmonic)
        analysis["bar_start"] = seg.get("bar_start")
        analysis["bar_end"] = seg.get("bar_end")
        analysis["end"] = seg["end"]

        # Add energy analysis (use full mix, not harmonic-only)
        if include_energy:
            try:
                y_full, _ = load_audio_segment(path, sr, True, seg["start"], duration)
                if y_full is not None and len(y_full) > sr:
                    analysis["energy"] = analyze_energy_segment(y_full, sr)
            except Exception:
                pass  # Energy analysis is optional

        results.append(analysis)
    return results


def merge_timeline_segments(segments: List[Dict[str, Any]],
                            min_confidence: float = 0.1,
                            min_run_length: int = 2) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments with the same key to reduce noise.

    Logic:
    - Consecutive segments with same camelot code get merged
    - Single-segment "blips" with low confidence get absorbed into neighbors
    - Genuine transitions are preserved when confidence is reasonable

    Args:
        segments: Raw analyzed segments
        min_confidence: Below this, a segment is considered uncertain
        min_run_length: Minimum consecutive segments to be considered stable
    """
    if not segments:
        return []

    # First pass: identify runs of same key
    runs = []
    current_run = [segments[0]]

    for seg in segments[1:]:
        if seg.get("error"):
            # Error segments break runs
            if current_run:
                runs.append(current_run)
            runs.append([seg])
            current_run = []
        elif not current_run:
            current_run = [seg]
        elif seg.get("camelot") == current_run[-1].get("camelot"):
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]

    if current_run:
        runs.append(current_run)

    # Second pass: merge small uncertain runs into neighbors
    merged_runs = []
    i = 0
    while i < len(runs):
        run = runs[i]

        # Check if this is a short, low-confidence run
        avg_conf = sum(s.get("confidence", 0) for s in run) / len(run) if run else 0
        is_uncertain = len(run) < min_run_length and avg_conf < min_confidence

        if is_uncertain and merged_runs and i + 1 < len(runs):
            # Check if neighbors have same key - if so, absorb this blip
            prev_key = merged_runs[-1][-1].get("camelot") if merged_runs[-1] else None
            next_key = runs[i + 1][0].get("camelot") if runs[i + 1] else None

            if prev_key == next_key and prev_key is not None:
                # Absorb into previous run (mark as uncertain)
                for seg in run:
                    seg["absorbed"] = True
                    seg["original_camelot"] = seg.get("camelot")
                    seg["camelot"] = prev_key
                    seg["key"] = merged_runs[-1][-1].get("key")
                merged_runs[-1].extend(run)
                i += 1
                continue

        merged_runs.append(run)
        i += 1

    # Third pass: create final timeline entries
    timeline = []
    for run in merged_runs:
        if not run:
            continue

        # Check for errors
        if run[0].get("error"):
            timeline.append({
                "start": run[0]["start"],
                "end": run[-1].get("end", run[-1]["start"] + run[-1].get("duration", 0)),
                "type": "error",
                "error": run[0]["error"]
            })
            continue

        # Calculate aggregate stats for the run
        avg_confidence = sum(s.get("confidence", 0) for s in run) / len(run)
        avg_score = sum(s.get("score", 0) for s in run) / len(run)
        max_confidence = max(s.get("confidence", 0) for s in run)

        # Determine segment type
        if len(run) >= min_run_length and avg_confidence >= min_confidence:
            seg_type = "stable"
        elif avg_confidence < min_confidence * 0.5:
            seg_type = "uncertain"
        else:
            seg_type = "transition"

        entry = {
            "start": round(run[0]["start"], 3),
            "end": round(run[-1].get("end", run[-1]["start"] + run[-1].get("duration", 0)), 3),
            "key": run[0].get("key"),
            "camelot": run[0].get("camelot"),
            "type": seg_type,
            "confidence": {
                "average": round(avg_confidence, 3),
                "max": round(max_confidence, 3)
            },
            "score": round(avg_score, 4),
            "segments_merged": len(run)
        }

        # Add bar info if available
        if run[0].get("bar_start") is not None:
            entry["bar_start"] = run[0]["bar_start"]
            entry["bar_end"] = run[-1].get("bar_end")

        # Flag if any segments were absorbed
        absorbed_count = sum(1 for s in run if s.get("absorbed"))
        if absorbed_count > 0:
            entry["absorbed_uncertain"] = absorbed_count

        timeline.append(entry)

    return timeline


def compute_timeline_summary(timeline: List[Dict[str, Any]],
                             total_duration: float) -> Dict[str, Any]:
    """
    Compute summary statistics for the timeline.
    Returns dominant key, coverage stats, mix points (intro/outro keys).
    """
    if not timeline:
        return {"error": "No timeline data"}

    # Filter to valid key entries
    valid = [t for t in timeline if t.get("camelot") and t.get("type") != "error"]
    if not valid:
        return {"error": "No valid key regions found"}

    # Calculate coverage by key
    key_durations = Counter()
    for entry in valid:
        duration = entry["end"] - entry["start"]
        key_durations[entry["camelot"]] += duration

    # Dominant key
    dominant_camelot, dominant_duration = key_durations.most_common(1)[0]
    dominant_entry = next(t for t in valid if t["camelot"] == dominant_camelot)

    # Key changes
    key_changes = []
    for i in range(1, len(valid)):
        if valid[i]["camelot"] != valid[i-1]["camelot"]:
            key_changes.append({
                "time": valid[i]["start"],
                "bar": valid[i].get("bar_start"),
                "from": valid[i-1]["camelot"],
                "to": valid[i]["camelot"]
            })

    # Mix points (intro = first stable, outro = last stable)
    stable = [t for t in valid if t.get("type") == "stable"]
    intro_key = stable[0] if stable else valid[0]
    outro_key = stable[-1] if stable else valid[-1]

    return {
        "dominant": {
            "key": dominant_entry["key"],
            "camelot": dominant_camelot,
            "duration": round(dominant_duration, 2),
            "coverage": round(dominant_duration / total_duration, 3)
        },
        "key_changes": key_changes,
        "total_regions": len(valid),
        "stable_regions": len([t for t in valid if t.get("type") == "stable"]),
        "mix_points": {
            "intro": {
                "time": intro_key["start"],
                "key": intro_key["key"],
                "camelot": intro_key["camelot"]
            },
            "outro": {
                "time": outro_key["start"],
                "key": outro_key["key"],
                "camelot": outro_key["camelot"]
            }
        }
    }


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"


def format_timeline_human(timeline_data: Dict[str, Any]) -> str:
    """
    Format timeline analysis as human-readable text.

    Format: start-end (bar_start:bar_end) -> camelot (key) - type (score)
    Example: 0:00-1:30 (0:48) -> 8A (A minor) - stable (0.9416)
    """
    lines = []

    # Header with summary
    summary = timeline_data.get("summary", {})
    dominant = summary.get("dominant", {})
    if dominant:
        lines.append(f"Dominant: {dominant.get('camelot')} ({dominant.get('key')}) - {dominant.get('coverage', 0)*100:.1f}% coverage")

    bpm = timeline_data.get("bpm")
    duration = timeline_data.get("duration", 0)
    if bpm and duration:
        lines.append(f"Duration: {format_time(duration)} | BPM: {bpm}")

    # Key changes summary
    key_changes = summary.get("key_changes", [])
    energy_events = timeline_data.get("energy_events", [])
    if key_changes:
        lines.append(f"Key changes: {len(key_changes)}")
    else:
        lines.append("Key changes: none")
    lines.append(f"Energy events: {len(energy_events)}")

    lines.append("")
    lines.append("Timeline (Key):")
    lines.append("-" * 60)

    # Build a map of energy events by bar for interleaving
    energy_by_bar = {}
    for event in energy_events:
        bar = event.get("bar")
        if bar is not None:
            energy_by_bar[bar] = event

    # Timeline entries
    for entry in timeline_data.get("timeline", []):
        start_time = format_time(entry.get("start", 0))
        end_time = format_time(entry.get("end", 0))

        bar_start = entry.get("bar_start")
        bar_end = entry.get("bar_end")
        bar_info = f" ({bar_start}:{bar_end})" if bar_start is not None else ""

        camelot = entry.get("camelot", "?")
        key = entry.get("key", "unknown")
        seg_type = entry.get("type", "unknown")
        score = entry.get("score", 0)

        # Confidence info
        conf = entry.get("confidence", {})
        avg_conf = conf.get("average", 0) if isinstance(conf, dict) else conf

        line = f"{start_time}-{end_time}{bar_info} -> {camelot} ({key}) - {seg_type} ({score:.4f})"
        lines.append(line)

    lines.append("-" * 60)

    # Energy events section
    if energy_events:
        lines.append("")
        lines.append("Energy Events (Structural):")
        lines.append("-" * 60)
        for event in energy_events:
            time_str = format_time(event.get("time", 0))
            bar = event.get("bar", "?")
            event_types = event.get("types", [])
            label = classify_energy_event(event_types)
            lines.append(f"{time_str} (bar {bar}) -> {label.upper()}")
        lines.append("-" * 60)

    # Mix points
    mix_points = summary.get("mix_points", {})
    intro = mix_points.get("intro", {})
    outro = mix_points.get("outro", {})
    if intro:
        lines.append(f"Intro: {intro.get('camelot')} ({intro.get('key')}) @ {format_time(intro.get('time', 0))}")
    if outro:
        lines.append(f"Outro: {outro.get('camelot')} ({outro.get('key')}) @ {format_time(outro.get('time', 0))}")

    return "\n".join(lines)


def timeline_analysis(path: str, sr: int, bpm: float,
                      bars_per_segment: int = 8,
                      beats_per_bar: float = 4.0,
                      use_harmonic: bool = True,
                      min_confidence: float = 0.1,
                      min_stable_segments: int = 2,
                      bpm_info: Dict[str, Any] = None,
                      use_dynamic_segments: bool = True,
                      detect_micro: bool = True,
                      micro_bars: int = 2,
                      classify_rhythm_flag: bool = True) -> Dict[str, Any]:
    """
    Full timeline analysis of a track.

    1. Divides track into bar-aligned segments (dynamic beat-based by default)
    2. Analyzes key per segment
    3. Merges stable regions, identifies transitions
    4. Optionally detects micro-events (fills, vocals) at finer resolution
    5. Returns timeline + summary

    Args:
        bpm_info: Enhanced BPM data with beat_times for dynamic segment alignment
        use_dynamic_segments: If True, use dynamic beat-based segments (recommended)
        detect_micro: If True, run fine-grained event detection (fills, vocals)
        micro_bars: Bars per micro-segment for fine-grained detection (default 2)
    """
    # Get duration
    duration = get_audio_duration(path, sr)

    # Extract beat alignment info if available
    first_downbeat = 0.0
    first_downbeat_idx = 0
    beat_times = None
    median_beat_interval = None
    downbeat_method = None
    downbeat_confidence = None

    if bpm_info:
        first_downbeat = bpm_info.get("first_downbeat", 0.0)
        first_downbeat_idx = bpm_info.get("first_downbeat_idx", 0)
        beat_times = bpm_info.get("beat_times")
        median_beat_interval = bpm_info.get("median_beat_interval")
        downbeat_method = bpm_info.get("downbeat_method")
        downbeat_confidence = bpm_info.get("downbeat_confidence")

    # Generate segments
    # Dynamic approach uses actual beat positions (no drift over time)
    # Synthetic approach uses BPM calculation (may drift)
    segments = generate_segment_times(
        duration=duration,
        bpm=bpm,
        bars_per_segment=bars_per_segment,
        beats_per_bar=beats_per_bar,
        first_downbeat=first_downbeat,
        beat_times=beat_times,
        first_downbeat_idx=first_downbeat_idx,
        median_beat_interval=median_beat_interval,
        use_dynamic=use_dynamic_segments
    )

    segment_method = "dynamic_beats" if (use_dynamic_segments and beat_times) else "synthetic_bpm"

    # Analyze each segment (key + energy)
    raw_segments = analyze_segments(path, sr, segments, use_harmonic, include_energy=True)

    # Detect energy events (structural changes at 8-bar resolution)
    energy_events = detect_energy_events(raw_segments)

    # Detect micro-events (fills, vocals at 2-bar resolution)
    micro_events = []
    if detect_micro and beat_times and len(beat_times) >= 16:
        micro_events = detect_micro_events(
            path=path,
            sr=sr,
            beat_times=beat_times,
            first_downbeat_idx=first_downbeat_idx,
            beats_per_bar=int(beats_per_bar),
            bars_per_micro=micro_bars
        )

    # Rhythmic pattern classification (per-bar kick analysis)
    rhythm_data = None
    if classify_rhythm_flag and beat_times and len(beat_times) >= 8:
        try:
            from rhythm_detect import classify_rhythm, summarize_rhythm, summarize_rhythm_sections

            # Build 1-bar measures from beat grid
            bar_measures = []
            beats_per_bar_int = int(beats_per_bar)
            beat_idx = first_downbeat_idx
            bar_num = 0
            while beat_idx + beats_per_bar_int < len(beat_times):
                bar_start_time = beat_times[beat_idx]
                bar_end_idx = beat_idx + beats_per_bar_int
                bar_end_time = beat_times[bar_end_idx] if bar_end_idx < len(beat_times) else bar_start_time + median_beat_interval * beats_per_bar_int
                bar_measures.append({
                    "measure_num": bar_num,
                    "start": bar_start_time,
                    "end": bar_end_time,
                })
                beat_idx = bar_end_idx
                bar_num += 1

            if bar_measures:
                # Load full audio for rhythm analysis
                y_full, _ = librosa.load(path, sr=sr, mono=True)
                bar_rhythms = classify_rhythm(
                    y_full, sr, bar_measures,
                    beat_interval=median_beat_interval or (60.0 / bpm),
                    beats_per_bar=beats_per_bar_int
                )
                rhythm_summary = summarize_rhythm(bar_rhythms)
                rhythm_sections = summarize_rhythm_sections(bar_rhythms, bar_measures)
                rhythm_data = {
                    "summary": rhythm_summary,
                    "sections": rhythm_sections,
                    "per_bar": bar_rhythms,
                }
                print(f"Rhythm classification: {rhythm_summary['dominant_pattern']} "
                      f"({len(bar_rhythms)} bars)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Rhythm classification failed: {e}", file=sys.stderr)

    # Merge to reduce noise
    timeline = merge_timeline_segments(
        raw_segments,
        min_confidence=min_confidence,
        min_run_length=min_stable_segments
    )

    # Compute summary
    summary = compute_timeline_summary(timeline, duration)

    result = {
        "duration": round(duration, 2),
        "bpm": bpm,
        "bars_per_segment": bars_per_segment,
        "total_segments_analyzed": len(raw_segments),
        "segment_method": segment_method,
        "summary": summary,
        "timeline": timeline,
        "energy_events": energy_events,
        "micro_events": micro_events,
        "raw_segments": raw_segments
    }

    if rhythm_data:
        result["rhythm"] = rhythm_data

    # Include beat grid info if available
    if bpm_info:
        result["beat_grid"] = {
            "first_downbeat": first_downbeat,
            "first_downbeat_idx": first_downbeat_idx,
            "downbeat_method": downbeat_method,
            "downbeat_confidence": downbeat_confidence,
            "bpm_confidence": bpm_info.get("bpm_confidence"),
            "tempo_stable": bpm_info.get("tempo_stable"),
            "tempo_changes": bpm_info.get("tempo_changes"),
            "beat_count": bpm_info.get("beat_count"),
            "beat_intervals": bpm_info.get("beat_intervals")
        }

    return result


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Estimate musical key + Camelot from audio (single or strict consensus).")

    ap.add_argument("--audio", required=False, help="Path to local audio file (WAV/MP3/FLAC recommended). Required unless --compare.")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate for analysis (default: 22050).")

    ap.add_argument("--start", type=float, default=90.0, help="Start time for single-window analysis (seconds).")
    ap.add_argument("--duration", type=float, default=120.0, help="Duration for single-window analysis (seconds).")

    ap.add_argument("--consensus", action="store_true", help="Run strict consensus (multi-window) analysis.")
    ap.add_argument("--timeline", action="store_true", help="Run timeline analysis (key changes over time).")
    ap.add_argument("--window", type=float, default=120.0, help="Window length for consensus analysis (seconds).")
    ap.add_argument("--starts", help="Comma-separated explicit start times for consensus, e.g. '90,148,206'.")

    # BPM/bars window generation
    ap.add_argument("--bpm", help="BPM used to generate starts from bars. Use 'auto' to detect from audio.")
    ap.add_argument("--bar-start", type=int, help="Starting bar number (e.g., 48).")
    ap.add_argument("--bar-step", type=int, default=32, help="Step in bars between windows (default: 32).")
    ap.add_argument("--num-windows", type=int, default=4, help="Number of windows to generate (default: 4).")
    ap.add_argument("--beats-per-bar", type=float, default=4.0, help="Beats per bar (default: 4.0).")
    ap.add_argument("--full-track-beats", action="store_true",
                    help="Analyze full track for beat grid (slower but more accurate alignment).")
    ap.add_argument("--first-downbeat", type=float, default=None,
                    help="Manual first downbeat time in seconds (overrides auto-detection).")
    ap.add_argument("--simple-detection", action="store_true",
                    help="Use simple extrapolation instead of hybrid downbeat detection.")
    ap.add_argument("--synthetic-segments", action="store_true",
                    help="Use synthetic BPM-based segments instead of dynamic beat-based.")

    # Timeline analysis options
    ap.add_argument("--bars-per-segment", type=int, default=8,
                    help="Bars per analysis segment for timeline mode (default: 8).")
    ap.add_argument("--min-stable-segments", type=int, default=2,
                    help="Minimum consecutive segments to be considered stable (default: 2).")

    # Strict consensus tuning
    ap.add_argument("--min-confidence", type=float, default=0.05,
                    help="Minimum confidence for a window to count in strict consensus (default: 0.05).")

    # Harmonic-only toggle
    ap.add_argument("--full-mix", action="store_true", help="Use full mix (disable harmonic-only).")

    # Optional YouTube metadata
    ap.add_argument("--url", help="YouTube URL (optional; metadata only).")
    ap.add_argument("--yt-api-key", default=os.environ.get("YT_API_KEY"),
                    help="YouTube Data API key (optional; defaults to YT_API_KEY env var).")

    # Rhythm classification
    ap.add_argument("--no-rhythm", action="store_true",
                    help="Skip rhythmic pattern classification in timeline mode.")

    # Two-track comparison mode
    compare_group = ap.add_argument_group("Two-track comparison (--compare)")
    compare_group.add_argument("--compare", action="store_true",
        help="Run collision + phase alignment between two tracks.")
    compare_group.add_argument("--audio-a",
        help="Path to Track A audio file (for --compare).")
    compare_group.add_argument("--bars-a",
        help="Bar range for Track A (e.g. '216-240').")
    compare_group.add_argument("--audio-b",
        help="Path to Track B audio file (for --compare).")
    compare_group.add_argument("--bars-b",
        help="Bar range for Track B (e.g. '0-24').")
    compare_group.add_argument("--analysis-a",
        help="Track A analysis cache dir (has analysis_cache.npz). If omitted, computes on the fly.")
    compare_group.add_argument("--analysis-b",
        help="Track B analysis cache dir. If omitted, computes on the fly.")
    compare_group.add_argument("--bpm-a", type=float,
        help="Track A BPM (for phase alignment).")
    compare_group.add_argument("--bpm-b", type=float,
        help="Track B BPM (for phase alignment).")
    compare_group.add_argument("--first-beat-a", type=int,
        help="Track A first_beat_sample (for phase alignment).")
    compare_group.add_argument("--first-beat-b", type=int,
        help="Track B first_beat_sample (for phase alignment).")
    compare_group.add_argument("--min-gap-db", type=float, default=10.0,
        help="Minimum dB gap for spectral gap detection (default: 10).")
    compare_group.add_argument("--max-offset-ms", type=float, default=100.0,
        help="Maximum phase offset search window in ms (default: 100).")
    compare_group.add_argument("--save-collision-plot",
        help="Save collision heatmap to PNG.")

    # Output
    ap.add_argument("--json-out", help="Write JSON output to a file.")
    ap.add_argument("--plot", choices=["summary", "detailed", "both"],
                    help="Generate visualization plots (requires matplotlib).")
    ap.add_argument("--plot-save", help="Save plots to files (prefix, e.g., 'track1' -> track1_summary.png)")

    args = ap.parse_args()

    # --compare mode doesn't need --audio
    if not args.compare and not args.audio:
        raise SystemExit("--audio is required unless using --compare mode.")

    audio_path = args.audio
    if audio_path:
        ext = Path(audio_path).suffix.lower()
        if ext in UNSUPPORTED_HINT_EXTS:
            raise SystemExit(
                f"Input format {ext} often requires ffmpeg. Convert to WAV/MP3/FLAC and re-run."
            )

    yt_meta = {}
    if args.url and args.yt_api_key:
        vid = extract_youtube_id(args.url)
        yt_meta = fetch_youtube_metadata(vid, args.yt_api_key) if vid else {"error": "Could not parse video ID from URL."}

    use_harmonic = not args.full_mix

    # Handle BPM: could be a number, "auto", or None
    bpm_value = None
    bpm_info = None
    if args.bpm is not None:
        if args.bpm.lower() == "auto":
            bpm_info = detect_bpm(audio_path, sr=args.sr)
            if bpm_info.get("bpm"):
                bpm_value = bpm_info["bpm"]
            else:
                print(f"Warning: BPM auto-detection failed: {bpm_info.get('error', 'unknown error')}", file=sys.stderr)
        else:
            try:
                bpm_value = float(args.bpm)
            except ValueError:
                raise SystemExit(f"Invalid --bpm value: {args.bpm}. Use a number or 'auto'.")

    result: Dict[str, Any] = {
        "youtube": yt_meta,
        "audio": {
            "path": audio_path,
            "sr": args.sr,
            "mono": True,
            "harmonic_only": use_harmonic
        }
    }

    # Include BPM detection info if auto was used
    if bpm_info is not None:
        result["bpm_detection"] = bpm_info

    if args.timeline:
        # Timeline mode: requires BPM
        # Use enhanced BPM detection with hybrid downbeat detection for beat alignment
        enhanced_bpm_info = None

        # Determine options from CLI args
        full_track_beats = getattr(args, 'full_track_beats', False)
        manual_first_downbeat = getattr(args, 'first_downbeat', None)
        use_hybrid_downbeat = not getattr(args, 'simple_detection', False)
        use_dynamic_segments = not getattr(args, 'synthetic_segments', False)

        # For dynamic segments, we need full track beats
        if use_dynamic_segments and not full_track_beats:
            print("Dynamic segments enabled - using full track beat analysis...", file=sys.stderr)
            full_track_beats = True

        if bpm_value is None:
            # Auto-detect if not provided
            if full_track_beats:
                print("Timeline mode: Auto-detecting BPM with full track beat analysis...", file=sys.stderr)
            else:
                print("Timeline mode: Auto-detecting BPM with beat grid analysis...", file=sys.stderr)

            enhanced_bpm_info = detect_bpm_enhanced(
                audio_path, sr=args.sr,
                full_track=full_track_beats,
                use_hybrid_downbeat=use_hybrid_downbeat,
                manual_first_downbeat=manual_first_downbeat,
                beats_per_bar=int(args.beats_per_bar),
                verbose=True
            )

            if enhanced_bpm_info.get("bpm"):
                bpm_value = enhanced_bpm_info["bpm"]
                result["bpm_detection"] = {
                    "bpm": enhanced_bpm_info["bpm"],
                    "bpm_librosa": enhanced_bpm_info.get("bpm_librosa"),
                    "bpm_confidence": enhanced_bpm_info.get("bpm_confidence"),
                    "detected_from": enhanced_bpm_info.get("detected_from"),
                    "beat_count": enhanced_bpm_info.get("beat_count"),
                    "first_downbeat": enhanced_bpm_info.get("first_downbeat"),
                    "first_downbeat_idx": enhanced_bpm_info.get("first_downbeat_idx"),
                    "downbeat_method": enhanced_bpm_info.get("downbeat_method"),
                    "downbeat_confidence": enhanced_bpm_info.get("downbeat_confidence"),
                    "tempo_stable": enhanced_bpm_info.get("tempo_stable"),
                    "tempo_changes": enhanced_bpm_info.get("tempo_changes"),
                    "beat_intervals": enhanced_bpm_info.get("beat_intervals")
                }
                # Include downbeat analysis details (extrapolation info, etc.)
                if enhanced_bpm_info.get("downbeat_analysis"):
                    result["bpm_detection"]["downbeat_analysis"] = enhanced_bpm_info["downbeat_analysis"]
            else:
                raise SystemExit(f"Timeline mode requires --bpm. Auto-detection failed: {enhanced_bpm_info.get('error')}")
        else:
            # BPM was provided manually, but still get beat grid for alignment
            if full_track_beats:
                print("Running full track beat grid analysis for segment alignment...", file=sys.stderr)
            else:
                print("Running beat grid analysis for segment alignment...", file=sys.stderr)

            enhanced_bpm_info = detect_bpm_enhanced(
                audio_path, sr=args.sr,
                full_track=full_track_beats,
                use_hybrid_downbeat=use_hybrid_downbeat,
                manual_first_downbeat=manual_first_downbeat,
                beats_per_bar=int(args.beats_per_bar),
                verbose=True
            )

            if enhanced_bpm_info.get("bpm"):
                # Keep user-provided BPM but use detected beat grid for alignment
                result["bpm_detection"] = {
                    "bpm_provided": bpm_value,
                    "bpm_detected": enhanced_bpm_info["bpm"],
                    "bpm_confidence": enhanced_bpm_info.get("bpm_confidence"),
                    "first_downbeat": enhanced_bpm_info.get("first_downbeat"),
                    "first_downbeat_idx": enhanced_bpm_info.get("first_downbeat_idx"),
                    "downbeat_method": enhanced_bpm_info.get("downbeat_method"),
                    "downbeat_confidence": enhanced_bpm_info.get("downbeat_confidence"),
                    "tempo_stable": enhanced_bpm_info.get("tempo_stable"),
                    "tempo_changes": enhanced_bpm_info.get("tempo_changes"),
                    "beat_intervals": enhanced_bpm_info.get("beat_intervals")
                }
                # Include downbeat analysis details (extrapolation info, etc.)
                if enhanced_bpm_info.get("downbeat_analysis"):
                    result["bpm_detection"]["downbeat_analysis"] = enhanced_bpm_info["downbeat_analysis"]

        # Determine segment method for analysis info
        segment_method = "dynamic_beats" if use_dynamic_segments else "synthetic_bpm"
        downbeat_method = enhanced_bpm_info.get("downbeat_method", "unknown") if enhanced_bpm_info else "none"

        result["analysis"] = {
            "type": "timeline",
            "bpm": bpm_value,
            "bars_per_segment": args.bars_per_segment,
            "min_confidence": float(args.min_confidence),
            "min_stable_segments": args.min_stable_segments,
            "segment_method": segment_method,
            "downbeat_method": downbeat_method
        }

        result["estimate"] = timeline_analysis(
            path=audio_path,
            sr=int(args.sr),
            bpm=bpm_value,
            bars_per_segment=args.bars_per_segment,
            beats_per_bar=float(args.beats_per_bar),
            use_harmonic=use_harmonic,
            min_confidence=float(args.min_confidence),
            min_stable_segments=args.min_stable_segments,
            bpm_info=enhanced_bpm_info,
            use_dynamic_segments=use_dynamic_segments,
            classify_rhythm_flag=not getattr(args, 'no_rhythm', False)
        )

    elif args.compare:
        # Two-track comparison mode: collision + phase alignment
        if not args.audio_a or not args.audio_b:
            raise SystemExit("--compare requires --audio-a and --audio-b")

        result["analysis"] = {"type": "compare"}

        compare_result = {}

        # Collision analysis (requires analysis caches or bar ranges)
        if args.analysis_a and args.analysis_b and args.bars_a and args.bars_b:
            try:
                from collision import (load_track_spectra, compute_collision,
                                       find_spectral_gaps, generate_eq_recommendations,
                                       collision_report as make_collision_report,
                                       collision_to_json, plot_collision)

                print(f"Loading Track A spectra: {args.analysis_a}", file=sys.stderr)
                data_a = load_track_spectra(args.analysis_a)
                print(f"  {data_a['name']} -- {data_a['bar_spectra'].shape[0]} bars", file=sys.stderr)

                print(f"Loading Track B spectra: {args.analysis_b}", file=sys.stderr)
                data_b = load_track_spectra(args.analysis_b)
                print(f"  {data_b['name']} -- {data_b['bar_spectra'].shape[0]} bars", file=sys.stderr)

                col = compute_collision(data_a, data_b, args.bars_a, args.bars_b,
                                        track_a_data=data_a, track_b_data=data_b)
                gaps = find_spectral_gaps(data_a, data_b, args.bars_a, args.bars_b,
                                          track_a_data=data_a, track_b_data=data_b,
                                          min_gap_db=args.min_gap_db)
                eq_recs = generate_eq_recommendations(gaps, col,
                                                      data_a["name"], data_b["name"])

                report = make_collision_report(col, gaps, data_a["name"], data_b["name"],
                                               eq_recs=eq_recs)
                print(f"\n{report}", file=sys.stderr)

                compare_result["collision"] = collision_to_json(
                    col, gaps, eq_recs,
                    data_a["name"], data_b["name"],
                    args.bars_a, args.bars_b
                )

                if args.save_collision_plot:
                    plot_collision(col, gaps, data_a["name"], data_b["name"],
                                  save_path=args.save_collision_plot)

            except Exception as e:
                print(f"Warning: Collision analysis failed: {e}", file=sys.stderr)
                compare_result["collision_error"] = str(e)

        # Phase alignment (requires audio files + BPM + first_beat_sample)
        if (args.bpm_a and args.bpm_b and
                args.first_beat_a is not None and args.first_beat_b is not None and
                args.bars_a and args.bars_b):
            try:
                from alignment import multi_probe_alignment, alignment_to_json

                print(f"\nLoading Track A audio: {args.audio_a}", file=sys.stderr)
                y_a, _sr = librosa.load(args.audio_a, sr=int(args.sr), mono=True)
                print(f"Loading Track B audio: {args.audio_b}", file=sys.stderr)
                y_b, _ = librosa.load(args.audio_b, sr=int(args.sr), mono=True)

                # Parse bar ranges
                bars_a_parts = args.bars_a.split("-")
                bar_start_a = int(bars_a_parts[0])
                bars_b_parts = args.bars_b.split("-")
                bar_start_b = int(bars_b_parts[0])
                n_overlap = int(bars_a_parts[1]) - int(bars_a_parts[0])

                align_result = multi_probe_alignment(
                    y_a, y_b, int(args.sr), args.bpm_a,
                    args.first_beat_a, args.first_beat_b,
                    bar_start_a, bar_start_b,
                    n_probes=max(1, n_overlap // 4),
                    probe_length_bars=4,
                    max_offset_ms=args.max_offset_ms,
                )

                name_a = Path(args.audio_a).stem
                name_b = Path(args.audio_b).stem
                compare_result["phase_alignment"] = alignment_to_json(
                    align_result, name_a, name_b,
                    first_beat_a=args.first_beat_a,
                    first_beat_b=args.first_beat_b,
                )

                rec = align_result["recommendation"]
                print(f"\nPhase alignment: {align_result['offset_ms']}ms "
                      f"({align_result['offset_samples']} samples)", file=sys.stderr)
                print(f"Consistency: {align_result.get('consistency', 'N/A')}", file=sys.stderr)
                print(f"Recommendation: {rec['rationale']}", file=sys.stderr)

            except Exception as e:
                print(f"Warning: Phase alignment failed: {e}", file=sys.stderr)
                compare_result["alignment_error"] = str(e)

        result["compare"] = compare_result

    elif args.consensus:
        # Determine starts precedence:
        # 1) explicit --starts
        # 2) BPM/bars if bpm + bar-start provided
        # 3) fallback: derive starts from the single start + bar-step (seconds) guess
        starts = parse_starts_arg(args.starts)

        if starts is None:
            if bpm_value is not None and args.bar_start is not None:
                starts = make_bar_starts(
                    bpm=bpm_value,
                    bar_start=int(args.bar_start),
                    bar_step=int(args.bar_step),
                    num_windows=int(args.num_windows),
                    beats_per_bar=float(args.beats_per_bar)
                )
            else:
                # fallback: simple rolling starts from --start stepping by window length/2
                base = float(args.start)
                step = max(30.0, float(args.window) / 2.0)
                starts = [round(base + i * step, 3) for i in range(int(args.num_windows))]

        result["analysis"] = {
            "type": "consensus_strict",
            "window": float(args.window),
            "starts": starts,
            "min_confidence": float(args.min_confidence),
            "bpm": bpm_value,
            "generated_from": (
                "explicit --starts" if args.starts else
                "bpm/bars (auto-detected)" if (args.bpm and args.bpm.lower() == "auto" and bpm_value and args.bar_start is not None) else
                "bpm/bars" if (bpm_value is not None and args.bar_start is not None) else
                "fallback rolling"
            )
        }

        result["estimate"] = consensus_analysis(
            path=audio_path,
            sr=int(args.sr),
            mono=True,
            window=float(args.window),
            starts=starts,
            use_harmonic=use_harmonic,
            min_conf=float(args.min_confidence)
        )
    else:
        result["analysis"] = {
            "type": "single",
            "start": float(args.start),
            "duration": float(args.duration)
        }

        result["estimate"] = analyze_window(
            path=audio_path,
            sr=int(args.sr),
            mono=True,
            start=float(args.start),
            duration=float(args.duration),
            use_harmonic=use_harmonic
        )

    out = json.dumps(result, indent=2)

    if args.json_out:
        # Write full JSON to file
        Path(args.json_out).write_text(out, encoding="utf-8")

        # Print human-readable summary to console for timeline mode
        if args.timeline:
            print(format_timeline_human(result["estimate"]))
            print(f"\nFull JSON saved to: {args.json_out}")
        else:
            print(out)
    else:
        # No file specified - print JSON to console
        print(out)

    # Generate plots if requested
    if args.plot and args.timeline:
        try:
            from visualize_timeline import plot_summary_view, plot_detailed_view

            if args.plot in ["summary", "both"]:
                save_path = f"{args.plot_save}_summary.png" if args.plot_save else None
                plot_summary_view(result, save_path)

            if args.plot in ["detailed", "both"]:
                save_path = f"{args.plot_save}_detailed.png" if args.plot_save else None
                plot_detailed_view(result, save_path)

        except ImportError:
            print("Warning: Could not import visualize_timeline. Make sure matplotlib is installed.")
            print("Run: pip install matplotlib")
    elif args.plot and not args.timeline:
        print("Note: --plot requires --timeline mode")


if __name__ == "__main__":
    main()
