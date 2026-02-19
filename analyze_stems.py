#!/usr/bin/env python3
"""
Stem Analyzer - Classify stem content and detect vocals vs instruments
======================================================================
Analyzes separated stems to determine if "vocals" stem actually contains
vocals or just high-pitched instruments (synths, leads, etc.)

Features analyzed:
- Formant patterns (vocal tract resonances)
- Pitch variation / vibrato
- MFCCs (voice characteristics)
- Spectral centroid stability
- Harmonic-to-noise ratio
- Zero crossing rate patterns

Usage:
    python analyze_stems.py --stems-dir "path/to/stems_folder"
    python analyze_stems.py --audio "path/to/vocals.wav"
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def analyze_vocal_likelihood(audio_path: str, sr: int = 22050, segment_duration: float = 5.0):
    """
    Analyze audio to determine likelihood it contains vocals vs instruments.

    Returns a dict with:
    - vocal_likelihood: 0-1 score (1 = definitely vocals)
    - features: detailed feature analysis
    - segments: per-segment analysis for timeline
    """
    import librosa

    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y) / sr

    print(f"Analyzing: {Path(audio_path).name}")
    print(f"Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    # Skip silence detection - find active regions
    rms = librosa.feature.rms(y=y)[0]
    rms_threshold = np.percentile(rms, 25)  # Bottom 25% considered quiet

    results = {
        "file": str(audio_path),
        "duration": duration,
        "features": {},
        "segments": [],
        "vocal_likelihood": 0.0,
        "classification": "unknown"
    }

    # === GLOBAL FEATURES ===

    # 1. MFCCs - voice characteristics
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)

    # Vocals typically have higher variance in MFCCs 2-6 (formant-related)
    formant_mfcc_var = np.mean(mfcc_var[1:6])
    results["features"]["mfcc_formant_variance"] = float(formant_mfcc_var)

    # 2. Spectral Centroid - "brightness" of sound
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)
    centroid_cv = centroid_std / centroid_mean if centroid_mean > 0 else 0  # Coefficient of variation

    # Vocals have more centroid variation than steady synths
    results["features"]["spectral_centroid_mean"] = float(centroid_mean)
    results["features"]["spectral_centroid_cv"] = float(centroid_cv)

    # 3. Spectral Bandwidth - frequency spread
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = np.mean(spectral_bandwidth)
    results["features"]["spectral_bandwidth"] = float(bandwidth_mean)

    # 4. Spectral Flatness - tonal vs noisy
    # Vocals are more tonal (lower flatness) than noise/breathy sounds
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = np.mean(spectral_flatness)
    results["features"]["spectral_flatness"] = float(flatness_mean)

    # 5. Zero Crossing Rate - roughness/noisiness
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    results["features"]["zcr_mean"] = float(zcr_mean)
    results["features"]["zcr_std"] = float(zcr_std)

    # 6. Pitch tracking - vocals have characteristic pitch patterns
    # Use pyin for better pitch tracking
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1047 Hz
            sr=sr
        )

        # Filter to voiced regions
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 10:
            pitch_mean = np.nanmean(voiced_f0)
            pitch_std = np.nanstd(voiced_f0)
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)

            # Vocals: pitch in 100-500Hz range, moderate variation, high voiced ratio
            results["features"]["pitch_mean_hz"] = float(pitch_mean)
            results["features"]["pitch_cv"] = float(pitch_cv)
            results["features"]["voiced_ratio"] = float(voiced_ratio)
        else:
            results["features"]["pitch_mean_hz"] = 0.0
            results["features"]["pitch_cv"] = 0.0
            results["features"]["voiced_ratio"] = 0.0
    except Exception as e:
        print(f"  Pitch tracking failed: {e}")
        results["features"]["pitch_mean_hz"] = 0.0
        results["features"]["pitch_cv"] = 0.0
        results["features"]["voiced_ratio"] = 0.0

    # 7. Harmonic content analysis
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(harmonic**2) / (np.sum(y**2) + 1e-10)
    results["features"]["harmonic_ratio"] = float(harmonic_ratio)

    # 8. Spectral Contrast - difference between peaks and valleys
    # Vocals have characteristic contrast patterns
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(spectral_contrast, axis=1)
    results["features"]["spectral_contrast_bands"] = [float(c) for c in contrast_mean]

    # === VOCAL LIKELIHOOD SCORING ===
    # Recalibrated based on actual stem analysis:
    # - Bass: ~100Hz, high formant var due to sub harmonics
    # - Drums: ~88Hz, percussive
    # - Other (synths): ~115Hz, very stable pitch
    # - Vocals: 150-600Hz typical, moderate pitch variation

    score = 0.0
    reasons = []

    pitch_hz = results["features"].get("pitch_mean_hz", 0)
    voiced_ratio = results["features"].get("voiced_ratio", 0)
    pitch_cv = results["features"].get("pitch_cv", 0)
    centroid_cv = results["features"].get("spectral_centroid_cv", 0)
    mfcc_var = results["features"].get("mfcc_formant_variance", 0)
    harmonic = results["features"].get("harmonic_ratio", 0)
    flatness = results["features"].get("spectral_flatness", 0)
    centroid_mean = results["features"].get("spectral_centroid_mean", 0)

    # 1. PITCH RANGE - Most important discriminator
    # Vocals: 150-600Hz fundamental, Bass: <150Hz, Synths: variable
    if 150 < pitch_hz < 600:
        score += 0.25
        reasons.append(f"pitch_in_vocal_range ({pitch_hz:.0f}Hz)")
    elif pitch_hz < 150:
        score -= 0.3  # Strong negative - this is bass range
        reasons.append(f"pitch_in_bass_range ({pitch_hz:.0f}Hz)")
    elif pitch_hz > 600:
        score -= 0.1  # Could be high vocals or high synths
        reasons.append(f"pitch_high ({pitch_hz:.0f}Hz)")

    # 2. SPECTRAL CENTROID - Where the "center of mass" of frequencies is
    # Vocals: typically 1000-4000Hz centroid, Bass: <500Hz, Drums: variable
    if 1000 < centroid_mean < 4000:
        score += 0.15
        reasons.append(f"centroid_in_vocal_range ({centroid_mean:.0f}Hz)")
    elif centroid_mean < 500:
        score -= 0.2
        reasons.append(f"centroid_too_low ({centroid_mean:.0f}Hz)")

    # 3. PITCH VARIATION - Vocals have natural vibrato, synths are stable
    # But be careful: bass also has variation from notes
    if 0.1 < pitch_cv < 0.5 and pitch_hz > 150:  # Only if not bass
        score += 0.15
        reasons.append(f"natural_pitch_variation ({pitch_cv:.2f})")
    elif pitch_cv < 0.05 and pitch_hz > 150:  # Very stable = synth lead
        score -= 0.15
        reasons.append(f"pitch_too_stable_for_vocals ({pitch_cv:.2f})")

    # 4. VOICED RATIO - How much of the signal has pitched content
    # Drums have LOW voiced ratio, vocals have HIGH
    if voiced_ratio > 0.4:
        score += 0.1
        reasons.append(f"high_voiced_ratio ({voiced_ratio:.2f})")
    elif voiced_ratio < 0.2:
        score -= 0.15
        reasons.append(f"low_voiced_ratio ({voiced_ratio:.2f})")

    # 5. TIMBRE VARIATION - Vocals vary their timbre (formants) when speaking
    # Synths tend to have stable timbre
    if centroid_cv > 0.4:
        score += 0.1
        reasons.append(f"varied_timbre ({centroid_cv:.2f})")
    elif centroid_cv < 0.2:
        score -= 0.05
        reasons.append(f"stable_timbre ({centroid_cv:.2f})")

    # 6. MFCC VARIANCE - Voice has characteristic formant patterns
    # High variance in formant bands (2-6) suggests vocal tract resonances
    # But bass also has high variance - so weight less
    if mfcc_var > 500 and pitch_hz > 150:
        score += 0.1
        reasons.append(f"high_formant_variance ({mfcc_var:.1f})")

    # 7. HARMONIC CONTENT - Both vocals and synths are harmonic
    # Drums are NOT harmonic
    if harmonic < 0.5:
        score -= 0.2
        reasons.append(f"percussive_content ({harmonic:.2f})")

    # 8. SPECTRAL FLATNESS - Breathy/noisy content
    # Vocals have SOME breathiness, pure synths are very tonal
    if 0.02 < flatness < 0.15:
        score += 0.05
        reasons.append(f"slight_breathiness ({flatness:.3f})")
    elif flatness < 0.01:
        score -= 0.05
        reasons.append(f"too_pure_tonal ({flatness:.3f})")

    # Normalize score to 0-1 (start from 0.5 baseline)
    score = max(0, min(1, score + 0.5))

    results["vocal_likelihood"] = float(score)
    results["scoring_reasons"] = reasons

    # Classification
    if score > 0.65:
        results["classification"] = "likely_vocals"
    elif score > 0.45:
        results["classification"] = "mixed_or_uncertain"
    else:
        results["classification"] = "likely_instruments"

    # === SEGMENT ANALYSIS ===
    # Analyze in segments to find where vocals actually appear
    segment_samples = int(segment_duration * sr)
    n_segments = int(np.ceil(len(y) / segment_samples))

    for i in range(n_segments):
        start_sample = i * segment_samples
        end_sample = min((i + 1) * segment_samples, len(y))
        segment = y[start_sample:end_sample]

        if len(segment) < sr:  # Skip very short segments
            continue

        start_time = start_sample / sr
        end_time = end_sample / sr

        # Quick analysis per segment
        seg_rms = np.sqrt(np.mean(segment**2))
        seg_zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])

        # Pitch in segment
        try:
            seg_f0, seg_voiced, _ = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr
            )
            seg_voiced_ratio = np.sum(seg_voiced) / len(seg_voiced) if len(seg_voiced) > 0 else 0
            seg_pitch = np.nanmean(seg_f0[seg_voiced]) if np.sum(seg_voiced) > 0 else 0
        except:
            seg_voiced_ratio = 0
            seg_pitch = 0

        # Segment scoring (simplified)
        seg_score = 0.5
        if 80 < seg_pitch < 800:
            seg_score += 0.2
        if seg_voiced_ratio > 0.3:
            seg_score += 0.2
        if seg_rms > 0.01:  # Has energy
            seg_score += 0.1

        seg_score = max(0, min(1, seg_score))

        results["segments"].append({
            "start": float(start_time),
            "end": float(end_time),
            "rms": float(seg_rms),
            "voiced_ratio": float(seg_voiced_ratio),
            "pitch_hz": float(seg_pitch) if seg_pitch else None,
            "vocal_likelihood": float(seg_score),
            "has_content": bool(seg_rms > 0.005)  # Ensure native Python bool
        })

    return results


def analyze_all_stems(stems_dir: str, output_json: str = None):
    """Analyze all stems in a directory."""
    stems_dir = Path(stems_dir)

    if not stems_dir.exists():
        print(f"Error: Directory not found: {stems_dir}")
        return None

    # Find stem files
    stem_files = list(stems_dir.glob("*.wav"))
    if not stem_files:
        print(f"No .wav files found in {stems_dir}")
        return None

    results = {
        "stems_dir": str(stems_dir),
        "stems": {}
    }

    for stem_file in sorted(stem_files):
        stem_name = stem_file.stem
        print(f"\n{'='*50}")
        print(f"Analyzing stem: {stem_name}")
        print('='*50)

        analysis = analyze_vocal_likelihood(str(stem_file))
        results["stems"][stem_name] = analysis

        print(f"\nResults for {stem_name}:")
        print(f"  Vocal likelihood: {analysis['vocal_likelihood']:.2f}")
        print(f"  Classification: {analysis['classification']}")
        print(f"  Reasons: {', '.join(analysis.get('scoring_reasons', []))}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)

    for stem_name, analysis in results["stems"].items():
        likelihood = analysis['vocal_likelihood']
        classification = analysis['classification']
        bar = '#' * int(likelihood * 20) + '-' * (20 - int(likelihood * 20))
        print(f"{stem_name:10s}: [{bar}] {likelihood:.2f} - {classification}")

    # Save results
    if output_json:
        output_path = Path(output_json)
    else:
        output_path = stems_dir / "stem_analysis.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stems to classify vocals vs instruments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all stems in a folder
  python analyze_stems.py --stems-dir "NothingLeftBreeder_stems"

  # Analyze a single file
  python analyze_stems.py --audio "vocals.wav"

  # Custom output location
  python analyze_stems.py --stems-dir "stems" --output "analysis.json"
        """
    )

    parser.add_argument(
        "--stems-dir", "-d",
        help="Directory containing stem .wav files"
    )
    parser.add_argument(
        "--audio", "-a",
        help="Single audio file to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=5.0,
        help="Duration of analysis segments in seconds (default: 5.0)"
    )

    args = parser.parse_args()

    if not args.stems_dir and not args.audio:
        parser.error("Either --stems-dir or --audio is required")

    if args.audio:
        results = analyze_vocal_likelihood(args.audio, segment_duration=args.segment_duration)
        print(f"\nResults:")
        print(f"  Vocal likelihood: {results['vocal_likelihood']:.2f}")
        print(f"  Classification: {results['classification']}")
        print(f"  Reasons: {', '.join(results.get('scoring_reasons', []))}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to: {args.output}")
    else:
        analyze_all_stems(args.stems_dir, args.output)


if __name__ == "__main__":
    main()
