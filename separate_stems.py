#!/usr/bin/env python3
"""
Stem Separation Tool using Demucs
=================================
Separates audio into stems (vocals, drums, bass, other) using Facebook's Demucs model.

Usage:
    python separate_stems.py --audio "path/to/audio.wav"
    python separate_stems.py --audio "path/to/audio.wav" --model htdemucs_ft --output-dir ./stems

Models available:
    - htdemucs (default): Hybrid Transformer Demucs - good balance of speed/quality
    - htdemucs_ft: Fine-tuned version - best quality, slower
    - htdemucs_6s: 6-stem model (vocals, drums, bass, guitar, piano, other)
    - mdx_extra: MDX architecture - alternative model

Output:
    Creates a folder with separated stems:
    - vocals.wav
    - drums.wav
    - bass.wav
    - other.wav (guitar, synths, etc.)
    - (piano.wav, guitar.wav if using 6-stem model)

Notes:
    - Requires ~4GB VRAM for GPU processing
    - Falls back to CPU if GPU memory insufficient
    - Uses segment mode for long tracks to manage memory

Environment:
    This script requires the env_demucs environment (Python 3.9 with PyTorch CUDA)
    Run with: env_demucs\\Scripts\\python.exe separate_stems.py ...
"""

import argparse
import os
import sys
import time
from pathlib import Path


def check_environment():
    """Check that required packages are available."""
    try:
        import torch
        import demucs
        return True, torch.cuda.is_available()
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Run this script with the env_demucs environment:")
        print("  env_demucs\\Scripts\\python.exe separate_stems.py ...")
        return False, False


def get_device(prefer_gpu=True):
    """Determine the best device to use."""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        # Check available GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

        if gpu_mem >= 4:
            return "cuda"
        else:
            print(f"Warning: GPU has only {gpu_mem:.1f}GB VRAM, may need to use segments")
            return "cuda"
    else:
        print("Using CPU (slower)")
        return "cpu"


def separate_audio(
    audio_path: str,
    output_dir: str = None,
    model_name: str = "htdemucs",
    device: str = None,
    segment_length: int = None,
    overlap: float = 0.25,
    mp3_output: bool = False,
    shifts: int = 1,
):
    """
    Separate audio file into stems using Demucs.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory for output stems (default: same as input)
        model_name: Demucs model to use
        device: 'cuda' or 'cpu' (auto-detect if None)
        segment_length: Segment length in seconds for long files (None=auto)
        overlap: Overlap between segments (0-1)
        mp3_output: Output MP3 instead of WAV
        shifts: Number of random shifts for better quality (1-10, higher=slower)

    Returns:
        dict: Paths to separated stem files
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import save_audio

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_stems"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device is None:
        device = get_device()

    print(f"\nLoading model: {model_name}")
    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Get model info
    sources = model.sources
    sample_rate = model.samplerate
    print(f"Model sources: {sources}")
    print(f"Sample rate: {sample_rate} Hz")

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        print(f"Resampling from {sr} Hz to {sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    duration = waveform.shape[1] / sample_rate
    print(f"Duration: {duration:.1f} seconds")

    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)

    # Determine if we need to use segment mode for memory management
    use_split = False
    if segment_length is None:
        # Auto-detect based on available memory
        if device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem < 6:
                use_split = True
                print(f"Using split mode for memory efficiency (4GB GPU)")

    # Apply model
    print(f"\nSeparating with {shifts} shift(s)...")
    start_time = time.time()

    with torch.no_grad():
        try:
            if segment_length:
                # Explicit segment length provided
                estimates = apply_model(
                    model,
                    waveform,
                    device=device,
                    shifts=shifts,
                    split=True,
                    overlap=overlap,
                    segment=segment_length,
                    progress=True
                )
            elif use_split:
                # Let Demucs decide segment size, just enable splitting
                estimates = apply_model(
                    model,
                    waveform,
                    device=device,
                    shifts=shifts,
                    split=True,
                    overlap=overlap,
                    progress=True
                )
            else:
                estimates = apply_model(
                    model,
                    waveform,
                    device=device,
                    shifts=shifts,
                    progress=True
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"\nGPU memory error, falling back to CPU...")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                waveform = waveform.to("cpu")
                estimates = apply_model(
                    model,
                    waveform,
                    device="cpu",
                    shifts=shifts,
                    split=True,
                    overlap=overlap,
                    progress=True
                )
            else:
                raise

    elapsed = time.time() - start_time
    print(f"Separation completed in {elapsed:.1f} seconds")

    # Save stems
    print(f"\nSaving stems to: {output_dir}")
    stem_paths = {}

    for i, source in enumerate(sources):
        stem = estimates[0, i]  # Remove batch dimension

        ext = "mp3" if mp3_output else "wav"
        stem_path = output_dir / f"{source}.{ext}"

        save_audio(
            stem.cpu(),
            stem_path,
            samplerate=sample_rate,
            clip="clamp"  # Prevent clipping
        )

        stem_paths[source] = str(stem_path)
        print(f"  Saved: {source}.{ext}")

    # Also save a summary JSON
    import json
    summary = {
        "input": str(audio_path),
        "model": model_name,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "device": device,
        "processing_time_seconds": elapsed,
        "stems": stem_paths
    }

    summary_path = output_dir / "separation_info.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return stem_paths


def main():
    parser = argparse.ArgumentParser(
        description="Separate audio into stems using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic separation
  python separate_stems.py --audio song.wav

  # High quality (slower)
  python separate_stems.py --audio song.wav --model htdemucs_ft --shifts 5

  # 6-stem separation (includes guitar and piano)
  python separate_stems.py --audio song.wav --model htdemucs_6s

  # Force CPU processing
  python separate_stems.py --audio song.wav --cpu
        """
    )

    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for stems (default: <audio>_stems/)"
    )
    parser.add_argument(
        "--model", "-m",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"],
        help="Demucs model to use (default: htdemucs)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing (slower but uses less memory)"
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Segment length in seconds (for memory management)"
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for averaging (1-10, higher=better quality but slower)"
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Output MP3 instead of WAV"
    )

    args = parser.parse_args()

    # Check environment
    ok, has_gpu = check_environment()
    if not ok:
        sys.exit(1)

    # Determine device
    device = "cpu" if args.cpu else None

    try:
        stem_paths = separate_audio(
            audio_path=args.audio,
            output_dir=args.output_dir,
            model_name=args.model,
            device=device,
            segment_length=args.segment,
            shifts=args.shifts,
            mp3_output=args.mp3,
        )

        print("\n" + "="*50)
        print("Stem separation complete!")
        print("="*50)
        for name, path in stem_paths.items():
            print(f"  {name}: {path}")

    except Exception as e:
        print(f"\nError during separation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
