# Camelot from YouTube

Automated BPM, key, and event detection for DJ tracks -- with Rekordbox export.

## What It Does

Camelot from YouTube analyzes audio files to detect BPM, musical key (as Camelot codes), beat grids, and structural events like drops, breakdowns, and builds. Results can be edited in an interactive GUI and exported directly to Rekordbox XML, so your cue points and metadata are ready before you even open Rekordbox.

## Features

- **BPM detection** with beat grid and bar alignment
- **Key detection** using chroma-based analysis -- single key, consensus, or full timeline mode for tracks with key changes
- **Event detection** identifying drops, breakdowns, builds, fills, transitions, and stem-level events (bass in/out, vocal in/out, etc.)
- **Stem separation** via Demucs for more accurate structural analysis
- **Interactive GUI** with waveform display, event markers, and right-click editing
- **Rekordbox XML export** with hot cues, memory cues, BPM, and key metadata
- **Accumulating export file** -- one XML file grows across all your tracks; Rekordbox only needs to be pointed at it once

## Requirements

- Python 3
- Core dependencies: `librosa`, `numpy`, `scipy`, `matplotlib`, `pyrekordbox`
- Optional: `demucs` (for stem separation), `requests` + `python-dotenv` (for YouTube metadata lookup)

## Setup

1. Clone the repository and create a virtual environment:
   ```
   python -m venv env1
   ```

2. Activate the environment:
   ```powershell
   # PowerShell
   .\env1\Scripts\Activate.ps1
   ```
   ```bash
   # Bash
   source env1/Scripts/activate
   ```

3. Install dependencies:
   ```
   pip install librosa numpy scipy matplotlib pyrekordbox demucs requests python-dotenv
   ```

   Then the shared analysis library, cloned beside this repo
   ([dj-score](https://github.com/chrisaacson69/dj-score)):
   ```
   pip install --no-deps -e ../dj-score
   ```

4. *(Optional)* For YouTube metadata support, copy `.env.example` to `.env` and add your [YouTube Data API key](https://console.cloud.google.com/apis/credentials).

## Usage

### GUI

```
python audio_ui.py
```

Load an audio file, run BPM/key/event detection from the UI, edit events by right-clicking markers on the waveform, and export to Rekordbox via the File menu.

### Command line

The analysis CLIs moved to [dj-score](https://github.com/chrisaacson69/dj-score); the originals are
in `archive/`. For example:

```bash
dj-score key-window "track.mp3"                          # single-window key
dj-score key-window "track.mp3" --consensus --bpm auto   # multi-window consensus
dj-score timeline "track.mp3"                            # key changes over time
dj-score export --project "Track Name" --audio "Track Name.mp3" --output rekordbox_collection.xml
```

## Rekordbox Integration

1. In Rekordbox, go to **Preferences > Advanced > rekordbox xml** and browse to `rekordbox_collection.xml` in the project folder.
2. Under **Preferences > View > Layout**, enable the rekordbox xml panel in the sidebar.
3. After exporting from Camelot, right-click the track in the xml panel and select **Import to Collection**.

Re-exporting a track replaces its existing entry in the XML -- no duplicates, no reconfiguration needed.

## License

This project is not yet licensed. All rights reserved.
