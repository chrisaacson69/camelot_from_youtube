# CLAUDE.md

**Vault:** `C:\Users\Chris.Isaacson\Vault\projects\camelot-from-youtube\README.md`

## Project Overview

**camelot_from_youtube** is a Python toolkit for DJ track analysis. It detects BPM, musical key (Camelot codes), beat grids, structural events (drops, breakdowns, fills, etc.), and exports everything to Rekordbox XML. Includes a Tkinter GUI (`audio_ui.py`) for interactive analysis and editing.

## Tech Stack

- **Language**: Python 3
- **Core Libraries**:
  - `librosa` - Audio analysis and feature extraction
  - `numpy` / `scipy` - Numerical computations
  - `matplotlib` - Waveform and chart rendering in UI
  - `pyrekordbox` - Rekordbox XML generation
  - `demucs` - Stem separation (vocals, drums, bass, other)
- **Optional**:
  - `requests` / `python-dotenv` - YouTube metadata via Data API

## Project Structure

```
camelot_from_youtube/
├── audio_ui.py               # Tkinter GUI - main application
├── camelot_from_youtube.py   # Original CLI for key detection
├── rekordbox_export.py       # Rekordbox XML export (CLI + library)
├── bpm_detect.py             # BPM and beat grid detection
├── key_detect.py             # Key/Camelot detection (timeline, consensus)
├── detect_events.py          # Feature-based event detection
├── detect_stem_events.py     # Stem-based event detection
├── event_detect.py           # Event detection orchestrator
├── separate_stems.py         # Demucs stem separation
├── analyze_stems.py          # Stem energy analysis
├── analyze_onset_drops.py    # Onset/drop analysis
├── visualize_audio.py        # Audio visualization helpers
├── visualize_stems.py        # Stem visualization
├── visualize_timeline.py     # Key timeline visualization
├── check_beats.py            # Beat grid validation utility
├── CLAUDE.md                 # This file
├── .env                      # API keys (not committed)
├── .env.example              # Template for .env
├── rekordbox_collection.xml  # Accumulating Rekordbox export file
├── env1/                     # Active virtual environment
└── <track_name>/             # Per-track project directories
    └── analysis_cache.json   # Persisted analysis data
```

## Development Setup

1. Activate the virtual environment:
   ```powershell
   .\env1\Scripts\Activate.ps1
   ```

2. The project uses `env1` as the active Python environment (configured in `.vscode/settings.json`)

## GUI Application (audio_ui.py)

Launch:
```bash
python audio_ui.py
```

### Key Features
- **BPM detection** with beat grid overlay on waveform
- **Key detection** with timeline showing key changes over time
- **Event detection** (feature-based or stem-based) identifying drops, breakdowns, builds, fills, transitions
- **Interactive event editing** — right-click chart to add/edit/delete events
- **Export to Rekordbox** via File menu — writes to accumulating XML

### Event Editor
Right-click any event marker on the chart to edit. Fields:
- **Time / Bar** — with snap modes (bar, phrase, free)
- **Type** — drop, breakdown, build, fill, transition, bass_in/out, vocal_in/out, melodic_in/out, increase, decrease, other
- **Description** — free text
- **Cue** — Auto / Hot / Memory (controls Rekordbox cue type on export)
- **Score** — read-only confidence from detection

### AnalysisStore
Central data model (`AnalysisStore` class in audio_ui.py). Holds all analysis results, persists to `analysis_cache.json` per project directory. Key fields:
- `tempo`, `beats`, `measures`, `segments` — BPM/beat data
- `key` — key detection results with `key_summary` (dominant key, key changes)
- `events` — list of event dicts with `time`, `bar`, `type`, `description`, `score`, `cue`, `source`
- `duration` — track length in seconds

## Rekordbox Export (rekordbox_export.py)

### Accumulating XML
All exports write to a single XML file (`rekordbox_collection.xml` in project folder). Rekordbox is pointed at this file once via Preferences > Advanced > rekordbox xml. Re-exporting a track replaces its entry (matched by file path).

### CLI Usage
```bash
python rekordbox_export.py --project "Alone (Extended Mix)" --audio "Alone (Extended Mix).mp3"
```

### Cue Point Logic
- **Hot cues** (Num 0-7, max 8): Assigned by tier priority
  - Tier 1: drop, breakdown, build, bass_in/out, melodic_in/out, vocal_in/out, transition
  - Tier 2: fill
  - Manual override: events with `cue: "hot"` are forced into hot cue slots first
- **Memory cues** (Num -1, max 10): Remaining events
  - Key changes get priority labels: `KEY 6A >> 9B` format
  - Co-timed events merge: `KEY 4A >> 9B | decrease`
  - Key changes fill memory slots first, then other events
  - Manual override: events with `cue: "memory"` are forced into memory pool
- Events with `cue: "auto"` (default) use the tier logic above

### pyrekordbox API Notes
- Use `track.set("Tonality", "Gm")` — NOT `track.Tonality = "Gm"` (sets Python attr, not XML)
- Use `Type="cue"` for `add_mark()` — NOT `Type=0` (string type names required)
- `get_track(Location=...)` crashes on miss — iterate `get_tracks()` and compare paths manually
- `RekordboxXml(path=...)` to load existing, `RekordboxXml(name=..., version=...)` for new

### Rekordbox 7 Setup
1. Preferences > Advanced > rekordbox xml > Browse — point to `rekordbox_collection.xml`
2. Preferences > View > Layout — enable rekordbox xml panel in sidebar
3. Right-click imported tracks > Import to Collection

## Key Detection (camelot_from_youtube.py / key_detect.py)

### Algorithm
1. Load audio segment via librosa
2. Extract CQT chroma features
3. Correlate against Krumhansl major/minor key profiles for all 12 keys
4. Return best match with confidence score

### Camelot Wheel
- Minor keys = "A" codes (1A-12A), Major keys = "B" codes (1B-12B)
- Follows the circle of fifths

### Analysis Modes
- **Single window** — one segment at a configurable start/duration
- **Consensus** — multiple time windows with strict voting
- **Timeline** — bar-aligned segments detecting key changes over time

## CLI Arguments (camelot_from_youtube.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | required | Path to audio file |
| `--consensus` | false | Multi-window consensus analysis |
| `--timeline` | false | Timeline analysis (key changes) |
| `--bpm` | none | BPM (number or "auto") |
| `--bar-start` | none | Starting bar for consensus windows |
| `--bars-per-segment` | 8 | Bars per segment for timeline |
| `--full-track-beats` | false | Full track beat analysis (slower) |
| `--json-out` | none | Write JSON output to file |
| `--url` | none | YouTube URL for metadata |

## Important Notes

- Avoid `.weba`/`.webm`/`.opus` formats (require ffmpeg which may not be available)
- Harmonic-only analysis (default) works better for EDM/electronic music
- Per-track data lives in `<track_name>/analysis_cache.json` — this is the source of truth for all analysis
- The `rekordbox_collection.xml` file accumulates across exports — no need to reconfigure Rekordbox
