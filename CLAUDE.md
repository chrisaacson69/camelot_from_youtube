# CLAUDE.md

**Vault page:** `Vault/projects/camelot-from-youtube/README.md` (logical name `Vault`; resolve the absolute
mount via `Vault/.claude/local-paths.md` — never hardcode it, the old `C:\Users\Chris.Isaacson\...` path was
dead after the machine migration).

**Related research:** `Vault/research/dj-eq-blending.md`, `Vault/research/programmatic-dj-mixing-tools.md`.

**Data lives in the shared music library** (logical name `music`, sibling `../music`, local only, never
pushed): originals at its top level, each track's analysis under `analysis/<name>/` (dj-score's
`adapters/library` owns that layout; the UI opens a track's project there). CyborgDJ reads the same library
since 2026-10-22; its old copies and forks of this repo's code are gone.

**Shared library — `dj-score` (exists, 2026-10-21):** logical name `dj-score`, sibling `../dj-score`,
`chrisaacson69/dj-score`. All analysis code is migrating there (copy → repoint → archive; nothing is
purged, retired tools and their outputs move to `archive/<tool>/`); this repo keeps the Tk UI and per-track
data and is a consumer, as CyborgDJ will be. **The per-module keep / fold / retire verdicts and the
migration order live in `../dj-score/MIGRATION.md`** — read it before touching any analysis module here,
and add new analysis code to dj-score, never to this repo.

**`audio_ui.py` runs on dj-score (2026-10-21).** Its store holds a dj-score `TrackAnalysis`; BPM/key/events/
spectrum/export call `dj_score.services`. **The old analysis modules are archived (2026-10-22)** in
`archive/<tool>/`, with their outputs. The whole archive is local (gitignored; git history keeps the code);
see `archive/README.md`. Their CLIs are `dj-score` subcommands now, including Rekordbox export. Install into `env1` with **`pip install --no-deps -e ../dj-score`**: `--no-deps`
keeps env1's pinned numpy/librosa. `tests/ui_repoint_oracle.py` replays one scripted session through the
pre-repoint UI (from git) and the current one and requires identical caches, detail files, chart pixels and
text; run it after any UI change that should not alter behaviour.

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
├── audio_ui.py               # Tkinter GUI - main application (runs on dj-score)
├── bench/                    # beat-grid benchmark against Rekordbox (its own README)
├── tests/ui_repoint_oracle.py # old-vs-new UI replay oracle
├── archive/<tool>/           # retired tools + their outputs (local only, gitignored)
├── CLAUDE.md                 # This file
├── .env                      # API keys (not committed)
├── .env.example              # Template for .env
├── rekordbox_collection.xml  # Accumulating Rekordbox export file
├── env1/                     # The UI environment (librosa 0.11, matplotlib, pygame, pyrekordbox, dj-score --no-deps)
└── env_mir/                  # The one GPU/ML environment: beat_this + Demucs on CUDA 12.8 (recipe: bench/requirements-mir.txt)
(per-track data: ../music/analysis/<track>/analysis_cache.*, in the shared music library)
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

## Rekordbox Export (dj-score `export`; the UI's File menu)

### Accumulating XML
All exports write to a single XML file (`rekordbox_collection.xml` in project folder). Rekordbox is pointed at this file once via Preferences > Advanced > rekordbox xml. Re-exporting a track replaces its entry (matched by file path).

### CLI Usage
```bash
dj-score export --project "../music/analysis/Alone (Extended Mix)" --audio "../music/Alone (Extended Mix).mp3" --output rekordbox_collection.xml
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

## Key Detection (now dj-score `domain/key`; CLIs `dj-score key | key-window | timeline`)

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

## Command-line tools

The old CLIs (`camelot_from_youtube.py`, `bpm_detect.py`, `key_detect.py`, …) are archived. Their
replacements are `dj-score` subcommands, which take the same arguments (`dj-score <cmd> --help`); the
mapping and the deliberate differences are in `../dj-score/MIGRATION.md` step 5.

## Important Notes

- Avoid `.weba`/`.webm`/`.opus` formats (require ffmpeg which may not be available)
- Harmonic-only analysis (default) works better for EDM/electronic music
- Per-track data lives in `../music/analysis/<track>/analysis_cache.json` (the shared music library): the source of truth for all analysis
- The `rekordbox_collection.xml` file accumulates across exports — no need to reconfigure Rekordbox
