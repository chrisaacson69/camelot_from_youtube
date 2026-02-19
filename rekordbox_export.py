#!/usr/bin/env python3
"""
rekordbox_export.py - Export camelot analysis data to Rekordbox XML format.

Exports track metadata (BPM, key), beat grid, and section markers as
cue points that Rekordbox can import.

All exports write to a single accumulating XML file. Point Rekordbox at
this file once (Preferences > Advanced > rekordbox xml) and every new
export appears automatically.

Default location: ~/Music/camelot_rekordbox.xml

Usage (standalone):
    python rekordbox_export.py --project "path/to/track_project_dir" --audio "path/to/track.wav"

Usage (from audio_ui):
    Called via AudioAnalysisApp menu: File > Export to Rekordbox
"""

import argparse
import json
from pathlib import Path
from urllib.request import pathname2url


# ---------------------------------------------------------------------------
# Default export path
# ---------------------------------------------------------------------------

DEFAULT_EXPORT_PATH = Path(__file__).resolve().parent / "rekordbox_collection.xml"


# ---------------------------------------------------------------------------
# Camelot -> Rekordbox Tonality mapping
# ---------------------------------------------------------------------------

CAMELOT_TO_TONALITY = {
    # Minor keys (A codes)
    "1A": "Abm", "2A": "Ebm", "3A": "Bbm", "4A": "Fm",
    "5A": "Cm",  "6A": "Gm",  "7A": "Dm",  "8A": "Am",
    "9A": "Em",  "10A": "Bm", "11A": "F#m", "12A": "Dbm",
    # Major keys (B codes)
    "1B": "B",   "2B": "F#",  "3B": "Db",  "4B": "Ab",
    "5B": "Eb",  "6B": "Bb",  "7B": "F",   "8B": "C",
    "9B": "G",   "10B": "D",  "11B": "A",  "12B": "E",
}


def _get_or_create_xml(output_path):
    """Load existing XML or create a fresh one."""
    from pyrekordbox.rbxml import RekordboxXml

    output_path = Path(output_path)
    if output_path.exists():
        return RekordboxXml(path=str(output_path))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return RekordboxXml(name="camelot_from_youtube", version="1.0.0")


def _next_track_id(xml):
    """Find the next available TrackID."""
    existing = xml.get_track_ids()
    return max(existing) + 1 if existing else 1


def _event_label(ev):
    """Build a cue point label from an event dict."""
    ev_type = ev.get("type", "cue")
    ev_desc = ev.get("description", "")
    ev_bar = ev.get("bar", "")
    if ev_desc:
        return f"{ev_type}: {ev_desc}"
    label = ev_type
    if ev_bar:
        label += f" @{ev_bar}"
    return label


def _build_track(xml, audio_path, store_data, include_cues=True,
                 max_hot_cues=8):
    """Add a track to the XML with metadata, beat grid, and cue points.

    If the track already exists (matched by file path), it is removed
    and re-added with fresh data.
    """
    audio_path = Path(audio_path).resolve()

    # Remove existing entry for this track if present
    for t in xml.get_tracks():
        if Path(t.Location).resolve() == audio_path:
            xml.remove_track(t)
            break

    # --- Track metadata ---
    track = xml.add_track(str(audio_path))
    track.set("TrackID", _next_track_id(xml))
    track.set("Name", audio_path.stem)
    track.set("TotalTime", int(store_data.get("duration", 0)))
    track.set("AverageBpm", round(float(store_data.get("tempo", 0)), 2))

    # Key
    key_block = store_data.get("key", {})
    key_summary = key_block.get("key_summary", {}) if key_block else {}
    dominant = key_summary.get("dominant", {})
    camelot = dominant.get("camelot", "")
    key_name = dominant.get("key_name", "")

    if camelot:
        tonality = CAMELOT_TO_TONALITY.get(camelot, "")
        if tonality:
            track.set("Tonality", tonality)
        track.set("Comments", f"{camelot} {key_name}")

    # --- Beat grid ---
    bpm_segments = store_data.get("segments", [])
    tempo = store_data.get("tempo")

    if bpm_segments and len(bpm_segments) > 1:
        for seg in bpm_segments:
            track.add_tempo(
                Inizio=round(float(seg.get("start_time", 0)), 3),
                Bpm=round(float(seg.get("bpm", tempo or 120)), 2),
                Metro="4/4",
                Battito=1,
            )
    elif tempo:
        measures = store_data.get("measures", [])
        first_beat = 0.0
        if measures and len(measures) > 0:
            first_beat = float(measures[0].get("start", 0))
        track.add_tempo(
            Inizio=round(first_beat, 3),
            Bpm=round(float(tempo), 2),
            Metro="4/4",
            Battito=1,
        )

    # --- Cue points ---
    if include_cues:
        MAX_MEMORY_CUES = 10

        events = []
        ev_block = store_data.get("events_block", {})
        if ev_block:
            events = list(ev_block.get("events", []))

        # Build key change lookup by time (rounded to 3 decimals)
        key_change_map = {}  # time -> key change info
        if key_summary:
            for kc in key_summary.get("key_changes", []):
                t = round(float(kc.get("time", 0)), 3)
                key_change_map[t] = {
                    "from": kc.get("from_camelot", "?"),
                    "to": kc.get("to_camelot", "?"),
                    "bar": kc.get("bar", 0),
                }

        # --- Merge events with co-timed key changes ---
        # Key change label takes priority in the merged label
        events.sort(key=lambda e: e.get("time", 0))
        merged_events = []
        used_kc_times = set()

        for ev in events:
            t = round(float(ev.get("time", 0)), 3)
            kc = key_change_map.get(t)
            if kc:
                used_kc_times.add(t)
                # Merge: KEY label first, event description after
                key_label = f"KEY {kc['from']} >> {kc['to']}"
                ev_type = ev.get("type", "")
                ev_desc = ev.get("description", "")
                suffix = f" | {ev_type}" if ev_type else ""
                merged_events.append({
                    **ev,
                    "_label": f"{key_label}{suffix}",
                    "_has_key_change": True,
                })
            else:
                merged_events.append({
                    **ev,
                    "_label": _event_label(ev),
                    "_has_key_change": False,
                })

        # Add standalone key changes (no co-timed event)
        for t, kc in key_change_map.items():
            if t not in used_kc_times:
                bar = kc.get("bar", "")
                label = f"KEY {kc['from']} >> {kc['to']}"
                if bar:
                    label += f" @{bar}"
                merged_events.append({
                    "time": t,
                    "bar": bar,
                    "type": "key_change",
                    "_label": label,
                    "_has_key_change": True,
                })

        merged_events.sort(key=lambda e: float(e.get("time", 0)))

        # --- Cue assignment: manual overrides then tier-based auto ---
        # Tier 1: structural moments a DJ cues to
        HOT_CUE_TIER1 = {
            "drop", "breakdown", "build", "bass_in", "bass_out",
            "melodic_in", "melodic_out", "vocal_in", "vocal_out",
            "transition",
        }
        # Tier 2: fills signal transitions
        HOT_CUE_TIER2 = {"fill"}

        # Separate manual overrides from auto events
        forced_hot = [e for e in merged_events if e.get("cue") == "hot"]
        forced_mem = [e for e in merged_events if e.get("cue") == "memory"]
        auto_events = [e for e in merged_events
                       if e.get("cue", "auto") == "auto"]

        # Auto tier assignment
        tier1 = [e for e in auto_events if e.get("type") in HOT_CUE_TIER1]
        tier2 = [e for e in auto_events if e.get("type") in HOT_CUE_TIER2]
        auto_rest = [e for e in auto_events
                     if e.get("type") not in HOT_CUE_TIER1
                     and e.get("type") not in HOT_CUE_TIER2]

        # Build hot cues: forced first, then tier 1, then tier 2
        hot_cues = list(forced_hot)
        hot_slot = len(hot_cues)
        auto_memory = []
        for ev in tier1:
            if hot_slot < max_hot_cues:
                hot_cues.append(ev)
                hot_slot += 1
            else:
                auto_memory.append(ev)
        for ev in tier2:
            if hot_slot < max_hot_cues:
                hot_cues.append(ev)
                hot_slot += 1
            else:
                auto_memory.append(ev)
        auto_memory.extend(auto_rest)

        # Write hot cues sorted by time
        hot_cues.sort(key=lambda e: float(e.get("time", 0)))
        for i, ev in enumerate(hot_cues):
            track.add_mark(
                Name=ev["_label"][:40],
                Type="cue",
                Start=round(float(ev.get("time", 0)), 3),
                Num=i,
            )

        # Memory pool: forced memory + auto remainder
        memory_pool = forced_mem + auto_memory
        memory_pool.sort(key=lambda e: float(e.get("time", 0)))
        kc_memory = [e for e in memory_pool if e.get("_has_key_change")]
        other_memory = [e for e in memory_pool if not e.get("_has_key_change")]

        # Key changes take priority slots, fill remainder with others
        memory_cues = kc_memory[:MAX_MEMORY_CUES]
        remaining_slots = MAX_MEMORY_CUES - len(memory_cues)
        if remaining_slots > 0:
            memory_cues.extend(other_memory[:remaining_slots])

        memory_cues.sort(key=lambda e: float(e.get("time", 0)))
        for ev in memory_cues:
            track.add_mark(
                Name=ev["_label"][:40],
                Type="cue",
                Start=round(float(ev.get("time", 0)), 3),
                Num=-1,
            )

    return track


def export_track(audio_path, store_data, output_path=None,
                 include_cues=True, max_hot_cues=8):
    """Export a single track to the accumulating Rekordbox XML.

    If the track already exists in the XML, it is replaced with fresh data.

    Parameters
    ----------
    audio_path : str or Path
        Path to the original audio file.
    store_data : dict
        Analysis data from AnalysisStore.save / analysis_cache.json.
    output_path : str or Path, optional
        XML file path. Defaults to ~/Music/camelot_rekordbox.xml.
    include_cues : bool
        Whether to include event markers as cue points.
    max_hot_cues : int
        Maximum hot cues (A-H = 0-7). Remaining become memory cues.

    Returns
    -------
    output_path : Path
        The XML file that was written.
    num_tracks : int
        Total number of tracks now in the XML.
    """
    output_path = Path(output_path) if output_path else DEFAULT_EXPORT_PATH
    xml = _get_or_create_xml(output_path)
    _build_track(xml, audio_path, store_data, include_cues, max_hot_cues)
    xml.save(str(output_path))
    return output_path, len(xml.get_tracks())


def export_set(tracks, output_path=None, playlist_name="Camelot Set"):
    """Export multiple tracks to the accumulating Rekordbox XML with a playlist.

    Parameters
    ----------
    tracks : list of dict
        Each dict has: audio_path, store_data (analysis dict).
    output_path : str or Path, optional
        XML file path. Defaults to ~/Music/camelot_rekordbox.xml.
    playlist_name : str
        Name for the playlist in Rekordbox.
    """
    output_path = Path(output_path) if output_path else DEFAULT_EXPORT_PATH
    xml = _get_or_create_xml(output_path)

    track_ids = []
    for t in tracks:
        track = _build_track(xml, t["audio_path"], t["store_data"])
        track_ids.append(track.TrackID)

    # Create/replace playlist
    playlist = xml.add_playlist(playlist_name)
    for tid in track_ids:
        playlist.add_track(xml.get_track(TrackID=tid))

    xml.save(str(output_path))
    return output_path, len(xml.get_tracks())


def load_store_data(project_dir):
    """Load analysis data from a project directory's cache."""
    d = Path(project_dir)
    json_path = d / "analysis_cache.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No analysis cache at {json_path}")
    with open(json_path) as f:
        return json.load(f)


def get_export_path():
    """Return the default export path."""
    return DEFAULT_EXPORT_PATH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export camelot analysis to Rekordbox XML")
    parser.add_argument("--project", "-p", required=True,
                        help="Project directory with analysis_cache.json")
    parser.add_argument("--audio", "-a", required=True,
                        help="Path to the audio file")
    parser.add_argument("--output", "-o", default=None,
                        help=f"Output XML path (default: {DEFAULT_EXPORT_PATH})")
    parser.add_argument("--no-cues", action="store_true",
                        help="Skip exporting event markers as cue points")
    args = parser.parse_args()

    data = load_store_data(args.project)

    result_path, num_tracks = export_track(
        audio_path=args.audio,
        store_data=data,
        output_path=args.output,
        include_cues=not args.no_cues,
    )
    print(f"Exported to: {result_path}")
    print(f"  Tracks in XML: {num_tracks}")
    print(f"  BPM: {data.get('tempo', '?')}")
    key_block = data.get("key", {})
    if key_block:
        dom = key_block.get("key_summary", {}).get("dominant", {})
        print(f"  Key: {dom.get('key_name', '?')} ({dom.get('camelot', '?')})")
    ev_block = data.get("events_block", {})
    if ev_block:
        print(f"  Events: {len(ev_block.get('events', []))}")
    print(f"\nPoint Rekordbox at: {result_path}")
    print("  Preferences > Advanced > rekordbox xml > Browse")


if __name__ == "__main__":
    main()
