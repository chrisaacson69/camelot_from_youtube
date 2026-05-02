#!/usr/bin/env python3
"""
audio_ui.py - Desktop UI for DJ audio analysis.

Architecture (3 layers in one file):
  Layer 1: AnalysisStore   — pure data container, no matplotlib
  Layer 2: ChartBuilder    — lazy figure factory, version-cached
  Layer 3: AudioAnalysisApp — slim tkinter UI, overlay during analysis

Usage:
    python audio_ui.py
    python audio_ui.py --audio "path/to/track.wav"

Requires: pygame-ce (pip install pygame-ce)
"""

import argparse
import bisect
import json
import sys
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, str(Path(__file__).parent))
from bpm_detect import (
    detect_beats, group_beats_into_measures, compute_span_bpm,
    find_bpm_segments, format_time,
)
from key_detect import (
    compute_measure_chroma, compute_phrase_chroma, find_key_segments,
    compute_summary as key_compute_summary,
)
from event_detect import (
    find_or_create_stems, compute_stem_phrase_energy, compute_stem_presence,
    detect_stem_events, compute_bar_features, compute_phrase_features,
    detect_feature_events, normalize_feature,
    compute_summary as event_compute_summary,
)
from spectral_analysis import (
    compute_bar_spectra, compute_spectral_summary, NOISE_FLOOR_DB,
)

import pygame


# =========================================================================
# VolumeSlider — canvas-based slider with red clip zone
# =========================================================================

class VolumeSlider(tk.Canvas):
    """Horizontal volume slider with dual-colour trough.

    Draws normal (grey) below *safe_volume* and red above it.
    The thumb is draggable; clicking the trough jumps to that position.
    """

    WIDTH = 100
    HEIGHT = 22
    PAD_X = 6          # horizontal padding inside canvas
    THUMB_W = 8        # thumb width
    TROUGH_H = 6       # trough bar height
    COLOR_NORMAL = "#888888"
    COLOR_SAFE_FILL = "#aacfaa"
    COLOR_CLIP = "#cc3333"
    COLOR_CLIP_FILL = "#ee9999"
    COLOR_THUMB = "#444444"
    COLOR_THUMB_CLIP = "#cc2222"

    def __init__(self, parent, variable: tk.DoubleVar, command=None,
                 length=100, **kw):
        self.WIDTH = length
        super().__init__(parent, width=self.WIDTH, height=self.HEIGHT,
                         highlightthickness=0, bd=0, **kw)
        self._var = variable
        self._command = command
        self._safe = 1.0        # safe_volume threshold (0-1)
        self._dragging = False

        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._var.trace_add("write", lambda *_: self._redraw())
        self._redraw()

    # -- public API -------------------------------------------------------

    def set_safe(self, safe: float):
        """Update the safe-volume threshold (0-1) and redraw."""
        self._safe = max(0.0, min(1.0, safe))
        self._redraw()

    # -- internal ---------------------------------------------------------

    def _val_to_x(self, v: float) -> float:
        """Map value 0-1 to canvas x coordinate."""
        usable = self.WIDTH - 2 * self.PAD_X - self.THUMB_W
        return self.PAD_X + self.THUMB_W / 2 + v * usable

    def _x_to_val(self, x: float) -> float:
        usable = self.WIDTH - 2 * self.PAD_X - self.THUMB_W
        v = (x - self.PAD_X - self.THUMB_W / 2) / usable
        return max(0.0, min(1.0, v))

    def _redraw(self):
        self.delete("all")
        cy = self.HEIGHT / 2
        t_top = cy - self.TROUGH_H / 2
        t_bot = cy + self.TROUGH_H / 2
        x_left = self._val_to_x(0)
        x_right = self._val_to_x(1)
        x_safe = self._val_to_x(self._safe)

        # Trough: safe zone (left part)
        if self._safe > 0:
            self.create_rectangle(x_left, t_top, x_safe, t_bot,
                                  fill=self.COLOR_SAFE_FILL,
                                  outline=self.COLOR_NORMAL, width=1)
        # Trough: clip zone (right part)
        if self._safe < 1.0:
            self.create_rectangle(x_safe, t_top, x_right, t_bot,
                                  fill=self.COLOR_CLIP_FILL,
                                  outline=self.COLOR_CLIP, width=1)

        # Safe threshold tick mark
        if 0.01 < self._safe < 0.99:
            self.create_line(x_safe, t_top - 2, x_safe, t_bot + 2,
                             fill=self.COLOR_CLIP, width=1)

        # Thumb
        val = self._var.get()
        tx = self._val_to_x(val)
        in_clip = val > self._safe + 0.005
        tc = self.COLOR_THUMB_CLIP if in_clip else self.COLOR_THUMB
        self.create_rectangle(tx - self.THUMB_W / 2, cy - 8,
                              tx + self.THUMB_W / 2, cy + 8,
                              fill=tc, outline=tc, width=1)

    def _set_from_x(self, x):
        v = self._x_to_val(x)
        self._var.set(round(v, 3))
        if self._command:
            self._command(v)

    def _on_click(self, event):
        self._dragging = True
        self._set_from_x(event.x)

    def _on_drag(self, event):
        if self._dragging:
            self._set_from_x(event.x)

    def _on_release(self, event):
        self._dragging = False


# =========================================================================
# EventDialog — modal dialog for add/edit event
# =========================================================================

_EVENT_TYPES = [
    "drop", "breakdown", "build", "increase", "decrease",
    "bass_in", "bass_out", "drums_in", "drums_out",
    "vocal_in", "vocal_out", "melodic_in", "melodic_out",
    "fill", "transition", "other",
]


class EventDialog(tk.Toplevel):
    """Modal dialog for adding or editing an event.

    Parameters
    ----------
    parent : tk widget
    store  : AnalysisStore — for measures, duration, event_mode
    phrase_bars : int — bars per phrase (for phrase snap)
    title  : str — window title
    event  : dict or None — existing event to edit (pre-fills fields)
    initial_time : float or None — starting time for a new event
    """

    def __init__(self, parent, store, phrase_bars, snap_var,
                 title="Add Event", event=None, initial_time=None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._store = store
        self._phrase_bars = phrase_bars
        self._snap_var = snap_var   # shared global snap StringVar
        self._updating = False      # guard against feedback loops
        self.result = None          # set on OK

        pad = dict(padx=6, pady=3)

        # -- Row 0: Time --
        row = 0
        ttk.Label(self, text="Time (sec):").grid(row=row, column=0,
                                                  sticky="e", **pad)
        self._time_var = tk.StringVar()
        self._time_entry = ttk.Entry(self, textvariable=self._time_var,
                                      width=12)
        self._time_entry.grid(row=row, column=1, sticky="w", **pad)
        self._time_var.trace_add("write", self._on_time_changed)

        # -- Row 1: Bar --
        row = 1
        ttk.Label(self, text="Bar:").grid(row=row, column=0,
                                           sticky="e", **pad)
        self._bar_var = tk.StringVar()
        self._bar_entry = ttk.Entry(self, textvariable=self._bar_var,
                                     width=8)
        self._bar_entry.grid(row=row, column=1, sticky="w", **pad)
        self._bar_var.trace_add("write", self._on_bar_changed)

        # -- Row 2: Snap mode (bound to global snap_var) --
        row = 2
        ttk.Label(self, text="Snap:").grid(row=row, column=0,
                                            sticky="e", **pad)
        snap_frame = ttk.Frame(self)
        snap_frame.grid(row=row, column=1, sticky="w", **pad)
        for val, label in [("bar", "Bar"), ("phrase", "Phrase"),
                           ("free", "Free")]:
            ttk.Radiobutton(snap_frame, text=label, variable=self._snap_var,
                            value=val,
                            command=self._on_snap_changed).pack(side="left",
                                                                 padx=4)

        # -- Row 3: Type --
        row = 3
        ttk.Label(self, text="Type:").grid(row=row, column=0,
                                            sticky="e", **pad)
        self._type_var = tk.StringVar()
        self._type_combo = ttk.Combobox(self, textvariable=self._type_var,
                                         values=_EVENT_TYPES, width=18)
        self._type_combo.grid(row=row, column=1, sticky="w", **pad)

        # -- Row 4: Description --
        row = 4
        ttk.Label(self, text="Description:").grid(row=row, column=0,
                                                    sticky="e", **pad)
        self._desc_var = tk.StringVar()
        ttk.Entry(self, textvariable=self._desc_var,
                  width=30).grid(row=row, column=1, sticky="w", **pad)

        # -- Row 5: Cue priority (auto / hot / memory) --
        row = 5
        ttk.Label(self, text="Cue:").grid(row=row, column=0,
                                           sticky="e", **pad)
        self._cue_var = tk.StringVar(value="auto")
        cue_frame = ttk.Frame(self)
        cue_frame.grid(row=row, column=1, sticky="w", **pad)
        for val, label in [("auto", "Auto"), ("hot", "Hot"),
                           ("memory", "Memory")]:
            ttk.Radiobutton(cue_frame, text=label, variable=self._cue_var,
                            value=val).pack(side="left", padx=4)

        # -- Row 6: Score (read-only, shown in edit mode) --
        row = 6
        self._edit_score = None  # preserve original score on edit
        if event and "score" in event:
            self._edit_score = event["score"]
            ttk.Label(self, text="Score:").grid(row=row, column=0,
                                                 sticky="e", **pad)
            ttk.Label(self, text=f"{event['score']:.2f}",
                      foreground="#666666").grid(row=row, column=1,
                                                 sticky="w", **pad)

        # -- Buttons --
        row = 7
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="OK",
                   command=self._on_ok).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Cancel",
                   command=self.destroy).pack(side="left", padx=8)

        # -- Pre-fill --
        if event:
            self._time_var.set(str(event.get("time", 0)))
            self._bar_var.set(str(event.get("bar", 0)))
            self._type_var.set(event.get("type", ""))
            self._desc_var.set(event.get("description", ""))
            self._cue_var.set(event.get("cue", "auto"))
        elif initial_time is not None:
            self._set_time_with_snap(initial_time)

        self._time_entry.focus_set()
        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self.destroy())

        # Center on parent
        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"+{px - self.winfo_width() // 2}"
                      f"+{py - self.winfo_height() // 2}")

    # -- Snap helpers (delegate to app-level utilities via store) ----------

    def _snap_to_bar(self, t):
        measures = self._store.measures
        if not measures:
            return (t, 0)
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t)
        if idx == 0:
            best = 0
        elif idx >= len(starts):
            best = len(starts) - 1
        else:
            best = idx if abs(starts[idx] - t) < abs(starts[idx - 1] - t) \
                else idx - 1
        m = measures[best]
        return (m["start"], m["measure_num"])

    def _snap_to_phrase(self, t):
        measures = self._store.measures
        if not measures:
            return (t, 0)
        pb = self._phrase_bars
        phrase_m = [measures[i] for i in range(0, len(measures), pb)]
        starts = [m["start"] for m in phrase_m]
        idx = bisect.bisect_right(starts, t)
        if idx == 0:
            best = 0
        elif idx >= len(starts):
            best = len(starts) - 1
        else:
            best = idx if abs(starts[idx] - t) < abs(starts[idx - 1] - t) \
                else idx - 1
        m = phrase_m[best]
        return (m["start"], m["measure_num"])

    def _time_to_bar(self, t):
        measures = self._store.measures
        if not measures:
            return 0
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t) - 1
        idx = max(0, min(idx, len(measures) - 1))
        return measures[idx]["measure_num"]

    # -- Callbacks --------------------------------------------------------

    def _set_time_with_snap(self, t):
        """Apply snap mode and update both time and bar fields."""
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
        else:
            bar = self._time_to_bar(t)
        self._time_var.set(f"{t:.2f}")
        self._bar_var.set(str(bar))
        self._updating = False

    def _on_time_changed(self, *_args):
        if self._updating:
            return
        try:
            t = float(self._time_var.get())
        except ValueError:
            return
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
            self._time_var.set(f"{t:.2f}")
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
            self._time_var.set(f"{t:.2f}")
        else:
            bar = self._time_to_bar(t)
        self._bar_var.set(str(bar))
        self._updating = False

    def _on_bar_changed(self, *_args):
        if self._updating:
            return
        try:
            bar = int(self._bar_var.get())
        except ValueError:
            return
        measures = self._store.measures
        if not measures:
            return
        self._updating = True
        for m in measures:
            if m["measure_num"] == bar:
                self._time_var.set(f"{m['start']:.2f}")
                break
        self._updating = False

    def _on_snap_changed(self):
        """Re-snap the current time when snap mode changes."""
        try:
            t = float(self._time_var.get())
        except ValueError:
            return
        self._set_time_with_snap(t)

    # -- OK / result ------------------------------------------------------

    def _on_ok(self):
        try:
            t = float(self._time_var.get())
        except ValueError:
            return
        t = max(0.0, min(t, self._store.duration))
        try:
            bar = int(self._bar_var.get())
        except ValueError:
            bar = self._time_to_bar(t)

        type_str = self._type_var.get().strip()
        if not type_str:
            type_str = "other"
        desc = self._desc_var.get().strip()
        if not desc:
            desc = type_str
        # Preserve original score on edit; default 1.0 for new events
        score = self._edit_score if self._edit_score is not None else 1.0

        cue = self._cue_var.get()

        self.result = {
            "bar": bar,
            "time": round(t, 2),
            "time_fmt": format_time(t),
            "type": type_str,
            "score": round(float(score), 3),
            "description": desc,
            "source": "manual",
            "cue": cue,
        }
        # Add mode-appropriate extra fields
        mode = self._store.event_mode or "feature"
        if mode == "stems":
            self.result["n_stems"] = 0
            self.result["stems_in"] = []
            self.result["stems_out"] = []
        else:
            self.result["n_features"] = 0
            self.result["pass"] = 0

        self.destroy()


class ExportDialog(tk.Toplevel):
    """Modal dialog for adjusting clip boundaries and choosing format before
    exporting audio.

    Parameters
    ----------
    parent      : tk widget
    store       : AnalysisStore — for measures, duration info
    phrase_bars : int — bars per phrase (for phrase snap)
    snap_var    : tk.StringVar — shared global snap setting
    audio_path  : Path — original audio file (for detecting specs)
    start_time  : float — initial start marker time
    end_time    : float — initial end marker time
    """

    _FORMATS = [("WAV (.wav)", ".wav"),
                ("FLAC (.flac)", ".flac"),
                ("OGG Vorbis (.ogg)", ".ogg"),
                ("MP3 (.mp3)", ".mp3")]

    def __init__(self, parent, store, phrase_bars, snap_var,
                 audio_path, start_time, end_time):
        super().__init__(parent)
        self.title("Export Audio Clip")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._store = store
        self._phrase_bars = phrase_bars
        self._snap_var = snap_var
        self._updating = False
        self.result = None  # dict on OK, None on cancel

        pad = dict(padx=6, pady=3)

        # -- Row 0: Start time --
        row = 0
        ttk.Label(self, text="Start (sec):").grid(row=row, column=0,
                                                    sticky="e", **pad)
        self._start_var = tk.StringVar()
        self._start_entry = ttk.Entry(self, textvariable=self._start_var,
                                       width=12)
        self._start_entry.grid(row=row, column=1, sticky="w", **pad)
        self._start_var.trace_add("write", self._on_start_changed)

        # Start bar
        ttk.Label(self, text="Bar:").grid(row=row, column=2,
                                           sticky="e", **pad)
        self._start_bar_var = tk.StringVar()
        self._start_bar_entry = ttk.Entry(
            self, textvariable=self._start_bar_var, width=6)
        self._start_bar_entry.grid(row=row, column=3, sticky="w", **pad)
        self._start_bar_var.trace_add("write", self._on_start_bar_changed)

        # -- Row 1: End time --
        row = 1
        ttk.Label(self, text="End (sec):").grid(row=row, column=0,
                                                  sticky="e", **pad)
        self._end_var = tk.StringVar()
        self._end_entry = ttk.Entry(self, textvariable=self._end_var,
                                     width=12)
        self._end_entry.grid(row=row, column=1, sticky="w", **pad)
        self._end_var.trace_add("write", self._on_end_changed)

        # End bar
        ttk.Label(self, text="Bar:").grid(row=row, column=2,
                                           sticky="e", **pad)
        self._end_bar_var = tk.StringVar()
        self._end_bar_entry = ttk.Entry(
            self, textvariable=self._end_bar_var, width=6)
        self._end_bar_entry.grid(row=row, column=3, sticky="w", **pad)
        self._end_bar_var.trace_add("write", self._on_end_bar_changed)

        # -- Row 2: Snap mode (bound to global snap_var) --
        row = 2
        ttk.Label(self, text="Snap:").grid(row=row, column=0,
                                            sticky="e", **pad)
        snap_frame = ttk.Frame(self)
        snap_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        for val, label in [("bar", "Bar"), ("phrase", "Phrase"),
                           ("free", "Free")]:
            ttk.Radiobutton(snap_frame, text=label, variable=self._snap_var,
                            value=val,
                            command=self._on_snap_changed).pack(side="left",
                                                                 padx=4)

        # -- Row 3: Duration (computed, read-only) --
        row = 3
        ttk.Label(self, text="Duration:").grid(row=row, column=0,
                                                sticky="e", **pad)
        self._dur_label = ttk.Label(self, text="", foreground="#336699")
        self._dur_label.grid(row=row, column=1, columnspan=3,
                             sticky="w", **pad)

        # -- Row 4: Format --
        row = 4
        ttk.Label(self, text="Format:").grid(row=row, column=0,
                                              sticky="e", **pad)
        self._fmt_var = tk.StringVar()
        fmt_labels = [f[0] for f in self._FORMATS]
        self._fmt_combo = ttk.Combobox(
            self, textvariable=self._fmt_var, values=fmt_labels,
            state="readonly", width=20)
        self._fmt_combo.grid(row=row, column=1, columnspan=3,
                             sticky="w", **pad)
        # Default to input format
        in_ext = audio_path.suffix.lower() if audio_path else ".wav"
        default_idx = 0
        for i, (_, ext) in enumerate(self._FORMATS):
            if ext == in_ext:
                default_idx = i
                break
        self._fmt_combo.current(default_idx)

        # -- Row 5: Source info (read-only) --
        row = 5
        info_text = ""
        if audio_path and audio_path.exists():
            try:
                import soundfile as sf
                info = sf.info(str(audio_path))
                ch = "stereo" if info.channels == 2 else (
                    "mono" if info.channels == 1 else f"{info.channels}ch")
                info_text = (f"Source: {info.samplerate} Hz, {ch}, "
                             f"{info.subtype}")
            except Exception:
                info_text = f"Source: {audio_path.name}"
        if info_text:
            ttk.Label(self, text=info_text, foreground="#666666",
                      font=("Segoe UI", 8)).grid(
                row=row, column=0, columnspan=4, sticky="w", padx=6, pady=1)

        # -- Buttons --
        row = 6
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=8)
        ttk.Button(btn_frame, text="Export",
                   command=self._on_ok).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Cancel",
                   command=self.destroy).pack(side="left", padx=8)

        # -- Pre-fill --
        self._set_start_with_snap(start_time)
        self._set_end_with_snap(end_time)
        self._update_duration()

        self._start_entry.focus_set()
        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self.destroy())

        # Center on parent
        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"+{px - self.winfo_width() // 2}"
                      f"+{py - self.winfo_height() // 2}")

    # -- Snap helpers (same as EventDialog) ---------------------------------

    def _snap_to_bar(self, t):
        measures = self._store.measures
        if not measures:
            return (t, 0)
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t)
        if idx == 0:
            best = 0
        elif idx >= len(starts):
            best = len(starts) - 1
        else:
            best = idx if abs(starts[idx] - t) < abs(starts[idx - 1] - t) \
                else idx - 1
        m = measures[best]
        return (m["start"], m["measure_num"])

    def _snap_to_phrase(self, t):
        measures = self._store.measures
        if not measures:
            return (t, 0)
        pb = self._phrase_bars
        phrase_m = [measures[i] for i in range(0, len(measures), pb)]
        starts = [m["start"] for m in phrase_m]
        idx = bisect.bisect_right(starts, t)
        if idx == 0:
            best = 0
        elif idx >= len(starts):
            best = len(starts) - 1
        else:
            best = idx if abs(starts[idx] - t) < abs(starts[idx - 1] - t) \
                else idx - 1
        m = phrase_m[best]
        return (m["start"], m["measure_num"])

    def _time_to_bar(self, t):
        measures = self._store.measures
        if not measures:
            return 0
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t) - 1
        idx = max(0, min(idx, len(measures) - 1))
        return measures[idx]["measure_num"]

    def _bar_to_time(self, bar_num):
        for m in self._store.measures:
            if m["measure_num"] == bar_num:
                return m["start"]
        return None

    # -- Callbacks ----------------------------------------------------------

    def _set_start_with_snap(self, t):
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
        else:
            bar = self._time_to_bar(t)
        self._start_var.set(f"{t:.2f}")
        self._start_bar_var.set(str(bar))
        self._updating = False

    def _set_end_with_snap(self, t):
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
        else:
            bar = self._time_to_bar(t)
        self._end_var.set(f"{t:.2f}")
        self._end_bar_var.set(str(bar))
        self._updating = False

    def _on_start_changed(self, *_args):
        if self._updating:
            return
        try:
            t = float(self._start_var.get())
        except ValueError:
            return
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
            self._start_var.set(f"{t:.2f}")
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
            self._start_var.set(f"{t:.2f}")
        else:
            bar = self._time_to_bar(t)
        self._start_bar_var.set(str(bar))
        self._updating = False
        self._update_duration()

    def _on_end_changed(self, *_args):
        if self._updating:
            return
        try:
            t = float(self._end_var.get())
        except ValueError:
            return
        self._updating = True
        snap = self._snap_var.get()
        if snap == "bar":
            t, bar = self._snap_to_bar(t)
            self._end_var.set(f"{t:.2f}")
        elif snap == "phrase":
            t, bar = self._snap_to_phrase(t)
            self._end_var.set(f"{t:.2f}")
        else:
            bar = self._time_to_bar(t)
        self._end_bar_var.set(str(bar))
        self._updating = False
        self._update_duration()

    def _on_start_bar_changed(self, *_args):
        if self._updating:
            return
        try:
            bar = int(self._start_bar_var.get())
        except ValueError:
            return
        t = self._bar_to_time(bar)
        if t is not None:
            self._updating = True
            self._start_var.set(f"{t:.2f}")
            self._updating = False
            self._update_duration()

    def _on_end_bar_changed(self, *_args):
        if self._updating:
            return
        try:
            bar = int(self._end_bar_var.get())
        except ValueError:
            return
        t = self._bar_to_time(bar)
        if t is not None:
            self._updating = True
            self._end_var.set(f"{t:.2f}")
            self._updating = False
            self._update_duration()

    def _on_snap_changed(self):
        try:
            ts = float(self._start_var.get())
        except ValueError:
            ts = 0.0
        try:
            te = float(self._end_var.get())
        except ValueError:
            te = 0.0
        self._set_start_with_snap(ts)
        self._set_end_with_snap(te)
        self._update_duration()

    def _update_duration(self):
        try:
            ts = float(self._start_var.get())
            te = float(self._end_var.get())
        except ValueError:
            self._dur_label.config(text="—")
            return
        dur = te - ts
        if dur <= 0:
            self._dur_label.config(text="Invalid (end ≤ start)",
                                    foreground="#cc2222")
        else:
            bars = self._time_to_bar(te) - self._time_to_bar(ts)
            self._dur_label.config(
                text=f"{format_time(dur)}  ({bars} bars)",
                foreground="#336699")

    # -- OK / result --------------------------------------------------------

    def _on_ok(self):
        try:
            ts = float(self._start_var.get())
            te = float(self._end_var.get())
        except ValueError:
            return
        ts = max(0.0, min(ts, self._store.duration))
        te = max(0.0, min(te, self._store.duration))
        if te <= ts:
            messagebox.showwarning("Export", "End must be after start.")
            return

        # Get selected format extension
        sel = self._fmt_combo.current()
        ext = self._FORMATS[sel][1]

        self.result = {
            "start": round(ts, 4),
            "end": round(te, 4),
            "ext": ext,
            "bar_start": self._time_to_bar(ts),
            "bar_end": self._time_to_bar(te),
        }
        self.destroy()


# =========================================================================
# Layer 1: AnalysisStore — pure data, no matplotlib
# =========================================================================

class AnalysisStore:
    """Holds all analysis results.  No UI or plotting logic."""

    def __init__(self):
        self.reset()

    def reset(self):
        # BPM
        self.y = None
        self.sr = None
        self.beat_times = None
        self.measures = None
        self.tempo = None
        self.segments = None
        self.span_data = None
        self.downbeat_phase = 0
        self.bars_trimmed = 0
        self.loudness_dbfs = None  # RMS dBFS
        self.peak_dbfs = None      # Peak dBFS
        self.safe_volume = 1.0     # Max volume before clipping
        self.lufs = None           # Integrated LUFS (if pyloudnorm available)
        self.markers = []          # Temporary markers (session-only, not saved)
        # Waveform cache
        self.y_ds = None
        self.t_ds = None
        self.duration = 0.0
        # Key
        self.measure_keys = None
        self.phrases = None
        self.key_segments = None
        self.key_summary = None
        # Events
        self.events = None
        self.event_phrases = None
        self.event_mode = None
        self.event_summary = None
        self.event_extra = {}
        # Spectrum
        self.bar_spectra = None       # np.ndarray (n_bars, n_bands) dB
        self.band_centers = None      # np.ndarray (n_bands,) Hz
        self.band_edges = None        # np.ndarray (n_bands, 2) Hz
        self.spectral_summary = None  # dict
        # Version counter
        self.version = 0

    @property
    def has_bpm(self):
        return self.y is not None

    @property
    def has_key(self):
        return self.key_segments is not None

    @property
    def has_events(self):
        return self.events is not None

    @property
    def has_spectrum(self):
        return self.bar_spectra is not None

    def store_bpm(self, r):
        self.y = r["y"]
        self.sr = r["sr"]
        self.beat_times = r["beat_times"]
        self.tempo = r["tempo"]
        self.downbeat_phase = r.get("downbeat_phase", 0)
        self.bars_trimmed = r.get("bars_trimmed", 0)
        self.measures = r["measures"]
        self.span_data = r["span_data"]
        self.segments = r["segments"]
        self.duration = len(self.y) / self.sr
        # Loudness
        self._compute_loudness()
        # Pre-downsample waveform (~4000 points)
        n = len(self.y)
        factor = max(1, n // 4000)
        self.y_ds = self.y[::factor]
        self.t_ds = np.arange(len(self.y_ds)) * (factor / self.sr)
        self.version += 1

    def _compute_loudness(self):
        """Compute RMS dBFS, peak dBFS, and optionally LUFS."""
        y = self.y
        if y is None or len(y) == 0:
            return
        rms = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y)))
        self.loudness_dbfs = round(20 * np.log10(rms), 1) if rms > 0 else -120.0
        self.peak_dbfs = round(20 * np.log10(peak), 1) if peak > 0 else -120.0
        # Safe playback volume: pull peaks to just below 0 dBFS
        if self.peak_dbfs is not None and self.peak_dbfs > -0.1:
            # 10^(-peak_dbfs/20) gives multiplier that brings peak to 0 dBFS
            # subtract 0.5 dB headroom
            self.safe_volume = round(
                min(1.0, 10 ** (-(self.peak_dbfs + 0.5) / 20)), 2)
        else:
            self.safe_volume = 1.0
        # LUFS via pyloudnorm (optional)
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self.sr)
            # pyloudnorm expects shape (samples,) for mono
            self.lufs = round(float(meter.integrated_loudness(y)), 1)
        except ImportError:
            self.lufs = None
        except Exception:
            self.lufs = None

    def store_key(self, r):
        self.measure_keys = r["measure_keys"]
        self.phrases = r["phrases"]
        self.key_segments = r["segments"]
        self.key_summary = r["summary"]
        self.version += 1

    def store_events(self, r):
        self.events = r["events"]
        self.event_phrases = r["phrases"]
        self.event_mode = r["mode"]
        self.event_summary = r["summary"]
        self.event_extra = {k: r[k] for k in ("stem_energies", "stem_presence")
                            if k in r}
        self.version += 1

    def store_spectrum(self, r):
        self.bar_spectra = r["bar_spectra"]
        self.band_centers = r["band_centers"]
        self.band_edges = r["band_edges"]
        self.spectral_summary = r["summary"]
        self.version += 1

    # -- persistence -----------------------------------------------------

    def save(self, project_dir):
        """Save analysis data to project_dir/analysis_cache.*"""
        d = Path(project_dir)

        # Numpy arrays → compressed .npz
        arrays = {}
        if self.y is not None:
            arrays["y"] = self.y
        if self.beat_times is not None:
            arrays["beat_times"] = np.asarray(self.beat_times)
        if self.y_ds is not None:
            arrays["y_ds"] = self.y_ds
        if self.t_ds is not None:
            arrays["t_ds"] = self.t_ds
        if self.bar_spectra is not None:
            arrays["bar_spectra"] = self.bar_spectra
            arrays["band_centers"] = self.band_centers
            arrays["band_edges"] = self.band_edges
        if arrays:
            np.savez_compressed(str(d / "analysis_cache.npz"), **arrays)

        # Everything else → JSON
        def _to_list(v):
            """Convert numpy arrays/scalars to JSON-safe types."""
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        # span_data is a dict with numpy arrays — convert to plain types
        span_save = None
        if self.span_data:
            span_save = {}
            for k, v in self.span_data.items():
                span_save[k] = _to_list(v)

        data = {
            "sr": self.sr,
            "tempo": _to_list(self.tempo),
            "duration": self.duration,
            "downbeat_phase": self.downbeat_phase,
            "bars_trimmed": self.bars_trimmed,
            "loudness_dbfs": self.loudness_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "safe_volume": self.safe_volume,
            "lufs": self.lufs,
            "measures": self.measures,
            "segments": self.segments,
            "span_data": span_save,
        }

        # Key
        if self.has_key:
            data["key"] = {
                "measure_keys": self.measure_keys,
                "phrases": self.phrases,
                "key_segments": self.key_segments,
                "key_summary": self.key_summary,
            }

        # Events
        if self.has_events:
            ev_data = {
                "events": self.events,
                "event_phrases": self.event_phrases,
                "event_mode": self.event_mode,
                "event_summary": self.event_summary,
            }
            # Stem extra: energies and presence are dicts of lists
            if self.event_extra:
                se = {}
                for k, v in self.event_extra.items():
                    if isinstance(v, dict):
                        se[k] = {sk: [_to_list(x) for x in sv]
                                 if isinstance(sv, list) else _to_list(sv)
                                 for sk, sv in v.items()}
                    else:
                        se[k] = _to_list(v)
                ev_data["event_extra"] = se
            data["events_block"] = ev_data

        # Spectrum
        if self.has_spectrum:
            data["spectrum"] = {"spectral_summary": self.spectral_summary}

        # Metadata for cache validation
        data["_audio_name"] = getattr(self, "_audio_name", None)

        with open(d / "analysis_cache.json", "w") as f:
            json.dump(data, f, indent=2, default=_to_list)

    def load(self, project_dir):
        """Load cached analysis data.  Returns True if loaded, False if
        no cache or cache is invalid."""
        d = Path(project_dir)
        json_path = d / "analysis_cache.json"
        npz_path = d / "analysis_cache.npz"

        if not json_path.exists() or not npz_path.exists():
            return False

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            npz = np.load(str(npz_path), allow_pickle=False)

            # BPM data
            self.y = npz["y"]
            self.sr = data["sr"]
            self.beat_times = npz["beat_times"]
            self.tempo = data["tempo"]
            self.downbeat_phase = data.get("downbeat_phase", 0)
            self.bars_trimmed = data.get("bars_trimmed", 0)
            self.loudness_dbfs = data.get("loudness_dbfs")
            self.peak_dbfs = data.get("peak_dbfs")
            self.safe_volume = data.get("safe_volume", 1.0)
            self.lufs = data.get("lufs")
            self.measures = data["measures"]
            self.segments = data["segments"]
            # Restore span_data with numpy arrays
            sd = data.get("span_data")
            if sd and isinstance(sd, dict):
                if "times" in sd:
                    sd["times"] = np.asarray(sd["times"])
                if "bpms" in sd:
                    sd["bpms"] = np.asarray(sd["bpms"])
            self.span_data = sd
            self.duration = data["duration"]
            self.y_ds = npz["y_ds"]
            self.t_ds = npz["t_ds"]
            self._audio_name = data.get("_audio_name", "")

            # Key data
            key_block = data.get("key")
            if key_block:
                self.measure_keys = key_block["measure_keys"]
                self.phrases = key_block["phrases"]
                self.key_segments = key_block["key_segments"]
                self.key_summary = key_block["key_summary"]

            # Event data
            ev_block = data.get("events_block")
            if ev_block:
                self.events = ev_block["events"]
                self.event_phrases = ev_block["event_phrases"]
                self.event_mode = ev_block["event_mode"]
                self.event_summary = ev_block["event_summary"]
                self.event_extra = ev_block.get("event_extra", {})

            # Spectrum data
            if "bar_spectra" in npz:
                self.bar_spectra = npz["bar_spectra"]
                self.band_centers = npz["band_centers"]
                self.band_edges = npz["band_edges"]
                spec_block = data.get("spectrum")
                if spec_block:
                    self.spectral_summary = spec_block.get("spectral_summary")

            # Recompute loudness if not in cache (backward compat)
            if self.loudness_dbfs is None and self.y is not None:
                self._compute_loudness()

            self.version += 1
            npz.close()
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False


# =========================================================================
# Layer 2: ChartBuilder — lazy figure factory (single-panel templates)
# =========================================================================

# Chart names — each maps to exactly ONE panel
CHART_WAVEFORM = "Waveform"
CHART_BPM_SPAN = "BPM Span"
CHART_KEY_TIMELINE = "Key Timeline"
CHART_KEY_CONFIDENCE = "Key Confidence"
CHART_PERC_ONSET = "Perc Onset"
CHART_HARM_ONSET = "Harm Onset"
CHART_RMS_ENERGY = "RMS Energy"
CHART_CHANGE_SCORE = "Change Score"
CHART_SPECTRAL = "Spectral"
# Stem charts (dynamic names)
_STEM_PREFIX = "Stem: "  # e.g. "Stem: drums", "Stem: bass"

# Group labels for the view selector
_GROUP_BPM = [CHART_BPM_SPAN]
_GROUP_KEY = [CHART_KEY_TIMELINE, CHART_KEY_CONFIDENCE]
_GROUP_EVENTS_FEATURES = [CHART_PERC_ONSET, CHART_HARM_ONSET,
                          CHART_RMS_ENERGY, CHART_CHANGE_SCORE]
_GROUP_SPECTRUM = [CHART_SPECTRAL]


class ChartBuilder:
    """Builds single-panel matplotlib figures on demand, cached by version."""

    def __init__(self, store, beats_per_bar=4, phrase_bars=4):
        self.store = store
        self.beats_per_bar = beats_per_bar
        self.phrase_bars = phrase_bars
        self._cache = {}  # name -> (figure, version)

    # -- public API ------------------------------------------------------

    def available(self):
        """Chart names that can be built with current store data."""
        s = self.store
        out = []
        if s.has_bpm:
            out.append(CHART_WAVEFORM)
            out.append(CHART_BPM_SPAN)
        if s.has_key:
            out.extend(_GROUP_KEY)
        if s.has_events:
            if s.event_mode == "stems" and s.event_extra:
                se = s.event_extra.get("stem_energies", {})
                for stem in sorted(se.keys()):
                    out.append(f"{_STEM_PREFIX}{stem}")
            else:
                out.extend(_GROUP_EVENTS_FEATURES)
        if s.has_spectrum:
            out.extend(_GROUP_SPECTRUM)
        return out

    def get(self, name):
        """Return a (possibly cached) figure.  Builds if stale/missing."""
        cached = self._cache.get(name)
        if cached and cached[1] == self.store.version:
            return cached[0]
        if cached:
            plt.close(cached[0])
        fig = self._build(name)
        if fig:
            self._cache[name] = (fig, self.store.version)
        return fig

    def invalidate(self, *names):
        targets = names or list(self._cache.keys())
        for n in targets:
            if n in self._cache:
                plt.close(self._cache[n][0])
                del self._cache[n]

    def close_all(self):
        for _, (fig, __) in self._cache.items():
            plt.close(fig)
        self._cache.clear()

    # -- routing ---------------------------------------------------------

    def _build(self, name):
        s = self.store
        if name == CHART_WAVEFORM:
            return self._build_waveform()
        if name == CHART_BPM_SPAN and s.has_bpm:
            return self._build_bpm_span()
        if name == CHART_KEY_TIMELINE and s.has_key:
            return self._build_key_timeline()
        if name == CHART_KEY_CONFIDENCE and s.has_key:
            return self._build_key_confidence()
        if name == CHART_PERC_ONSET and s.has_events:
            return self._build_phrase_bar("onset_percussive",
                                          "Percussive Onset per Phrase",
                                          "Perc", "#cc6666", "#ff4444")
        if name == CHART_HARM_ONSET and s.has_events:
            return self._build_phrase_bar("onset_harmonic",
                                          "Harmonic Onset per Phrase",
                                          "Harm", "#6688cc", "#ff4444")
        if name == CHART_RMS_ENERGY and s.has_events:
            return self._build_phrase_bar("rms",
                                          "RMS Energy per Phrase",
                                          "RMS", "#66aa66", "#ff4444")
        if name == CHART_CHANGE_SCORE and s.has_events:
            return self._build_change_score()
        if name.startswith(_STEM_PREFIX) and s.has_events:
            stem = name[len(_STEM_PREFIX):]
            return self._build_stem_energy(stem)
        if name == CHART_SPECTRAL and s.has_spectrum:
            return self._build_spectral()
        return None

    # -- helpers ---------------------------------------------------------

    def _audio_name(self):
        s = self.store
        name = s._audio_name if hasattr(s, "_audio_name") else ""
        # Normalize fullwidth Unicode chars (e.g. ？→?) so matplotlib fonts
        # don't warn about missing glyphs.
        return unicodedata.normalize("NFKC", name) if name else ""

    def _make_fig(self, height=3.5):
        fig, ax = plt.subplots(1, 1, figsize=(14, height))
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18)
        return fig, ax

    def _add_event_lines(self, ax, y_frac=0.96, label=True):
        """Overlay event markers on any time-axis chart."""
        s = self.store
        if not s.events:
            return
        for e in s.events:
            et = e.get("type", "")
            c = ("#cc2222" if "decrease" in et or "breakdown" in et
                 else "#2266cc" if "increase" in et or "drop" in et
                 else "#888800")
            ax.axvline(x=e["time"], color=c, linewidth=1.2, alpha=0.5,
                       linestyle="--")
            if label:
                ax.text(e["time"], y_frac,
                        f'Bar {e.get("bar","")}\n{et}',
                        transform=ax.get_xaxis_transform(),
                        fontsize=5.5, ha="center", va="top", color=c,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="white", alpha=0.8,
                                  linewidth=0))

    def _add_markers(self, ax):
        """Overlay temporary markers and selection region on any chart."""
        markers = self.store.markers
        if not markers:
            return
        for t in markers:
            ax.axvline(x=t, color="#ff8800", linewidth=2.0, alpha=0.85,
                       linestyle="-", zorder=8)
        if len(markers) == 2:
            ax.axvspan(markers[0], markers[1], alpha=0.12, color="#ff8800",
                       zorder=1)

    def _phrase_times(self):
        """Return (mid_times[], widths[]) for bar charts aligned to phrases."""
        s = self.store
        if not s.event_phrases:
            return [], []
        mids, ws = [], []
        for p in s.event_phrases:
            t0 = p["start"]
            t1 = p["end"]
            mids.append((t0 + t1) / 2)
            ws.append(t1 - t0)
        return mids, ws

    # -- individual chart builders ---------------------------------------

    def _build_waveform(self):
        s = self.store
        if s.y_ds is None:
            return None

        fig, ax = self._make_fig(3.5)
        ax.plot(s.t_ds, s.y_ds, color="#888888", linewidth=0.3, alpha=0.6)

        # BPM segment shading
        if s.segments:
            cols = plt.cm.Pastel1(np.linspace(0, 0.8,
                                              max(len(s.segments), 1)))
            for i, seg in enumerate(s.segments):
                ax.axvspan(seg["start_time"], seg["end_time"],
                           alpha=0.15, color=cols[i % len(cols)])

        # Key segment labels (top)
        if s.key_segments:
            for seg in s.key_segments:
                mid = (seg["start_time"] + seg["end_time"]) / 2
                ax.text(mid, 1.02, seg["camelot"], fontsize=7,
                        ha="center", va="bottom", color="#224488",
                        fontweight="bold",
                        transform=ax.get_xaxis_transform())

        # Phrase boundaries
        if s.measures:
            for i in range(0, len(s.measures), self.phrase_bars):
                ax.axvline(x=s.measures[i]["start"], color="#2244aa",
                           linewidth=0.4, alpha=0.2)

        # Event markers (with labels)
        self._add_event_lines(ax, y_frac=0.96, label=True)
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"Waveform  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig

    def _build_bpm_span(self):
        """BPM per measure (thin bars) + smoothed span curve + segment
        medians.  Single panel version of the old BPM Detail panel 2."""
        s = self.store
        fig, ax = self._make_fig(3.5)

        # Per-measure BPM thin bars
        if s.measures:
            for m in s.measures:
                bpm = m.get("bpm")
                if bpm and bpm > 0:
                    ax.bar(m["start"], bpm, width=(m["end"] - m["start"]),
                           align="edge", color="#aaccff", alpha=0.5,
                           linewidth=0)

        # Span BPM curve
        if s.span_data:
            times = s.span_data.get("times")
            bpms = s.span_data.get("bpms")
            if times is not None and bpms is not None:
                ax.plot(times, bpms, color="#cc3333", linewidth=1.5,
                        label=f'{s.span_data.get("span", 8)}-beat span')

        # Segment BPM lines
        if s.segments:
            cols = plt.cm.Set2(np.linspace(0, 0.8,
                                           max(len(s.segments), 1)))
            for i, seg in enumerate(s.segments):
                seg_bpm = seg["bpm"]
                ax.hlines(seg_bpm, seg["start_time"],
                          seg["end_time"], colors=cols[i % len(cols)],
                          linewidth=2.5, zorder=5)
                ax.axvspan(seg["start_time"], seg["end_time"],
                           alpha=0.08, color=cols[i % len(cols)])
                mid = (seg["start_time"] + seg["end_time"]) / 2
                ax.text(mid, seg_bpm + 0.3,
                        f'{seg_bpm:.1f}', ha="center",
                        fontsize=8, fontweight="bold",
                        color=cols[i % len(cols)])
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylabel("BPM", fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"BPM Span  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        if s.span_data:
            ax.legend(fontsize=8, loc="upper right")
        return fig

    def _build_key_timeline(self):
        """Colored horizontal bars for each key segment."""
        s = self.store
        fig, ax = self._make_fig(2.5)

        if s.key_segments:
            # Assign colors per camelot code
            codes = list({seg["camelot"] for seg in s.key_segments})
            cmap = plt.cm.Set3(np.linspace(0, 1, max(len(codes), 1)))
            code_color = {c: cmap[i % len(cmap)] for i, c in enumerate(codes)}

            for seg in s.key_segments:
                t0, t1 = seg["start_time"], seg["end_time"]
                c = code_color[seg["camelot"]]
                ax.barh(0.5, t1 - t0, left=t0, height=0.6, color=c,
                        edgecolor="white", linewidth=0.5)
                mid = (t0 + t1) / 2
                label = seg["camelot"]
                if (t1 - t0) > s.duration * 0.06:
                    label += f'\n{seg.get("key_name", "")}'
                ax.text(mid, 0.5, label, ha="center", va="center",
                        fontsize=8, fontweight="bold")
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"Key Timeline  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig

    def _build_key_confidence(self):
        """Per-phrase key detection confidence bars with color legend."""
        s = self.store
        fig, ax = self._make_fig(2.5)

        if s.phrases:
            codes = sorted({p.get("camelot", "?") for p in s.phrases})
            cmap = plt.cm.Set3(np.linspace(0, 1, max(len(codes), 1)))
            code_color = {c: cmap[i % len(cmap)] for i, c in enumerate(codes)}

            # Track which codes actually appear (for legend)
            seen = set()
            for p in s.phrases:
                t0 = p["start"]
                t1 = p["end"]
                conf = p.get("confidence", 0)
                cam = p.get("camelot", "?")
                key_name = p.get("key_name", "")
                lbl = f"{cam} {key_name}" if cam not in seen else None
                seen.add(cam)
                ax.bar((t0 + t1) / 2, conf, width=(t1 - t0) * 0.9,
                       color=code_color.get(cam, "#aaaaaa"), alpha=0.8,
                       edgecolor="white", linewidth=0.3,
                       label=lbl)

            # Legend — show all unique camelot codes
            ncol = min(len(codes), 6)
            ax.legend(fontsize=7, loc="upper right", ncol=ncol,
                      framealpha=0.85)
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Confidence", fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"Key Confidence  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig

    def _build_phrase_bar(self, field, title, ylabel, color, event_color):
        """Generic single-panel phrase-level bar chart for a named field."""
        s = self.store
        fig, ax = self._make_fig(3.0)

        mids, ws = self._phrase_times()
        if not mids:
            return fig

        # Gather values
        event_times = {e["time"] for e in (s.events or [])}
        vals, colors = [], []
        for p in s.event_phrases:
            vals.append(p.get(field, 0))
            # Highlight phrases that contain an event
            is_ev = any(p["start"] <= t <= p["end"] for t in event_times)
            colors.append(event_color if is_ev else color)

        ax.bar(mids, vals, width=[w * 0.9 for w in ws],
               color=colors, alpha=0.75, edgecolor="white", linewidth=0.3)
        self._add_event_lines(ax, y_frac=0.95, label=False)
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"{title}  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig

    def _build_change_score(self):
        """Combined feature change score per phrase with threshold line.
        Recomputes scores from raw features (same algorithm as
        plot_feature_analysis in event_detect.py)."""
        s = self.store
        fig, ax = self._make_fig(3.0)

        mids, ws = self._phrase_times()
        if not mids or not s.event_phrases:
            return fig

        # Compute scores from raw phrase features
        feature_names = ["onset_percussive", "onset_harmonic", "rms",
                         "centroid", "low_freq_ratio"]
        weights = {"onset_percussive": 0.25, "onset_harmonic": 0.20,
                   "rms": 0.25, "centroid": 0.15, "low_freq_ratio": 0.15}
        context = 2

        # Gather raw values, skip any missing features gracefully
        available = [f for f in feature_names
                     if f in s.event_phrases[0]]
        if not available:
            return fig

        raw = {f: np.array([p.get(f, 0) for p in s.event_phrases])
               for f in available}
        normed = {f: normalize_feature(raw[f]) for f in available}

        scores = np.zeros(len(s.event_phrases))
        for i in range(context, len(s.event_phrases)):
            for f in available:
                before = np.mean(normed[f][max(0, i - context):i])
                scores[i] += abs(normed[f][i] - before) * weights.get(f, 0.1)

        event_times = {e["time"] for e in (s.events or [])}
        colors = []
        for p in s.event_phrases:
            is_ev = any(p["start"] <= t <= p["end"] for t in event_times)
            colors.append("#ff4444" if is_ev else "#ccaa66")

        ax.bar(mids, scores, width=[w * 0.9 for w in ws],
               color=colors, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axhline(y=0.15, color="#cc0000", linewidth=1, linestyle="--",
                   alpha=0.6, label="min_score (0.15)")
        self._add_event_lines(ax, y_frac=0.95, label=False)
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"Change Score  - {self._audio_name()}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=8)
        return fig

    def _build_stem_energy(self, stem):
        """Per-phrase energy for a single stem (drums, bass, etc.)."""
        s = self.store
        fig, ax = self._make_fig(3.0)

        se = s.event_extra.get("stem_energies", {})
        spr = s.event_extra.get("stem_presence", {})
        energies = se.get(stem, [])
        presence = spr.get(stem, [])

        mids, ws = self._phrase_times()
        if not mids or not energies:
            return fig

        event_times = {e["time"] for e in (s.events or [])}
        colors, alphas = [], []
        for i, p in enumerate(s.event_phrases):
            is_ev = any(p["start"] <= t <= p["end"] for t in event_times)
            colors.append("#ff4444" if is_ev else "#6699cc")
            pres = presence[i] if i < len(presence) else True
            alphas.append(0.85 if pres else 0.3)

        vals = [energies[i] if i < len(energies) else 0
                for i in range(len(mids))]
        bars = ax.bar(mids, vals, width=[w * 0.9 for w in ws],
                      color=colors, edgecolor="white", linewidth=0.3)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)
        self._add_event_lines(ax, y_frac=0.95, label=False)
        self._add_markers(ax)

        ax.set_xlim(0, s.duration)
        ax.set_ylabel(stem.capitalize(), fontsize=9)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(
            f"{stem.capitalize()} Energy  - {self._audio_name()}  "
            f"(shaded = present)", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig

    def _build_spectral(self):
        """Heatmap of per-bar spectral energy (third-octave bands)."""
        s = self.store
        fig, ax = self._make_fig(4.0)

        spectra = s.bar_spectra       # (n_bars, n_bands)
        edges = s.band_edges          # (n_bands, 2)

        # Build time edges from measures
        bar_times = [(m["start"], m["end"]) for m in s.measures]
        t_edges = [bar_times[0][0]] + [t[1] for t in bar_times]
        # Build frequency edges from band edges
        f_edges = [edges[0, 0]] + [edges[i, 1] for i in range(len(edges))]

        T, F = np.meshgrid(t_edges, f_edges)
        pcm = ax.pcolormesh(T, F, spectra.T, shading="flat", cmap="inferno",
                            vmin=NOISE_FLOOR_DB, vmax=0)
        ax.set_yscale("log")
        ax.set_ylim(20, 20000)
        ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
        ax.set_yticklabels(["50", "100", "200", "500", "1k", "2k",
                            "5k", "10k"])

        self._add_event_lines(ax, y_frac=0.96, label=True)
        self._add_markers(ax)

        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label("dB", fontsize=9)

        ax.set_xlim(0, s.duration)
        ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_title(f"Spectral Energy  - {self._audio_name()}", fontsize=10)
        ax.tick_params(labelsize=8)
        return fig


# =========================================================================
# Layer 3: AudioAnalysisApp — slim tkinter UI
# =========================================================================

class AudioAnalysisApp:

    def __init__(self, root, initial_audio=None):
        self.root = root
        self.root.title("DJ Audio Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(900, 600)

        self.audio_path = None
        self.project_dir = None

        self.beats_per_bar = 4
        self.phrase_bars = 4
        self.hop_length = 512

        # Data + chart layers
        self.store = AnalysisStore()
        self.builder = ChartBuilder(self.store, self.beats_per_bar,
                                     self.phrase_bars)

        # Playback
        self.playing = False
        self.play_start_offset = 0.0
        self.play_start_time = 0.0
        self.paused = False
        self.pause_position = 0.0
        self._seeking = False
        self._analyzing = False
        self._markers = []  # Temporary markers (session-only, max 2)

        # Display — per-chart widget cache for independent zoom/pan state
        self.active_chart = None
        self._chart_widgets = {}  # name -> {canvas, toolbar_frame, cursor_line, blit_bg}
        self.overlay_label = None

        # View controls
        self._programmatic_zoom = False
        self.auto_follow = False
        self._zoom_bars_map = {
            "4 bars": 4, "8 bars": 8, "16 bars": 16,
            "32 bars": 32, "64 bars": 64, "Full": None, "Custom": None,
        }
        self._zoom_options = ["4 bars", "8 bars", "16 bars", "32 bars",
                              "64 bars", "Full"]

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self._build_ui()

        if initial_audio:
            self.root.after(100, lambda: self._open_file(initial_audio))

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Menu
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open Audio...",
                       command=self._open_file_dialog,
                       accelerator="Ctrl+O")
        fm.add_command(label="Export to Rekordbox...",
                       command=self._export_rekordbox)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.quit)
        mb.add_cascade(label="File", menu=fm)
        self.root.config(menu=mb)
        self.root.bind("<Control-o>", lambda e: self._open_file_dialog())

        # Top: toolbar + view selector
        top = ttk.Frame(self.root, padding=2)
        top.pack(side="top", fill="x")

        tb = ttk.Frame(top)
        tb.pack(side="top", fill="x", pady=(0, 2))

        self.btn_open = ttk.Button(tb, text="\U0001F4C2 Open",
                                    command=self._open_file_dialog)
        self.btn_open.pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.btn_bpm = ttk.Button(tb, text="Run BPM",
                                   command=self._run_bpm, state="disabled")
        self.btn_bpm.pack(side="left", padx=2)
        self.btn_key = ttk.Button(tb, text="Run Key",
                                   command=self._run_key, state="disabled")
        self.btn_key.pack(side="left", padx=2)
        self.btn_events = ttk.Button(tb, text="Run Events",
                                      command=self._run_events,
                                      state="disabled")
        self.btn_events.pack(side="left", padx=2)
        self.btn_spectrum = ttk.Button(tb, text="Run Spectrum",
                                        command=self._run_spectrum,
                                        state="disabled")
        self.btn_spectrum.pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.btn_all = ttk.Button(tb, text="Run All",
                                   command=self._run_all, state="disabled")
        self.btn_all.pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.btn_add_event = ttk.Button(tb, text="+Event",
                                         command=self._add_event_at_playhead,
                                         state="disabled")
        self.btn_add_event.pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.btn_mark = ttk.Button(tb, text="Mark",
                                    command=self._place_marker,
                                    state="disabled")
        self.btn_mark.pack(side="left", padx=2)
        self.btn_clear_marks = ttk.Button(tb, text="Clear Marks",
                                           command=self._clear_markers,
                                           state="disabled")
        self.btn_clear_marks.pack(side="left", padx=2)
        self.btn_export_clip = ttk.Button(tb, text="Export Clip",
                                           command=self._export_clip,
                                           state="disabled")
        self.btn_export_clip.pack(side="left", padx=2)

        self.view_frame = ttk.Frame(top)
        self.view_frame.pack(side="top", fill="x")
        self.view_var = tk.StringVar(value="")
        self.view_buttons = {}

        # Persistent track info (right side of view bar)
        self.info_var = tk.StringVar(value="")
        self.info_label = ttk.Label(
            self.view_frame, textvariable=self.info_var,
            font=("Consolas", 9), foreground="#336699")
        self.info_label.pack(side="right", padx=(8, 4))

        # Bottom: status
        sf = ttk.Frame(self.root, padding=(4, 2))
        sf.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(sf, textvariable=self.status_var,
                  font=("Segoe UI", 9)).pack(side="left")

        # Transport
        tr = ttk.Frame(self.root, padding=4)
        tr.pack(side="bottom", fill="x")
        self.btn_start = ttk.Button(tr, text="\u23EE", width=3,
                                     command=self._go_start, state="disabled")
        self.btn_start.pack(side="left", padx=1)
        self.btn_play = ttk.Button(tr, text="\u25B6", width=3,
                                    command=self._toggle_play, state="disabled")
        self.btn_play.pack(side="left", padx=1)
        self.btn_stop = ttk.Button(tr, text="\u25A0", width=3,
                                    command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=1)
        self.btn_end = ttk.Button(tr, text="\u23ED", width=3,
                                   command=self._go_end, state="disabled")
        self.btn_end.pack(side="left", padx=1)
        ttk.Separator(tr, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.time_label = ttk.Label(tr, text="0:00 / 0:00",
                                     font=("Consolas", 11), width=16)
        self.time_label.pack(side="left", padx=4)
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_slider = ttk.Scale(tr, from_=0, to=1,
                                      variable=self.seek_var,
                                      orient="horizontal",
                                      command=self._on_seek)
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=4)
        self.seek_slider.config(state="disabled")

        # View controls (right side of transport)
        ttk.Separator(tr, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        self.btn_center = ttk.Button(tr, text="\u2316", width=3,
                                      command=self._center_on_playhead,
                                      state="disabled")
        self.btn_center.pack(side="left", padx=1)

        self.follow_var = tk.BooleanVar(value=False)
        self.chk_follow = ttk.Checkbutton(
            tr, text="Follow", variable=self.follow_var,
            command=self._toggle_follow)
        self.chk_follow.pack(side="left", padx=(4, 2))

        # Global snap toggle
        ttk.Separator(tr, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        ttk.Label(tr, text="Snap:", font=("Segoe UI", 9)).pack(
            side="left", padx=(2, 0))
        self.snap_var = tk.StringVar(value="bar")
        for val, lbl in [("bar", "Bar"), ("phrase", "Phr"), ("free", "Free")]:
            ttk.Radiobutton(tr, text=lbl, variable=self.snap_var,
                            value=val).pack(side="left", padx=1)

        # Volume control
        ttk.Separator(tr, orient="vertical").pack(side="left", fill="y",
                                                    padx=6)
        ttk.Label(tr, text="Vol:", font=("Segoe UI", 9)).pack(
            side="left", padx=(2, 0))
        self.volume_var = tk.DoubleVar(value=0.7)
        self.volume_slider = VolumeSlider(
            tr, variable=self.volume_var,
            command=self._on_volume_changed, length=100)
        self.volume_slider.pack(side="left", padx=(2, 2))
        self.volume_pct = ttk.Label(tr, text="70%", font=("Consolas", 8),
                                     width=5)
        self.volume_pct.pack(side="left", padx=(0, 4))
        pygame.mixer.music.set_volume(0.7)

        # Zoom is per-chart — created in _show_chart per toolbar_frame
        self.zoom_var = tk.StringVar(value="Full")

        # Chart area
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(side="top", fill="both", expand=True,
                               padx=4, pady=2)
        self.placeholder = ttk.Label(
            self.chart_frame,
            text="Open an audio file to begin.\n\nFile > Open Audio  or  Ctrl+O",
            font=("Segoe UI", 14), anchor="center", justify="center")
        self.placeholder.pack(expand=True)

        # Keys
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Escape>", lambda e: self._stop())
        self.root.bind("<Home>", lambda e: self._go_start())
        self.root.bind("<End>", lambda e: self._go_end())
        self.root.bind("c", lambda e: self._center_on_playhead())
        self.root.bind("f", lambda e: self._toggle_follow())
        self.root.bind("e", lambda e: self._add_event_at_playhead())
        self.root.bind("m", lambda e: self._place_marker())

    # ------------------------------------------------------------------
    # Rekordbox export
    # ------------------------------------------------------------------

    def _export_rekordbox(self):
        if not self.audio_path or not self.project_dir:
            messagebox.showwarning("Export", "No track loaded.")
            return
        if not self.store.tempo:
            messagebox.showwarning("Export",
                                   "Run at least BPM analysis before exporting.")
            return

        # Save current analysis so the cache file is up to date
        self.store.save(self.project_dir)

        try:
            from rekordbox_export import export_track, load_store_data, \
                get_export_path
            data = load_store_data(self.project_dir)
            result_path, num_tracks = export_track(
                audio_path=self.audio_path,
                store_data=data,
            )
            messagebox.showinfo("Export",
                                f"{self.audio_path.name}\n\n"
                                f"Exported to: {result_path}\n"
                                f"Tracks in collection: {num_tracks}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ------------------------------------------------------------------
    # File open
    # ------------------------------------------------------------------

    def _open_file_dialog(self):
        p = filedialog.askopenfilename(
            title="Open Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                       ("All files", "*.*")])
        if p:
            self._open_file(p)

    def _open_file(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showerror("Error", f"Not found:\n{path}")
            return

        self._stop()
        self.audio_path = path
        self.project_dir = path.parent / path.stem
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.root.title(f"DJ Audio Analysis - {path.name}")

        # Reset
        self.builder.close_all()
        self.store.reset()
        self.store._audio_name = path.name
        self._clear_display()
        self._update_view_selector()
        # Reset volume slider to full (safe zone unknown until BPM runs)
        self.volume_slider.set_safe(1.0)
        self.volume_var.set(1.0)
        pygame.mixer.music.set_volume(1.0)
        self.volume_pct.config(text="100%", foreground="")
        self.btn_add_event.config(state="disabled")
        self._markers = []
        self.store.markers = []
        self._update_marker_buttons()

        self.placeholder.config(
            text=f"File: {path.name}\nProject: {self.project_dir}\n\n"
                 "Click 'Run BPM' to start, or 'Run All'.")
        self.placeholder.pack(expand=True)

        try:
            pygame.mixer.music.load(str(path))
            for b in (self.btn_play, self.btn_stop, self.btn_start,
                      self.btn_end, self.btn_center):
                b.config(state="normal")
            self.seek_slider.config(state="normal")
        except Exception as e:
            messagebox.showwarning("Playback", f"Cannot load:\n{e}")

        self.btn_bpm.config(state="normal")
        self.btn_all.config(state="normal")

        # Try loading cached analysis data
        if self.store.load(self.project_dir):
            self._on_cache_loaded()
            return

        try:
            import librosa
            dur = librosa.get_duration(path=str(path))
            self.store.duration = dur
            self.seek_slider.config(to=dur)
            self.time_label.config(text=f"0:00 / {format_time(dur)}")
        except Exception:
            pass

        self._set_status(f"Loaded: {path.name}")
        self._update_info()

    def _on_cache_loaded(self):
        """Restore UI from cached analysis data."""
        s = self.store
        self.seek_slider.config(to=s.duration)
        self.time_label.config(text=f"0:00 / {format_time(s.duration)}")

        self.builder.invalidate()
        self._invalidate_chart_widgets()
        self._update_view_selector()
        self._show_waveform()

        # Enable appropriate buttons
        if s.has_bpm:
            self.btn_key.config(state="normal")
            self.btn_events.config(state="normal")
            self._apply_safe_volume()
            self._update_marker_buttons()

        # Build status summary
        parts = []
        if s.tempo:
            parts.append(f"BPM: {s.tempo:.1f}")
        if s.has_key and s.key_summary:
            dom = s.key_summary.get("dominant", {})
            parts.append(
                f'Key: {dom.get("key_name","?")} ({dom.get("camelot","?")})')
        if s.has_events:
            self.btn_add_event.config(state="normal")
            parts.append(f"Events: {len(s.events)} ({s.event_mode})")
        parts.append("(cached)")

        self._set_status("  |  ".join(parts))
        self._update_info()

    def _update_info(self):
        """Update the persistent track info label with BPM, Key, loudness, duration."""
        s = self.store
        parts = []
        if s.duration > 0:
            parts.append(format_time(s.duration))
        if s.tempo:
            parts.append(f"{s.tempo:.1f} BPM")
        if s.has_key and s.key_summary:
            dom = s.key_summary.get("dominant", {})
            key_name = dom.get("key_name", "")
            camelot = dom.get("camelot", "")
            if key_name:
                parts.append(f"{camelot}  {key_name}")
        if s.loudness_dbfs is not None:
            loud = f"{s.loudness_dbfs:+.1f} dBFS"
            if s.lufs is not None:
                loud += f" / {s.lufs:+.1f} LUFS"
            if s.peak_dbfs is not None and s.peak_dbfs > -0.1:
                loud += " \u26A0"  # warning sign for clipping
            parts.append(loud)
        if s.has_events and s.events:
            parts.append(f"{len(s.events)} events")
        self.info_var.set("    ".join(parts))

    # ------------------------------------------------------------------
    # View selector + display
    # ------------------------------------------------------------------

    def _update_view_selector(self):
        for w in self.view_frame.winfo_children():
            if w is self.info_label:
                continue
            w.destroy()
        self.view_buttons.clear()
        names = self.builder.available()
        if not names:
            return
        ttk.Label(self.view_frame, text="View:",
                  font=("Segoe UI", 9, "bold")).pack(side="left",
                                                       padx=(4, 8))
        prev_group = None
        for n in names:
            # Determine group for separator
            if n == CHART_WAVEFORM:
                group = "wave"
            elif n == CHART_BPM_SPAN:
                group = "bpm"
            elif n in _GROUP_KEY:
                group = "key"
            elif n in _GROUP_SPECTRUM:
                group = "spectrum"
            else:
                group = "events"
            if prev_group and group != prev_group:
                ttk.Separator(self.view_frame, orient="vertical").pack(
                    side="left", fill="y", padx=4)
            prev_group = group

            rb = ttk.Radiobutton(self.view_frame, text=n,
                                  variable=self.view_var, value=n,
                                  command=lambda n=n: self._show_chart(n))
            rb.pack(side="left", padx=2)
            self.view_buttons[n] = rb

    def _active_widgets(self):
        """Return widget dict for active chart, or None."""
        return self._chart_widgets.get(self.active_chart)

    def _destroy_chart_widgets(self, name):
        """Destroy and remove cached widgets for a single chart."""
        w = self._chart_widgets.pop(name, None)
        if w is None:
            return
        # Cancel pending matplotlib idle-draw / event-loop callbacks before
        # destroying the widget to avoid orphaned Tcl "after" errors.
        try:
            canvas = w["canvas"]
            tk_widget = canvas.get_tk_widget()
            for attr in ("_idle_draw_id", "_event_loop_id"):
                after_id = getattr(canvas, attr, None)
                if after_id:
                    tk_widget.after_cancel(after_id)
                    setattr(canvas, attr, None)
        except Exception:
            pass
        try:
            w["canvas"].get_tk_widget().destroy()
        except tk.TclError:
            pass
        try:
            w["toolbar_frame"].destroy()
        except tk.TclError:
            pass

    def _clear_display(self):
        """Destroy all chart widgets (e.g., on file open or full rebuild)."""
        for name in list(self._chart_widgets):
            self._destroy_chart_widgets(name)
        self.active_chart = None

    def _invalidate_chart_widgets(self, *names):
        """Discard cached widgets for specific charts (data changed).

        If no names given, discard all.
        """
        targets = names or list(self._chart_widgets)
        for n in list(targets):
            self._destroy_chart_widgets(n)
        if self.active_chart and self.active_chart not in self._chart_widgets:
            self.active_chart = None

    def _show_chart(self, name=None):
        if name is None:
            name = self.view_var.get()
        if not name:
            return

        # Ask builder for figure (cached or freshly built)
        fig = self.builder.get(name)
        if fig is None:
            return

        # Already displayed and up-to-date?
        if name == self.active_chart and name in self._chart_widgets:
            return

        self.placeholder.pack_forget()
        self._hide_overlay()

        # Hide current chart (don't destroy — keep zoom state)
        if self.active_chart and self.active_chart in self._chart_widgets:
            w = self._chart_widgets[self.active_chart]
            w["zoom_level"] = self.zoom_var.get()  # save per-chart zoom
            w["canvas"].get_tk_widget().pack_forget()
            w["toolbar_frame"].pack_forget()

        self.active_chart = name
        self.view_var.set(name)

        # Reuse existing widgets or create new ones
        if name in self._chart_widgets:
            w = self._chart_widgets[name]
            # Guard against on_xlim_changed firing during re-pack
            self._programmatic_zoom = True
            try:
                w["toolbar_frame"].pack(side="top", fill="x")
                w["canvas"].get_tk_widget().pack(side="top", fill="both",
                                                  expand=True)
                # Restore per-chart zoom level and re-apply to axes
                saved_zoom = w.get("zoom_level", "Full")
                self.zoom_var.set(saved_zoom)
                self._apply_zoom()
                # Mark blit_bg stale — the draw_event handler will
                # re-capture it when the deferred draw fires with
                # correct geometry.  _draw_cursor falls back to
                # draw_idle for one frame until blit_bg is valid.
                w["blit_bg"] = None
            finally:
                self._programmatic_zoom = False
            if self.playing or self.paused:
                self._draw_cursor(self._cur_pos())
            return

        # --- Create new widgets for this chart ---
        toolbar_frame = ttk.Frame(self.chart_frame)
        toolbar_frame.pack(side="top", fill="x")

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        nav_toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        nav_toolbar.update()

        # Per-chart zoom combobox (right side of toolbar)
        ttk.Separator(toolbar_frame, orient="vertical").pack(
            side="left", fill="y", padx=6)
        ttk.Label(toolbar_frame, text="Zoom:",
                  font=("Segoe UI", 9)).pack(side="left", padx=(2, 2))
        zoom_combo = ttk.Combobox(
            toolbar_frame, textvariable=self.zoom_var,
            values=self._zoom_options, state="readonly", width=8)
        zoom_combo.pack(side="left", padx=2)
        zoom_combo.bind("<<ComboboxSelected>>", self._on_zoom_changed)

        # Cursor on first axes
        ax0 = fig.get_axes()[0]
        cursor_line = ax0.axvline(x=0, color="#00cc00", linewidth=2,
                                   alpha=0.85, visible=False)
        canvas.draw()
        blit_bg = canvas.copy_from_bbox(fig.bbox)

        self._chart_widgets[name] = {
            "canvas": canvas,
            "toolbar_frame": toolbar_frame,
            "nav_toolbar": nav_toolbar,
            "cursor_line": cursor_line,
            "blit_bg": blit_bg,
            "zoom_level": "Full",  # new charts always start at full view
        }
        self.zoom_var.set("Full")

        # Click-to-seek (left) and event edit (right)
        def on_click(event):
            if self.active_chart != name:
                return  # click on inactive chart — ignore
            w = self._active_widgets()
            tb = w["nav_toolbar"] if w else None
            if not event.inaxes or event.xdata is None:
                return
            if tb and tb.mode != "":
                return  # pan/zoom active
            if event.button == 1:
                self._seek_to(event.xdata)
            elif event.button == 3:
                self._on_chart_right_click(event, canvas)
        canvas.mpl_connect("button_press_event", on_click)

        # Re-cache blit bg on resize/zoom
        def on_draw(event):
            w = self._chart_widgets.get(name)
            if w:
                w["blit_bg"] = w["canvas"].copy_from_bbox(fig.bbox)
        canvas.mpl_connect("draw_event", on_draw)

        # Detect manual toolbar zoom → switch dropdown to "Custom"
        def on_xlim_changed(ax_):
            if self._programmatic_zoom:
                return
            if self.active_chart == name:
                w = self._chart_widgets.get(name)
                if w:
                    w["zoom_level"] = "Custom"
                self.zoom_var.set("Custom")
        ax0.callbacks.connect("xlim_changed", on_xlim_changed)

        if self.playing or self.paused:
            self._draw_cursor(self._cur_pos())

    def _show_waveform(self):
        """Show the waveform view (or refresh if active)."""
        if self.active_chart == CHART_WAVEFORM:
            # Invalidate and rebuild in-place
            self.builder.invalidate(CHART_WAVEFORM)
            self._invalidate_chart_widgets(CHART_WAVEFORM)
        self._show_chart(CHART_WAVEFORM)

    # ------------------------------------------------------------------
    # Analysis overlay
    # ------------------------------------------------------------------

    def _show_overlay(self, msg):
        if self.overlay_label is None:
            self.overlay_label = ttk.Label(
                self.chart_frame, text=msg,
                font=("Segoe UI", 16),
                background="#ffffcc", foreground="#333333",
                padding=20, relief="solid", borderwidth=1)
        else:
            self.overlay_label.config(text=msg)
        self.overlay_label.place(relx=0.5, rely=0.5, anchor="center")

    def _hide_overlay(self):
        if self.overlay_label:
            self.overlay_label.place_forget()

    # ------------------------------------------------------------------
    # Thread helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _set_analysis_btns(self, state):
        for b in (self.btn_bpm, self.btn_key, self.btn_events,
                  self.btn_spectrum, self.btn_all):
            b.config(state=state)

    def _run_in_thread(self, fn, callback, msg):
        self._analyzing = True
        self._set_analysis_btns("disabled")
        self._set_status(msg)
        self._show_overlay(msg)

        def worker():
            try:
                result = fn()
                self.root.after(0, lambda: self._done(callback, result, None))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._done(callback, None, e))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, callback, result, error):
        self._analyzing = False
        self._hide_overlay()
        self._set_analysis_btns("normal" if self.audio_path else "disabled")
        if error:
            self._set_status(f"Error: {error}")
            messagebox.showerror("Analysis Error", str(error))
        else:
            callback(result)

    # ------------------------------------------------------------------
    # BPM
    # ------------------------------------------------------------------

    def _run_bpm(self):
        if not self.audio_path:
            return

        def do():
            d = detect_beats(str(self.audio_path), sr=44100,
                             hop_length=self.hop_length)
            phase = d.get("downbeat_phase", 0)
            m = group_beats_into_measures(d["beat_times"],
                                           self.beats_per_bar, phase=phase)
            sp = compute_span_bpm(d["beat_times"], 8)
            seg = find_bpm_segments(m, tolerance=2.0, min_segment_bars=4)
            return {"y": d["y"], "sr": d["sr"],
                    "beat_times": d["beat_times"], "tempo": d["tempo"],
                    "downbeat_phase": phase,
                    "bars_trimmed": d.get("bars_trimmed", 0),
                    "measures": m, "span_data": sp, "segments": seg}

        self._run_in_thread(do, self._on_bpm, "Running BPM analysis...")

    def _on_bpm(self, r):
        self.store.store_bpm(r)
        self.builder.invalidate()
        self._invalidate_chart_widgets()
        self.seek_slider.config(to=self.store.duration)
        self._update_view_selector()
        self._show_waveform()
        self._save_detail("bpm")
        try:
            self.store.save(self.project_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Cache save error: {e}", file=sys.stderr)
        self.btn_key.config(state="normal")
        self.btn_events.config(state="normal")
        self._apply_safe_volume()
        self._update_marker_buttons()
        bpm = f"{self.store.tempo:.1f}" if self.store.tempo else "?"
        phase_info = (f"  |  downbeat phase: {self.store.downbeat_phase}"
                      if self.store.downbeat_phase > 0 else "")
        trim_info = (f"  |  trimmed {self.store.bars_trimmed} intro bar(s)"
                     if self.store.bars_trimmed > 0 else "")
        self._set_status(
            f"BPM: {bpm}  |  {len(self.store.measures)} measures  |  "
            f"{len(self.store.segments)} seg(s){phase_info}{trim_info}")
        self._update_info()

    # ------------------------------------------------------------------
    # Key
    # ------------------------------------------------------------------

    def _run_key(self):
        if not self.store.has_bpm:
            messagebox.showinfo("Info", "Run BPM first.")
            return

        def do():
            s = self.store
            mk = compute_measure_chroma(s.y, s.sr, s.measures,
                                         use_harmonic=True)
            ph = compute_phrase_chroma(s.y, s.sr, s.measures,
                                        phrase_bars=self.phrase_bars,
                                        use_harmonic=True)
            seg = find_key_segments(ph, min_segment_phrases=2)
            sm = key_compute_summary(seg, ph, s.duration)
            return {"measure_keys": mk, "phrases": ph,
                    "segments": seg, "summary": sm}

        self._run_in_thread(do, self._on_key, "Running Key analysis...")

    def _on_key(self, r):
        self.store.store_key(r)
        self.builder.invalidate(CHART_WAVEFORM, CHART_KEY_TIMELINE,
                                CHART_KEY_CONFIDENCE)
        self._invalidate_chart_widgets(CHART_WAVEFORM, CHART_KEY_TIMELINE,
                                        CHART_KEY_CONFIDENCE)
        self._update_view_selector()
        self._show_waveform()
        self._save_detail("key")
        try:
            self.store.save(self.project_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Cache save error: {e}", file=sys.stderr)
        dom = r["summary"].get("dominant", {})
        self._set_status(
            f'Key: {dom.get("key_name","?")} ({dom.get("camelot","?")})  |  '
            f'{len(r["segments"])} seg(s)')
        self._update_info()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _run_events(self):
        if not self.store.has_bpm:
            messagebox.showinfo("Info", "Run BPM first.")
            return

        def do():
            s = self.store
            stems_dir, stem_paths = find_or_create_stems(
                str(self.audio_path))
            if stem_paths:
                import librosa as _lr
                se = {}
                for sn, sp in stem_paths.items():
                    ys, _ = _lr.load(sp, sr=s.sr, mono=True)
                    se[sn] = compute_stem_phrase_energy(
                        ys, s.sr, s.measures, phrase_bars=self.phrase_bars)
                spr = compute_stem_presence(se, presence_threshold=0.15)
                phrases = []
                for i in range(0, len(s.measures), self.phrase_bars):
                    g = s.measures[i:i + self.phrase_bars]
                    if g:
                        phrases.append({
                            "phrase_num": len(phrases) + 1,
                            "start_bar": g[0]["measure_num"],
                            "end_bar": g[-1]["measure_num"],
                            "start": g[0]["start"], "end": g[-1]["end"]})
                ev = detect_stem_events(spr, phrases, min_gap_phrases=2)
                mode = "stems"
                extra = {"stem_energies": se, "stem_presence": spr}
            else:
                bf = compute_bar_features(s.y, s.sr, s.measures)
                phrases = compute_phrase_features(
                    bf, phrase_bars=self.phrase_bars)
                ev = detect_feature_events(
                    phrases, context=2, min_score=0.15,
                    min_features=2, min_gap_phrases=2, bar_features=bf)
                mode = "features"
                extra = {}
            sm = event_compute_summary(ev, phrases, s.duration, mode=mode)
            return {"events": ev, "phrases": phrases,
                    "mode": mode, "summary": sm, **extra}

        self._run_in_thread(do, self._on_events, "Running Event analysis...")

    def _on_events(self, r):
        self.store.store_events(r)
        # Invalidate waveform + all event-related charts (features + stems)
        ev_charts = [CHART_WAVEFORM] + _GROUP_EVENTS_FEATURES
        stem_charts = [n for n in self._chart_widgets
                       if n.startswith(_STEM_PREFIX)]
        self.builder.invalidate(*(ev_charts + stem_charts))
        self._invalidate_chart_widgets(*(ev_charts + stem_charts))
        self._update_view_selector()
        self._show_waveform()
        self._save_detail("events")
        try:
            self.store.save(self.project_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Cache save error: {e}", file=sys.stderr)
        self.btn_add_event.config(state="normal")
        self._set_status(
            f'Events ({r["mode"]}): {len(r["events"])} detected')
        self._update_info()

    # ------------------------------------------------------------------
    # Spectrum
    # ------------------------------------------------------------------

    def _run_spectrum(self):
        if not self.store.has_bpm:
            messagebox.showinfo("Info", "Run BPM first.")
            return

        def do():
            s = self.store
            r = compute_bar_spectra(s.y, s.sr, s.measures)
            summary = compute_spectral_summary(r)
            return {"bar_spectra": r["bar_spectra"],
                    "band_centers": r["band_centers"],
                    "band_edges": r["band_edges"],
                    "summary": summary}

        self._run_in_thread(do, self._on_spectrum,
                            "Running Spectral analysis...")

    def _on_spectrum(self, r):
        self.store.store_spectrum(r)
        self.builder.invalidate(CHART_SPECTRAL)
        self._invalidate_chart_widgets(CHART_SPECTRAL)
        self._update_view_selector()
        self._show_waveform()
        try:
            self.store.save(self.project_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Cache save error: {e}", file=sys.stderr)
        summary = r["summary"]
        self._set_status(f'Spectrum: {summary["energy_profile"]}')
        self._update_info()

    # ------------------------------------------------------------------
    # Run All
    # ------------------------------------------------------------------

    def _run_all(self):
        if not self.audio_path:
            return

        def after_bpm():
            if not self.store.has_bpm:
                self.root.after(200, after_bpm)
            else:
                self._run_key()
                self.root.after(200, after_key)

        def after_key():
            if not self.store.has_key:
                self.root.after(200, after_key)
            else:
                self._run_events()
                self.root.after(200, after_events)

        def after_events():
            if not self.store.has_events:
                self.root.after(200, after_events)
            else:
                self._run_spectrum()

        self._run_bpm()
        self.root.after(300, after_bpm)

    # ------------------------------------------------------------------
    # Save detail charts + JSON
    # ------------------------------------------------------------------

    def _save_detail(self, which):
        """Save individual chart PNGs + JSON for the given analysis step."""
        s = self.store
        d = self.project_dir
        if which == "bpm":
            fig = self.builder.get(CHART_BPM_SPAN)
            if fig:
                fig.savefig(str(d / "bpm_analysis.png"), dpi=150,
                             bbox_inches="tight")
        elif which == "key":
            for name, fname in [(CHART_KEY_TIMELINE, "key_timeline.png"),
                                (CHART_KEY_CONFIDENCE, "key_confidence.png")]:
                fig = self.builder.get(name)
                if fig:
                    fig.savefig(str(d / fname), dpi=150,
                                 bbox_inches="tight")
            with open(d / "key_analysis.json", "w") as f:
                json.dump({"file": self.audio_path.name,
                           "duration": round(s.duration, 2),
                           "bpm": round(s.tempo, 2) if s.tempo else None,
                           "summary": s.key_summary,
                           "segments": s.key_segments}, f, indent=2)
        elif which == "events":
            if s.event_mode == "stems" and s.event_extra:
                se = s.event_extra.get("stem_energies", {})
                for stem in sorted(se.keys()):
                    fig = self.builder.get(f"{_STEM_PREFIX}{stem}")
                    if fig:
                        fig.savefig(str(d / f"stem_{stem}.png"), dpi=150,
                                     bbox_inches="tight")
            else:
                for name, fname in [
                    (CHART_PERC_ONSET, "perc_onset.png"),
                    (CHART_HARM_ONSET, "harm_onset.png"),
                    (CHART_RMS_ENERGY, "rms_energy.png"),
                    (CHART_CHANGE_SCORE, "change_score.png"),
                ]:
                    fig = self.builder.get(name)
                    if fig:
                        fig.savefig(str(d / fname), dpi=150,
                                     bbox_inches="tight")
            with open(d / "event_analysis.json", "w") as f:
                json.dump({"file": self.audio_path.name,
                           "duration": round(s.duration, 2),
                           "bpm": round(s.tempo, 2) if s.tempo else None,
                           "summary": s.event_summary,
                           "events": s.events}, f, indent=2)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if not self.audio_path:
            return
        if self.playing and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
            self.pause_position = self._cur_pos()
            self.btn_play.config(text="\u25B6")
            self._set_status(
                f"Paused at {format_time(self.pause_position)}")
        elif self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
            self.play_start_offset = self.pause_position
            self.play_start_time = time.time()
            self.btn_play.config(text="\u23F8")
            self._set_status("Playing...")
            self._tick()
        else:
            self._play_from(self.seek_var.get())

    def _play_from(self, pos):
        pos = max(0.0, min(pos, self.store.duration))
        try:
            pygame.mixer.music.play(start=pos)
            self.playing = True
            self.paused = False
            self.play_start_offset = pos
            self.play_start_time = time.time()
            self.btn_play.config(text="\u23F8")
            self._set_status("Playing...")
            self._tick()
        except Exception as e:
            self._set_status(f"Playback error: {e}")

    def _stop(self):
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        self.pause_position = 0.0
        self.btn_play.config(text="\u25B6")
        self._set_status("Stopped")
        self._hide_cursor()

    def _go_start(self):
        self._seek_to(0.0)

    def _go_end(self):
        self._seek_to(max(0, self.store.duration - 5.0))

    def _seek_to(self, pos):
        pos = max(0.0, min(pos, self.store.duration))
        self._seeking = True
        self.seek_var.set(pos)
        self._seeking = False
        self.time_label.config(
            text=f"{format_time(pos)} / {format_time(self.store.duration)}")
        if self.playing and not self.paused:
            self._play_from(pos)
        else:
            self.pause_position = pos
            self._draw_cursor(pos)

    def _on_seek(self, val):
        if self._seeking:
            return
        pos = float(val)
        self.time_label.config(
            text=f"{format_time(pos)} / {format_time(self.store.duration)}")
        if self.playing and not self.paused:
            self._play_from(pos)
        else:
            self.pause_position = pos
            self._draw_cursor(pos)

    def _on_volume_changed(self, val):
        v = float(val)
        pygame.mixer.music.set_volume(v)
        pct = int(round(v * 100))
        safe = self.store.safe_volume
        if v > safe + 0.005 and safe < 1.0:
            self.volume_pct.config(text=f"{pct}%", foreground="#cc2222")
        else:
            self.volume_pct.config(text=f"{pct}%", foreground="")

    def _apply_safe_volume(self):
        """Set volume slider to safe level and update the clip-zone marker."""
        safe = self.store.safe_volume
        self.volume_slider.set_safe(safe)
        # Auto-set to safe volume (user can still raise it)
        self.volume_var.set(safe)
        pygame.mixer.music.set_volume(safe)
        pct = int(round(safe * 100))
        self.volume_pct.config(text=f"{pct}%", foreground="")

    def _cur_pos(self):
        if not self.playing:
            return self.pause_position
        return self.play_start_offset + (time.time() - self.play_start_time)

    # ------------------------------------------------------------------
    # Snap-to utilities (for event editing)
    # ------------------------------------------------------------------

    def _snap_to_bar(self, t):
        """Return (snapped_time, bar_num) for the nearest measure start."""
        measures = self.store.measures
        if not measures:
            return (t, 0)
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t)
        # Compare candidate on the left and right
        best_idx = 0
        if idx == 0:
            best_idx = 0
        elif idx >= len(starts):
            best_idx = len(starts) - 1
        else:
            if abs(starts[idx] - t) < abs(starts[idx - 1] - t):
                best_idx = idx
            else:
                best_idx = idx - 1
        m = measures[best_idx]
        return (m["start"], m["measure_num"])

    def _snap_to_phrase(self, t):
        """Return (snapped_time, bar_num) for the nearest phrase boundary."""
        measures = self.store.measures
        if not measures:
            return (t, 0)
        pb = self.phrase_bars
        phrase_measures = [measures[i] for i in range(0, len(measures), pb)]
        starts = [m["start"] for m in phrase_measures]
        idx = bisect.bisect_right(starts, t)
        best_idx = 0
        if idx == 0:
            best_idx = 0
        elif idx >= len(starts):
            best_idx = len(starts) - 1
        else:
            if abs(starts[idx] - t) < abs(starts[idx - 1] - t):
                best_idx = idx
            else:
                best_idx = idx - 1
        m = phrase_measures[best_idx]
        return (m["start"], m["measure_num"])

    def _time_to_bar(self, t):
        """Return the bar number that contains time t."""
        measures = self.store.measures
        if not measures:
            return 0
        starts = [m["start"] for m in measures]
        idx = bisect.bisect_right(starts, t) - 1
        idx = max(0, min(idx, len(measures) - 1))
        return measures[idx]["measure_num"]

    def _bar_to_time(self, bar_num):
        """Return the start time of a given bar number."""
        measures = self.store.measures
        if not measures:
            return 0.0
        for m in measures:
            if m["measure_num"] == bar_num:
                return m["start"]
        # If bar_num exceeds last measure, return last measure start
        return measures[-1]["start"]

    def _snap_time(self, t):
        """Apply global snap setting. Returns (snapped_time, bar_num)."""
        mode = self.snap_var.get()
        if mode == "bar":
            return self._snap_to_bar(t)
        elif mode == "phrase":
            return self._snap_to_phrase(t)
        else:
            return (t, self._time_to_bar(t))

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def _add_event(self, event_dict):
        """Append a new event and refresh."""
        self.store.events.append(event_dict)
        self.store.events.sort(key=lambda e: e["time"])
        self._refresh_after_event_edit()

    def _edit_event(self, old_event, new_dict):
        """Replace old_event with new_dict and refresh."""
        try:
            idx = self.store.events.index(old_event)
        except ValueError:
            # Fallback: match by time + type
            for i, e in enumerate(self.store.events):
                if (abs(e["time"] - old_event["time"]) < 0.01
                        and e["type"] == old_event["type"]):
                    idx = i
                    break
            else:
                return
        self.store.events[idx] = new_dict
        self.store.events.sort(key=lambda e: e["time"])
        self._refresh_after_event_edit()

    def _delete_event(self, event):
        """Remove an event and refresh."""
        try:
            self.store.events.remove(event)
        except ValueError:
            # Fallback: match by time + type
            self.store.events = [
                e for e in self.store.events
                if not (abs(e["time"] - event["time"]) < 0.01
                        and e["type"] == event["type"])
            ]
        self._refresh_after_event_edit()

    def _refresh_after_event_edit(self):
        """Rebuild charts and save after any event add/edit/delete."""
        s = self.store
        s.event_summary = event_compute_summary(
            s.events, s.event_phrases, s.duration, mode=s.event_mode)
        s.version += 1

        # Invalidate charts that show events
        ev_charts = [CHART_WAVEFORM] + _GROUP_EVENTS_FEATURES
        stem_charts = [n for n in self._chart_widgets
                       if n.startswith(_STEM_PREFIX)]
        all_affected = ev_charts + stem_charts
        self.builder.invalidate(*all_affected)
        self._invalidate_chart_widgets(*all_affected)

        # Re-show the current active chart (or waveform)
        if self.active_chart:
            self._show_chart(self.active_chart)
        else:
            self._show_waveform()

        # Persist
        self._save_detail("events")
        try:
            self.store.save(self.project_dir)
        except Exception as e:
            print(f"Cache save error: {e}", file=sys.stderr)

        self._update_info()
        self._set_status(f"Events: {len(s.events)}")

    # ------------------------------------------------------------------
    # Marker management
    # ------------------------------------------------------------------

    def _place_marker(self):
        """Place a marker at the current playhead position (snapped)."""
        if not self.store.has_bpm:
            return
        pos = self._cur_pos()
        snapped, bar = self._snap_time(pos)
        snapped = max(0.0, min(snapped, self.store.duration))

        if len(self._markers) < 2:
            self._markers.append(snapped)
        else:
            # Replace the nearest marker
            dists = [abs(m - snapped) for m in self._markers]
            idx = dists.index(min(dists))
            self._markers[idx] = snapped
        self._markers.sort()
        self._refresh_markers()

        # Status feedback
        if len(self._markers) == 1:
            b = self._time_to_bar(self._markers[0])
            self._set_status(
                f"Marker 1: bar {b} ({format_time(self._markers[0])})")
        elif len(self._markers) == 2:
            b1 = self._time_to_bar(self._markers[0])
            b2 = self._time_to_bar(self._markers[1])
            dur = self._markers[1] - self._markers[0]
            self._set_status(
                f"Selection: bar {b1}\u2013{b2} ({format_time(dur)})")

    def _clear_markers(self):
        """Remove all markers."""
        self._markers = []
        self._refresh_markers()
        self._set_status("Markers cleared")

    def _refresh_markers(self):
        """Sync markers to store and rebuild all visible charts."""
        self.store.markers = list(self._markers)
        self.store.version += 1
        # Invalidate all charts so markers re-draw
        self.builder.invalidate()
        self._invalidate_chart_widgets()
        # Re-show active chart (or waveform as fallback)
        active = self.active_chart or CHART_WAVEFORM
        self._show_chart(active)
        self._update_marker_buttons()

    def _update_marker_buttons(self):
        """Enable/disable marker buttons based on current state."""
        has_bpm = self.store.has_bpm
        has_markers = len(self._markers) > 0
        has_pair = len(self._markers) == 2

        self.btn_mark.config(state="normal" if has_bpm else "disabled")
        self.btn_clear_marks.config(
            state="normal" if has_markers else "disabled")
        self.btn_export_clip.config(
            state="normal" if has_pair else "disabled")

    # ------------------------------------------------------------------
    # Event right-click interaction
    # ------------------------------------------------------------------

    def _on_chart_right_click(self, mpl_event, canvas):
        """Handle right-click on chart: interact with marker, event, or add."""
        if not self.store.has_bpm:
            return
        click_t = mpl_event.xdata
        # Proximity threshold: 1.5% of duration, clamped 2-10s
        threshold = max(2.0, min(10.0, self.store.duration * 0.015))

        # Find nearest event
        nearest_event = None
        event_dist = float("inf")
        if self.store.has_events:
            for e in self.store.events:
                d = abs(e["time"] - click_t)
                if d < event_dist:
                    nearest_event = e
                    event_dist = d

        # Find nearest marker
        nearest_marker_t = None
        marker_dist = float("inf")
        for mt in self._markers:
            d = abs(mt - click_t)
            if d < marker_dist:
                nearest_marker_t = mt
                marker_dist = d

        # Convert matplotlib pixel coords to screen coords for popup
        cw = canvas.get_tk_widget()
        screen_x = cw.winfo_rootx() + int(mpl_event.x)
        screen_y = (cw.winfo_rooty() + cw.winfo_height()
                    - int(mpl_event.y))

        # Pick whichever is closer (event wins ties)
        if (nearest_event and event_dist <= threshold
                and event_dist <= marker_dist):
            self._show_event_context_menu(nearest_event, screen_x, screen_y)
        elif nearest_marker_t is not None and marker_dist <= threshold:
            self._show_marker_context_menu(
                nearest_marker_t, screen_x, screen_y)
        elif self.store.has_events:
            self._open_add_dialog(click_t)

    def _show_event_context_menu(self, event, screen_x, screen_y):
        """Show right-click context menu for an existing event."""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f"Edit \"{event['type']}\" (bar {event['bar']})...",
                         command=lambda: self._open_edit_dialog(event))
        menu.add_separator()
        menu.add_command(label="Delete Event",
                         command=lambda: self._confirm_delete_event(event))
        menu.tk_popup(screen_x, screen_y)

    def _show_marker_context_menu(self, marker_time, screen_x, screen_y):
        """Show right-click context menu for a marker."""
        bar = self._time_to_bar(marker_time)
        menu = tk.Menu(self.root, tearoff=0)
        if self.store.has_events:
            menu.add_command(
                label=f"Convert to Event (bar {bar})...",
                command=lambda: self._convert_marker_to_event(marker_time))
        menu.add_command(
            label="Remove Marker",
            command=lambda: self._remove_marker(marker_time))
        menu.tk_popup(screen_x, screen_y)

    def _convert_marker_to_event(self, marker_time):
        """Open EventDialog pre-filled with marker time; on OK add event and
        remove the marker."""
        if not self.store.has_events:
            return
        dlg = EventDialog(self.root, self.store, self.phrase_bars,
                          self.snap_var,
                          title="Add Event from Marker",
                          initial_time=marker_time)
        self.root.wait_window(dlg)
        if dlg.result:
            self._add_event(dlg.result)
            self._remove_marker(marker_time)

    def _remove_marker(self, marker_time):
        """Remove a specific marker by its time value."""
        self._markers = [m for m in self._markers if m != marker_time]
        self._refresh_markers()
        if self._markers:
            b = self._time_to_bar(self._markers[0])
            self._set_status(
                f"Marker removed. Remaining: bar {b} "
                f"({format_time(self._markers[0])})")
        else:
            self._set_status("Marker removed")

    # ------------------------------------------------------------------
    # Audio clip export
    # ------------------------------------------------------------------

    def _export_clip(self):
        """Open ExportDialog, then export audio from the *original* file."""
        if len(self._markers) != 2:
            return
        if not self.audio_path or not self.audio_path.exists():
            messagebox.showinfo("Export", "No audio file loaded.")
            return

        # Show export dialog for marker adjustment + format choice
        dlg = ExportDialog(
            self.root, self.store, self.phrase_bars, self.snap_var,
            self.audio_path, self._markers[0], self._markers[1])
        self.root.wait_window(dlg)
        if not dlg.result:
            return

        t_start = dlg.result["start"]
        t_end = dlg.result["end"]
        ext = dlg.result["ext"]
        bar_start = dlg.result["bar_start"]
        bar_end = dlg.result["bar_end"]

        # Build suggested filename
        stem = self.audio_path.stem
        suggested = f"{stem}_bar{bar_start}-bar{bar_end}{ext}"

        # Map extension to filetypes for the save dialog
        ft_map = {
            ".wav": [("WAV", "*.wav")],
            ".flac": [("FLAC", "*.flac")],
            ".ogg": [("OGG Vorbis", "*.ogg")],
            ".mp3": [("MP3", "*.mp3")],
        }
        filetypes = ft_map.get(ext, [("WAV", "*.wav")])

        out_path = filedialog.asksaveasfilename(
            title="Save Audio Clip",
            initialdir=str(self.audio_path.parent),
            initialfile=suggested,
            filetypes=filetypes,
            defaultextension=ext,
        )
        if not out_path:
            return

        out_path = Path(out_path)
        out_ext = out_path.suffix.lower()

        try:
            import soundfile as sf

            # Read the clip from the *original* file to preserve
            # stereo, sample rate, and bit depth
            info = sf.info(str(self.audio_path))
            orig_sr = info.samplerate

            i_start = int(t_start * orig_sr)
            i_end = int(t_end * orig_sr)
            n_frames = i_end - i_start

            clip, _ = sf.read(str(self.audio_path),
                              start=i_start, frames=n_frames,
                              always_2d=True)

            if out_ext == ".mp3":
                self._export_mp3(clip, orig_sr, out_path, info.subtype)
            else:
                fmt_map = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}
                fmt = fmt_map.get(out_ext, "WAV")
                # Preserve original subtype (e.g. PCM_16, PCM_24)
                subtype = info.subtype if out_ext != ".ogg" else "VORBIS"
                sf.write(str(out_path), clip, orig_sr,
                         format=fmt, subtype=subtype)

            dur = t_end - t_start
            ch = "stereo" if clip.shape[1] == 2 else "mono"
            self._set_status(
                f"Exported: {out_path.name} "
                f"(bar {bar_start}\u2013{bar_end}, {format_time(dur)}, "
                f"{orig_sr} Hz, {ch})")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_mp3(self, clip, sr, output_path, subtype="PCM_16"):
        """Export clip as MP3 via ffmpeg (temp WAV → ffmpeg → MP3)."""
        import subprocess
        import tempfile
        import soundfile as sf

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        try:
            sf.write(str(tmp_wav), clip, sr,
                     format="WAV", subtype=subtype)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_wav),
                 "-b:a", "320k", str(output_path)],
                capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg error:\n{result.stderr[:500]}")
        except FileNotFoundError:
            # ffmpeg not found — fallback to WAV
            fallback = output_path.with_suffix(".wav")
            sf.write(str(fallback), clip, sr,
                     format="WAV", subtype=subtype)
            messagebox.showwarning(
                "MP3 Export",
                f"ffmpeg not found. Saved as WAV instead:\n{fallback.name}")
            self._set_status(f"Exported: {fallback.name} (WAV fallback)")
        finally:
            if tmp_wav.exists():
                tmp_wav.unlink()

    def _open_add_dialog(self, initial_time):
        """Open EventDialog in add mode."""
        dlg = EventDialog(self.root, self.store, self.phrase_bars,
                          self.snap_var,
                          title="Add Event", initial_time=initial_time)
        self.root.wait_window(dlg)
        if dlg.result:
            self._add_event(dlg.result)

    def _open_edit_dialog(self, event):
        """Open EventDialog in edit mode for an existing event."""
        dlg = EventDialog(self.root, self.store, self.phrase_bars,
                          self.snap_var,
                          title="Edit Event", event=event)
        self.root.wait_window(dlg)
        if dlg.result:
            self._edit_event(event, dlg.result)

    def _confirm_delete_event(self, event):
        """Ask for confirmation, then delete."""
        if messagebox.askyesno(
                "Delete Event",
                f'Delete "{event["type"]}" at bar {event["bar"]}?'):
            self._delete_event(event)

    def _add_event_at_playhead(self):
        """Open add-event dialog at the current playhead position."""
        if not self.store.has_events:
            return
        pos = self._cur_pos()
        self._open_add_dialog(pos)

    # ------------------------------------------------------------------
    # View controls: zoom, center, auto-follow
    # ------------------------------------------------------------------

    def _bars_to_seconds(self, n_bars):
        """Convert a bar count to approximate seconds using current tempo."""
        if self.store.tempo and self.store.tempo > 0:
            secs_per_beat = 60.0 / self.store.tempo
            return n_bars * self.beats_per_bar * secs_per_beat
        return n_bars * 2.0  # fallback ~120 BPM

    def _apply_zoom(self, center=None):
        """Set the x-axis limits on the active chart based on zoom level.

        If center is None, uses current playhead position.
        """
        w = self._active_widgets()
        if not w:
            return
        canvas = w["canvas"]
        axes = canvas.figure.get_axes()
        if not axes:
            return
        ax = axes[0]

        zoom = self.zoom_var.get()
        n_bars = self._zoom_bars_map.get(zoom)

        was_programmatic = self._programmatic_zoom
        self._programmatic_zoom = True
        try:
            if n_bars is None:
                # "Full" or "Custom" — restore full view
                if zoom == "Full":
                    ax.set_xlim(0, self.store.duration)
                    canvas.draw_idle()
                return

            half_window = self._bars_to_seconds(n_bars) / 2.0
            if center is None:
                center = self._cur_pos()
            center = max(0.0, min(center, self.store.duration))

            lo = center - half_window
            hi = center + half_window
            # Clamp to track boundaries
            if lo < 0:
                lo, hi = 0, min(half_window * 2, self.store.duration)
            if hi > self.store.duration:
                hi = self.store.duration
                lo = max(0, hi - half_window * 2)

            ax.set_xlim(lo, hi)
            canvas.draw_idle()
        finally:
            self._programmatic_zoom = was_programmatic

    def _center_on_playhead(self):
        """Center the active chart view on the current playhead position."""
        w = self._active_widgets()
        if not w:
            return
        canvas = w["canvas"]
        axes = canvas.figure.get_axes()
        if not axes:
            return
        ax = axes[0]
        pos = self._cur_pos()

        zoom = self.zoom_var.get()
        n_bars = self._zoom_bars_map.get(zoom)

        if n_bars is not None:
            # Bar-based zoom: re-center at current pos
            self._apply_zoom(center=pos)
        else:
            # Full or Custom: use current view width, re-center
            was_programmatic = self._programmatic_zoom
            self._programmatic_zoom = True
            try:
                lo, hi = ax.get_xlim()
                half_w = (hi - lo) / 2.0
                new_lo = pos - half_w
                new_hi = pos + half_w
                if new_lo < 0:
                    new_lo, new_hi = 0, min(half_w * 2, self.store.duration)
                if new_hi > self.store.duration:
                    new_hi = self.store.duration
                    new_lo = max(0, new_hi - half_w * 2)
                ax.set_xlim(new_lo, new_hi)
                canvas.draw_idle()
            finally:
                self._programmatic_zoom = was_programmatic

    def _toggle_follow(self):
        """Toggle auto-follow mode."""
        self.auto_follow = not self.auto_follow
        self.follow_var.set(self.auto_follow)

    def _on_zoom_changed(self, event=None):
        """Handle zoom combobox selection."""
        self._apply_zoom()

    def _auto_follow_tick(self, pos):
        """If auto-follow is on and we're zoomed in, scroll to keep playhead visible."""
        if not self.auto_follow:
            return
        if self._programmatic_zoom:
            return  # another zoom operation in progress — avoid re-entry
        w = self._active_widgets()
        if not w:
            return
        canvas = w["canvas"]
        axes = canvas.figure.get_axes()
        if not axes:
            return
        ax = axes[0]
        lo, hi = ax.get_xlim()
        window = hi - lo

        # If showing full track, nothing to scroll
        if window >= self.store.duration * 0.95:
            return

        # Scroll when playhead reaches 75% of the visible window
        scroll_at = lo + window * 0.75
        if pos >= scroll_at:
            self._programmatic_zoom = True
            try:
                new_lo = pos - window * 0.25
                new_hi = new_lo + window
                if new_hi > self.store.duration:
                    new_hi = self.store.duration
                    new_lo = max(0, new_hi - window)
                ax.set_xlim(new_lo, new_hi)
                # Synchronous draw + capture — cursor blit on the
                # same tick needs a fresh background immediately
                canvas.draw()
                w["blit_bg"] = canvas.copy_from_bbox(canvas.figure.bbox)
            finally:
                self._programmatic_zoom = False

    # ------------------------------------------------------------------
    # Cursor (blitting)
    # ------------------------------------------------------------------

    def _tick(self):
        if not self.playing or self.paused:
            return
        if self._analyzing:
            # Back off during analysis — only update time label
            pos = self._cur_pos()
            self._seeking = True
            self.seek_var.set(pos)
            self._seeking = False
            self.time_label.config(
                text=f"{format_time(pos)} / "
                     f"{format_time(self.store.duration)}")
            self.root.after(200, self._tick)
            return

        pos = self._cur_pos()
        if pos >= self.store.duration or not pygame.mixer.music.get_busy():
            self._stop()
            return

        self._seeking = True
        self.seek_var.set(pos)
        self._seeking = False
        self.time_label.config(
            text=f"{format_time(pos)} / {format_time(self.store.duration)}")
        self._auto_follow_tick(pos)
        self._draw_cursor(pos)
        self.root.after(50, self._tick)

    def _draw_cursor(self, pos):
        w = self._active_widgets()
        if not w or w["cursor_line"] is None:
            return
        cursor = w["cursor_line"]
        canvas = w["canvas"]
        cursor.set_xdata([pos, pos])
        cursor.set_visible(True)
        if w["blit_bg"] is not None:
            canvas.restore_region(w["blit_bg"])
            axes = canvas.figure.get_axes()
            if axes:
                axes[0].draw_artist(cursor)
            canvas.blit(canvas.figure.bbox)
        else:
            canvas.draw_idle()

    def _hide_cursor(self):
        w = self._active_widgets()
        if not w or w["cursor_line"] is None:
            return
        cursor = w["cursor_line"]
        canvas = w["canvas"]
        cursor.set_visible(False)
        if w["blit_bg"] is not None:
            canvas.restore_region(w["blit_bg"])
            canvas.blit(canvas.figure.bbox)
        else:
            canvas.draw_idle()


# =========================================================================
# Entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DJ Audio Analysis UI")
    parser.add_argument("--audio", "-a", default=None,
                        help="Audio file to open on launch")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass

    app = AudioAnalysisApp(root, initial_audio=args.audio)

    # Suppress harmless "invalid command name <id>_tick" TclErrors that
    # occur when matplotlib chart widgets are destroyed while pending
    # after-idle callbacks are still queued.
    _orig_report = root.report_callback_exception

    def _quiet_report(exc_type, exc_value, exc_tb):
        if (exc_type is tk.TclError
                and "invalid command name" in str(exc_value)):
            return  # suppress orphaned-callback noise
        _orig_report(exc_type, exc_value, exc_tb)

    root.report_callback_exception = _quiet_report

    # Also suppress at the Tcl level — some "after" errors bypass
    # Python's report_callback_exception and are reported by Tcl's
    # bgerror directly to stderr.
    try:
        root.tk.call('proc', 'bgerror', 'msg', '')
    except tk.TclError:
        pass

    def on_close():
        # Stop playback first to halt the _tick chain
        app._stop()
        app.playing = False
        app.paused = False
        # Cancel all pending after callbacks on chart canvases
        for w in app._chart_widgets.values():
            try:
                tk_widget = w["canvas"].get_tk_widget()
                for attr in ("_idle_draw_id", "_event_loop_id"):
                    after_id = getattr(w["canvas"], attr, None)
                    if after_id:
                        tk_widget.after_cancel(after_id)
                        setattr(w["canvas"], attr, None)
            except Exception:
                pass
        # Flush any remaining idle events before teardown
        try:
            root.update_idletasks()
        except tk.TclError:
            pass
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        app.builder.close_all()
        plt.close("all")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
