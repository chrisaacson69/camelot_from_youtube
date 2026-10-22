#!/usr/bin/env python3
"""
Behavioural oracle for repointing audio_ui.py onto dj-score.

Runs one scripted session through the pre-repoint UI (audio_ui.py at REFERENCE, read
from git) and through the working-copy UI, on temporary copies of a few tracks, and
compares everything each side persists or renders: analysis caches, detail JSON,
detail PNGs and chart figures (pixels), status and info text, snapping, event
add/edit/delete, and the Rekordbox export. Tracks are copied from the music library
(DJ_SCORE_MUSIC, default the sibling `../music`); the library and camelot's
rekordbox_collection.xml are never touched. Nothing is screen-captured; figures are
rendered by the app's own matplotlib code.

    env1\\Scripts\\python.exe tests\\ui_repoint_oracle.py [--reference REV] [--self-check]

--self-check runs the reference against itself, to show the harness is deterministic.
"""

import argparse
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from dj_score.adapters import library

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MUSIC = Path(os.environ.get("DJ_SCORE_MUSIC", ROOT.parent / "music"))

import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import tkinter as tk  # noqa: E402

REFERENCE = "150422a"  # last commit before the repoint

# (audio file, copy its stems too, run the full BPM analysis too)
TRACKS = [
    ("NL123D.wav", None, True),                                         # features mode
    ("Waves.mp3", True, False),                                         # stems mode
    ("Quivver - Brothers & Sisters (Original Mix).mp3", None, False),   # spectrum
    ("Kasablanca - Am I Dreaming？ (@Kasablanca).mp3", None, False),  # legacy cache: no loudness keys
]


def load_module(name, source_path):
    spec = importlib.util.spec_from_file_location(name, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_module(rev, workdir):
    """The UI at `rev`, importing that revision's modules too (they are archived since)."""
    import tarfile
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", rev],
                         capture_output=True, check=True).stdout
    tree = workdir / f"tree_{rev}"
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tree, filter="data")
    sys.path.insert(0, str(tree))
    path = workdir / "audio_ui_reference.py"
    shutil.copyfile(tree / "audio_ui.py", path)
    return load_module("audio_ui_reference", path)


def pixels(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    buf.seek(0)
    return plt.imread(buf)


def events_of(app):
    a = getattr(app.store, "analysis", None)
    if a is not None:
        return a.events.events if a.events else []
    return app.store.events or []


def collection(path):
    """Rekordbox XML as data; TrackIDs replaced by location (they only link playlists)."""
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    tracks = root.find("COLLECTION").findall("TRACK")
    return {Path(unquote(t.get("Location"))).name: (
        {k: v for k, v in t.attrib.items() if k not in ("TrackID", "Location")},
        [(c.tag, dict(c.attrib)) for c in t]) for t in tracks}


class Session:
    """One side's run: its module, Tk root, temp folder, and everything observed."""

    def __init__(self, mod, workdir):
        self.mod = mod
        self.dir = workdir
        self.obs = {}
        self.dialogs = []
        mod.messagebox = types.SimpleNamespace(
            showinfo=self._dialog, showwarning=self._dialog, showerror=self._dialog,
            askyesno=lambda *a, **k: True)
        self.xml = workdir / "rekordbox_collection.xml"
        if hasattr(mod, "REKORDBOX_XML"):
            mod.REKORDBOX_XML = self.xml
        else:
            import rekordbox_export
            rekordbox_export.DEFAULT_EXPORT_PATH = self.xml

    def _dialog(self, title, msg, *a, **k):
        self.dialogs.append((title, str(msg).replace(str(self.dir), "<TMP>")))

    def record(self, key, value):
        self.obs[key] = value

    def snapshot(self, app, key):
        """Everything persisted in the project folder, plus the UI's text."""
        d = app.project_dir
        files = {}
        for p in sorted(d.iterdir()):
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if p.name == "analysis_cache.json":
                    data.pop("beatgrid_source", None)
                files[p.name] = data
            elif p.suffix == ".npz":
                with np.load(p) as z:
                    files[p.name] = {k: z[k] for k in z.files}
            elif p.suffix == ".png":
                files[p.name] = plt.imread(p)
        self.record(key, {"files": files, "status": app.status_var.get(), "info": app.info_var.get()})

    def run(self):
        for audio, stems, run_bpm in TRACKS:
            src = MUSIC / audio
            if not src.exists():  # a skipped track would make the comparison vacuous
                raise FileNotFoundError(f"{src} not in the music library")
            # the old layout (project and stems beside the audio), which both UIs read
            work = self.dir / Path(audio).stem
            work.mkdir()
            shutil.copy2(src, work / audio)
            if library.project_dir(src).exists():
                shutil.copytree(library.project_dir(src), work / Path(audio).stem)
            if stems:
                shutil.copytree(library.stems_dir(src), work / library.stems_dir(src).name)
            self.track(work / audio, run_bpm)
        self.record("dialogs", self.dialogs)
        self.record("rekordbox", collection(self.xml))

    def track(self, path, run_bpm):
        t = path.stem
        root = tk.Tk()
        root.withdraw()
        app = self.mod.AudioAnalysisApp(root)
        app._run_in_thread = lambda fn, cb, msg: cb(fn())
        try:
            app._open_file(path)
            root.update()
            self.snapshot(app, f"{t}/opened")
            self.charts(app, f"{t}/opened")

            probe = [0.0, 1.0, 7.3, 33.3, 61.0, 1e6]
            self.record(f"{t}/snap", [(app._snap_to_bar(x), app._snap_to_phrase(x), app._time_to_bar(x))
                                      for x in probe] + [app._bar_to_time(b) for b in (0, 1, 5, 9999)])

            if events_of(app):
                new = {"bar": 9, "time": 20.0, "time_fmt": "0:20", "type": "drop", "score": 1.0,
                       "description": "test drop", "source": "manual", "cue": "hot",
                       "n_features": 0, "pass": 0}
                app._add_event(dict(new))
                self.snapshot(app, f"{t}/event_added")
                added = next(e for e in events_of(app) if _get(e, "description") == "test drop")
                app._edit_event(added, {**new, "type": "breakdown", "cue": "memory"})
                self.snapshot(app, f"{t}/event_edited")
                app._delete_event(events_of(app)[0])
                self.snapshot(app, f"{t}/event_deleted")

            app._export_rekordbox()

            if run_bpm:
                app._run_bpm()
                self.snapshot(app, f"{t}/bpm")
            app._run_key()
            self.snapshot(app, f"{t}/key")
            app._run_events()
            self.snapshot(app, f"{t}/events")
            app._run_spectrum()
            self.snapshot(app, f"{t}/spectrum")
            self.charts(app, f"{t}/final")
        finally:
            app._stop()
            self.mod.pygame.mixer.music.unload()  # release the audio file
            app.builder.close_all()
            plt.close("all")
            root.destroy()

    def charts(self, app, key):
        app.store.markers = [5.0, 12.5]
        app.store.version += 1
        out = {}
        for name in app.builder.available():
            fig = app.builder.get(name)
            out[name] = pixels(fig) if fig else None
        self.record(f"{key}/charts", out)


def _get(e, k):
    return e.get(k) if isinstance(e, dict) else getattr(e, k)


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
            out.append(f"{path}: arrays differ (shape {a.shape} vs {b.shape})")
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in b else 'reference'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {str(a)[:120]!r} vs {str(b)[:120]!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--only", help="run just the track whose file name contains this")
    args = ap.parse_args()
    if args.only:
        TRACKS[:] = [t for t in TRACKS if args.only in t[0]]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        ref = reference_module(args.reference, tmp)
        new = load_module("audio_ui_reference2", tmp / "audio_ui_reference.py") if args.self_check \
            else load_module("audio_ui_new", ROOT / "audio_ui.py")
        sides = {}
        for label, mod in (("reference", ref), ("new", new)):
            (tmp / label).mkdir()
            print(f"running {label} ...")
            s = Session(mod, tmp / label)
            s.run()
            sides[label] = s.obs

        problems = diff(sides["reference"], sides["new"])
        n = sum(len(v) if isinstance(v, dict) else 1 for v in sides["reference"].values())
        print(f"\ncompared {len(sides['reference'])} observations ({n} entries)")
        if problems:
            print(f"{len(problems)} differences:")
            for p in problems[:60]:
                print("  " + p)
            sys.exit(1)
        print("identical")


if __name__ == "__main__":
    main()
