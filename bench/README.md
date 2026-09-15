# bench — beat-grid benchmark against Rekordbox

Scores beat/downbeat trackers against the grid Rekordbox holds for each track (`PQTZ` tag in
`ANLZ0000.DAT`, read via `pyrekordbox`). Results and env-specific outputs live in `results/`
(gitignored); the scripts and `tracks.json` are the durable part.

| Script | Env | Does |
|---|---|---|
| `beat_bench.py reference` | env1 | dump Rekordbox grids → `results/rekordbox/`, write `tracks.json` |
| `beat_bench.py camelot` | env1 | dump camelot's cached grid (`analysis_cache.npz` beats + measure starts) |
| `run_tracker.py librosa\|beat_this` | env1 / env_mir | run a tracker, write the same JSON shape |
| `rigidify.py <tracker>…` | any | robust line fit → constant-tempo grid from any tracker's beats |
| `rigid_grid.py [--comb]` | env1 | Rekordbox's method in camelot's stack: one tempo, phase from onsets |
| `beat_bench.py score` | any | F-measure (70 ms) for beats and downbeats, median offset, per tracker |

**Caveat that governs everything (2026-09-15):** of the 14 tracks, only the three added to
Rekordbox on 2026-03-07 (Flip of the Coin, Waves, Quicksilver) carry a *native* Rekordbox
analysis. The other 11 were imported from this repo's own `rekordbox_collection.xml` on
2026-02-20 and never re-analysed — their "Rekordbox grid" is camelot's export of itself
(tempogram bin-centre BPMs: 120.185 / 123.047 / 126.048 / 129.199). Re-analyse those in
Rekordbox before treating them as references. Findings are written up in the vault:
`research/dj-blend-tool-test-plan.md` → Phase 3 results.

`env_mir` = Python 3.14 + torch 2.11 cu128 + `beat_this` (git). `allin1` is not runnable here:
NATTEN ships no Windows wheels and there is no CUDA toolkit on this machine.
