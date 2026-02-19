#!/usr/bin/env python3
"""
Analyze onset drops at known event points to help tune detection parameters.
"""
import json
import numpy as np

# Known events from manual analysis (bar numbers)
KNOWN_EVENTS = {
    # From bars 0-32 analysis
    "bar_1": {"type": "intro_peek", "description": "intro peek"},
    "bar_9": {"type": "fill_new_element", "description": "fill + hi-hat in"},
    "bar_17": {"type": "fill_new_element", "description": "fill + bass in"},
    "bar_25": {"type": "fill_new_element", "description": "fill + kick in"},

    # From bars 64-96 analysis
    "bar_72": {"type": "fill", "description": "fill (2:13)"},
    "bar_80": {"type": "fill", "description": "fill (2:28)"},
    "bar_88": {"type": "fill", "description": "fill (2:43)"},

    # From bars 96-128 analysis
    "bar_104": {"type": "vocals_in", "description": "vocals in (3:12)"},
    "bar_112": {"type": "fill", "description": "fill"},
    "bar_121": {"type": "drums_out", "description": "drums out / breakdown"},
}

def load_features(json_file):
    with open(json_file) as f:
        return json.load(f)

def analyze_onset_at_bar(features, target_bar, context_bars=2):
    """
    Analyze onset values around a specific bar.
    Returns stats about the drop.
    """
    start_bar = features["start_bar"]
    beats_per_bar = 4
    onset = np.array(features["onset_strength"])

    # Calculate beat index for target bar
    bar_offset = target_bar - start_bar
    if bar_offset < 0 or bar_offset >= features["num_bars"]:
        return None

    target_beat = bar_offset * beats_per_bar

    # Get context window
    context_beats = context_bars * beats_per_bar
    start_idx = max(0, target_beat - context_beats)
    end_idx = min(len(onset), target_beat + context_beats + beats_per_bar)

    if target_beat >= len(onset):
        return None

    # Calculate stats
    # "Before" = average of 1-2 bars before the event
    before_start = max(0, target_beat - context_beats)
    before_end = target_beat
    before_vals = onset[before_start:before_end] if before_end > before_start else []

    # "At event" = the bar where the event occurs (and maybe next bar)
    event_start = target_beat
    event_end = min(len(onset), target_beat + beats_per_bar * 2)  # 2 bars
    event_vals = onset[event_start:event_end] if event_end > event_start else []

    # "After" = 1-2 bars after the event
    after_start = event_end
    after_end = min(len(onset), after_start + context_beats)
    after_vals = onset[after_start:after_end] if after_end > after_start else []

    if len(before_vals) == 0 or len(event_vals) == 0:
        return None

    before_mean = np.mean(before_vals)
    before_max = np.max(before_vals)
    event_min = np.min(event_vals)
    event_mean = np.mean(event_vals)
    after_mean = np.mean(after_vals) if len(after_vals) > 0 else event_mean

    # Calculate drop metrics
    abs_drop = before_mean - event_min
    pct_drop = abs_drop / before_mean if before_mean > 0 else 0

    # Does it recover? Compare event min to after mean
    recovery = after_mean - event_min
    recovery_pct = recovery / abs_drop if abs_drop > 0 else 0

    return {
        "target_bar": target_bar,
        "before_mean": round(before_mean, 3),
        "before_max": round(before_max, 3),
        "event_min": round(event_min, 3),
        "event_mean": round(event_mean, 3),
        "after_mean": round(after_mean, 3),
        "abs_drop": round(abs_drop, 3),
        "pct_drop": round(pct_drop * 100, 1),
        "recovery": round(recovery, 3),
        "recovery_pct": round(recovery_pct * 100, 1),
        "raw_values": {
            "before": [round(v, 3) for v in before_vals[-8:]],  # Last 2 bars before
            "event": [round(v, 3) for v in event_vals[:8]],     # First 2 bars of event
            "after": [round(v, 3) for v in after_vals[:8]],     # First 2 bars after
        }
    }

def main():
    # Load all feature files
    files = {
        "0-32": "viz_bars0-32_v2.json",
        "64-96": "viz_bars64-96_v2.json",
        "96-128": "viz_bars96-128_v2.json",
    }

    features_by_range = {}
    for name, filename in files.items():
        try:
            features_by_range[name] = load_features(filename)
            print(f"Loaded {filename}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found")

    print("\n" + "=" * 80)
    print("ONSET DROP ANALYSIS AT KNOWN EVENT POINTS")
    print("=" * 80)

    results = []

    for bar_key, event_info in KNOWN_EVENTS.items():
        bar_num = int(bar_key.split("_")[1])

        # Find which file contains this bar
        analysis = None
        for name, features in features_by_range.items():
            start = features["start_bar"]
            end = start + features["num_bars"]
            if start <= bar_num < end:
                analysis = analyze_onset_at_bar(features, bar_num)
                break

        if analysis:
            analysis["event_type"] = event_info["type"]
            analysis["description"] = event_info["description"]
            results.append(analysis)

            print(f"\n--- Bar {bar_num}: {event_info['description']} ({event_info['type']}) ---")
            print(f"  Before (mean): {analysis['before_mean']:.3f}")
            print(f"  Event (min):   {analysis['event_min']:.3f}")
            print(f"  After (mean):  {analysis['after_mean']:.3f}")
            print(f"  Absolute drop: {analysis['abs_drop']:.3f}")
            print(f"  Percent drop:  {analysis['pct_drop']:.1f}%")
            print(f"  Recovery:      {analysis['recovery']:.3f} ({analysis['recovery_pct']:.1f}%)")
            print(f"  Raw before:    {analysis['raw_values']['before']}")
            print(f"  Raw event:     {analysis['raw_values']['event']}")
            print(f"  Raw after:     {analysis['raw_values']['after']}")
        else:
            print(f"\n--- Bar {bar_num}: {event_info['description']} --- NOT FOUND IN DATA")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results:
        abs_drops = [r["abs_drop"] for r in results]
        pct_drops = [r["pct_drop"] for r in results]

        print(f"\nAbsolute drops:")
        print(f"  Min:  {min(abs_drops):.3f}")
        print(f"  Max:  {max(abs_drops):.3f}")
        print(f"  Mean: {np.mean(abs_drops):.3f}")
        print(f"  Std:  {np.std(abs_drops):.3f}")

        print(f"\nPercent drops:")
        print(f"  Min:  {min(pct_drops):.1f}%")
        print(f"  Max:  {max(pct_drops):.1f}%")
        print(f"  Mean: {np.mean(pct_drops):.1f}%")
        print(f"  Std:  {np.std(pct_drops):.1f}%")

        # Group by event type
        print("\n" + "-" * 40)
        print("BY EVENT TYPE:")
        print("-" * 40)

        by_type = {}
        for r in results:
            t = r["event_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        for event_type, events in by_type.items():
            print(f"\n{event_type}:")
            drops = [e["pct_drop"] for e in events]
            recoveries = [e["recovery_pct"] for e in events]
            print(f"  Count: {len(events)}")
            print(f"  Avg drop: {np.mean(drops):.1f}%")
            print(f"  Avg recovery: {np.mean(recoveries):.1f}%")

if __name__ == "__main__":
    main()
