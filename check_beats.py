#!/usr/bin/env python3
"""Quick script to analyze beat intervals"""
import json

with open("viz_bars0-32.json") as f:
    data = json.load(f)

beats = data["beat_times"]
print("Beat intervals (should be ~0.464s at 129 BPM):")
print("=" * 55)
for i in range(min(48, len(beats)-1)):
    interval = beats[i+1] - beats[i]
    bar = i // 4
    beat_in_bar = (i % 4) + 1
    expected = 0.4644  # 60/129.2
    drift = interval - expected
    marker = " ***" if abs(drift) > 0.02 else ""
    print(f"Bar {bar:2}.{beat_in_bar}: {interval:.4f}s (drift: {drift:+.4f}){marker}")
