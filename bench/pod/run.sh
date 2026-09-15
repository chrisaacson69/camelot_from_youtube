#!/usr/bin/env bash
# run.sh <tracks_dir>  - run allin1 over the uploaded tracks and tar the results for download
set -euo pipefail
TRACKS="${1:-/workspace/tracks}"
cd "$(dirname "$0")"
python run.py "$TRACKS" /workspace/results
tar -czf /workspace/allin1-results.tgz -C /workspace results
echo "== download: /workspace/allin1-results.tgz"
