#!/usr/bin/env bash
# macOS double-click entry point. All the logic lives in start.sh; this only
# exists because Finder opens ".command" files in Terminal.
exec bash "$(cd "$(dirname "$0")" && pwd)/start.sh"
