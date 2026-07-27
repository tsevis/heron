#!/usr/bin/env bash
# Double-clickable macOS launcher — opens Terminal and runs Heron.
exec "$(cd "$(dirname "$0")" && pwd)/heron.sh"
