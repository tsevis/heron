#!/usr/bin/env bash
# HERON — Optical Bench launcher.
#
#   ./heron.sh                 start on port 8077 and open the browser
#   ./heron.sh --port=8090     use another port
#   ./heron.sh --no-open       start without opening the browser
#   PORT=8090 ./heron.sh       same as --port
#
# Ctrl-C stops the server. If Heron is already running, this just opens it.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8077}"
OPEN=1
for a in "$@"; do
  case "$a" in
    --no-open) OPEN=0 ;;
    --port=*)  PORT="${a#*=}" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  esac
done

PY="$DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "✗ Python venv not found at $DIR/.venv"
  echo "  Set it up once with:"
  echo "    cd \"$DIR\" && python3 -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install -r requirements-ai.txt"
  exit 1
fi

# Already running? Just open it.
if curl -s --max-time 1 "http://127.0.0.1:$PORT/api/meta" 2>/dev/null | grep -q '"instruments"'; then
  echo "✓ Heron is already running → http://127.0.0.1:$PORT"
  [ "$OPEN" = 1 ] && open "http://127.0.0.1:$PORT"
  exit 0
fi

# Port taken by something that is not Heron?
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "✗ Port $PORT is in use by another application."
  echo "  Try:  $0 --port=8090"
  exit 1
fi

# First run on a fresh clone: Layer A's optional models will not be here. Heron
# renders fine without them (the classical path needs none), so this informs and
# offers — it never downloads on its own. HERON_SKIP_MODEL_CHECK=1 silences it.
if [ "${HERON_SKIP_MODEL_CHECK:-0}" != "1" ]; then
  if ! "$PY" "$DIR/scripts/fetch_models.py" --check >/dev/null 2>&1; then
    echo
    "$PY" "$DIR/scripts/fetch_models.py" --check 2>/dev/null || true
    echo
    printf "  Download them now? [y/N] "
    read -r REPLY </dev/tty || REPLY=""
    case "$REPLY" in
      [yY]*) "$PY" "$DIR/scripts/fetch_models.py" --yes ;;
      *)     echo "  Skipped — starting with the classical path. Run"
             echo "  '$PY scripts/fetch_models.py' any time to add them." ;;
    esac
    echo
  fi
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "  _   _ _____ ____   ___  _   _"
echo " | | | | ____|  _ \\ / _ \\| \\ | |   Optical Bench"
echo " | |_| |  _| | |_) | | | |  \\| |   http://127.0.0.1:$PORT"
echo " |  _  | |___|  _ <| |_| | |\\  |   artistic simulation — not measurement"
echo " |_| |_|_____|_| \\_\\\\___/|_| \\_|   Ctrl-C stops the server"
echo

# Open the browser as soon as the server answers (first start loads AI deps, ~15 s).
if [ "$OPEN" = 1 ]; then
  (
    for _ in $(seq 1 60); do
      sleep 1
      if curl -s --max-time 1 "http://127.0.0.1:$PORT/api/meta" >/dev/null 2>&1; then
        open "http://127.0.0.1:$PORT"
        exit 0
      fi
    done
  ) &
fi

exec "$PY" -m uvicorn --app-dir "$DIR" webapp.server:app --host 127.0.0.1 --port "$PORT"
