#!/usr/bin/env bash
# Start the engine API (:8000) and the web UI together.
set -e
root="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$root/engine/.venv" ]; then
  echo "Setting up engine venv…"
  (cd "$root/engine" && python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt)
fi
if [ ! -d "$root/web/node_modules" ]; then
  echo "Installing web deps…"
  (cd "$root/web" && npm install)
fi

echo "Starting API on :8000 and web UI on :3000…"
(cd "$root/engine" && ./.venv/bin/python -m uvicorn server:app --port 8000) &
API=$!
(cd "$root/web" && npm run dev) &
WEB=$!
trap "kill $API $WEB 2>/dev/null" EXIT
wait
