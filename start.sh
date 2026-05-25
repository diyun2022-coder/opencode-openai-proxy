#!/usr/bin/env bash
set -e

# opencode runs under Bun, whose bundled CA bundle rejects certs newer than its
# release. Point Node at the system CA store so upstream TLS handshakes succeed.
export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-/etc/ssl/certs/ca-certificates.crt}"
export NODE_TLS_REJECT_UNAUTHORIZED="${NODE_TLS_REJECT_UNAUTHORIZED:-0}"

# Suppress MaxListenersExceededWarning from opencode's Effect-TS runtime.
# Each proxy SSE connection registers a listener on opencode's internal event bus;
# raising the limit silences the warning without affecting correctness.
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=512} --max-listeners=100"

# Start opencode server in the background if not already running
if ! curl -sf "${OPENCODE_BASE_URL:-http://localhost:4096}/global/health" >/dev/null 2>&1; then
  echo "[start.sh] starting opencode serve..."
  opencode serve --port 4096 --hostname 127.0.0.1 &
  OPENCODE_PID=$!
  trap "kill $OPENCODE_PID 2>/dev/null || true" EXIT

  # Wait for opencode to become healthy
  for i in {1..30}; do
    if curl -sf http://localhost:4096/global/health >/dev/null; then
      break
    fi
    sleep 0.5
  done
fi

exec "${UVICORN:-.venv/bin/uvicorn}" main:app --host 0.0.0.0 --port "${PROXY_PORT:-8000}"
