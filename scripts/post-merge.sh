#!/bin/bash
# Post-merge setup for Ophelian (Python + uv).
# Runs after every task merge to keep the dev environment in sync.

set -euo pipefail

echo "[post-merge] syncing Python dependencies with uv (locked)..."
uv sync --frozen --extra dev

echo "[post-merge] done."
