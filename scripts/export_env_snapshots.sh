#!/usr/bin/env bash
# Export conda env specs + pip freeze for merging into a single JanusMesh env.
# Usage:
#   bash scripts/export_env_snapshots.sh [TRELLIS_ENV_NAME] [SYNCTWEEDIES_ENV_NAME]
# Defaults: trellis synctweedies
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/environment/exports"
TRELLIS_ENV="${1:-trellis}"
ST_ENV="${2:-synctweedies}"

mkdir -p "${OUT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi

export_ts() {
  local name="$1"
  local slug
  slug="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
  echo "[INFO] Exporting conda env: ${name}"
  if ! conda env export -n "${name}" --from-history >/dev/null 2>&1; then
    echo "[WARN] Env '${name}' not found or not exportable. Skipping." >&2
    return 0
  fi
  conda env export -n "${name}" --from-history > "${OUT}/${slug}_from_history.yml"
  conda env export -n "${name}" --no-builds > "${OUT}/${slug}_full.yml"
  conda run -n "${name}" pip list --format=freeze > "${OUT}/${slug}_pip_freeze.txt" || true
  echo "       -> ${OUT}/${slug}_*.yml , ${slug}_pip_freeze.txt"
}

export_ts "${TRELLIS_ENV}"
export_ts "${ST_ENV}"

echo "[INFO] Done. Next: read docs/ENVIRONMENT.md and merge into environment.yml (name: janusmesh)."
