#!/usr/bin/env bash

# Ensures the React UI under web/client is built before embedding it in the Go binary.
# The script keeps a hash of the UI sources so we only rebuild when files actually change.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$(cd "$SCRIPT_DIR/../web/client" && pwd)"
DIST_DIR="$CLIENT_DIR/dist"
HASH_FILE="$CLIENT_DIR/.ui-build-hash"

if [[ "${SKIP_UI_BUILD:-}" == "1" ]]; then
  echo "[ui-build] SKIP_UI_BUILD=1, skipping web UI build step"
  exit 0
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[ui-build] npm is required to build the web UI" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ui-build] python3 is required to calculate the UI hash" >&2
  exit 1
fi

current_hash="$(
  python3 - "$CLIENT_DIR" <<'PY'
import hashlib
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
files_to_hash = [
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "tailwind.config.js",
    "postcss.config.js",
    "vite.config.ts",
    "eslint.config.js",
    "components.json",
    "index.html",
]
dirs_to_hash = ["src", "public"]

entries = []
for rel in files_to_hash:
    path = base / rel
    if path.exists():
        entries.append((path.relative_to(base).as_posix(), path))

for rel in dirs_to_hash:
    path = base / rel
    if path.exists():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                entries.append((child.relative_to(base).as_posix(), child))

entries.sort(key=lambda item: item[0])

digest = hashlib.sha256()
for rel, path in entries:
    digest.update(rel.encode())
    with path.open("rb") as fh:
        digest.update(fh.read())

print(digest.hexdigest())
PY
)"

previous_hash=""
if [[ -f "$HASH_FILE" ]]; then
  previous_hash="$(<"$HASH_FILE")"
fi

if [[ -n "$current_hash" && "$current_hash" == "$previous_hash" && -d "$DIST_DIR" ]]; then
  echo "[ui-build] Web UI already built (cache hit ${current_hash:0:7})"
  exit 0
fi

echo "[ui-build] Building web UI (hash ${current_hash:0:7})"
(
  cd "$CLIENT_DIR"
  npm ci --no-progress
  npm run build --silent
)

echo "$current_hash" > "$HASH_FILE"
echo "[ui-build] UI artifacts ready in $DIST_DIR"
