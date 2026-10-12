#!/usr/bin/env bash
set -euo pipefail

# Render deploy/*.service templates with repository root and install into ~/.config/systemd/user
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$TARGET_DIR"

echo "Installing systemd user units for repository: $REPO_DIR"
for unit in gaja-llm.service gaja-adapter.service; do
  src="$REPO_DIR/deploy/$unit"
  dst="$TARGET_DIR/$unit"
  if [[ ! -f "$src" ]]; then
    echo "Error: source unit template not found: $src" >&2
    exit 1
  fi
  sed "s|@REPO@|$REPO_DIR|g" "$src" > "$dst"
  echo "Rendered $dst"
done

echo ""
echo "Done. To reload and start services, run:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now gaja-llm gaja-adapter"
