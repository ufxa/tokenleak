#!/usr/bin/env bash
# Build the paper PDF using tectonic
# Usage: bash scripts/build.sh
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")/../paper" && pwd)"

echo "[*] Building TokenLeak paper with tectonic..."
cd "$PAPER_DIR"
tectonic main.tex
echo "[+] Done: $PAPER_DIR/main.pdf"
