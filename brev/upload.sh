#!/usr/bin/env bash
# Upload the AIM4D repo + non-tracked data to a Brev instance.
# Usage:  BREV_HOST=user@brev-host ./brev/upload.sh
#         (optional) BREV_PATH=~/AIM4D
set -euo pipefail

: "${BREV_HOST:?Set BREV_HOST=user@host}"
BREV_PATH="${BREV_PATH:-~/AIM4D}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading $LOCAL_ROOT -> $BREV_HOST:$BREV_PATH"

ssh "$BREV_HOST" "mkdir -p $BREV_PATH"

rsync -avh --partial --progress \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/objects/pack/*.pack' \
  --exclude 'data/gdelt_cache/' \
  --exclude '*.zip' \
  "$LOCAL_ROOT/" \
  "$BREV_HOST:$BREV_PATH/"

echo
echo "Done. Next:"
echo "  ssh $BREV_HOST"
echo "  cd $BREV_PATH && bash brev/setup.sh"
echo "  bash brev/run_gdelt.sh"
