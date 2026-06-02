#!/usr/bin/env bash
# List local branches whose upstream remote tracking ref is [gone]
# (i.e. the remote branch was deleted after merge). These are safe-to-review
# orphans per the branch hygiene policy in CONTRIBUTING.md.
#
# Usage:
#   bash scripts/list_orphans.sh
#
# Tip: run `git fetch --prune` first so tracking refs are up to date.
set -euo pipefail

git fetch --prune --quiet 2>/dev/null || true

orphans=$(git branch -vv | grep -E '\[[^]]*: gone\]' || true)

if [ -z "$orphans" ]; then
    echo "No orphaned [gone] branches. Working tree is clean."
    exit 0
fi

echo "Orphaned local branches (remote deleted). Confirm merge to main, then delete with 'git branch -d <branch>':"
echo "$orphans"
