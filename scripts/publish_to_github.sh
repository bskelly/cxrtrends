#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <repo-name> [public|private]" >&2
  exit 1
fi

REPO_NAME="$1"
VISIBILITY="${2:-private}"
GITHUB_USER="${GITHUB_USER:-bskelly}"

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Visibility must be 'public' or 'private'." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found." >&2
  exit 1
fi

git init
git add .
git commit -m "Initial commit for CXR forecasting adaptation"
git branch -M main

echo
echo "Create an empty GitHub repository named '$REPO_NAME' under '$GITHUB_USER', then run:"
echo
echo "  git remote add origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
echo "  git push -u origin main"
