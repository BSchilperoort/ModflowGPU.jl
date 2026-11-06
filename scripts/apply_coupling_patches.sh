#!/usr/bin/env bash
# Reset the reference ModflowCoupling clone to the patch base commit and apply
# patches/modflow-coupling/*.patch in order with `git am` (PLAN.md section 2).
#
# The clone is created by validation/fetch_reference.sh at $REFERENCE_DIR/Modflow-coupling
# (default <repo root>/reference/Modflow-coupling). The patches are applied on a local
# branch `modflowgpu` that starts at the base commit. Any uncommitted work in the clone
# is discarded, so export your commits with `git format-patch` first.
#
# Usage:
#   scripts/apply_coupling_patches.sh
#   REFERENCE_DIR=/some/other/place scripts/apply_coupling_patches.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFERENCE_DIR="${REFERENCE_DIR:-$REPO_ROOT/reference}"
CLONE="$REFERENCE_DIR/Modflow-coupling"
PATCH_DIR="$REPO_ROOT/patches/modflow-coupling"
# Keep in sync with validation/fetch_reference.sh and patches/modflow-coupling/README.md.
COUPLING_BASE="8dd8c010796fd3648b855bed51195c0f2e61dd60"
BRANCH="modflowgpu"

log() { printf '[apply_coupling_patches] %s\n' "$*" >&2; }
die() {
    printf '[apply_coupling_patches] ERROR: %s\n' "$*" >&2
    exit 1
}

# Make sure every git command below acts on the clone, not on an inherited repository.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE
[[ -d "$CLONE/.git" ]] || die "$CLONE is not a git clone; run validation/fetch_reference.sh first"
CLONE="$(cd "$CLONE" && pwd -P)"
toplevel="$(git -C "$CLONE" rev-parse --show-toplevel)" || die "git rev-parse failed in $CLONE"
[[ "$toplevel" == "$CLONE" ]] || die "$CLONE is not the top level of its git repository ($toplevel)"
git -C "$CLONE" cat-file -e "$COUPLING_BASE^{commit}" 2>/dev/null ||
    die "base commit $COUPLING_BASE not found in $CLONE"

# Abort a half-finished `git am` from an earlier failed run.
if [[ -d "$(git -C "$CLONE" rev-parse --git-path rebase-apply)" ]]; then
    log "aborting an unfinished git am in the clone"
    git -C "$CLONE" am --abort
fi

log "reset $CLONE to $COUPLING_BASE on branch $BRANCH"
git -C "$CLONE" checkout --quiet --force -B "$BRANCH" "$COUPLING_BASE"
git -C "$CLONE" clean --quiet -fdx

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob
if [[ ${#patches[@]} -eq 0 ]]; then
    log "no patches in $PATCH_DIR; clone left at the base commit"
else
    # The glob expands in lexical order, which is the numbered order of format-patch.
    log "applying ${#patches[@]} patch(es)"
    git -C "$CLONE" -c user.name="ModflowGPU patches" -c user.email="modflowgpu@localhost" \
        am --quiet --committer-date-is-author-date "${patches[@]}" ||
        die "git am failed; inspect $CLONE (git am --abort to undo)"
fi
log "HEAD is now $(git -C "$CLONE" log -1 --format='%h %s')"
