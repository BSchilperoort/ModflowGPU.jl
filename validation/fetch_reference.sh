#!/usr/bin/env bash
# Download the reference toolchain for ModflowGPU (task card T0.2, PLAN.md section 2 and 4).
#
# Fetches into $REFERENCE_DIR (default: <repo root>/reference, gitignored):
#
#   archives/mf6.8.2_linux/mf6.8.2_linux.zip        MODFLOW 6.8.2 release asset (mf6 + libmf6.so)
#   archives/mf6.8.2_dfns/mf6.8.2_dfns.zip          input definition files
#   archives/mf6examples/mf6examples.zip            modflow6-examples release `current`
#   mf6.8.2_linux/                                  extracted: bin/mf6, bin/libmf6.so, ...
#   mf6.8.2_dfns/                                   extracted *.dfn files
#   mf6examples/                                    extracted example directories (ex-gwf-toth, ...)
#   modflow6/                                       shallow clone of MODFLOW-ORG/modflow6 at tag 6.8.2
#   Modflow-coupling/                               clone of SWIFT-WUR/Modflow-coupling at the patch base commit
#   checksums.txt                                   sha256 of every downloaded archive
#
# Every download and every extraction goes into its own new directory: archives are
# untrusted data, so they are never unpacked on top of existing files. The script is
# idempotent: anything already present (and verified) is skipped. Any error aborts.
#
# Usage:
#   validation/fetch_reference.sh
#   REFERENCE_DIR=/some/other/place validation/fetch_reference.sh
#
# Optional environment:
#   REFERENCE_DIR   target directory (default <repo root>/reference)
#   GITHUB_TOKEN    used for the GitHub API call (rate limits) if set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFERENCE_DIR="${REFERENCE_DIR:-$REPO_ROOT/reference}"

MF6_VERSION="6.8.2"
MF6_RELEASE_URL="https://github.com/MODFLOW-ORG/modflow6/releases/download/$MF6_VERSION"
# sha256 digests of the (immutable) 6.8.2 release assets, from the GitHub release API.
MF6_LINUX_SHA256="69847a003a4259d69f0ae637bc191b99097650062847f39ed17d234fd8596f37"
MF6_DFNS_SHA256="a89f39584c4a125e6fba3bd819a38ed9d63d870fdbaa6a2365963f4080a6bc17"

EXAMPLES_REPO="MODFLOW-ORG/modflow6-examples"
EXAMPLES_TAG="current" # rolling release: its digest is looked up at download time

MF6_GIT_URL="https://github.com/MODFLOW-ORG/modflow6.git"
COUPLING_GIT_URL="https://github.com/SWIFT-WUR/Modflow-coupling.git"
# Patch base commit (2026-10-19); keep in sync with scripts/apply_coupling_patches.sh
# and patches/modflow-coupling/README.md.
COUPLING_BASE="8dd8c010796fd3648b855bed51195c0f2e61dd60"

log() { printf '[fetch_reference] %s\n' "$*" >&2; }
die() {
    printf '[fetch_reference] ERROR: %s\n' "$*" >&2
    exit 1
}

for tool in curl unzip sha256sum git python3; do
    command -v "$tool" >/dev/null || die "required tool '$tool' not found"
done

mkdir -p "$REFERENCE_DIR/archives"
REFERENCE_DIR="$(cd "$REFERENCE_DIR" && pwd)"
CHECKSUMS="$REFERENCE_DIR/checksums.txt"
touch "$CHECKSUMS"
log "reference directory: $REFERENCE_DIR"

# sha256 digest of a release asset, from the GitHub release API.
release_asset_digest() {
    local repo="$1" tag="$2" asset="$3" json
    local auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    json="$(curl -fsSL "${auth[@]}" -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$repo/releases/tags/$tag")" ||
        die "GitHub API request for $repo@$tag failed"
    printf '%s' "$json" | python3 -I -c '
import json, sys
name = sys.argv[1]
for a in json.load(sys.stdin)["assets"]:
    if a["name"] == name:
        d = a.get("digest") or ""
        if not d.startswith("sha256:"):
            sys.exit("asset %s has no sha256 digest" % name)
        print(d[len("sha256:"):])
        sys.exit(0)
sys.exit("asset %s not found in release" % name)
' "$asset" || die "could not get digest of $asset from $repo@$tag"
}

# fetch_archive <url> <name> <sha256>
# Downloads <url> into archives/<name>/<file> (a new directory) and checks its sha256.
fetch_archive() {
    local url="$1" name="$2" sha="$3"
    local file dir tmp
    file="$(basename "$url")"
    dir="$REFERENCE_DIR/archives/$name"
    if [[ -f "$dir/$file" ]]; then
        log "skip download: archives/$name/$file exists"
    else
        [[ -e "$dir" ]] && die "archives/$name exists but has no $file; remove it and rerun"
        tmp="$(mktemp -d "$REFERENCE_DIR/archives/.tmp.$name.XXXXXX")"
        log "download $url"
        curl -fL --retry 3 --progress-bar -o "$tmp/$file" "$url" || {
            rm -rf "$tmp"
            die "download of $url failed"
        }
        echo "$sha  $tmp/$file" | sha256sum --check --quiet - || {
            rm -rf "$tmp"
            die "sha256 mismatch for $file (expected $sha)"
        }
        mv "$tmp" "$dir"
    fi
    # Verify again on every run (cheap) so a corrupted archive is caught.
    echo "$sha  $dir/$file" | sha256sum --check --quiet - ||
        die "sha256 mismatch for archives/$name/$file (expected $sha)"
    grep -q "  archives/$name/$file\$" "$CHECKSUMS" ||
        echo "$sha  archives/$name/$file" >>"$CHECKSUMS"
}

# extract_archive <name> <zipfile>
# Extracts into $REFERENCE_DIR/<name> (a new directory). If the archive has a single
# top-level directory, its contents become the contents of <name>.
extract_archive() {
    local name="$1" zip="$2" dest tmp entries
    dest="$REFERENCE_DIR/$name"
    if [[ -e "$dest" ]]; then
        log "skip extract: $name/ exists"
        return
    fi
    # Refuse archives with absolute paths or parent-directory components.
    if unzip -Z1 "$zip" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
        die "$zip contains unsafe paths"
    fi
    tmp="$(mktemp -d "$REFERENCE_DIR/.tmp.$name.XXXXXX")"
    log "extract $(basename "$zip") -> $name/"
    unzip -q "$zip" -d "$tmp/x" || {
        rm -rf "$tmp"
        die "unzip of $zip failed"
    }
    mapfile -t entries < <(find "$tmp/x" -mindepth 1 -maxdepth 1)
    if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" && ! -L "${entries[0]}" ]]; then
        mv "${entries[0]}" "$dest"
    else
        mv "$tmp/x" "$dest"
    fi
    rm -rf "$tmp"
}

# --- MODFLOW 6.8.2 binaries and dfns -----------------------------------------------

fetch_archive "$MF6_RELEASE_URL/mf${MF6_VERSION}_linux.zip" "mf${MF6_VERSION}_linux" "$MF6_LINUX_SHA256"
extract_archive "mf${MF6_VERSION}_linux" "$REFERENCE_DIR/archives/mf${MF6_VERSION}_linux/mf${MF6_VERSION}_linux.zip"
for f in "$REFERENCE_DIR/mf${MF6_VERSION}_linux/bin/mf6" "$REFERENCE_DIR/mf${MF6_VERSION}_linux/bin/libmf6.so"; do
    [[ -f "$f" ]] || die "expected $f after extraction"
done
version_line="$("$REFERENCE_DIR/mf${MF6_VERSION}_linux/bin/mf6" --version 2>&1 | grep -m1 'mf6:' || true)"
[[ "$version_line" == *"$MF6_VERSION"* ]] || die "bin/mf6 --version gave '$version_line', expected $MF6_VERSION"
log "$version_line"

fetch_archive "$MF6_RELEASE_URL/mf${MF6_VERSION}_dfns.zip" "mf${MF6_VERSION}_dfns" "$MF6_DFNS_SHA256"
extract_archive "mf${MF6_VERSION}_dfns" "$REFERENCE_DIR/archives/mf${MF6_VERSION}_dfns/mf${MF6_VERSION}_dfns.zip"

# --- MF6 examples (rolling release `current`) -----------------------------------------

examples_zip="$REFERENCE_DIR/archives/mf6examples/mf6examples.zip"
if [[ -f "$examples_zip" ]]; then
    # Already downloaded: verify against the digest recorded at download time.
    examples_sha="$(awk '$2 == "archives/mf6examples/mf6examples.zip" {print $1}' "$CHECKSUMS")"
    [[ -n "$examples_sha" ]] || die "no recorded checksum for $examples_zip; remove archives/mf6examples and rerun"
else
    examples_sha="$(release_asset_digest "$EXAMPLES_REPO" "$EXAMPLES_TAG" mf6examples.zip)"
fi
fetch_archive "https://github.com/$EXAMPLES_REPO/releases/download/$EXAMPLES_TAG/mf6examples.zip" \
    mf6examples "$examples_sha"
extract_archive mf6examples "$examples_zip"
[[ -d "$REFERENCE_DIR/mf6examples/ex-gwf-toth" ]] || die "mf6examples/ex-gwf-toth missing after extraction"

# --- MODFLOW 6 source at tag 6.8.2 ---------------------------------------------------

mf6_src="$REFERENCE_DIR/modflow6"
if [[ -e "$mf6_src" ]]; then
    tag_commit="$(git -C "$mf6_src" rev-parse -q --verify "refs/tags/$MF6_VERSION^{commit}")" ||
        die "modflow6/ exists but has no tag $MF6_VERSION; remove it and rerun"
    head_commit="$(git -C "$mf6_src" rev-parse HEAD)"
    [[ "$tag_commit" == "$head_commit" ]] ||
        die "modflow6/ HEAD ($head_commit) is not tag $MF6_VERSION ($tag_commit)"
    log "skip clone: modflow6/ is at tag $MF6_VERSION"
else
    log "clone $MF6_GIT_URL at tag $MF6_VERSION (shallow)"
    tmp="$(mktemp -d "$REFERENCE_DIR/.tmp.modflow6.XXXXXX")"
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$MF6_VERSION" "$MF6_GIT_URL" "$tmp/modflow6" || {
        rm -rf "$tmp"
        die "clone of $MF6_GIT_URL failed"
    }
    mv "$tmp/modflow6" "$mf6_src"
    rm -rf "$tmp"
fi

# --- ModflowCoupling at the patch base commit ----------------------------------------

coupling="$REFERENCE_DIR/Modflow-coupling"
if [[ -e "$coupling" ]]; then
    git -C "$coupling" cat-file -e "$COUPLING_BASE^{commit}" 2>/dev/null ||
        die "Modflow-coupling/ exists but lacks base commit $COUPLING_BASE; remove it and rerun"
    # Leave HEAD alone: it may carry applied patches (scripts/apply_coupling_patches.sh).
    log "skip clone: Modflow-coupling/ exists (HEAD $(git -C "$coupling" rev-parse --short HEAD))"
else
    log "clone $COUPLING_GIT_URL at $COUPLING_BASE"
    tmp="$(mktemp -d "$REFERENCE_DIR/.tmp.Modflow-coupling.XXXXXX")"
    {
        git clone --quiet "$COUPLING_GIT_URL" "$tmp/Modflow-coupling" &&
            git -C "$tmp/Modflow-coupling" -c advice.detachedHead=false checkout --quiet "$COUPLING_BASE"
    } || {
        rm -rf "$tmp"
        die "clone/checkout of $COUPLING_GIT_URL at $COUPLING_BASE failed"
    }
    mv "$tmp/Modflow-coupling" "$coupling"
    rm -rf "$tmp"
fi

log "done. Contents of $REFERENCE_DIR:"
(cd "$REFERENCE_DIR" && du -sh -- * | sed 's/^/    /' >&2)
