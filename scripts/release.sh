#!/usr/bin/env bash
# scripts/release.sh — Nous release pipeline
#
# Usage:
#   ./scripts/release.sh <major|minor|patch|volatile> [--note="..."|--note-file=path] [--yes] [--dry-run]
#   ./scripts/release.sh --repair[=vX.Y.Z.W] [--yes]
#
# Versioning: vMAJOR.MINOR.PATCH.VOLATILE (Hermit federation standard)
#   major    v0.1.0.0 → v1.0.0.0  (resets minor, patch, volatile)
#   minor    v0.1.0.0 → v0.2.0.0  (resets patch, volatile)
#   patch    v0.1.0.0 → v0.1.1.0  (resets volatile)
#   volatile v0.1.0.0 → v0.1.0.1  (no resets)
#
# Remote topology: single remote — origin → github.com/ologos-repos/Nous (GitHub)
# Languages detected at runtime: go.mod → Go binary, pyproject.toml → Python wheel
# If both are present (after Go merges to main), both are built and uploaded.
#
# GitHub CLI (gh) must be authenticated (gh auth status).
# Python: uv must be on PATH or at /home/bobbyhiddn/.local/bin/uv

set -euo pipefail

# ──────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[release]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[release]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[release]${RESET} $*"; }
die()     { echo -e "${RED}${BOLD}[release] ERROR:${RESET} $*" >&2; exit 1; }

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
BUMP=""
DRY_RUN=false
NOTE_TEXT=""
NOTE_FILE=""
ASSUME_YES=false
REPAIR=false
REPAIR_TAG=""

for arg in "$@"; do
  case "$arg" in
    major|minor|patch|volatile) BUMP="$arg" ;;
    --dry-run) DRY_RUN=true ;;
    --yes|-y) ASSUME_YES=true ;;
    --repair) REPAIR=true ;;
    --repair=*) REPAIR=true; REPAIR_TAG="${arg#--repair=}" ;;
    --note=*) NOTE_TEXT="${arg#--note=}" ;;
    --note-file=*) NOTE_FILE="${arg#--note-file=}" ;;
    --help|-h)
      echo "Usage: $0 <major|minor|patch|volatile> [--note=\"...\"|--note-file=path] [--yes] [--dry-run]"
      echo "       $0 --repair[=vX.Y.Z.W] [--yes]   # complete a partial release without bumping"
      exit 0
      ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

if [[ -z "$BUMP" && "$REPAIR" == false ]]; then
  echo "Usage: $0 <major|minor|patch|volatile> [--note=\"...\"|--note-file=path] [--yes] [--dry-run]"
  echo "       $0 --repair[=vX.Y.Z.W] [--yes]"
  echo ""
  echo "  major    v0.1.0.0 → v1.0.0.0"
  echo "  minor    v0.1.0.0 → v0.2.0.0"
  echo "  patch    v0.1.0.0 → v0.1.1.0"
  echo "  volatile v0.1.0.0 → v0.1.0.1"
  echo ""
  echo "  --note=\"...\"         Inline release note (required for non-volatile)"
  echo "  --note-file=path     Read release note from file"
  echo "  RELEASE_NOTE file    Auto-read from repo root if present"
  echo "  --yes / -y           Skip the interactive confirmation (non-interactive)"
  echo "  --repair[=TAG]       Re-run push + release for an existing tag (idempotent)"
  exit 1
fi

if $REPAIR && [[ -n "$BUMP" ]]; then
  die "--repair cannot be combined with a bump type (major/minor/patch/volatile)."
fi

# ──────────────────────────────────────────────
# Resolve release note (required for non-volatile)
# Priority: --note flag > --note-file flag > RELEASE_NOTE file
# ──────────────────────────────────────────────
RELEASE_NOTE=""
if [[ -n "$NOTE_TEXT" ]]; then
  RELEASE_NOTE="$NOTE_TEXT"
elif [[ -n "$NOTE_FILE" ]]; then
  [[ -f "$NOTE_FILE" ]] || die "Release note file not found: $NOTE_FILE"
  RELEASE_NOTE="$(cat "$NOTE_FILE")"
elif [[ -f "RELEASE_NOTE" ]]; then
  RELEASE_NOTE="$(cat RELEASE_NOTE)"
fi

if [[ -z "$RELEASE_NOTE" && "$BUMP" != "volatile" && "$REPAIR" == false ]]; then
  die "Release note required for $BUMP releases. Use --note=\"...\" or create a RELEASE_NOTE file."
fi

# ──────────────────────────────────────────────
# Dependency checks + repo root
# ──────────────────────────────────────────────
command -v git &>/dev/null || die "git is not installed or not on PATH"
command -v gh  &>/dev/null || die "gh (GitHub CLI) is not installed or not on PATH"
git rev-parse --git-dir &>/dev/null || die "Not inside a git repository"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ──────────────────────────────────────────────
# Language auto-detection (build both if both present)
# ──────────────────────────────────────────────
HAS_GO=false
HAS_PYTHON=false
[[ -f "go.mod" ]] && HAS_GO=true
[[ -f "pyproject.toml" ]] && HAS_PYTHON=true

$HAS_GO || $HAS_PYTHON || die "Cannot detect project language. Expected go.mod and/or pyproject.toml in repo root."

# Go: discover module path, binary name, and build target
GO_MODULE=""
GO_BINARY=""
GO_BUILD_TARGET=""
if $HAS_GO; then
  command -v go &>/dev/null || die "go is not installed or not on PATH"
  GO_MODULE="$(awk '/^module /{print $2}' go.mod)"
  if [[ -d "cmd" ]] && ls cmd/ 2>/dev/null | grep -q .; then
    GO_BINARY="$(ls cmd/ | head -1)"
    GO_BUILD_TARGET="./cmd/${GO_BINARY}"
  else
    GO_BINARY="${GO_MODULE##*/}"
    GO_BUILD_TARGET="."
  fi
fi

# Python: resolve uv
UV=""
if $HAS_PYTHON; then
  UV="${UV_BIN:-/home/bobbyhiddn/.local/bin/uv}"
  [[ -x "$UV" ]] || UV="$(command -v uv 2>/dev/null || echo '')"
  [[ -n "$UV" && -x "$UV" ]] || die "uv not found. Install uv or set UV_BIN=/path/to/uv"
fi

# ──────────────────────────────────────────────
# Remote topology — GitHub only (origin)
# ──────────────────────────────────────────────
GITHUB_REMOTE="origin"
git remote get-url "${GITHUB_REMOTE}" >/dev/null 2>&1 \
  || die "Remote '${GITHUB_REMOTE}' not found. Is this the right repository?"

REMOTE_URL="$(git remote get-url "${GITHUB_REMOTE}")"
echo "$REMOTE_URL" | grep -q "github\.com" \
  || warn "Remote '${GITHUB_REMOTE}' URL does not look like GitHub: ${REMOTE_URL}"

# ──────────────────────────────────────────────
# Default branch detection
# ──────────────────────────────────────────────
DEFAULT_BRANCH=""
DEFAULT_BRANCH="$(git symbolic-ref "refs/remotes/${GITHUB_REMOTE}/HEAD" 2>/dev/null \
  | sed "s|refs/remotes/${GITHUB_REMOTE}/||" || true)"
if [[ -z "$DEFAULT_BRANCH" ]]; then
  DEFAULT_BRANCH="$(timeout 5s git remote show "$GITHUB_REMOTE" 2>/dev/null \
    | awk '/HEAD branch/{print $NF}' || true)"
  [[ "$DEFAULT_BRANCH" == "(unknown)" ]] && DEFAULT_BRANCH=""
fi
[[ -z "$DEFAULT_BRANCH" ]] && DEFAULT_BRANCH="$(git branch --show-current)"

# ──────────────────────────────────────────────
# Branch guard — must be on the default release branch
# ──────────────────────────────────────────────
CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$DEFAULT_BRANCH" ]] || \
  die "Must be on '${DEFAULT_BRANCH}' to release (currently on '${CURRENT_BRANCH}')"

# ──────────────────────────────────────────────
# Working tree must be clean
# ──────────────────────────────────────────────
git diff --quiet HEAD || die "Working tree has uncommitted changes. Commit or stash before releasing."
git diff --cached --quiet || die "Staged changes exist. Commit or stash before releasing."

# ──────────────────────────────────────────────
# Fetch latest tag + parse vMAJOR.MINOR.PATCH.VOLATILE
# ──────────────────────────────────────────────
LATEST_TAG="$(git tag --sort=-v:refname | head -1)"
[[ -z "$LATEST_TAG" ]] && die "No tags found. Seed the baseline: git tag v0.0.0.0 && git push origin v0.0.0.0"

info "Latest tag: ${BOLD}${LATEST_TAG}${RESET}"

VERSION_RE='^v([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)$'
[[ "$LATEST_TAG" =~ $VERSION_RE ]] || \
  die "Latest tag '${LATEST_TAG}' does not match vMAJOR.MINOR.PATCH.VOLATILE"

MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"
VOLATILE="${BASH_REMATCH[4]}"

# ──────────────────────────────────────────────
# Bump or repair
# ──────────────────────────────────────────────
if $REPAIR; then
  if [[ -n "$REPAIR_TAG" ]]; then
    NEW_TAG="$REPAIR_TAG"
  else
    NEW_TAG="$LATEST_TAG"
  fi
  git rev-parse -q --verify "refs/tags/${NEW_TAG}" >/dev/null \
    || die "Repair tag '${NEW_TAG}' does not exist locally."
  info "Repair mode: completing release ${BOLD}${NEW_TAG}${RESET} (no version bump)"
else
  case "$BUMP" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0; VOLATILE=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0; VOLATILE=0 ;;
    patch) PATCH=$((PATCH + 1)); VOLATILE=0 ;;
    volatile) VOLATILE=$((VOLATILE + 1)) ;;
  esac
  NEW_TAG="v${MAJOR}.${MINOR}.${PATCH}.${VOLATILE}"
fi

VERSION_PLAIN="${NEW_TAG#v}"   # strip leading 'v' for PEP 440 / pyproject

# ──────────────────────────────────────────────
# Commits since last tag
# ──────────────────────────────────────────────
COMMIT_LOG="$(git log "${LATEST_TAG}..HEAD" --oneline --no-decorate 2>/dev/null || true)"

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
REPO_NAME="$(basename "$REPO_ROOT")"
LANGS=""
$HAS_GO && LANGS="${LANGS}go "
$HAS_PYTHON && LANGS="${LANGS}python"

echo ""
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${BOLD}  ${REPO_NAME} Release Summary${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "  Languages : ${YELLOW}${LANGS}${RESET}"
echo -e "  Remote    : ${GITHUB_REMOTE} → ${REMOTE_URL}"
echo -e "  Bump type : ${YELLOW}${BUMP:-repair (no bump)}${RESET}"
echo -e "  Old tag   : ${CYAN}${LATEST_TAG}${RESET}"
echo -e "  New tag   : ${GREEN}${NEW_TAG}${RESET}"
echo -e "  Branch    : ${CURRENT_BRANCH}"
echo -e "  HEAD      : $(git rev-parse --short HEAD)"
if $HAS_GO; then
  echo -e "  Go binary : ${GO_BINARY}  (target: ${GO_BUILD_TARGET})"
fi
echo ""

if [[ -z "$COMMIT_LOG" ]]; then
  warn "No commits since ${LATEST_TAG} — releasing HEAD as-is."
else
  echo -e "${BOLD}Commits since ${LATEST_TAG}:${RESET}"
  echo "$COMMIT_LOG" | while IFS= read -r line; do
    echo -e "  ${CYAN}•${RESET} ${line}"
  done
fi
echo ""

# ──────────────────────────────────────────────
# Dry-run exit
# ──────────────────────────────────────────────
if $DRY_RUN; then
  warn "--dry-run mode: nothing will be created, tagged, or pushed."
  echo ""
  [[ -n "$RELEASE_NOTE" ]] && { echo -e "Release note: ${CYAN}${RELEASE_NOTE}${RESET}"; echo ""; }
  echo "Would execute:"
  if $HAS_PYTHON; then
    echo "  sed __version__ → ${VERSION_PLAIN}  (src/nous/__init__.py)"
    echo "  sed version → ${VERSION_PLAIN}       (pyproject.toml)"
    echo "  git commit -m 'chore: bump version to ${NEW_TAG}'"
  fi
  echo "  git tag ${NEW_TAG}"
  echo "  git push ${GITHUB_REMOTE} ${DEFAULT_BRANCH}"
  echo "  git push ${GITHUB_REMOTE} ${NEW_TAG}"
  echo "  gh release create ${NEW_TAG} --title '${NEW_TAG}' (with release notes)"
  if $HAS_GO; then
    echo "  [build] ${GO_BINARY}-linux-amd64, ${GO_BINARY}-linux-arm64, ${GO_BINARY}-darwin-amd64, ${GO_BINARY}-darwin-arm64"
  fi
  if $HAS_PYTHON; then
    echo "  [build] uv build → dist/*.whl + dist/*.tar.gz"
  fi
  echo "  [upload] artifacts to release ${NEW_TAG}"
  echo ""
  success "Dry run complete. New tag would be: ${NEW_TAG}"
  exit 0
fi

# ──────────────────────────────────────────────
# Confirmation prompt
# ──────────────────────────────────────────────
if $ASSUME_YES; then
  info "--yes given — proceeding non-interactively"
elif [[ -t 0 ]]; then
  echo -e "${BOLD}Proceed with release ${GREEN}${NEW_TAG}${RESET}${BOLD}? [y/N]${RESET} "
  read -r CONFIRM
  case "$CONFIRM" in
    y|Y) ;;
    *) warn "Release aborted."; exit 1 ;;
  esac
else
  die "No TTY for confirmation. Re-run with --yes."
fi

echo ""

# ──────────────────────────────────────────────
# Python version bump (before tagging)
# Bumps __version__ in src/nous/__init__.py and version in pyproject.toml,
# commits the change so the tag lands on the bumped commit.
# Skipped in --repair mode (already done in the original bump run).
# ──────────────────────────────────────────────
if $HAS_PYTHON && ! $REPAIR; then
  INIT_FILE="${REPO_ROOT}/src/nous/__init__.py"
  PYPROJECT="${REPO_ROOT}/pyproject.toml"

  if [[ -f "$INIT_FILE" ]]; then
    info "Bumping __version__ in src/nous/__init__.py → ${VERSION_PLAIN}..."
    sed -i "s/^__version__ = .*/__version__ = \"${VERSION_PLAIN}\"/" "$INIT_FILE"
  fi

  if [[ -f "$PYPROJECT" ]]; then
    info "Bumping version in pyproject.toml → ${VERSION_PLAIN}..."
    sed -i "s/^version = .*/version = \"${VERSION_PLAIN}\"/" "$PYPROJECT"
  fi

  # Commit version bump
  git add "${INIT_FILE}" "${PYPROJECT}" 2>/dev/null || true
  git commit -m "chore: bump version to ${NEW_TAG}" \
    --author "release-bot <release@nous>" \
    --no-verify 2>/dev/null || warn "Nothing to commit for version bump (already at ${NEW_TAG}?)"
  success "Version bump committed"
fi

# ──────────────────────────────────────────────
# Tag — refuse to move a published tag
# ──────────────────────────────────────────────
if git rev-parse -q --verify "refs/tags/${NEW_TAG}" >/dev/null; then
  EXISTING_C="$(git rev-parse "refs/tags/${NEW_TAG}^{commit}")"
  HEAD_C="$(git rev-parse "HEAD^{commit}")"
  if $REPAIR || [[ "$EXISTING_C" == "$HEAD_C" ]]; then
    info "Tag ${NEW_TAG} already exists — skipping creation (idempotent)"
  else
    die "Tag ${NEW_TAG} exists at ${EXISTING_C:0:9} but HEAD is ${HEAD_C:0:9}. Refusing to move a published tag. Use --repair=${NEW_TAG} to complete its release, or pick a new bump."
  fi
else
  info "Creating tag ${NEW_TAG}..."
  git tag "${NEW_TAG}"
  success "Tagged HEAD as ${NEW_TAG}"
fi

# ──────────────────────────────────────────────
# Push to GitHub
# ──────────────────────────────────────────────
info "Pushing ${DEFAULT_BRANCH} to ${GITHUB_REMOTE} (GitHub)..."
git push "${GITHUB_REMOTE}" "${DEFAULT_BRANCH}"
success "Branch pushed to ${GITHUB_REMOTE}"

info "Pushing tag ${NEW_TAG} to ${GITHUB_REMOTE}..."
git push "${GITHUB_REMOTE}" "${NEW_TAG}"
success "Tag pushed to ${GITHUB_REMOTE}"

# Small delay — let GitHub register the tag push before creating the release
sleep 2

# ──────────────────────────────────────────────
# GitHub release
# ──────────────────────────────────────────────
RELEASE_BODY=""
[[ -n "$RELEASE_NOTE" ]] && RELEASE_BODY="$(printf "## Release Notes\n\n%s\n" "$RELEASE_NOTE")"
if [[ -n "$COMMIT_LOG" ]]; then
  COMMIT_SECTION="$(printf "## Changes since %s\n\n%s" "${LATEST_TAG}" \
    "$(echo "$COMMIT_LOG" | sed 's/^/- /')")"
  if [[ -n "$RELEASE_BODY" ]]; then
    RELEASE_BODY="$(printf "%s\n\n---\n\n%s" "$RELEASE_BODY" "$COMMIT_SECTION")"
  else
    RELEASE_BODY="$COMMIT_SECTION"
  fi
elif [[ -z "$RELEASE_BODY" ]]; then
  RELEASE_BODY="Release ${NEW_TAG} (no new commits since ${LATEST_TAG})"
fi

info "Creating GitHub release ${NEW_TAG}..."
if gh release view "${NEW_TAG}" >/dev/null 2>&1; then
  info "GitHub release already exists — updating notes (idempotent)"
  gh release edit "${NEW_TAG}" --title "${NEW_TAG}" --notes "${RELEASE_BODY}"
else
  gh release create "${NEW_TAG}" --verify-tag --title "${NEW_TAG}" --notes "${RELEASE_BODY}"
fi
success "GitHub release created: ${NEW_TAG}"

# Clean up RELEASE_NOTE file if consumed from repo root
if [[ -f "RELEASE_NOTE" && -z "$NOTE_TEXT" && -z "$NOTE_FILE" ]]; then
  rm -f RELEASE_NOTE
  info "Removed RELEASE_NOTE file (consumed into release)"
fi

# ──────────────────────────────────────────────
# Build artifacts
# ──────────────────────────────────────────────

# ── Go: cross-compile linux/darwin × amd64/arm64 ──────────────────────────
if $HAS_GO; then
  BUILD_DIR="/tmp/${GO_BINARY}-release"
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"

  BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  LDFLAGS="-s -w -X ${GO_MODULE}/internal/version.Version=${NEW_TAG} -X ${GO_MODULE}/internal/version.BuildTime=${BUILD_TIME}"

  info "Cross-compiling ${GO_BINARY} for linux/darwin × amd64/arm64..."
  for OS in linux darwin; do
    for ARCH in amd64 arm64; do
      OUTPUT="${BUILD_DIR}/${GO_BINARY}-${OS}-${ARCH}"
      info "  Building ${GO_BINARY}-${OS}-${ARCH}..."
      CGO_ENABLED=0 GOOS="${OS}" GOARCH="${ARCH}" go build \
        -ldflags "${LDFLAGS}" \
        -o "${OUTPUT}" \
        "${GO_BUILD_TARGET}"
    done
  done

  (cd "${BUILD_DIR}" && sha256sum "${GO_BINARY}"-* > checksums-go.txt)
  info "Go checksums written"

  gh release upload "${NEW_TAG}" \
    "${BUILD_DIR}/${GO_BINARY}-linux-amd64" \
    "${BUILD_DIR}/${GO_BINARY}-linux-arm64" \
    "${BUILD_DIR}/${GO_BINARY}-darwin-amd64" \
    "${BUILD_DIR}/${GO_BINARY}-darwin-arm64" \
    "${BUILD_DIR}/checksums-go.txt" \
    --clobber
  success "Go binaries uploaded to release ${NEW_TAG}"
fi

# ── Python: uv build → wheel + sdist ─────────────────────────────────────
if $HAS_PYTHON; then
  PYTHON_DIST="${REPO_ROOT}/dist"
  rm -rf "${PYTHON_DIST}"
  mkdir -p "${PYTHON_DIST}"

  info "Building Python wheel + sdist (version ${VERSION_PLAIN})..."
  "${UV}" build --out-dir "${PYTHON_DIST}" "${REPO_ROOT}"
  success "uv build complete"

  WHEEL_FILE="$(ls "${PYTHON_DIST}"/*.whl 2>/dev/null | head -1)"
  SDIST_FILE="$(ls "${PYTHON_DIST}"/*.tar.gz 2>/dev/null | head -1)"
  [[ -n "$WHEEL_FILE" ]] || die "No .whl found in dist/"
  [[ -n "$SDIST_FILE" ]] || die "No .tar.gz found in dist/"

  (cd "${PYTHON_DIST}" && sha256sum ./*.whl ./*.tar.gz > checksums-python.txt)

  gh release upload "${NEW_TAG}" \
    "${WHEEL_FILE}" "${SDIST_FILE}" "${PYTHON_DIST}/checksums-python.txt" --clobber
  success "Python artifacts uploaded to release ${NEW_TAG}"
fi

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  Release ${NEW_TAG} complete!${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}•${RESET} Tag       : ${NEW_TAG}"
echo -e "  ${CYAN}•${RESET} Pushed to : ${GITHUB_REMOTE} (${REMOTE_URL})"
echo -e "  ${CYAN}•${RESET} GH release: $(gh release view "${NEW_TAG}" --json url -q .url 2>/dev/null || echo 'created')"
echo ""
