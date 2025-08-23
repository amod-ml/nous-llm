#!/usr/bin/env bash
# Script to prepare a release for nous-llm

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    print_error "Not authenticated with GitHub. Run 'gh auth login' first."
    exit 1
fi

# Check GPG signing
if ! git config --get commit.gpgsign &> /dev/null; then
    print_warn "GPG signing not enabled. Setting up..."
    git config --global commit.gpgsign true
    git config --global tag.gpgsign true
fi

# Get current version
CURRENT_VERSION=$(grep "^version = " pyproject.toml | cut -d'"' -f2)
print_info "Current version: $CURRENT_VERSION"

# Prompt for release type
echo "Select release type:"
echo "1) patch (bug fixes)"
echo "2) minor (new features)"
echo "3) major (breaking changes)"
echo "4) prerelease (alpha/beta/rc)"
echo "5) custom version"
read -p "Choice [1-5]: " choice

case $choice in
    1) RELEASE_TYPE="patch";;
    2) RELEASE_TYPE="minor";;
    3) RELEASE_TYPE="major";;
    4) RELEASE_TYPE="prerelease";;
    5) 
        read -p "Enter version (without 'v' prefix): " NEW_VERSION
        RELEASE_TYPE="custom"
        ;;
    *) 
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Calculate new version if not custom
if [ "$RELEASE_TYPE" != "custom" ]; then
    IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
    patch=${patch%%[a-z]*}  # Remove any prerelease suffix
    
    case "$RELEASE_TYPE" in
        major)
            NEW_VERSION="$((major + 1)).0.0"
            ;;
        minor)
            NEW_VERSION="$major.$((minor + 1)).0"
            ;;
        patch)
            NEW_VERSION="$major.$minor.$((patch + 1))"
            ;;
        prerelease)
            read -p "Prerelease type (alpha/beta/rc) [alpha]: " PRE_TYPE
            PRE_TYPE=${PRE_TYPE:-alpha}
            NEW_VERSION="$major.$minor.$((patch + 1))-${PRE_TYPE}.1"
            ;;
    esac
fi

print_info "New version will be: $NEW_VERSION"

# Confirmation
read -p "Continue with release v$NEW_VERSION? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    print_info "Release cancelled"
    exit 0
fi

# Run tests
print_info "Running tests..."
if command -v uv &> /dev/null; then
    uv run pytest tests/ -v || {
        print_error "Tests failed. Fix issues before releasing."
        exit 1
    }
else
    print_warn "uv not found, skipping tests"
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "You have uncommitted changes. Commit or stash them first."
    exit 1
fi

# Update version in pyproject.toml
print_info "Updating version in pyproject.toml..."
sed -i.bak "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update version in __init__.py
print_info "Updating version in __init__.py..."
sed -i.bak "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" src/nous_llm/__init__.py
rm src/nous_llm/__init__.py.bak

# Commit version bump
print_info "Committing version bump..."
git add pyproject.toml src/nous_llm/__init__.py
git commit -m "chore: bump version to $NEW_VERSION"

# Create and push tag
print_info "Creating tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Push changes
print_info "Pushing changes to GitHub..."
git push origin main
git push origin "v$NEW_VERSION"

print_info "✅ Release v$NEW_VERSION prepared successfully!"
print_info "GitHub Actions will now:"
print_info "  1. Build the package"
print_info "  2. Run tests on multiple platforms"
print_info "  3. Publish to TestPyPI"
print_info "  4. Publish to PyPI"
print_info "  5. Create GitHub release"
print_info ""
print_info "Monitor progress at: https://github.com/amod-ml/nous-llm/actions"

# Open Actions page
read -p "Open GitHub Actions page? [y/N]: " open_actions
if [[ "$open_actions" =~ ^[Yy]$ ]]; then
    gh repo view --web
fi
