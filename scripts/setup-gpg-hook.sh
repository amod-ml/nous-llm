#!/bin/bash
#
# Setup script for GPG signing enforcement
# This script sets up the pre-commit hook that enforces GPG signing for all commits
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
HOOK_FILE="$HOOKS_DIR/pre-commit"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔒 Setting up GPG signing enforcement for the repository${NC}"
echo ""

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo -e "${RED}❌ ERROR: Not in a git repository!${NC}"
    echo "Please run this script from within the repository."
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Copy the pre-commit hook
echo -e "${YELLOW}📋 Installing pre-commit hook...${NC}"
cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash
#
# Pre-commit hook to enforce GPG signing
# This hook ensures all commits are GPG signed before allowing them to be committed
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔒 Checking GPG signing configuration...${NC}"

# Check if commit.gpgsign is enabled globally
GLOBAL_GPG_SIGN=$(git config --global --get commit.gpgsign 2>/dev/null)

# Check if commit.gpgsign is enabled locally (repository level)
LOCAL_GPG_SIGN=$(git config --get commit.gpgsign 2>/dev/null)

# Check if user has a signing key configured
GLOBAL_SIGNING_KEY=$(git config --global --get user.signingkey 2>/dev/null)
LOCAL_SIGNING_KEY=$(git config --get user.signingkey 2>/dev/null)

# Determine effective GPG signing setting (local overrides global)
if [[ -n "$LOCAL_GPG_SIGN" ]]; then
    GPG_SIGN="$LOCAL_GPG_SIGN"
elif [[ -n "$GLOBAL_GPG_SIGN" ]]; then
    GPG_SIGN="$GLOBAL_GPG_SIGN"
else
    GPG_SIGN="false"
fi

# Determine effective signing key (local overrides global)
if [[ -n "$LOCAL_SIGNING_KEY" ]]; then
    SIGNING_KEY="$LOCAL_SIGNING_KEY"
elif [[ -n "$GLOBAL_SIGNING_KEY" ]]; then
    SIGNING_KEY="$GLOBAL_SIGNING_KEY"
else
    SIGNING_KEY=""
fi

echo -e "📋 GPG signing enabled: ${GREEN}${GPG_SIGN}${NC}"
echo -e "🔑 Signing key: ${GREEN}${SIGNING_KEY:-"Not configured"}${NC}"

# Check if GPG signing is enabled
if [[ "$GPG_SIGN" != "true" ]]; then
    echo -e "${RED}❌ ERROR: GPG signing is not enabled!${NC}"
    echo -e "${YELLOW}To fix this, run one of the following commands:${NC}"
    echo -e "  ${GREEN}git config --global commit.gpgsign true${NC}  (global setting)"
    echo -e "  ${GREEN}git config commit.gpgsign true${NC}           (repository setting)"
    echo ""
    echo -e "${YELLOW}💡 Also ensure you have a GPG signing key configured:${NC}"
    echo -e "  ${GREEN}git config --global user.signingkey YOUR_KEY_ID${NC}"
    echo -e "  ${GREEN}git config user.signingkey YOUR_KEY_ID${NC}"
    exit 1
fi

# Check if signing key is configured
if [[ -z "$SIGNING_KEY" ]]; then
    echo -e "${RED}❌ ERROR: No GPG signing key configured!${NC}"
    echo -e "${YELLOW}To fix this, run one of the following commands:${NC}"
    echo -e "  ${GREEN}git config --global user.signingkey YOUR_KEY_ID${NC}  (global setting)"
    echo -e "  ${GREEN}git config user.signingkey YOUR_KEY_ID${NC}           (repository setting)"
    echo ""
    echo -e "${YELLOW}💡 To list your available GPG keys:${NC}"
    echo -e "  ${GREEN}gpg --list-secret-keys --keyid-format=long${NC}"
    exit 1
fi

# Test if the GPG key is available and can be used for signing
echo -e "${YELLOW}🔍 Verifying GPG key is available and functional...${NC}"
if ! echo "test" | gpg --clearsign --default-key "$SIGNING_KEY" >/dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: Cannot sign with the configured GPG key: ${SIGNING_KEY}${NC}"
    echo -e "${YELLOW}Possible issues:${NC}"
    echo -e "  • Key does not exist in your GPG keyring"
    echo -e "  • Key has expired"
    echo -e "  • GPG agent is not running or accessible"
    echo -e "  • Passphrase prompt may be required"
    echo ""
    echo -e "${YELLOW}💡 To debug:${NC}"
    echo -e "  ${GREEN}gpg --list-secret-keys --keyid-format=long${NC}  (list available keys)"
    echo -e "  ${GREEN}gpg --card-status${NC}                            (if using hardware key)"
    echo -e "  ${GREEN}echo 'test' | gpg --clearsign --default-key ${SIGNING_KEY}${NC}  (test signing)"
    exit 1
fi

echo -e "${GREEN}✅ GPG signing configuration is valid!${NC}"
echo -e "${GREEN}🚀 Proceeding with signed commit...${NC}"
echo ""

# All checks passed, allow the commit to proceed
exit 0
EOF

# Make the hook executable
chmod +x "$HOOK_FILE"

echo -e "${GREEN}✅ Pre-commit hook installed successfully!${NC}"
echo ""

# Check current user's GPG configuration
echo -e "${YELLOW}🔍 Checking your current GPG configuration...${NC}"
CURRENT_GPG_SIGN=$(git config --get commit.gpgsign 2>/dev/null || echo "not set")
CURRENT_SIGNING_KEY=$(git config --get user.signingkey 2>/dev/null || echo "not set")

echo -e "📋 Your GPG signing enabled: ${GREEN}${CURRENT_GPG_SIGN}${NC}"
echo -e "🔑 Your signing key: ${GREEN}${CURRENT_SIGNING_KEY}${NC}"

if [[ "$CURRENT_GPG_SIGN" != "true" ]] || [[ "$CURRENT_SIGNING_KEY" == "not set" ]]; then
    echo ""
    echo -e "${YELLOW}⚠️  Your GPG configuration needs setup!${NC}"
    echo -e "${YELLOW}To configure GPG signing for this repository:${NC}"
    echo ""
    echo -e "1. ${BLUE}Enable GPG signing:${NC}"
    echo -e "   ${GREEN}git config commit.gpgsign true${NC}"
    echo ""
    echo -e "2. ${BLUE}Set your signing key (replace with your key ID):${NC}"
    echo -e "   ${GREEN}git config user.signingkey YOUR_KEY_ID${NC}"
    echo ""
    echo -e "3. ${BLUE}Find your key ID:${NC}"
    echo -e "   ${GREEN}gpg --list-secret-keys --keyid-format=long${NC}"
    echo ""
    echo -e "${YELLOW}💡 For global configuration (all repositories), add --global flag to the commands above.${NC}"
else
    echo ""
    echo -e "${GREEN}🎉 Your GPG configuration looks good!${NC}"
fi

echo ""
echo -e "${BLUE}📝 The pre-commit hook is now active and will:${NC}"
echo -e "  • Check that GPG signing is enabled before each commit"
echo -e "  • Verify that a valid GPG signing key is configured"
echo -e "  • Test that the GPG key can actually be used for signing"
echo -e "  • Block any commits that would not be GPG signed"
echo ""
echo -e "${GREEN}🔒 Repository is now protected against unsigned commits!${NC}"
