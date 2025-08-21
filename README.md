# nous-llm
Intelligent No Frills LLM Router

## 🔒 Security & Authentication

### GPG Signing Required

**ALL commits to this repository MUST be GPG-signed.** This is automatically enforced by a pre-commit hook that cannot be bypassed.

#### 🛡️ What This Means:
- **Authentication**: Every commit is cryptographically verified
- **Integrity**: Commits cannot be tampered with after signing
- **Non-repudiation**: Contributors cannot deny authorship of signed commits
- **Supply Chain Security**: Protection against commit spoofing attacks

#### 🚀 Quick Setup for Contributors:

**New to the project? Run this first:**
```bash
# Automated setup - installs hook and guides through GPG configuration
./scripts/setup-gpg-hook.sh
```

**Already have GPG configured? Just ensure it's enabled:**
```bash
# Enable GPG signing for this repository
git config commit.gpgsign true
git config user.signingkey YOUR_KEY_ID
```

#### ⚠️ Important Notes:
- **Unsigned commits will be automatically rejected**
- **The pre-commit hook validates your GPG setup before every commit**
- **You must add your GPG public key to your GitHub account**
- **The hook cannot be bypassed with `--no-verify`**

#### 📚 Need Help?
- **Full Setup Guide**: [GPG Signing Documentation](docs/GPG-SIGNING.md)
- **Troubleshooting**: Run `./scripts/setup-gpg-hook.sh` for diagnostics
- **Quick Test**: Try making a commit - the hook will guide you if anything's wrong

---

*This security measure ensures the authenticity and integrity of all code contributions.*
