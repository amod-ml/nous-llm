# GPG Signing Requirements

This repository requires all commits to be GPG signed for security and authenticity verification. A pre-commit hook enforces this requirement automatically.

## 🔒 Why GPG Signing?

GPG signing provides:
- **Authentication**: Proves commits come from verified contributors
- **Integrity**: Ensures commits haven't been tampered with
- **Non-repudiation**: Prevents denial of authorship
- **Security**: Protects against commit spoofing and supply chain attacks

## 🚀 Quick Setup

### For New Contributors

1. **Run the setup script:**
   ```bash
   ./scripts/setup-gpg-hook.sh
   ```

2. **Follow the on-screen instructions to configure your GPG key**

### Manual Setup

If you prefer manual setup or need to troubleshoot:

#### 1. Generate a GPG Key (if you don't have one)

```bash
# Generate a new GPG key
gpg --full-generate-key

# List your keys to find the Key ID
gpg --list-secret-keys --keyid-format=long
```

#### 2. Configure Git to Use GPG Signing

```bash
# Enable GPG signing (choose one):
git config --global commit.gpgsign true  # For all repositories
git config commit.gpgsign true           # For this repository only

# Set your GPG key ID (replace YOUR_KEY_ID with actual key ID):
git config --global user.signingkey YOUR_KEY_ID  # Global setting
git config user.signingkey YOUR_KEY_ID           # Repository setting
```

#### 3. Add Your GPG Key to GitHub

1. Export your public key:
   ```bash
   gpg --armor --export YOUR_KEY_ID
   ```

2. Copy the output and add it to your GitHub account:
   - Go to GitHub → Settings → SSH and GPG keys → New GPG key
   - Paste your public key and save

#### 4. Test Your Setup

```bash
# Create a test commit
echo "test" > test.txt
git add test.txt
git commit -m "test: GPG signing verification"

# Verify the signature
git log --show-signature -1
```

## 🛠️ Pre-commit Hook

The repository includes a pre-commit hook that automatically:

- ✅ Checks if GPG signing is enabled
- ✅ Verifies a valid signing key is configured  
- ✅ Tests that the GPG key can actually sign commits
- ❌ **Blocks unsigned commits**

### Hook Installation

The pre-commit hook is installed automatically when you run:
```bash
./scripts/setup-gpg-hook.sh
```

### Manual Hook Installation

If you need to install the hook manually:
```bash
# Copy the hook
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit
```

## 🔧 Troubleshooting

### Common Issues

#### "GPG signing is not enabled!"
```bash
# Solution:
git config commit.gpgsign true
```

#### "No GPG signing key configured!"
```bash
# List your keys:
gpg --list-secret-keys --keyid-format=long

# Configure the key:
git config user.signingkey YOUR_KEY_ID
```

#### "Cannot sign with the configured GPG key"
```bash
# Check if key exists:
gpg --list-secret-keys YOUR_KEY_ID

# Test signing:
echo "test" | gpg --clearsign --default-key YOUR_KEY_ID

# Restart GPG agent if needed:
gpgconf --kill gpg-agent
gpgconf --launch gpg-agent
```

#### Commits show as "Unverified" on GitHub
- Make sure you've added your GPG public key to your GitHub account
- Verify the email in your GPG key matches your Git commit email
- Check that your key hasn't expired

### Getting Help

If you encounter issues:

1. **Run the setup script** - it provides detailed diagnostics:
   ```bash
   ./scripts/setup-gpg-hook.sh
   ```

2. **Check the hook output** when committing - it provides specific error messages and solutions

3. **Verify your GPG setup**:
   ```bash
   # List available keys
   gpg --list-secret-keys --keyid-format=long
   
   # Check Git configuration
   git config --list | grep -E "(gpgsign|signingkey)"
   
   # Test GPG signing
   echo "test" | gpg --clearsign
   ```

## 📋 Repository Policy

- **All commits MUST be GPG signed**
- **The pre-commit hook CANNOT be bypassed** (no `--no-verify` commits)
- **Contributors must have verified GPG keys on GitHub**
- **Unsigned commits will be rejected automatically**

This policy ensures the security and integrity of the codebase for all contributors and users.

## 🔗 Additional Resources

- [GitHub GPG Documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification)
- [Git GPG Signing Documentation](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [GPG Quick Start Guide](https://www.gnupg.org/gph/en/manual/c14.html)
