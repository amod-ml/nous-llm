# PyPI Release Process

This document describes the release process for publishing `nous-llm` to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. **GitHub Repository Settings**: Configure trusted publishing (see below)
3. **GPG Signing**: Ensure commits and tags are signed (see GPG-SIGNING.md)

## Setting Up PyPI Trusted Publishing

PyPI trusted publishing uses OpenID Connect (OIDC) to authenticate GitHub Actions workflows without API tokens.

### Step 1: Configure PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher with these settings:
   - **PyPI Project Name**: `nous-llm`
   - **Owner**: `amod-ml`
   - **Repository name**: `nous-llm`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi` (leave blank if not using environments)

3. Repeat for TestPyPI at https://test.pypi.org/manage/account/publishing/

### Step 2: Configure GitHub Repository

1. Go to Settings → Environments
2. Create two environments:
   - `test-pypi` - For TestPyPI releases
   - `pypi` - For production PyPI releases

3. For the `pypi` environment, add protection rules:
   - Required reviewers (optional)
   - Deployment branches: Only selected branches → `main`

## Release Workflows

### Automated Release (Recommended)

1. **Manual Release Trigger**:
   ```bash
   # Trigger via GitHub UI: Actions → Manual Release → Run workflow
   # Or use gh CLI:
   gh workflow run manual-release.yml -f release_type=patch
   gh workflow run manual-release.yml -f release_type=minor
   gh workflow run manual-release.yml -f release_type=major
   ```

2. **Tag-based Release**:
   ```bash
   # Create and push a version tag
   git tag -a v0.1.1 -m "Release v0.1.1"
   git push origin v0.1.1
   ```

### Release via GitHub CLI

```bash
# Create a release with the gh CLI
gh release create v0.1.1 \
  --title "v0.1.1" \
  --notes "Release notes here" \
  --verify-tag
```

## Release Process Flow

1. **Version Bump**: Update version in `pyproject.toml` and `__init__.py`
2. **Build**: GitHub Actions builds wheel and sdist
3. **Test**: Install and test on multiple platforms
4. **TestPyPI**: Publish to TestPyPI first
5. **PyPI**: Publish to production PyPI
6. **GitHub Release**: Create release with artifacts

## Version Management

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Pre-releases

For alpha/beta/rc releases:
```bash
gh workflow run manual-release.yml \
  -f release_type=prerelease \
  -f prerelease_token=alpha
```

## Troubleshooting

### Failed PyPI Upload

1. Check the GitHub Actions logs
2. Verify trusted publishing is configured
3. Ensure the package name is available
4. Check for existing version conflicts

### TestPyPI Issues

TestPyPI may have dependency resolution issues. This is normal and doesn't affect production releases.

### GPG Signing Issues

Ensure your commits and tags are signed:
```bash
git config --global commit.gpgsign true
git config --global tag.gpgsign true
```

## Manual Package Building

If needed, build locally:
```bash
# Install build tools
uv add --dev build twine

# Build distributions
python -m build

# Check distributions
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI (use trusted publishing instead)
# twine upload dist/*
```

## Post-Release Checklist

- [ ] Verify package on PyPI: https://pypi.org/project/nous-llm/
- [ ] Test installation: `pip install nous-llm`
- [ ] Update documentation if needed
- [ ] Announce release (if applicable)
- [ ] Plan next release milestones

## Rollback Procedure

If a release has critical issues:

1. **Yank the release** on PyPI (doesn't delete, marks as "avoid")
2. **Fix the issue** in a new commit
3. **Create a patch release** with the fix
4. **Document the issue** in the changelog

## Security Considerations

- Never commit API tokens or secrets
- Use trusted publishing (OIDC) instead of API tokens
- Keep GitHub Actions workflows minimal and auditable
- Review all dependency updates before releasing
- Sign all commits and tags with GPG
