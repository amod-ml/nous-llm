# PyPI Trusted Publishing Setup Instructions

## Current Status
✅ GitHub Actions workflow is configured and working
✅ GitHub release created with artifacts
❌ PyPI trusted publishing needs to be configured

## Setup Instructions

### For TestPyPI (Testing)

1. Go to https://test.pypi.org/manage/account/publishing/
2. Click "Add a new publisher"
3. Fill in these EXACT values:
   - **PyPI Project Name**: `nous-llm`
   - **Owner**: `amod-ml`
   - **Repository name**: `nous-llm`
   - **Workflow name**: `release.yml`
   - **Environment name**: `test-pypi`

### For PyPI (Production)

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new publisher"
3. Fill in these EXACT values:
   - **PyPI Project Name**: `nous-llm`
   - **Owner**: `amod-ml`
   - **Repository name**: `nous-llm`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

## Important Notes

- The environment names (`test-pypi` and `pypi`) must match EXACTLY what's in the workflow
- The workflow name must be `release.yml` (not the full path)
- This setup allows GitHub Actions to publish without API tokens (more secure)

## Verification

After setup, the workflow error showed these claims:
```
* `sub`: `repo:amod-ml/nous-llm:environment:test-pypi`
* `repository`: `amod-ml/nous-llm`
* `repository_owner`: `amod-ml`
* `workflow_ref`: `amod-ml/nous-llm/.github/workflows/release.yml@refs/tags/v0.1.0`
```

These match what you'll configure in PyPI.

## Next Steps

Once configured on PyPI:
1. Create a new release tag (e.g., v0.1.1) to trigger the workflow
2. The workflow will automatically publish to TestPyPI and PyPI

## Manual Publishing (If Needed)

If you want to manually publish the current release:

```bash
# Download the artifacts from GitHub release
gh release download v0.1.0 --repo amod-ml/nous-llm

# Install twine
pip install twine

# Upload to TestPyPI (requires API token)
twine upload --repository testpypi nous_llm-0.1.0*

# Upload to PyPI (requires API token)
twine upload nous_llm-0.1.0*
```

## GitHub Release Page

The release is live at: https://github.com/amod-ml/nous-llm/releases/tag/v0.1.0

Artifacts available:
- `nous_llm-0.1.0-py3-none-any.whl` (33.46 KiB) - Wheel distribution
- `nous_llm-0.1.0.tar.gz` (119.73 KiB) - Source distribution
