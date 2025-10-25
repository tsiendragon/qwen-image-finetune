# Test Data Management

## Overview

All test data for this project is stored and managed through **HuggingFace Hub**, not in the git repository. This approach keeps the repository lightweight while ensuring reproducible tests.

## Data Source

- **Repository**: [TsienDragon/qwen-image-finetune-test-resources](https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources)
- **Type**: HuggingFace Dataset
- **Configuration**: `tests/resources_config.yaml`

## How It Works

### Automatic Download

When you run tests, the `test_resources` fixture (defined in `tests/conftest.py`) automatically:

1. **First run**: Downloads required test data from HuggingFace Hub
2. **Subsequent runs**: Uses locally cached data from `tests/resources/`
3. **Smart caching**: Only downloads what's needed for the specific test

### Resource Groups

Test resources are organized into logical groups in `tests/resources_config.yaml`:

- `flux_input`: Flux transformer input test data (19MB)
- `flux_training_sample1`: Training sample 1 for face segmentation (11MB)
- `flux_training_sample2`: Training sample 2 for face segmentation (12MB)
- `flux_sampling`: Sampling and generation test data (7.3MB)
- `reference_outputs`: Reference output images (688KB)

### Test Dependencies

Each test file declares which resource groups it needs:

```yaml
test_dependencies:
  "tests/src/models/test_flux_transform_custom.py":
    - flux_input

  "tests/e2e/test_flux_loss.py":
    - flux_training_sample1
    - flux_training_sample2
```

## Usage

### Running Tests

Simply run pytest as usual:

```bash
pytest tests/src/models/test_flux_transform_custom.py
```

The first time you run a test, you'll see:

```
🔍 Checking test resources from HuggingFace Hub...
📦 Test test_flux_transform_custom.py requires groups: flux_input
🔄 Checking/downloading from HuggingFace Hub...
✅ Test resources ready at: /path/to/tests/resources
```

Subsequent runs will use cached data:

```
✅ Using cached test resources from: /path/to/tests/resources
```

### Environment Variables

- `SKIP_DOWNLOAD_TEST_RESOURCES=1`: Skip download check, use local cache only (useful in CI)

```bash
SKIP_DOWNLOAD_TEST_RESOURCES=1 pytest tests/
```

### Force Re-download

To force re-download all resources:

```python
from tests.utils.test_resources import download_all_resources
download_all_resources(force_download=True)
```

## Adding New Test Data

### 1. Upload to HuggingFace

Upload your test data to the HuggingFace dataset repository:

```bash
# Using huggingface_hub CLI
huggingface-cli upload TsienDragon/qwen-image-finetune-test-resources \
  /local/path/to/data.pt \
  flux_models/new_test/data.pt
```

### 2. Update Configuration

Add the new resource group to `tests/resources_config.yaml`:

```yaml
resource_groups:
  new_test_group:
    description: "Description of the new test data"
    files:
      - "flux_models/new_test/data.pt"
    size: "10MB"
    used_by:
      - "tests/path/to/test_file.py"

test_dependencies:
  "tests/path/to/test_file.py":
    - new_test_group
```

### 3. Use in Tests

Access the data through the `test_resources` fixture:

```python
@pytest.fixture
def my_test_data(test_resources):
    """Load test data from HuggingFace Hub."""
    data_path = test_resources / "flux_models" / "new_test" / "data.pt"
    return torch.load(data_path, map_location="cpu")
```

## Benefits

✅ **Lightweight Repository**: No large binary files in git
✅ **Version Control**: Test data versioned on HuggingFace Hub
✅ **Reproducibility**: Everyone uses the same test data
✅ **Efficient**: Smart caching and selective downloads
✅ **CI-Friendly**: Can pre-cache in CI environments

## Troubleshooting

### Download Fails

If download fails, the system will try to use local cache if available:

```
⚠️  Failed to download from HuggingFace, using local cache: /path/to/tests/resources
```

### Missing Resources

If no local cache exists and download fails:

```
RuntimeError: Failed to download test resources from HuggingFace and no local resources found
```

**Solution**: Check your internet connection and HuggingFace Hub access.

### Clear Cache

To clear cached test data:

```bash
rm -rf tests/resources/
```

Next test run will re-download from HuggingFace Hub.

## Implementation Details

### Key Files

- `tests/conftest.py`: Defines `test_resources` fixture
- `tests/utils/test_resources.py`: Download and caching logic
- `tests/resources_config.yaml`: Resource metadata and dependencies
- `tests/resources/`: Local cache directory (git-ignored)

### Download Flow

```
pytest run
    ↓
test_resources fixture (conftest.py)
    ↓
ensure_test_resources() (test_resources.py)
    ↓
Check local cache
    ↓
If missing → download_resource_groups()
    ↓
huggingface_hub.hf_hub_download()
    ↓
Cache to tests/resources/
    ↓
Return path to test
```

## References

- HuggingFace Hub Documentation: https://huggingface.co/docs/huggingface_hub
- Dataset Repository: https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources
