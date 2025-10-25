# Testing Guide

**Last Updated**: 2025-10-25
**Maintainer**: Development Team

## Overview

This guide covers the complete testing infrastructure for the qwen-image-finetune project, including test data management with HuggingFace Hub integration, data loading utilities, and testing best practices.

## Quick Start

### Running Tests

```bash
# Run all tests (resources auto-download on first run)
pytest tests/

# Run specific test file
pytest tests/e2e/test_flux_sampling.py -v

# Skip resource download (use local cache only)
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

### Writing Tests with Data

```python
from tests.utils.data_loader import load_flux_transformer_input

def test_example(test_resources):
    """test_resources fixture provides path to test data directory"""
    # Load data using convenience functions
    data = load_flux_transformer_input(test_resources)

    # Use the data in your test
    assert "latent_ids" in data
    assert data["prompt_embeds"].dim() == 3
```

## Test Resources Architecture

### How It Works

```
Test Execution
  ↓
Use test_resources fixture (from conftest.py)
  ↓
Check local cache (tests/resources/)
  ↓
If not exists → Download from HuggingFace Hub
  ↓
Return Path object to tests/resources/
  ↓
Test loads data using this path
```

### Data Storage

- **Remote**: HuggingFace Hub repository [`TsienDragon/qwen-image-finetune-test-resources`](https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources)
- **Local Cache**: `tests/resources/` (auto-created, gitignored)
- **Configuration**: `tests/resources_config.yaml`
- **Total Size**: ~50MB

### Directory Structure

```
tests/
├── conftest.py                      # test_resources fixture definition
├── resources_config.yaml            # Resource configuration
├── resources/                       # Local cache (gitignored)
│   ├── flux_models/
│   │   └── transformer/input/
│   ├── flux_training/
│   │   └── face_segmentation/
│   │       ├── sample1/
│   │       └── sample2/
│   ├── flux_sampling/embeddings/
│   ├── qwen_sampling/embeddings/
│   ├── qwen_plus_sampling/embeddings/
│   └── reference_outputs/images/
├── utils/
│   ├── test_resources.py           # Download logic
│   └── data_loader.py              # Data loading utilities
├── src/                            # Unit tests for source code
├── e2e/                            # End-to-end tests
└── test_configs/                   # Test configuration files
```

## Data Loading Utilities

### Core Functions

#### `load_torch_file` - Load Single File

```python
from tests.utils.data_loader import load_torch_file

def test_single_file(test_resources):
    data = load_torch_file(
        test_resources,
        "flux_models/transformer/input/flux_input.pth",
        map_location="cpu",
        weights_only=True
    )
```

#### `load_torch_directory` - Batch Load Directory

```python
from tests.utils.data_loader import load_torch_directory

def test_directory(test_resources):
    # Loads all .pt files in directory into a dict
    data = load_torch_directory(
        test_resources,
        "flux_training/face_segmentation/sample1",
        pattern="*.pt"
    )

    # Keys are filenames without extension
    control_ids = data["sample_control_ids"]
    noise = data["sample_noise"]
```

### Preset Convenience Functions

#### Flux Model Tests

```python
from tests.utils.data_loader import (
    load_flux_transformer_input,
    load_flux_training_sample,
    load_flux_sampling_embeddings
)

# Flux Transformer input
data = load_flux_transformer_input(test_resources)

# Flux training samples
sample1 = load_flux_training_sample(test_resources, "sample1")
sample2 = load_flux_training_sample(test_resources, "sample2")

# Flux sampling embeddings
embeddings = load_flux_sampling_embeddings(test_resources)
```

#### Qwen Model Tests

```python
from tests.utils.data_loader import (
    load_qwen_sampling_embeddings,
    load_qwen_plus_sampling_embeddings
)

# Qwen-Image-Edit sampling
qwen_data = load_qwen_sampling_embeddings(test_resources)

# Qwen-Image-Edit-Plus sampling
qwen_plus_data = load_qwen_plus_sampling_embeddings(test_resources)
```

### Device and Dtype Conversion

```python
from tests.utils.data_loader import prepare_test_data_for_device

def test_with_gpu(test_resources):
    # Load data (CPU by default)
    data = load_flux_training_sample(test_resources, "sample1")

    # Move to GPU and convert dtype
    data_gpu = prepare_test_data_for_device(
        data,
        device="cuda:0",
        dtype=torch.bfloat16,
        exclude_keys=["sample_loss"]  # Keep certain keys on CPU
    )

    assert data_gpu["sample_latent_model_input"].device.type == "cuda"
```

## Resource Configuration

### Configuration File (`tests/resources_config.yaml`)

```yaml
# HuggingFace repository
repository:
  repo_id: "TsienDragon/qwen-image-finetune-test-resources"
  repo_type: "dataset"
  revision: "main"

# Resource groups
resource_groups:
  flux_input:
    description: "Flux transformer input test data"
    files:
      - "flux_models/transformer/input/flux_input.pth"
    size: "19MB"
    used_by:
      - "tests/src/models/test_flux_transform_custom.py"

# Test dependencies
test_dependencies:
  "tests/src/models/test_flux_transform_custom.py":
    - flux_input
```

### Available Resource Groups

| Resource Group | Size | Description | Used By |
|---------------|------|-------------|---------|
| `flux_input` | 19MB | Flux transformer input | `test_flux_transform_custom.py` |
| `flux_training_sample1` | 11MB | Training sample 1 | `test_flux_loss.py`, `test_transformer_consistency.py` |
| `flux_training_sample2` | 12MB | Training sample 2 | `test_flux_loss.py`, `test_transformer_consistency.py` |
| `flux_sampling` | 7.3MB | Flux sampling embeddings | `test_flux_sampling.py` |
| `qwen_sampling` | TBD | Qwen-Image-Edit sampling | `test_qwen_image_edit_sampling.py` |
| `qwen_plus_sampling` | TBD | Qwen-Image-Edit-Plus sampling | `test_qwen_image_edit_plus_sampling.py` |
| `reference_outputs` | 688KB | Reference output images | `test_flux_sampling.py` |

## Managing Test Resources

### Adding New Test Resources

1. **Organize files locally**

```bash
# Put files in tests/resources/ or a separate staging directory
mkdir -p tests/resources/my_new_data/
cp my_data_files/* tests/resources/my_new_data/
```

2. **Update configuration**

Edit `tests/resources_config.yaml`:

```yaml
resource_groups:
  my_new_data:
    description: "Description of new data"
    files:
      - "my_new_data/file1.pt"
      - "my_new_data/file2.pt"
    size: "5MB"
    used_by:
      - "tests/my_new_test.py"

test_dependencies:
  "tests/my_new_test.py":
    - my_new_data
```

3. **Upload to HuggingFace Hub**

```bash
# Set your HuggingFace token
export HF_TOKEN=your_huggingface_token

# Upload using the script
python scripts/upload_test_resources.py --resources-dir tests/resources
```

The script will:
- Create/update the HuggingFace repository
- Upload all files from the specified directory
- Preserve directory structure

4. **Verify upload**

Visit: https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources

5. **Test automatic download**

```bash
# Delete local cache
rm -rf tests/resources/

# Run test (should auto-download)
pytest tests/my_new_test.py -v
```

### Manual Resource Management

```python
from tests.utils.test_resources import (
    ensure_test_resources,
    download_resource_groups,
    download_all_resources
)

# Download all resources
ensure_test_resources(download_all=True)

# Download specific resource groups
download_resource_groups(['flux_input', 'flux_sampling'])

# For specific test file
ensure_test_resources(test_file='tests/e2e/test_flux_sampling.py')

# Force re-download
download_all_resources(force_download=True)
```

## Complete Examples

### Example 1: Testing Model Forward Pass

```python
import pytest
import torch
from tests.utils.data_loader import (
    load_flux_transformer_input,
    prepare_test_data_for_device
)

class TestFluxTransformer:
    @pytest.fixture(scope="class")
    def test_data(self, test_resources):
        """Load test data once for the class"""
        return load_flux_transformer_input(test_resources)

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_pass(self, test_data, device):
        """Test model forward pass"""
        # Prepare data for device
        data = prepare_test_data_for_device(test_data, device, torch.bfloat16)

        # Run test
        model_input = data["latent_model_input"]
        # ... your test logic
```

### Example 2: Testing Training Loss

```python
from tests.utils.data_loader import load_flux_training_sample

@pytest.fixture
def sample_data(test_resources):
    return load_flux_training_sample(test_resources, "sample1")

def test_loss_computation(sample_data):
    """Test loss computation against reference"""
    image_latents = sample_data["sample_image_latents"]
    noise = sample_data["sample_noise"]
    expected_loss = sample_data["sample_loss"]

    # Compute loss
    target = noise - image_latents
    # ... compute and compare with expected_loss
```

### Example 3: Testing Sampling

```python
from tests.utils.data_loader import load_flux_sampling_embeddings

def test_sampling(test_resources):
    """Test sampling with precomputed embeddings"""
    embeddings = load_flux_sampling_embeddings(test_resources)

    # Use embeddings for sampling
    control_latents = embeddings["sample_control_latents"]
    prompt_embeds = embeddings["sample_prompt_embeds"]

    # Run sampling
    # ... your test logic
```

## Best Practices

### DO ✅

- Use `test_resources` fixture for all test data
- Use data loader utility functions (`load_torch_file`, `load_torch_directory`)
- Add new test data to HuggingFace Hub, not git
- Document test data in `resources_config.yaml`
- Keep test data files reasonable size (< 100MB per file)
- Use `prepare_test_data_for_device` for GPU tests

### DON'T ❌

- Don't commit test data to git repository
- Don't hardcode absolute paths in tests
- Don't manually download test data
- Don't duplicate test data across multiple locations
- Don't create test data on-the-fly if it can be reused

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SKIP_DOWNLOAD_TEST_RESOURCES` | Skip automatic download | `0` (enabled) |

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Cache test resources
  uses: actions/cache@v3
  with:
    path: tests/resources
    key: test-resources-${{ hashFiles('tests/resources_config.yaml') }}

- name: Run tests
  run: pytest tests/
  env:
    SKIP_DOWNLOAD_TEST_RESOURCES: ${{ steps.cache.outputs.cache-hit }}
```

## Troubleshooting

### Issue: Tests fail with "FileNotFoundError"

**Solution**: Ensure test data is downloaded

```bash
rm -rf tests/resources/
pytest tests/your_test.py -v
```

### Issue: Download is slow

**Solution**: Use local cache or CI cache

```bash
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

### Issue: Key name mismatch in loaded data

**Solution**: Check file naming
- Files loaded with `load_torch_directory()` use filename (without extension) as key
- Example: `sample_noise.pt` → key is `"sample_noise"`

### Issue: Cannot access HuggingFace Hub

**Solution**: Use local resources
1. Get `tests/resources/` directory from another source
2. Place in your project
3. Set `export SKIP_DOWNLOAD_TEST_RESOURCES=1`

## Related Documentation

- **PyTest Documentation**: https://docs.pytest.org/
- **HuggingFace Hub**: https://huggingface.co/docs/hub/
- **Test Resources Repository**: https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources
- **Testing Quick Reference**: `docs/guide/testing-quick-reference.md`

## Maintenance

### Regular Tasks

- Review and clean up unused test data (quarterly)
- Update documentation when adding new features
- Monitor HuggingFace Hub repository size
- Ensure CI cache is working properly

### When to Update

- Adding new test scenarios
- Changing test data format
- Updating test infrastructure
- Fixing test data issues
