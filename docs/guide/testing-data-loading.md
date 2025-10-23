# Testing Data Loading Guide

**Last Updated**: 2024-10-23
**Maintainer**: Development Team
**Purpose**: Developer guide for loading test data from HuggingFace Hub in pytest

## Overview

This guide explains how to load test data in pytest tests using the automated HuggingFace Hub download system.

Test data is centrally stored in HuggingFace Hub and automatically downloaded/cached locally:
- **Repository**: `TsienDragon/qwen-image-finetune-test-resources`
- **Type**: dataset
- **Configuration**: `tests/resources_config.yaml`

## Quick Start

### 1. Using the `test_resources` Fixture

All tests can use the global `test_resources` fixture to access the test resources directory:

```python
def test_example(test_resources):
    """test_resources is a Path object pointing to tests/resources/"""
    # Method 1: Direct path construction
    data_file = test_resources / "flux_models/transformer/input/flux_input.pth"
    data = torch.load(data_file, map_location="cpu")

    # Method 2: Using data loader utility functions (recommended)
    from tests.utils.data_loader import load_flux_transformer_input
    data = load_flux_transformer_input(test_resources)
```

### 2. Using Data Loader Utility Functions

`tests/utils/data_loader.py` provides convenient functions for loading common test data.

## Core Data Loading Functions

### `load_torch_file` - Load Single File

```python
from tests.utils.data_loader import load_torch_file

def test_single_file(test_resources):
    """Load a single PyTorch file"""
    data = load_torch_file(
        test_resources,
        "flux_models/transformer/input/flux_input.pth",
        map_location="cpu",
        weights_only=True
    )
```

**Parameters**:
- `test_resources`: Test resources root directory (from fixture)
- `relative_path`: File path relative to resources directory
- `map_location`: torch.load map_location parameter (default: "cpu")
- `weights_only`: Whether to load weights only (default: True, safer)

### `load_torch_directory` - Batch Load Directory

```python
from tests.utils.data_loader import load_torch_directory

def test_directory(test_resources):
    """Batch load all .pt files in a directory"""
    data = load_torch_directory(
        test_resources,
        "flux_training/face_segmentation/sample1",
        pattern="*.pt"
    )

    # data is a dict with keys as filenames (without extension)
    control_ids = data["sample_control_ids"]
    noise = data["sample_noise"]
    image_latents = data["sample_image_latents"]
```

**Parameters**:
- `test_resources`: Test resources root directory
- `relative_dir`: Directory path relative to resources directory
- `pattern`: Filename pattern (default: "*.pt")
- `map_location`: torch.load map_location parameter
- `weights_only`: Whether to load weights only

## Preset Convenience Functions

For common test scenarios, preset convenience functions with predefined paths are provided.

### `load_flux_transformer_input` - Flux Transformer Input

```python
from tests.utils.data_loader import load_flux_transformer_input

def test_transformer(test_resources):
    """Load Flux Transformer input test data"""
    data = load_flux_transformer_input(test_resources)

    # Data contains the following keys:
    # - latent_ids, latent_model_input, timestep_input
    # - prompt_embeds, pooled_prompt_embeds
    # - text_ids, guidance, full_attention_mask
    latent_ids = data["latent_ids"]
    prompt_embeds = data["prompt_embeds"]
```

**Use Case**: `tests/src/models/test_flux_transform_custom.py`

### `load_flux_training_sample` - Flux Training Sample

```python
from tests.utils.data_loader import load_flux_training_sample

def test_training(test_resources):
    """Load Flux training sample data"""
    sample1 = load_flux_training_sample(test_resources, "sample1")
    sample2 = load_flux_training_sample(test_resources, "sample2")

    # Each sample contains the following keys:
    # - sample_control_ids, sample_control_latents
    # - sample_image_latents, sample_noise
    # - sample_prompt_embeds, sample_pooled_prompt_embeds
    # - sample_text_ids, sample_model_pred
    # - sample_loss, sample_latent_model_input, sample_t
    # - sample_guidance, sample_latent_ids
    image_latents = sample1["sample_image_latents"]
    expected_loss = sample1["sample_loss"]
```

**Use Case**: `tests/e2e/test_flux_loss.py`

### `load_flux_sampling_embeddings` - Flux Sampling Embeddings

```python
from tests.utils.data_loader import load_flux_sampling_embeddings

def test_sampling(test_resources):
    """Load Flux sampling embedding data"""
    embeddings = load_flux_sampling_embeddings(test_resources)

    # embeddings contains:
    # - sample_control_latents, sample_latents
    # - sample_prompt_embeds, sample_pooled_prompt_embeds
    # - sample_text_ids, sample_control_ids, sample_latent_ids
    control_latents = embeddings["sample_control_latents"]
```

**Use Case**: `tests/e2e/test_flux_sampling.py`

## Data Movement to Device/Dtype Conversion

### `prepare_test_data_for_device` - Prepare Data

```python
from tests.utils.data_loader import (
    load_flux_training_sample,
    prepare_test_data_for_device
)

def test_with_gpu(test_resources):
    """Load data and move to GPU"""
    # Load data (CPU)
    data = load_flux_training_sample(test_resources, "sample1")

    # Move to GPU and convert dtype
    data_gpu = prepare_test_data_for_device(
        data,
        device="cuda:0",
        dtype=torch.bfloat16,
        exclude_keys=["sample_loss"]  # Keep certain keys on CPU
    )

    # Use GPU data for testing
    model_input = data_gpu["sample_latent_model_input"]
    assert model_input.device.type == "cuda"
    assert model_input.dtype == torch.bfloat16
```

### `move_dict_to_device` - In-place Movement

```python
from tests.utils.data_loader import move_dict_to_device

def test_move_inplace(test_resources):
    """Move data to device in-place (modifies original dict)"""
    data = {"tensor1": torch.randn(10), "tensor2": torch.randn(20)}

    # Modify data in-place
    move_dict_to_device(
        data,
        device="cuda",
        dtype=torch.float16,
        keys=["tensor1"]  # Only move specified keys
    )
```

## Complete Examples

### Example 1: Testing Transformer Model

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
        """Load test data"""
        return load_flux_transformer_input(test_resources)

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    def test_forward_pass(self, test_data, device, dtype):
        """Test forward pass"""
        # Prepare data
        data = prepare_test_data_for_device(test_data, device, dtype)

        # Run model
        # ... your test code
```

### Example 2: Testing Loss Computation

```python
import pytest
from tests.utils.data_loader import load_flux_training_sample

@pytest.fixture
def sample_data_1(test_resources):
    """Load sample 1"""
    return load_flux_training_sample(test_resources, "sample1")

@pytest.fixture
def sample_data_2(test_resources):
    """Load sample 2"""
    return load_flux_training_sample(test_resources, "sample2")

class TestLossComputation:
    def test_loss_sample1(self, sample_data_1):
        """Test loss computation for sample 1"""
        image_latents = sample_data_1["sample_image_latents"]
        noise = sample_data_1["sample_noise"]
        expected_loss = sample_data_1["sample_loss"]

        # Compute loss
        target = noise - image_latents
        # ... your test code

    def test_loss_sample2(self, sample_data_2):
        """Test loss computation for sample 2"""
        # Similar test logic
        pass
```

### Example 3: Custom Data Loading

```python
from tests.utils.data_loader import load_torch_file, load_torch_directory

def test_custom_data(test_resources):
    """Load custom data structure"""
    # Load specific file
    control_ids = load_torch_file(
        test_resources,
        "flux_training/face_segmentation/sample1/sample_control_ids.pt"
    )

    # Batch load
    all_samples = load_torch_directory(
        test_resources,
        "flux_training/face_segmentation/sample1",
        pattern="sample_*.pt"
    )
```

## Test Resources Configuration

Test resources are configured in `tests/resources_config.yaml`:

```yaml
# Resource group definition
resource_groups:
  flux_input:
    description: "Flux transformer input test data"
    files:
      - "flux_models/transformer/input/flux_input.pth"
    used_by:
      - "tests/src/models/test_flux_transform_custom.py"

# Test dependencies
test_dependencies:
  "tests/src/models/test_flux_transform_custom.py":
    - flux_input
```

### Adding New Test Resources

1. Upload data to HuggingFace Hub repository
2. Add resource group definition in `resources_config.yaml`
3. Declare test file dependencies in `test_dependencies`
4. Use data loader functions in tests to load data

## Environment Variables

### `SKIP_DOWNLOAD_TEST_RESOURCES`

Skip automatic download and use local cache directly (useful for CI environments):

```bash
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

## Common Issues

### Q: Do I need to download data on first test run?

A: Yes. On first run, data will automatically download from HuggingFace Hub. Subsequent runs will use local cache (`tests/resources/`).

### Q: How to force re-download test data?

A: Delete the `tests/resources/` directory, or call `download_all_resources(force_download=True)` in code.

### Q: Where is test data stored?

A:
- **Remote**: HuggingFace Hub repository `TsienDragon/qwen-image-finetune-test-resources`
- **Local**: Project directory `tests/resources/` (gitignored)

### Q: How to view available test resources?

A: Check the `resource_groups` section in `tests/resources_config.yaml`.

### Q: Will tests automatically download if I don't need test resources?

A: No. Only test files declared in `resources_config.yaml` dependencies will trigger downloads.

## Reference

- **Data loader utilities**: `tests/utils/data_loader.py`
- **Test resource management**: `tests/utils/test_resources.py`
- **Pytest configuration**: `tests/conftest.py`
- **Resource configuration**: `tests/resources_config.yaml`
- **HuggingFace repository**: https://huggingface.co/datasets/TsienDragon/qwen-image-finetune-test-resources
- **Architecture explanation**: `docs/guide/test-resources-architecture.md`
