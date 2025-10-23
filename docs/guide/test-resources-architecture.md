# Test Resources Architecture

**Last Updated**: 2024-10-23
**Maintainer**: Development Team
**Purpose**: Technical explanation of test resources system architecture

## What is test_resources?

`test_resources` is a **pytest fixture** that automatically downloads test data from HuggingFace Hub and returns the local cache directory path.

## Architecture Overview

```
Test Execution
  ↓
Use test_resources fixture
  ↓
Check local cache (tests/resources/)
  ↓
If not exists → Download from HuggingFace Hub
  ↓
Return local directory path (Path object)
  ↓
Test uses this path to load data
```

## Detailed Workflow

### 1. Fixture Definition Location

```python
# tests/conftest.py

@pytest.fixture(scope="session", autouse=True)
def test_resources():
    """
    Automatically ensure test resources are downloaded from HuggingFace Hub.

    Data Source:
        - HuggingFace Repository: TsienDragon/qwen-image-finetune-test-resources
        - Repository Type: dataset
        - Configuration: tests/resources_config.yaml
    """
    # ... implementation code
```

**Key Features**:
- `scope="session"`: Executed only once per test session (shared across all tests)
- `autouse=True`: Automatically used even if test functions don't explicitly declare it
- Return value: `Path` object pointing to local cache directory

### 2. Workflow Steps

#### Step 1: Check Environment Variable

```python
if os.environ.get("SKIP_DOWNLOAD_TEST_RESOURCES") == "1":
    # Skip download, use local cache directly
    resources_dir = Path(__file__).parent / "resources"  # tests/resources/
    return resources_dir
```

#### Step 2: Call Download Function

```python
from tests.utils.test_resources import ensure_test_resources

resources_dir = ensure_test_resources()
# Returns: /path/to/project/tests/resources/
```

#### Step 3: Download Logic (tests/utils/test_resources.py)

```python
def ensure_test_resources(test_file=None, download_all=False):
    """Ensure test resources are available, download from HuggingFace Hub if necessary"""

    resources_dir = get_test_resources_dir()  # tests/resources/

    # If exists locally, return directly
    if resources_dir.exists() and any(resources_dir.iterdir()):
        return resources_dir

    # Otherwise download from HuggingFace Hub
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="TsienDragon/qwen-image-finetune-test-resources",
        repo_type="dataset",
        local_dir=resources_dir,
        local_dir_use_symlinks=False,
    )

    return resources_dir
```

## Data Organization

### HuggingFace Hub Repository Structure

```
TsienDragon/qwen-image-finetune-test-resources (dataset)
├── flux_models/
│   └── transformer/
│       └── input/
│           └── flux_input.pth          # Transformer input data
├── flux_training/
│   └── face_segmentation/
│       ├── sample1/
│       │   ├── sample_control_ids.pt
│       │   ├── sample_noise.pt
│       │   ├── sample_image_latents.pt
│       │   └── ... (13 files total)
│       └── sample2/
│           └── ... (same 13 files)
└── flux_sampling/
    └── embeddings/
        ├── sample_control_latents.pt
        ├── sample_prompt_embeds.pt
        └── ...
```

### Local Cache Directory Structure

After download, data is cached locally:

```
qwen-image-finetune/
├── tests/
│   ├── resources/                    ← test_resources returns this path
│   │   ├── flux_models/
│   │   │   └── transformer/
│   │   │       └── input/
│   │   │           └── flux_input.pth
│   │   ├── flux_training/
│   │   │   └── face_segmentation/
│   │   │       ├── sample1/
│   │   │       └── sample2/
│   │   └── flux_sampling/
│   │       └── embeddings/
│   ├── conftest.py                   ← test_resources fixture definition
│   ├── resources_config.yaml         ← Resource configuration
│   └── utils/
│       ├── test_resources.py         ← Download logic
│       └── data_loader.py            ← Data loading utilities
└── .gitignore                        ← Contains tests/resources/
```

**Note**: `tests/resources/` directory is in `.gitignore` and not committed to git.

## Usage Examples

### Example 1: Basic Usage

```python
def test_example(test_resources):
    """test_resources is a Path object"""

    # Method 1: Direct path construction
    data_file = test_resources / "flux_models/transformer/input/flux_input.pth"
    data = torch.load(data_file, map_location="cpu")

    # Method 2: Using helper functions (recommended)
    from tests.utils.data_loader import load_flux_transformer_input
    data = load_flux_transformer_input(test_resources)

    # test_resources actual value example:
    # PosixPath('/home/user/qwen-image-finetune/tests/resources')
```

### Example 2: Your Test File

```python
# tests/src/models/test_flux_transform_custom.py

class TestFluxTransformerEquivalence:
    @pytest.fixture(scope="class")
    def test_input_data(self, test_resources):
        """Load test input data

        test_resources automatically provided by conftest.py
        Type: pathlib.Path
        Points to: tests/resources/
        """
        # Use data loader utility
        data = load_flux_transformer_input(test_resources)
        # Equivalent to:
        # data_path = test_resources / "flux_models/transformer/input/flux_input.pth"
        # data = torch.load(data_path, map_location="cpu")

        return data

    def test_something(self, test_input_data):
        """Use loaded data for testing"""
        latent_ids = test_input_data["latent_ids"]
        # ... test logic
```

## Configuration File (resources_config.yaml)

```yaml
# HuggingFace repository configuration
repository:
  repo_id: "TsienDragon/qwen-image-finetune-test-resources"
  repo_type: "dataset"
  revision: "main"

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

This configuration defines:
1. Which HuggingFace repository to download from
2. What resource groups exist
3. Which test files need which resources

## First Run Workflow

```bash
# First test run
pytest tests/src/models/test_flux_transform_custom.py -v

# Output:
# 🔍 Checking test resources from HuggingFace Hub...
# 📥 Downloading all test resources from HuggingFace Hub...
# [download progress...]
# ✅ Test resources ready at: /path/to/tests/resources
# 📦 Loading test data from HuggingFace-cached resources
# ✅ Test data loaded successfully with 8 keys
# [test execution...]
```

## Subsequent Run Workflow

```bash
# Second and later test runs
pytest tests/src/models/test_flux_transform_custom.py -v

# Output:
# 🔍 Checking test resources from HuggingFace Hub...
# ✅ Using cached test resources from: /path/to/tests/resources
# 📦 Loading test data from HuggingFace-cached resources
# ✅ Test data loaded successfully with 8 keys
# [test execution...]
```

## Data Loading Utility Functions

To simplify data loading, convenience functions are provided:

```python
# tests/utils/data_loader.py

def load_flux_transformer_input(test_resources: Path) -> Dict[str, torch.Tensor]:
    """Load Flux Transformer input data (preset path)"""
    return load_torch_file(
        test_resources,
        "flux_models/transformer/input/flux_input.pth",
        map_location="cpu",
    )

def load_flux_training_sample(test_resources: Path, sample_name: str) -> Dict:
    """Load Flux training sample data"""
    return load_torch_directory(
        test_resources,
        f"flux_training/face_segmentation/{sample_name}",
        pattern="*.pt",
    )
```

## Common Questions

### Q1: What type is test_resources?

**A**: `pathlib.Path` object pointing to `tests/resources/` directory.

```python
def test_example(test_resources):
    print(type(test_resources))  # <class 'pathlib.Path'>
    print(test_resources)        # /path/to/qwen-image-finetune/tests/resources
```

### Q2: Where is data stored?

**A**:
- **Remote**: HuggingFace Hub repository `TsienDragon/qwen-image-finetune-test-resources`
- **Local**: Project directory `tests/resources/` (gitignored)

### Q3: How to add new test data?

**Steps**:
1. Upload data to HuggingFace Hub repository
2. Add resource group definition in `resources_config.yaml`
3. Use `load_torch_file` or `load_torch_directory` to load

### Q4: How to skip automatic download?

**A**: Set environment variable:

```bash
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

### Q5: How to force re-download?

**A**: Delete local cache directory:

```bash
rm -rf tests/resources/
pytest tests/  # Will re-download
```

## Summary

Core value of `test_resources`:

1. **Automation**: No manual data download needed, pytest handles it automatically
2. **Caching**: Download once, use local cache thereafter
3. **Version control friendly**: Test data not committed to git, reducing repository size
4. **Ease of use**: Injected via fixture, test code stays clean
5. **Sharing**: Multiple tests share the same data

**Key Understanding**:
- `test_resources` = Local cache directory path (`Path` object)
- Data source = HuggingFace Hub repository
- First run = Automatic download
- Subsequent runs = Use cache

## Related Documentation

- **Usage Guide**: `docs/guide/testing-data-loading.md`
- **Data Loader Utilities**: `tests/utils/data_loader.py`
- **Test Resource Management**: `tests/utils/test_resources.py`
