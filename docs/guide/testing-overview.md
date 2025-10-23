# Testing Overview

**Last Updated**: 2024-10-23
**Maintainer**: Development Team
**Purpose**: Overview of testing infrastructure and guidelines

## Testing Documentation Index

### Core Testing Guides

1. **[Testing Data Loading Guide](testing-data-loading.md)**
   - How to load test data from HuggingFace Hub
   - Data loader utility functions
   - Complete usage examples
   - **Audience**: All developers writing tests

2. **[Test Resources Architecture](test-resources-architecture.md)**
   - Technical architecture of test resources system
   - How `test_resources` fixture works
   - Data organization and caching
   - **Audience**: Developers maintaining test infrastructure

### Testing Standards

3. **PyTest Testing Standards** (`tests/conftest.py` docstring)
   - Test naming conventions
   - Test organization (unit/integration/e2e)
   - Fixture usage guidelines
   - **Audience**: All developers

### Test Utilities

4. **Data Loading Utilities** (`tests/utils/data_loader.py`)
   - `load_torch_file()` - Load single file
   - `load_torch_directory()` - Batch load directory
   - `load_flux_transformer_input()` - Preset function for Flux Transformer
   - `load_flux_training_sample()` - Preset function for training samples
   - `prepare_test_data_for_device()` - Move data to GPU/CPU

5. **Test Resource Management** (`tests/utils/test_resources.py`)
   - `ensure_test_resources()` - Main download function
   - `download_resource_groups()` - Download specific groups
   - `load_resources_config()` - Load configuration

## Quick Start

### For Test Writers

If you're writing tests and need to load test data:

1. Read: [Testing Data Loading Guide](testing-data-loading.md)
2. Use the `test_resources` fixture in your tests
3. Use data loader functions from `tests.utils.data_loader`

```python
from tests.utils.data_loader import load_flux_transformer_input

def test_example(test_resources):
    data = load_flux_transformer_input(test_resources)
    # Your test code here
```

### For Infrastructure Maintainers

If you're maintaining the test infrastructure:

1. Read: [Test Resources Architecture](test-resources-architecture.md)
2. Understand the download workflow
3. Know how to add new test resources

## Test Organization

```
tests/
├── conftest.py                       # Global fixtures and configuration
├── resources_config.yaml             # Test resources configuration
├── resources/                        # Local cache (gitignored)
├── utils/
│   ├── test_resources.py            # Download logic
│   ├── data_loader.py               # Data loading utilities
│   └── README.md                    # Utils documentation
├── unit/                            # Unit tests
├── integration/                     # Integration tests
├── e2e/                             # End-to-end tests
└── src/                             # Source code tests
```

## Test Data Sources

### HuggingFace Hub Repository

- **Repository**: `TsienDragon/qwen-image-finetune-test-resources`
- **Type**: dataset
- **Access**: Public
- **Size**: ~50MB total

### Local Cache

- **Location**: `tests/resources/`
- **Git Status**: Ignored (in `.gitignore`)
- **Persistence**: Permanent (until manually deleted)

## Common Workflows

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/e2e/test_flux_loss.py -v

# Run with verbose output
pytest tests/ -v -s

# Skip resource download (use local cache)
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

### Adding New Test Data

1. **Upload to HuggingFace Hub**
   ```bash
   # Upload your test data files to the repository
   ```

2. **Update Configuration**
   ```yaml
   # tests/resources_config.yaml
   resource_groups:
     my_new_data:
       description: "My new test data"
       files:
         - "path/to/my_data.pt"
       used_by:
         - "tests/my_test.py"

   test_dependencies:
     "tests/my_test.py":
       - my_new_data
   ```

3. **Use in Tests**
   ```python
   from tests.utils.data_loader import load_torch_file

   def test_my_feature(test_resources):
       data = load_torch_file(test_resources, "path/to/my_data.pt")
   ```

### Debugging Test Data Issues

1. **Check if data exists locally**
   ```bash
   ls -la tests/resources/
   ```

2. **Force re-download**
   ```bash
   rm -rf tests/resources/
   pytest tests/your_test.py -v
   ```

3. **Check configuration**
   ```bash
   cat tests/resources_config.yaml
   ```

## Best Practices

### DO ✅

- Use `test_resources` fixture for all test data loading
- Use data loader utility functions instead of manual `torch.load()`
- Add new test data to HuggingFace Hub, not to git
- Document test data in `resources_config.yaml`
- Keep test data files small (< 100MB per file)

### DON'T ❌

- Don't commit test data to git repository
- Don't hardcode absolute paths in tests
- Don't manually download test data
- Don't duplicate test data across multiple locations
- Don't create test data on-the-fly if it can be reused

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SKIP_DOWNLOAD_TEST_RESOURCES` | Skip automatic download | `0` (download enabled) |

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

**Solution**: Check file naming in repository
- Files loaded with `load_torch_directory()` use filename (without extension) as key
- Example: `sample_noise.pt` → key is `"sample_noise"`

## Related Documentation

- **Testing Data Loading**: `docs/guide/testing-data-loading.md`
- **Test Resources Architecture**: `docs/guide/test-resources-architecture.md`
- **PyTest Documentation**: https://docs.pytest.org/
- **HuggingFace Hub**: https://huggingface.co/docs/hub/

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

## Contact

For questions or issues:
- Check documentation first
- Review existing tests for examples
- Ask in development team chat
- Create issue in project repository
