# Utils Module Tests

This directory contains comprehensive test suites for all utility modules in `src/utils/`.

## Test Coverage

### ✅ test_images.py
Tests for image processing utilities:
- `resize_bhw` - Tensor resizing
- `make_image_shape_devisible` - Shape adjustment for VAE
- `make_image_devisible` - Image divisibility for different input types
- `calculate_dimensions` - Dimension calculation with aspect ratios
- `calculate_best_resolution` - Best resolution calculation
- `image_adjust_best_resolution` - Image adjustment for optimal resolution

### ✅ test_seed.py
Tests for random seed management:
- `seed_everything` - Reproducible random state across PyTorch, NumPy, and Python
- Environment variable configuration
- CUDA random state (when available)

### ✅ test_options.py
Tests for command-line argument parsing:
- `parse_args` - Config file loading
- Training mode flags (fit/cache)
- Resume checkpoint handling
- Error handling for missing/invalid configs

### ✅ test_sampling.py
Tests for sampling and timestep utilities:
- `calculate_shift` - Shift calculation for different sequence lengths
- `retrieve_timesteps` - Timestep retrieval from schedulers
- Custom timesteps and sigmas support

### ✅ test_tools.py
Tests for general utility functions:
- `sample_indices_per_rank` - Distributed sampling
- Hash functions (MD5, SHA256, BLAKE3, perceptual hash)
- `get_git_info` - Git repository information
- `infer_image_tensor` - Image tensor layout and range inference
- `extract_batch_field` - Batch data extraction

### ✅ test_logger.py
Tests for logging utilities:
- `load_logger` - Logger initialization
- `log_images_auto` - Automatic image logging to W&B/TensorBoard
- `log_text_auto` - Automatic text/table logging

### ✅ test_lora_utils.py
Tests for LoRA (Low-Rank Adaptation) utilities:
- `classify_lora_weight` - LoRA weight format detection (PEFT/Diffusers)
- `get_lora_layers` - LoRA layer extraction
- `FpsLogger` - FPS tracking with warmup, EMA, and token counting

### ✅ test_model_summary.py
Tests for model inspection utilities:
- `gather_model_stats` - Comprehensive model statistics
- `print_model_summary_table` - Model summary visualization
- Dtype detection (fp32, fp16, bf16, int8, fp8, 4-bit)
- Module detection (attention, normalization, MLP)
- LoRA statistics
- Transformer architecture stats

### ✅ test_huggingface.py
Tests for HuggingFace Hub utilities:
- `is_huggingface_repo` - Repository pattern detection
- `_pick_first_existing` - File extension matching

## Running Tests

### Run all utils tests:
```bash
pytest tests/utils/ -v
```

### Run specific test file:
```bash
pytest tests/utils/test_images.py -v
```

### Run with coverage:
```bash
pytest tests/utils/ --cov=src/utils --cov-report=html
```

## Test Statistics

- **Total test files**: 9
- **Total test classes**: 30+
- **Total test methods**: 100+
- **Code coverage**: Comprehensive coverage of all public APIs

## Notes

### Python Version Compatibility
- Tests are designed for Python 3.8+
- Some project dependencies may require Python 3.10+ (e.g., `str | None` syntax in config files)
- If you encounter import errors related to type annotations, ensure the project code uses `Optional[str]` instead of `str | None` for Python 3.8 compatibility

### Dependencies
- pytest
- torch
- numpy
- PIL/Pillow
- safetensors
- blake3
- imagehash
- pandas (for HuggingFace dataset tests)

### Test Isolation
- All tests use fixtures for temporary files
- Mock objects are used for external dependencies (accelerator, trackers)
- No side effects on global state (except for seed tests by design)

## Contributing

When adding new utility functions:
1. Add corresponding tests in the appropriate test file
2. Follow the existing test structure and naming conventions
3. Use pytest fixtures for setup/teardown
4. Mock external dependencies
5. Test both success and error cases
6. Ensure all lint checks pass (`black`, `flake8`)
