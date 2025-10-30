# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the training stack: `data/` loaders, `models/` architectures, `trainer/` implementations, `utils/` helpers, plus the entry point `main.py` for launching runs.
- `configs/` and `tests/test_configs/` contain YAML examples that define LoRA setups, device mapping, and accelerate parameters; keep new configs scoped and documented.
- `tests/` mirrors the runtime layout (data, loss, models, trainer) with unit and regression suites; add fixtures here when extending functionality.
- `docs/` provides architecture notes and optimization guides, while `script/` and `run*.sh` scripts show repeatable launch patterns.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (or `./setup.sh /desired/path hf_token`) prepares a reproducible Python 3.12 environment with CUDA-ready PyTorch.
- `pip install -r requirements.txt` synchronizes dependencies; rerun after editing `requirements.txt` or `pyproject.toml`.
- `python -m qflux.main --config configs/face_seg_flux_kontext_fp16.yaml` runs training with the specified YAML; append `--cache` to precompute embeddings.
- `CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m qflux.main --config …` enables multi-GPU runs via Accelerate.
- `pytest` (or `pytest tests/trainer/test_flux_kontext_trainer.py`) executes the test suite with live logging configured in `pyproject.toml`.

## Coding Style & Naming Conventions
- Follow Black/Flake8 rules with a 120-character limit; run `black src tests` before submitting.
- Use 4-space indentation, type hints for public APIs, and descriptive snake_case for modules, packages, and function names.
- Align config keys with existing YAML patterns (e.g., `data.init_args.processor`); document novel options inline or in `docs/`.

## Testing Guidelines
- Prefer pytest parametrizations and fixtures to mirror training modes (fit, cache, predict).
- Co-locate tests beside the module under test (`tests/models`, `tests/trainer`, etc.) and name new files `test_<feature>.py`.
- Validate GPU-sensitive changes with config-driven tests in `tests/test_configs/`; when unavailable, stub devices via Accelerate’s CPU backend.

## Commit & Pull Request Guidelines
- Git history follows Conventional Commits (`feat:`, `fix:`, `chore:`); keep subjects under 72 characters and describe scope clearly.
- Each PR should summarize behavior changes, list configs affected, link related issues, and attach benchmark screenshots or logs when performance shifts.
- Ensure CI-successful `pytest` output and linting (Black/Flake8) are referenced in the PR description before requesting review.
