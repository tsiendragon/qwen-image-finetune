fix: scheduler state isolation and directory cleanup improvements (v3.2.1)

## Summary

This PR fixes two critical bugs that could affect training stability:

1. **Scheduler state pollution**: Validation and training were sharing the same scheduler instance, causing state modifications during validation to affect subsequent training steps
2. **Directory cleanup**: Invalid version directories were being completely deleted instead of just clearing their contents

## Changes

### 1. Scheduler State Isolation

**Problem**:
- Training and validation shared `self.scheduler`
- Validation calls `set_timesteps()` which modifies `scheduler.timesteps`
- Training relies on `scheduler.timesteps[indices]` for timestep sampling
- This caused incorrect timesteps in training after validation runs

**Solution**:
- Created separate scheduler instances:
  - `self.scheduler`: Used exclusively for training
  - `self.sampling_scheduler`: Used exclusively for validation/sampling (deepcopy of scheduler)
- Updated all sampling methods to use `sampling_scheduler`
- Modified `prepare_predict_timesteps()` to accept scheduler parameter

**Files Modified**:
- `src/qflux/trainer/base_trainer.py`: Added `sampling_scheduler` attribute, updated `prepare_predict_timesteps()`
- `src/qflux/trainer/flux_kontext_trainer.py`: Create independent schedulers, use `sampling_scheduler` in sampling methods
- `src/qflux/trainer/qwen_image_edit_trainer.py`: Create independent schedulers, use `sampling_scheduler` in sampling methods

### 2. Directory Cleanup Fix

**Problem**:
- `shutil.rmtree()` deleted entire version directories
- Could cause issues if directory structure needs to be preserved

**Solution**:
- Only remove contents within invalid version directories
- Preserve directory structure for version management

**Files Modified**:
- `src/qflux/trainer/base_trainer.py`: Updated `setup_versioned_logging_dir()` cleanup logic

## Impact

- ✅ **Training stability**: Scheduler state modifications during validation no longer affect training
- ✅ **Backward compatible**: No configuration changes required, existing scripts continue to work
- ✅ **Directory preservation**: Version directory structure is preserved during cleanup
- ✅ **No behavior changes**: Validation still runs at configured intervals with same results

## Testing

- ✅ Verified training uses `self.scheduler` consistently
- ✅ Verified validation uses `self.sampling_scheduler` exclusively
- ✅ Confirmed scheduler state isolation (validation modifications don't affect training)
- ✅ Tested directory cleanup preserves directory structure

## Version Update

- Updated version to `3.2.1` (PATCH - bug fixes)
- Added changelog entry in `docs/changelog/v3.2.1.md`
- Updated `docs/changelog/index.md` and `docs/TODO.md`

## Related

Fixes scheduler state pollution issue between training and validation phases.
