---
name: qwen-image-edit-aipc-finetune-07-training
description: >
  Step 7 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-06-quantize. Run the cache-first two-phase training: the cache
  phase (one-time per dataset) then the fit phase, including a small smoke-test run to catch the
  max_train_steps bug, XPU memory monitoring, and resume-from-checkpoint. Only use once step 6
  quantization (or its deliberate skip) is done and config_check passes.
---

# Step 7 — Training (Cache Phase + Fit Phase)

**Series position**: Step 7 of 8.

**Prerequisites**:
- Framework adaptation complete (Step 5)
- `config.yaml` fully filled, including `transformer_path`/`text_encoder_path` where applicable (Steps 4 + 6)
- Conda env and oneAPI confirmed (Steps 3 + 1)

**Next step**: when training completes and a checkpoint is written, proceed to skill
`qwen-image-edit-aipc-finetune-08-validation`.

By the time you reach this step: env is set up (Steps 1, 3), framework is adapted (Step 5),
dataset is validated (Step 2), and Step 4 emitted a YAML config. This step runs the actual training.

Each phase below has a machine-judgable PASS signal. **If a phase fails, stop and report rather
than proceed** — a silently-failed cache phase surfaces as confusing errors during fit that are
much costlier to debug than the original failure.

---

## 7.1 Cache-first workflow (why two phases)

**Required behavior**: `cache.use_cache: true` is mandatory for this skill's training path on every
AI PC tier. This is a boundary of the supported recipe, not a general size heuristic. For
Qwen-Image-Edit on AI PCs, live text/image encoding during fit keeps the Qwen2.5-VL text_encoder
active alongside the NF4 DiT and falls outside the validated cache-first workflow.

The text_encoder (Qwen2.5-VL) and VAE produce conditioning signals without themselves being trained.
Pre-compute their outputs once and reload from disk each step — this frees encoder memory entirely
during the training loop, leaving only the ~10 GB DiT (NF4) + LoRA + Adam8bit state on XPU during fit.

Two-phase flow:

```
[cache step]  python -m qflux.main --config <cfg> --cache
              ├─ load text_encoder + VAE (DiT skipped — see Step 5 mode-aware loading)
              ├─ encode all training samples → save embeddings to cache_dir
              └─ encoders freed after pass; build time scales with dataset size

[fit step]    accelerate launch ... -m qflux.main --config <cfg>
              ├─ cache detected → skip loading text_encoder (see Step 5)
              ├─ only DiT (NF4) + LoRA + optimizer resident on XPU
              └─ each step: load pre-computed embeddings from disk
```

> **`cache.use_cache: true` is two phases, not one — and an empty cache is NOT fixed by turning
> caching off.** If a fit launch reports the cache is missing or empty, the correct action is to
> **run the cache phase first** (§7.3), then fit. Setting `cache.use_cache: false` is **not a valid
> workaround** on an AI PC: it forces the text_encoder to stay resident alongside the NF4 DiT during
> fit (Step 5 mode-aware loading can no longer skip it — bf16 ~15 GB, or ~5.5 GB if pre-quantized
> per Step 6), tightening or over-committing the unified budget and risking a crash or severe
> stalls. Never flip `use_cache` to `false` to get past an empty-cache message.

Config block (already in the recommender's output):

```yaml
cache:
  devices:
    text_encoder: xpu:0
    vae: xpu:0
  cache_dir: "outputs/<run_name>/cache"
  use_cache: true
```

**Cache compatibility / invalidation**: cache files are valid only for the exact preprocessing and
frozen-encoder setup that produced them. Delete the entire `cache.cache_dir` and rebuild before fit
when any input to cached tensors changes:

- dataset contents, prompts, masks, controls, or `selected_control_indexes`;
- `data.processor` settings such as `target_size`, `controls_size`, `process_type`, crop policy, or aspect-ratio limits;
- VAE weights/config, text_encoder weights/precision (bf16 ↔ NF4), tokenizer/processor, or prompt template;
- cache schema or saved embedding key names.

Do **not** reuse a cache built at one resolution with a fit config using another resolution. The
cached prompt/image sequence lengths no longer match the DiT RoPE shapes and can fail as
`apply_rotary_emb_qwen` / RoPE dimension mismatch (for example `2016` vs `4608`). For LoRA rank or
learning-rate sweeps where the dataset, processor, model encoders, and cache settings are unchanged,
reusing the same cache is valid.

## 7.2 The launcher

Save this as `train_xpu.bat` in the project root. It activates oneAPI + conda and dispatches cache
vs fit mode. Update the oneAPI path / conda env name if yours differ:

```bat
@echo off
REM train_xpu.bat <config-path> [--cache]
REM Launches qwen-image-edit fine-tuning on Intel XPU on Windows.
REM   <config-path>: path to YAML config (output of recommend_config.py)
REM   --cache:       optional. If present, runs cache-build mode; otherwise fit (training) mode.
setlocal
if "%~1"=="" ( echo ERROR: config path required & echo Usage: train_xpu.bat ^<config^> [--cache] & exit /b 1 )
set CONFIG=%~1
set MODE=%~2
REM NoDefaultCurrentDirectoryInExePath guards against DLL loading from the current directory.
REM Cleared only for setvars.bat; restored below — DO NOT remove the restore line.
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
if errorlevel 1 ( echo ERROR: setvars.bat failed -- check oneAPI install path in this script & exit /b 1 )
call conda activate qwen-image-edit-xpu
if errorlevel 1 ( echo ERROR: conda activate failed -- expected env name 'qwen-image-edit-xpu' & exit /b 1 )
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1
set SYCL_CACHE_PERSISTENT=1
if /i "%MODE%"=="--cache" (
    echo [train_xpu] cache-build mode
    python -m qflux.main --config %CONFIG% --cache
) else (
    echo [train_xpu] fit mode
    accelerate launch --config_file accelerate_config_xpu.yaml -m qflux.main --config %CONFIG%
)
endlocal
```

## 7.3 Cache phase (one-time per dataset)

The cache phase **must run before the fit phase** — fit reads the embeddings it produces. If you
skip it, fit reports an empty/missing cache; the fix is to run this step, **not** to set
`cache.use_cache: false` (§7.1).

```bat
train_xpu.bat config.yaml --cache
```

Runs the text_encoder and VAE forward over every training sample and writes embeddings to
`cache.cache_dir`. Build time scales linearly with sample count and image resolution; a dataset of
~30–50 samples at default [384, 672] resolution completes in one pass.

**PASS**: process exits 0; `cache.cache_dir` contains a per-sample directory structure (one folder
per sample with embedding tensors). Re-running with the same dataset + config is a no-op (cache is
hash-keyed).

> **Corrupt cache (interrupted run)**: If a previous cache run was interrupted, it may leave empty or
> partial files. Re-running will fail with `json.JSONDecodeError` or `KeyError` on the first sample.
> Fix: delete the entire `cache.cache_dir` directory and re-run the cache phase from scratch.

## 7.4 Fit phase (smoke test first)

> **Do the first fit run with a small `max_train_steps` (e.g. 5–10) and confirm it stops there.**
> This catches the `train_epoch` `max_train_steps` bug early (Step 5 §5.2): if that guard wasn't
> applied, training ignores `max_train_steps` and runs `num_epochs × dataset_size` steps — you'd
> otherwise only notice after a long run. Set `checkpointing_steps` ≤ `max_train_steps` for this run
> (e.g. `5`) so a checkpoint is actually written; `config_check.py` warns if it isn't.

Temporarily edit `config.yaml` for the smoke test:
```yaml
train:
  max_train_steps: 5
  checkpointing_steps: 5
```

Then launch:

```bat
train_xpu.bat config.yaml
```

Logs go to stdout. Checkpoints land in a **per-run version directory**, not directly under
`output_dir`:

```
<logging.output_dir>/<logging.tracker_project_name>/v<N>/checkpoint-<epoch>-<step>/
```

The trainer picks `v<N>` itself on every launch (`setup_versioned_logging_dir`), so each run —
including a §7.6 resume — gets its own `v<N>` and runs never mix their checkpoints. A version
directory that never produced a checkpoint is treated as unused and its number is recycled.

**Smoke test PASS criteria**:
- **Training stops at `max_train_steps`** (not at `num_epochs × dataset_size`) — this is the key
  check on the small first run
- **Step 1 completes without exception**
- **`torch.xpu.memory_allocated() > 0` during the step** — sample from a second CMD:
  ```bat
  python -c "import torch; print(torch.xpu.memory_allocated()/2**30, 'GB (GiB)')"
  ```
  A near-zero value means compute is happening on CPU (typically a Step 5 adaptation not applied —
  check this before suspecting anything else).
- **One checkpoint written** to `<logging.output_dir>/checkpoint-<epoch>-<step>/`

Once the early run stops at the expected step and writes a checkpoint, raise both values for the real
run:
```yaml
train:
  max_train_steps: 1000
  checkpointing_steps: 100
```

Then launch the full training again:

```bat
train_xpu.bat config.yaml
```

**Full fit PASS criteria for a healthy run**:
- **Training stops at `max_train_steps`** (verified in the smoke test)
- **Loss is finite and shows a downward trend after the warmup period** (batch-level fluctuation is
  normal; warmup is set by `train.warmup_steps` in the config)
- **Checkpoints written**; final loss meaningfully below initial
- **Checkpoints are adapter-only, not a merged base model.** Two shapes are both normal: a
  `checkpointing_steps` checkpoint holds only the LoRA adapter (`pytorch_lora_weights.safetensors`),
  while a `checkpoint-last-<epoch>-<step>-last` additionally holds the resumable state
  (`model.safetensors`, `optimizer.bin`, `scheduler.bin`, RNG state) for §7.6. Either way the
  directory is far smaller than the base DiT and contains **no base weights**. A checkpoint
  approaching the base model's size means the LoRA was merged into the base — not intended here: it
  bloats disk and breaks the Step 8 `--lora-weight` flow, which applies a standalone adapter to the
  base. Do **not** add any merge-LoRA / save-full-model step; keep the default adapter-only
  checkpointing.
**Not a PASS criterion, but worth a glance on the smoke run**: how many
`checkpoint-last-*-last` directories the run left in its `v<N>`. A correct run leaves one — the
`train_epoch` guard and `fit()`'s closing save derive the same name from `(epoch, global_step)`, so
the second overwrites the first, and this holds however the run ends. Several of them sharing the
*same* step number and differing only by epoch means the Step 5 §5.2 epoch-loop break is missing.

That is a **disk and save-time problem, not a correctness one**: the redundant copies are
byte-identical, so the trained adapter, the resume state and Step 8 validation are all unaffected —
do not fail a run or re-train because of it. It is worth catching early only because nothing else
reports it, and because the pile grows with `num_epochs`.

Nothing prunes checkpoints automatically: `train.checkpoints_total_limit` is declared in the config
schema but never wired into accelerate's `ProjectConfiguration`, so setting it has no effect. Once a
run is finished and validated, intermediate checkpoints are safe to delete — §7.6 resume needs only
the `-last-` one, and Step 8 validation needs only `pytorch_lora_weights.safetensors`.

## 7.5 Monitoring across longer runs

The training log emits per-step `xpu memory: N.NN GB` lines. **Note the unit exception**: the
framework prints these in **decimal GB** (`bytes / 1e9`), ~7% higher than the GiB used everywhere
else in this series (and ~7% higher than Task Manager / the monitor script below, which are GiB). All
describe the same memory; just don't compare the framework log against the GiB figures
number-for-number. Memory should be stable across steps; linear growth indicates a tensor or gradient
leak.

Save this as `monitor_xpu_memory.py` and run it from a second CMD while training:

```python
# monitor_xpu_memory.py
# Poll XPU memory allocation at a fixed interval. Run from a second terminal during training/inference.
# Usage:
#   python monitor_xpu_memory.py
#   python monitor_xpu_memory.py --interval 1.0
#   python monitor_xpu_memory.py --count 10   # stop after 10 readings
# Output (values are GiB):
#   [HH:MM:SS] XPU alloc: <N> GB  reserved: <N> GB  (<capacity> GB capacity)  [GiB]
import argparse, time
from datetime import datetime


def main():
    p = argparse.ArgumentParser(description="Poll XPU memory allocation at a fixed interval.")
    p.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2.0)")
    p.add_argument("--count", type=int, default=0, help="Number of readings before stopping; 0 = unlimited")
    args = p.parse_args()
    try:
        import torch
    except ImportError:
        print("[monitor] torch is not installed in this environment.")
        return
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[monitor] XPU not available — nothing to monitor.")
        return
    # GiB (binary) to match Task Manager / the rest of this series' size figures.
    capacity_gb = torch.xpu.get_device_properties(0).total_memory / 2**30
    i = 0
    try:
        while args.count == 0 or i < args.count:
            ts = datetime.now().strftime("%H:%M:%S")
            alloc = torch.xpu.memory_allocated(0) / 2**30
            reserved = torch.xpu.memory_reserved(0) / 2**30
            print(f"[{ts}] XPU alloc: {alloc:.2f} GB  reserved: {reserved:.2f} GB  ({capacity_gb:.2f} GB capacity)  [GiB]")
            time.sleep(args.interval)
            i += 1
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")


if __name__ == "__main__":
    main()
```

If step time degrades mid-run, check the Triton kernel cache (Troubleshooting below) — stale kernels
after a oneAPI / torch upgrade can manifest as erratic slowdowns.

## 7.6 Resume from checkpoint

If interrupted, set `resume: <checkpoint-path>` in the config and re-launch. qflux's checkpoint loader
handles state restoration. Point it at a `checkpoint-last-<epoch>-<step>-last` directory — only those
carry the optimizer, scheduler and RNG state that resume needs; a plain `checkpoint-<epoch>-<step>`
holds the adapter alone.

Note: the cosine LR schedule restarts from its initial value on resume, which can cause smooth loss to
temporarily rise before continuing to decrease — expected behavior, not instability.

## Notes on judging convergence

For a small dataset (~30 samples), use **smooth loss** rather than per-step raw loss — raw loss is
noisy on small datasets due to repeated batches. A sustained downward trend in smooth loss is the key
signal, regardless of absolute values.

## PASS signal

- Cache phase: exit 0; `cache.cache_dir` contains per-sample subdirs
- Fit smoke test: training stops at `max_train_steps`; `xpu.memory_allocated() > 0`; checkpoints
  written under the run's `v<N>` directory — with `checkpointing_steps` equal to `max_train_steps`
  that is one `checkpoint-<epoch>-<step>` (adapter only) plus one `checkpoint-last-<epoch>-<step>-last`
  (resumable)

> Also glance at the `-last-` count in that `v<N>`: more than one is the tell that the Step 5 §5.2
> epoch-loop break is missing. It wastes disk and save time but does **not** make the run incorrect,
> so it is not a gate — see §7.4.
- Full fit: final loss meaningfully below initial; checkpoints written

**Proceed to**: `qwen-image-edit-aipc-finetune-08-validation`

---

## Troubleshooting

**`torch.xpu.memory_allocated()` returns 0 during training** — two distinct causes:
- A direct `torch.cuda.memory_allocated()` call not replaced with a `device_utils` wrapper — run the
  Step 5 §5.2 scan, locate it, replace with `torch.xpu.memory_allocated()` guarded by
  `if torch.xpu.is_available()`.
- (NF4 + LoRA flow) the trainer moves the model to CPU before LoRA injection and never moves it back.
  Check `src/qflux/trainer/base_trainer.py` for `.to("cpu")` calls before `get_peft_model(...)`. If
  found, add an explicit `.to(self.accelerator.device)` after LoRA injection.

**`accelerator.device` is `cuda:0` instead of `xpu:0`**: `accelerate_config_xpu.yaml` (Step 5 §5.3)
isn't passed to `accelerate launch`, or it contains `distributed_type: MULTI_GPU`. Confirm the
launcher passes `--config_file accelerate_config_xpu.yaml` and the config has `distributed_type: NO`
and `use_cpu: false`.

**`KeyError: Invalid key: 0` — `load_dataset` returned a `DatasetDict`**:
`load_dataset(name)` without a `split=` argument returns a `DatasetDict`; integer indexing / `len()`
then fails. Fix:
```python
from datasets import DatasetDict, load_dataset
dataset = load_dataset(dataset_name)
if isinstance(dataset, DatasetDict):
    split = "train" if "train" in dataset else next(iter(dataset))
    dataset = dataset[split]
```

**`KeyError: 'prompt_embed'` during cache / fit**: caption-dropout (`caption_dropout_rate > 0`) with
the upstream `CacheConfig` default `prompt_empty_drop_keys = ["prompt_embed", ...]` — the key is
missing an `s`; the cached key is `prompt_embeds`. Fix:
```yaml
cache:
  prompt_empty_drop_keys: ["prompt_embeds", "prompt_embeds_mask"]
```
The Step 4 recommender emits this automatically; you only hit the error with a hand-written config.

**Dtype mismatch / `RuntimeError` on the first fit step**: mixed dtypes reaching the DiT's joint
attention. Apply Step 5 §5.5.3 NF4-A2 (`torch.autocast` wrap); `adaptation_check.py` WARNs when it's
absent. It is the only autocast on the XPU path, since the `Accelerator` runs with
`mixed_precision="no"`. Do **not** try to fix this by setting a LoRA dtype in the YAML: `lora` has no
`dtype` field and the schema is `extra="forbid"`, so the config would be rejected outright.

**Online NF4 quantization fails while the model loads** — with `model.transformer_path` unset the DiT
is quantized on every run start, so it can fail here even if you skipped Step 6 entirely. Two forms:

- `InvalidModule: Invalid SPIR-V module: input SPIR-V module uses extension 'SPV_KHR_bfloat16' which
  were disabled by --spirv-ext option`. A runtime abort, not a Python exception — `try/except` does
  not catch it and the process simply dies.
- `torch.OutOfMemoryError: XPU out of memory. Tried to allocate <N> GiB`, raised from
  `caching_allocator_warmup`, often while the message itself reports more free memory than the block
  it failed to allocate. Adding memory does not help.

**Fix**: the online path has no switch for this — go back to **Step 6** and pre-quantize with
`--device cpu`, then set `model.transformer_path` in the config (and `model.text_encoder_path` if the
text_encoder was the component that failed). Step 6's Troubleshooting has the commands. Rebuild the
cache only if you changed the text_encoder; a transformer-only change leaves the cache valid.

> If you saw the `SPV_KHR_bfloat16` form, apply the next entry's fix as well before re-launching —
> the same limitation has a second trigger point at the first optimizer step.

**Fit dies at the first optimizer step (`SPV_KHR_bfloat16` SPIR-V module rejected).** The run loads
the model, logs the LoRA parameter count, constructs the optimizer — and then dies at the first
`Adam8bit.step()`, before the first step's loss line. It may print `InvalidModule: Invalid SPIR-V
module: input SPIR-V module uses extension 'SPV_KHR_bfloat16' which were disabled by --spirv-ext
option`, or nothing at all: this is a runtime abort rather than a Python exception, so the process
can vanish without a traceback.

**Identify it by where the run stops, not by the missing traceback.** A silent death somewhere else
— during model load (see the entry above), during the cache phase, or after several steps have
already completed — is a different problem, and the fix below does not apply to it.

**What to try**: the LoRA adapter is created in the base layer's dtype, which on this recipe is bf16.
Casting the trainable parameters to fp32 has workarounded this. Add the cast immediately after the adapter
is added, in `add_lora_adapter` (`src/qflux/trainer/base_trainer.py`):

```python
transformer.add_adapter(lora_config, adapter_name=adapter_name)
transformer.set_adapter(adapter_name)
# XPU: fp32 LoRA params — see this step's Troubleshooting
for p in transformer.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
```

Step 5 §5.5.3 NF4-A2's autocast already reconciles an fp32 adapter with the bf16 base at compute time,
so training precision does not change. The **cache stays valid** — do not rebuild it. Checkpoints
written after the change carry an fp32 adapter and are correspondingly larger, but the adapter still
applies to the base model unchanged, so Step 8 validation is unaffected.

If the abort persists, set `optimizer.class_path: torch.optim.AdamW`, which avoids the bnb Triton
optimizer kernels entirely at the cost of a slower step. As a last resort try a **different** Intel
Arc Graphics driver version (Step 1 §1.4).

**Cache-phase out-of-memory on tighter RAM tiers**: the bf16 text_encoder (~15 GB) + dataset + OS
overhead exceeds the RAM ceiling. On 32 GB **system-RAM** machines this is on the edge; 16 GB
system RAM is over budget. (Tier = total system RAM, not the Task-Manager "GPU (XX GB)" figure.)
Mitigations, in order of effort:
1. Confirm Step 1 §1.3 GPU shared memory ceiling has been raised.
2. Reduce dataset size for the first run (cache fewer samples; verify the pipeline before scaling up).
3. Use Step 6 §6.2 NF4 text_encoder (validated): reduces cache-phase RAM by ~8 GB.
4. Run cache phase with the text_encoder on CPU (much slower, lowest XPU pressure): set
   `cache.devices.text_encoder: cpu`.

**No checkpoint written** — usual causes, in order of likelihood:
- The run was interrupted before the first checkpoint boundary (wrapper/CI timeout, Ctrl-C, OOM, or an
  earlier crash). Checkpoints are only written every `checkpointing_steps` steps.
- `checkpointing_steps > max_train_steps` — the run ends before any boundary. Set
  `checkpointing_steps <= max_train_steps` (Step 4); `config_check.py` warns on this.
- Fit crashed on step 1 — e.g. the dtype mismatch above.

**Stale Triton cache after a version change** (erratic slowdowns or first-kernel errors after
upgrading oneAPI / `torch+xpu` / any `triton*` package):
```bat
rmdir /s /q "%USERPROFILE%\.triton"
rmdir /s /q "%LOCALAPPDATA%\Temp\torchinductor_%USERNAME%"
```
Both rebuild automatically. If `rmdir` fails, close all Python / training processes first.

**Machine reboots unexpectedly during training**: first thing to try is updating the Intel Arc
Graphics driver (see Step 1 §1.4). After updating and rebooting, re-verify the shared GPU memory registry setting (Step 1 §1.3) —
driver installation can reset it.
