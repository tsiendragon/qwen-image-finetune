---
name: qwen-image-edit-aipc-finetune-06-quantize
description: >
  Step 6 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-05-adaptation. This is a CONDITIONAL step: perform the one-time
  NF4 pre-quantization of the transformer (DiT) and, on 32 GB machines, the text_encoder, then
  wire the quantized paths into config.yaml. On a 64 GB+ machine using online NF4 you may skip
  it entirely. Only use once step 5 adaptation_check passes.
---

# Step 6 — NF4 Pre-Quantization (Conditional, One-Time)

**Series position**: Step 6 of 8. One-time per machine; reuse across all training runs.

**Prerequisites**:
- Framework adaptation complete (Step 5)
- `config.yaml` with paths filled (Step 4)
- RAM tier confirmed (Step 1); conda env active; oneAPI installed

**Next step**: when the quantizations that apply to your machine are done, config paths are
updated, and `config_check.py` exits 0, proceed to skill
`qwen-image-edit-aipc-finetune-07-training`.

---

## 6.0 Decision gate — do you need this step?

This whole step is optional/conditional. Decide before running anything:

| Your situation | §6.1 transformer NF4 | §6.2 text_encoder NF4 |
|---|---|---|
| **32 GB RAM machine** | Recommended (faster startup) | **Recommended** — relieves cache-phase RAM pressure |
| **64 GB+ RAM machine, will train several times** | Recommended (faster startup) | Optional / skip |
| **64 GB+ RAM machine, single exploratory run** | **Skip** — use online NF4 | Skip |

- If **both** columns say skip for you (64 GB+, one-off run): leave `model.transformer_path` and
  `model.text_encoder_path` as their TODO/absent values, confirm `config_check.py` still passes
  (it treats them as "unset" and only WARNs), and **go straight to Step 7** using online NF4.
- Otherwise do the applicable sections below. Neither section changes training-time memory — the
  transformer one only saves per-run startup time; the text_encoder one only relieves cache-phase
  RAM on 32 GB machines.

Both quantizations require the conda env active **and** oneAPI activated (`setvars.bat`), because
bnb's Triton-backed `quantize_4bit` needs the oneAPI runtime to JIT-compile the quantize kernel.

Both scripts quantize on the XPU by default. If that step fails, `--device cpu` runs the
whole load on the CPU instead — see this step's Troubleshooting.

## 6.1 Pre-quantize the transformer (DiT) — optional, recommended for repeated runs

This skill always uses NF4 for the transformer — the question is *how* NF4 weights are loaded at
each training start:

| | Pre-quantized (`transformer_path` set) | Online NF4 (`transformer_path` null) |
|---|---|---|
| Disk space | +~10 GB for the NF4 checkpoint dir | No extra |
| Per-run startup | Fast — load ~10 GB NF4 directly | Slower — stream bf16 shards from the original repo and quantize on the fly; Triton `quantize_4bit` JIT-compiles on first use |
| Training-time memory | Essentially the same as online NF4 | Essentially the same as pre-quantized |
| Original model needed on disk | Still required for text_encoder, VAE, scheduler | Still required |

The bf16 transformer checkpoint on disk is ~38 GB; the NF4 output is ~10 GB (~4× smaller). During
online quantization the model is loaded shard-by-shard so peak RAM is much less than 38 GB, but the
process still takes meaningful time on every run start.

**Save the quantize script** as `quantize_transformer_nf4.py` in the project root:

```python
# quantize_transformer_nf4.py
# Pre-quantize the Qwen-Image-Edit transformer (DiT) to NF4 and save to disk. Run once per machine.
# Prerequisites: qflux importable (run from project root with src/ on PYTHONPATH, or pip install -e .);
# conda env active + setvars.bat called (bnb's Triton quantize_4bit needs the oneAPI runtime).
# Usage:
#   python quantize_transformer_nf4.py --src <pipeline-dir> --dst <nf4-output-dir>
# After completion, set model.transformer_path: <DST> in your training config. The qflux loader
# auto-detects the saved quantization_config in <DST>/config.json and skips online quantization.
import argparse
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None


def main():
    p = argparse.ArgumentParser(description="Pre-quantize Qwen-Image-Edit transformer to NF4 (one-time).")
    p.add_argument("--src", required=True, help="Source pipeline dir (contains transformer/ subfolder)")
    p.add_argument("--dst", required=True, help="Destination dir for the NF4-quantized transformer")
    p.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16"],
                   help="bnb_4bit_compute_dtype (default: bfloat16)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"],
                   help="Where the load + quantization run. 'auto' (default) lets "
                        "bitsandbytes target the XPU; 'cpu' keeps the whole "
                        "load on the CPU. See this step's Troubleshooting")
    p.add_argument("--probe-interval", type=float, default=0.5,
                   help="Memory sampling interval (seconds); ignored if MemoryProbe unavailable")
    args = p.parse_args()
    weight_dtype = getattr(torch, args.compute_dtype)
    load_kwargs = {"device_map": {"": "cpu"}} if args.device == "cpu" else {}
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=weight_dtype, bnb_4bit_use_double_quant=True)

    def _quantize():
        print(f"[quantize] streaming-load & quantize on {args.device}: {args.src}/transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.src, subfolder="transformer",
            quantization_config=bnb_config, torch_dtype=weight_dtype, **load_kwargs)
        Path(args.dst).mkdir(parents=True, exist_ok=True)
        print(f"[quantize] saving NF4 weights to {args.dst}")
        transformer.save_pretrained(args.dst, safe_serialization=True)
        print(f'[quantize] done. Set in config:  model.transformer_path: "{args.dst}"')

    if MemoryProbe is not None:
        with MemoryProbe(interval=args.probe_interval, label="quantize"):
            _quantize()
    else:
        print("[quantize] (qflux.utils.memory_probe not installed; running without probe)")
        _quantize()


if __name__ == "__main__":
    main()
```

**Save the Windows launcher** as `quantize_xpu.bat` in the project root (activates oneAPI + conda
automatically). Update the oneAPI path and conda env name if yours differ:

```bat
@echo off
REM quantize_xpu.bat <src-pipeline-dir> <dst-nf4-dir> [extra args...]
REM One-time NF4 pre-quantization of the Qwen-Image-Edit transformer (DiT).
setlocal
if "%~1"=="" ( echo ERROR: source pipeline directory required. & echo Usage: quantize_xpu.bat ^<src^> ^<dst^> & exit /b 1 )
if "%~2"=="" ( echo ERROR: destination directory required. & echo Usage: quantize_xpu.bat ^<src^> ^<dst^> & exit /b 1 )
set SRC=%~1
set DST=%~2
REM Forward any further arguments to the Python script (e.g. --device cpu).
shift
shift
set "EXTRA="
:collect_extra
if "%~1"=="" goto collected_extra
set "EXTRA=%EXTRA% %1"
shift
goto collect_extra
:collected_extra
REM NoDefaultCurrentDirectoryInExePath guards against DLL loading from the current directory.
REM Cleared only for setvars.bat; restored below — DO NOT remove the restore line.
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
if errorlevel 1 ( echo ERROR: setvars.bat failed & exit /b 1 )
call conda activate qwen-image-edit-xpu
if errorlevel 1 ( echo ERROR: conda activate failed -- expected env name 'qwen-image-edit-xpu' & exit /b 1 )
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1
set SYCL_CACHE_PERSISTENT=1
echo [quantize] Quantizing transformer: %SRC% -^> %DST%
python quantize_transformer_nf4.py --src "%SRC%" --dst "%DST%"%EXTRA%
echo [quantize] Done. Exit code: %ERRORLEVEL%
endlocal
```

Run either the launcher or the script directly:

```bat
quantize_xpu.bat <SRC-pipeline-dir> <DST-nf4-transformer-dir>
REM or, after activating conda + setvars.bat manually:
python quantize_transformer_nf4.py --src <SRC> --dst <DST>
```

**After quantizing**, set in `config.yaml`:

```yaml
model:
  pretrained_model_name_or_path: <SRC>   # still needed for text_encoder, VAE, etc.
  transformer_path: <DST>                # NF4-quantized DiT
  quantize_type: "nf4"
```

The qflux loader reads `quantization_config` from `<DST>/config.json` and skips online quantization
automatically.

**PASS**: `<DST>` exists and contains `config.json` with a `quantization_config` field; ~10 GB total.

## 6.2 Pre-quantize the text_encoder (NF4) — 32 GB machines: recommended

**Trigger condition** (machine-judgable), by RAM tier (Step 1 `--ram-tier`; fallback: probe
`ram_gb` in GiB, where the 32 GB tier is `< 36`):
- **32 GB tier** (the minimum supported) → **recommended**. At this tier the cache phase loads the
  ~15 GB bf16 text_encoder alongside the dataset and VAE, pushing RAM near its limit and triggering
  virtual-memory paging that slows caching significantly.
- **64 GB+ tier** → optional / not needed (extra headroom).

The Qwen2.5-VL text_encoder weighs ~15 GB at bf16 (15.4 GiB in Windows Explorer). Pre-quantizing to
NF4 drops it to ~5.5 GB on disk.

**Save the quantize script** as `quantize_text_encoder_nf4.py` in the project root:

```python
# quantize_text_encoder_nf4.py
# Pre-quantize the Qwen2.5-VL text_encoder to NF4 and save to disk. Run once per machine. Saves
# ~8 GB CPU RAM vs bf16 load — the margin that prevents the ~31.5 GB ceiling crash on 32 GB machines.
# Prerequisites: qflux importable; conda env active + setvars.bat called (oneAPI runtime + bnb XPU).
# Usage:
#   python quantize_text_encoder_nf4.py --src <pipeline-dir> --dst <nf4-text-encoder-dir>
# After completion, set model.text_encoder_path: <DST> in your training config. The qflux loader
# auto-detects quantization_config in <DST>/config.json and skips online quantization.
import argparse
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None


def main():
    p = argparse.ArgumentParser(description="Pre-quantize Qwen2.5-VL text_encoder to NF4 (one-time).")
    p.add_argument("--src", required=True, help="Source pipeline dir (contains text_encoder/ subfolder)")
    p.add_argument("--dst", required=True, help="Destination dir for the NF4-quantized text_encoder")
    p.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16"],
                   help="bnb_4bit_compute_dtype (default: bfloat16)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"],
                   help="Where the load + quantization run. 'auto' (default) lets "
                        "bitsandbytes target the XPU; 'cpu' keeps the whole "
                        "load on the CPU. See this step's Troubleshooting")
    p.add_argument("--probe-interval", type=float, default=0.5,
                   help="Memory sampling interval (seconds); ignored if MemoryProbe unavailable")
    args = p.parse_args()
    weight_dtype = getattr(torch, args.compute_dtype)
    load_kwargs = {"device_map": {"": "cpu"}} if args.device == "cpu" else {}
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=weight_dtype, bnb_4bit_use_double_quant=True)

    def _quantize():
        print(f"[quantize_te] streaming-load & quantize on {args.device}: {args.src}/text_encoder")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.src, subfolder="text_encoder",
            quantization_config=bnb_config, torch_dtype=weight_dtype, **load_kwargs)
        Path(args.dst).mkdir(parents=True, exist_ok=True)
        print(f"[quantize_te] saving NF4 text_encoder to {args.dst}")
        model.save_pretrained(args.dst, safe_serialization=True)
        print(f'[quantize_te] done. Set in config:  model.text_encoder_path: "{args.dst}"')

    if MemoryProbe is not None:
        with MemoryProbe(interval=args.probe_interval, label="quantize_te"):
            _quantize()
    else:
        print("[quantize_te] (qflux.utils.memory_probe not installed; running without probe)")
        _quantize()


if __name__ == "__main__":
    main()
```

**Save the Windows launcher** as `quantize_text_encoder_xpu.bat` in the project root:

```bat
@echo off
REM quantize_text_encoder_xpu.bat <src-pipeline-dir> <dst-nf4-dir> [extra args...]
REM One-time NF4 pre-quantization of the Qwen2.5-VL text_encoder.
setlocal
if "%~1"=="" ( echo ERROR: source pipeline directory required. & echo Usage: quantize_text_encoder_xpu.bat ^<src^> ^<dst^> & exit /b 1 )
if "%~2"=="" ( echo ERROR: destination directory required. & echo Usage: quantize_text_encoder_xpu.bat ^<src^> ^<dst^> & exit /b 1 )
set SRC=%~1
set DST=%~2
REM Forward any further arguments to the Python script (e.g. --device cpu).
shift
shift
set "EXTRA="
:collect_extra
if "%~1"=="" goto collected_extra
set "EXTRA=%EXTRA% %1"
shift
goto collect_extra
:collected_extra
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
if errorlevel 1 ( echo ERROR: setvars.bat failed & exit /b 1 )
call conda activate qwen-image-edit-xpu
if errorlevel 1 ( echo ERROR: conda activate failed -- expected env name 'qwen-image-edit-xpu' & exit /b 1 )
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1
set SYCL_CACHE_PERSISTENT=1
echo [quantize_te] Quantizing text_encoder: %SRC% -^> %DST%
python quantize_text_encoder_nf4.py --src "%SRC%" --dst "%DST%"%EXTRA%
echo [quantize_te] Done. Exit code: %ERRORLEVEL%
endlocal
```

Run either:

```bat
quantize_text_encoder_xpu.bat <SRC-pipeline-dir> <DST-nf4-text-encoder-dir>
REM or, after activating conda + setvars.bat manually:
python quantize_text_encoder_nf4.py --src <SRC> --dst <DST>
```

**After quantizing**, add to `config.yaml`:

```yaml
model:
  text_encoder_path: "<DST>"
```

The qflux loader auto-detects `quantization_config` in the saved `config.json` and skips online
quantization on subsequent runs — identical to the `transformer_path` pattern in §6.1.

**Measured effect** (32 GB AI PC, validated):

| Phase | bf16 text_encoder | NF4 text_encoder |
|---|---|---|
| Cache phase RAM | near physical limit — may trigger virtual memory paging | well within limit — no paging |
| Disk size | ~15 GB | ~5.5 GB |
| Cache throughput | may slow significantly if paging occurs | normal |
| Fit phase RAM | unchanged | unchanged — fit does not load text_encoder |

**Note on cache compatibility**: NF4 embeddings differ slightly from bf16. If you have an existing
cache built with bf16 text_encoder, delete the cache dir and rebuild after switching to
`text_encoder_path`.

**32 GB machines — cache device**: with `text_encoder_path` set, keep `cache.devices.text_encoder:
xpu:0` (the NF4 text_encoder is small enough). Without it, set `cpu` as a slower fallback to avoid
placing the bf16 text_encoder on XPU.

**PASS**: `<DST>` exists and contains `config.json` with `quantization_config`; ~5.5 GB total.

## 6.3 Re-run config_check.py

With the quantized paths filled in `config.yaml`, re-run the linter (from Step 4):

```bat
python config_check.py config.yaml --ram-tier <16|32|64>
```

The previous `text_encoder_path` warning should now be gone.

## PASS signal

- §6.1 (if run): NF4 transformer output dir has `config.json` with `quantization_config`; ~10 GB;
  `model.transformer_path` points to it
- §6.2 (32 GB tier): NF4 text_encoder output dir has `config.json` with `quantization_config`;
  ~5.5 GB; `model.text_encoder_path` points to it
- OR (64 GB+ single run): step deliberately skipped; `config_check.py` still passes with online NF4
- `config_check.py config.yaml --ram-tier <N>` exits 0 (`PASS: 0 errors`)

**Proceed to**: `qwen-image-edit-aipc-finetune-07-training`

---

## Troubleshooting

**NF4 quantization fails: SPIR-V `SPV_KHR_bfloat16`, or XPU OOM on load.** Either of:

- `InvalidModule: Invalid SPIR-V module: input SPIR-V module uses extension 'SPV_KHR_bfloat16'
  which were disabled by --spirv-ext option`. A runtime abort, not a Python exception — `try/except`
  does not catch it and the process simply dies.
- `torch.OutOfMemoryError: XPU out of memory. Tried to allocate <N> GiB`, raised from
  `caching_allocator_warmup`, often while the message itself reports more free memory than the block
  it failed to allocate. Adding memory does not help.

Both have been workarounded by running the load on the CPU rather than the XPU — try that first,
with `--device cpu`:

```bat
python quantize_transformer_nf4.py   --src <SRC> --dst <DST> --device cpu
python quantize_text_encoder_nf4.py  --src <SRC> --dst <DST> --device cpu
REM the launchers forward extra arguments too:
quantize_xpu.bat <SRC> <DST> --device cpu
```

CPU quantization is slower, but it is a one-time step and the output is identical.

If instead it failed during **online NF4** in a Step 7 training run (`model.transformer_path`
unset), the online path has no equivalent switch: pre-quantize per §6.1 with `--device cpu` and set
`model.transformer_path` (and §6.2 / `model.text_encoder_path` if the text_encoder was the component
that failed).

> **If you saw the `SPV_KHR_bfloat16` variant, expect a second failure at the first fit optimizer
> step.** The same limitation has another trigger point inside the training loop, and quantizing on
> the CPU does nothing about it. See Step 7's Troubleshooting — you can apply that fix pre-emptively
> rather than waiting for the fit run to die.

If neither `--device cpu` nor the Step 7 fix helps, as a last resort try a **different** Intel Arc
Graphics driver version (Step 1 §1.4).


**Triton kernel JIT fails (`RuntimeError` on the first quantize op, or a Level Zero error)**:
- Confirm `setvars.bat` was called before running (the launcher does this).
- Check `probe_hw.py` output for `"level_zero_sdk": null` — see next entry.

**`level_zero_sdk: null` / Level Zero headers missing**: Triton XPU needs Level Zero headers to
JIT-compile kernels (`quantize_4bit`). Check first:
```bat
echo %LEVEL_ZERO_V1_SDK_PATH%
```
If it prints a real directory containing `include\level_zero\ze_api.h`, this is not the issue. If
null, download `level-zero-win-sdk-<version>.zip` from
https://github.com/oneapi-src/level-zero/releases (match your GPU driver's Level Zero loader),
extract to a path with **no spaces or non-ASCII characters**, then set `ZE_PATH` to the extracted
directory (containing `include/` and `lib/`) **before** calling `setvars.bat`:
```bat
set ZE_PATH=C:\path\to\level-zero-sdk
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
```
Add this `set ZE_PATH=...` line to your launchers so it's set on every run.

**Stale Triton cache** (after a oneAPI or torch+xpu upgrade — errors on the first kernel):
```bat
rmdir /s /q "%USERPROFILE%\.triton"
rmdir /s /q "%LOCALAPPDATA%\Temp\torchinductor_%USERNAME%"
```
Both rebuild automatically on the next Triton-backed op. If `rmdir` fails because a process holds
the directory, close all Python / training processes first.
