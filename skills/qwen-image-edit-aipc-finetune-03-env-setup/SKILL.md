---
name: qwen-image-edit-aipc-finetune-03-env-setup
description: >
  Step 3 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-02-dataset. Create the conda environment, install torch+xpu
  matched to the installed oneAPI, apply the two required requirements.txt fixes, install
  project dependencies, and run the hardware probe to confirm XPU and the bitsandbytes XPU
  backend are available. Only use once step 2 dataset validation is done.
---

# Step 3 — Environment Setup

**Series position**: Step 3 of 8. **One-time per machine**; reuse across runs.

Step 1 covered the system-level prerequisites (oneAPI, driver, conda). This step sets up the
Python training environment.

**Prerequisites from Step 1**:
- oneAPI version confirmed (needed to select the matching `torch+xpu` version)
- `qwen-image-finetune` project cloned to a local directory

**Next step**: when `probe_hw.py` passes all four PASS criteria, proceed to skill
`qwen-image-edit-aipc-finetune-04-config`.

---

## 3.1 Create the conda env

```bat
conda create -n qwen-image-edit-xpu python=3.12 -y
conda activate qwen-image-edit-xpu
```

(If a torch+xpu base env already exists locally, prefer cloning it —
`conda create -n qwen-image-edit-xpu --clone <existing-xpu-env>` — to avoid re-downloading
the torch wheel.)

## 3.2 Install torch+xpu

**Recommended: latest wheel (let pip pick the version)**

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

Minimum supported: `torch >= 2.9.0+xpu`. Confirm the installed version matches your oneAPI
version (Step 1 §1.7 version table). For example, `torch 2.11.0+xpu` requires oneAPI 2025.3.

**Pin a specific version** (if your oneAPI is already installed and you want to match it,
rather than upgrade oneAPI):

```bat
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/xpu
```

`torchvision` follows `torch` automatically from the same XPU channel — no need to pin it
separately.

## 3.3 Install project dependencies

Before running `pip install`, the upstream `requirements.txt` needs two edits (a fresh clone
of `tsiendragon/qwen-image-finetune` ships with both problems):

**1. Comment out `transformer_engine[pytorch]`** — CUDA-only, no XPU equivalent. Installing it
will fail on a machine without NVCC:

```diff
# requirements.txt
- transformer_engine[pytorch]
+ # XPU: transformer_engine is CUDA-only — comment out
+ # transformer_engine[pytorch]
```

**2. Pin `bitsandbytes>=0.48.2`** — the unversioned `bitsandbytes` line may resolve to an
older build that lacks the XPU backend:

```diff
- bitsandbytes
+ bitsandbytes>=0.48.2
```

Then install:

```bat
pip install -r requirements.txt
```

If Step 5 adaptation has already been applied (the repo has `device_utils.py` and the XPU
patches in place), this is the only required pre-install edit.

## 3.4 Activate oneAPI in the current shell

The training launcher (Step 7) handles this automatically; for this step's ad-hoc verification:

```bat
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
```

If this reports `'vars.bat' is not recognized` errors, clear
`NoDefaultCurrentDirectoryInExePath` across the call (shown above), and run from a CMD opened
directly (not from inside another wrapper shell).

## 3.5 Verify with probe_hw.py

Save this script as `probe_hw.py` in the project root and run it. It is re-used in Step 4, so
keep it.

```python
# probe_hw.py
# Probe the local Intel AI PC hardware + key software components. Emits a JSON dict to stdout.
# Consumed by recommend_config.py (Step 4). Network policy: does NOT touch the network.
#
# Output schema (keep field names + types stable):
#   xpu_model         str | null   the XPU device name string, or null if no XPU
#   xpu_available     bool         torch.xpu.is_available()
#   vram_gb           float        GiB (total_memory / 2**30); matches Task Manager GPU figure
#   ram_gb            float        GiB (virtual_memory().total / 2**30); matches Task Manager Memory
#   ram_available_gb  float        GiB (virtual_memory().available / 2**30)
#   oneapi_version    str | null   probed from `icpx --version`, null if not on PATH
#   conda_env         str | null   CONDA_DEFAULT_ENV, null if not in conda
#   bnb_version       str | null   bitsandbytes.__version__, null if not installed
#   bnb_xpu_available bool         bnb has XPU backend module
#   torch_version     str | null
#   platform          str          "windows" / "linux" / "darwin"
#   level_zero_sdk    str | null   LEVEL_ZERO_V1_SDK_PATH or ZE_PATH if set, else null
#
# Usage: python probe_hw.py > probe.json
import argparse, json, os, platform, re, subprocess


def _probe_xpu():
    info = {"xpu_model": None, "xpu_available": False, "vram_gb": 0.0, "torch_version": None}
    try:
        import torch
        info["torch_version"] = torch.__version__
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu_available"] = True
            props = torch.xpu.get_device_properties(0)
            info["xpu_model"] = props.name
            # GiB (binary) so the figure matches what Windows Task Manager shows.
            info["vram_gb"] = round(props.total_memory / 2**30, 2)
    except ImportError:
        pass
    return info


def _probe_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"ram_gb": round(vm.total / 2**30, 2), "ram_available_gb": round(vm.available / 2**30, 2)}
    except ImportError:
        try:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return {"ram_gb": round(ram / 2**30, 2), "ram_available_gb": -1.0}
        except (ValueError, AttributeError):
            return {"ram_gb": -1.0, "ram_available_gb": -1.0}


def _probe_oneapi():
    # Return the Intel oneAPI toolkit version (e.g. '2025.3'). Parse the InstalledDir path from
    # `icpx --version` (two-part toolkit version), falling back to the Compiler line. Requires
    # setvars.bat to have been called (Windows) or oneAPI on PATH. Returns None if icpx not found.
    try:
        which = "where" if platform.system() == "Windows" else "which"
        r = subprocess.run([which, "icpx"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        r2 = subprocess.run(["icpx", "--version"], capture_output=True, text=True, timeout=10)
        if r2.returncode != 0:
            return None
        output = r2.stdout + r2.stderr
        m = re.search(r"[Ii]nstalled[Dd]ir.*[/\\]compiler[/\\](\d+\.\d+)[/\\]", output)
        if m:
            return m.group(1)
        m = re.search(r"Compiler (\d+\.\d+)", output)
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _probe_level_zero():
    # Triton XPU requires Level Zero headers to JIT-compile kernels. Normally provided by the GPU
    # driver via LEVEL_ZERO_V1_SDK_PATH; ZE_PATH is the manual fallback. null here means Triton
    # kernel compilation will likely fail — see Step 6 troubleshooting.
    for var in ("LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _probe_bnb():
    info = {"bnb_version": None, "bnb_xpu_available": False}
    try:
        import bitsandbytes as bnb
        info["bnb_version"] = bnb.__version__
        try:
            import bitsandbytes.backends.xpu  # noqa: F401
            info["bnb_xpu_available"] = True
        except ImportError:
            pass
    except ImportError:
        pass
    return info


def main():
    argparse.ArgumentParser(description="Probe Intel AI PC hardware + key software components.").parse_args()
    result = {
        **_probe_xpu(),
        **_probe_ram(),
        "oneapi_version": _probe_oneapi(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        **_probe_bnb(),
        "platform": platform.system().lower(),
        "level_zero_sdk": _probe_level_zero(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

Run:

```bat
python probe_hw.py
```

> **Reading the output.** `ram_gb`/`vram_gb`/`ram_available_gb` are GiB (binary, bytes/2**30),
> matching Windows Task Manager. A GiB reading is a few GiB BELOW the GB tier number because
> firmware + the iGPU reserve memory — e.g. a 32 GB machine reads `ram_gb` ~30–32. `ram_gb` is
> the number the Step 4 recommender uses for the memory tier (overridable by `--ram-tier`).
> `vram_gb` is the XPU-visible carve-out of unified memory (often ~half of `ram_gb`); it is
> INFORMATIONAL ONLY — the recommender does NOT gate on it. Do not mistake a "GPU (16 GB)"
> label for a 16 GB machine.

## PASS signal

The JSON output must show all four of:

- `xpu_available: true`
- `torch_version` ending in `+xpu`
- `bnb_version` >= `0.48.2`
- `bnb_xpu_available: true`

> `oneapi_version` may be null here if `setvars.bat` hasn't run in this shell session — that's
> expected for a fresh session before the launcher runs. The training launcher handles
> `setvars.bat` in Step 7.

If any of the first four fail, do NOT proceed — fix the environment first.

**Keep `probe_hw.py`** in the project root — it is re-used in Step 4.

**Proceed to**: `qwen-image-edit-aipc-finetune-04-config`

---

## Troubleshooting

**`bnb_xpu_available: false`**: bitsandbytes is too old or the XPU backend isn't compiled in.
Re-run the §3.3 install with the correct version pin:
```bat
pip install "bitsandbytes>=0.48.2"
```

**`xpu_available: false`**: torch+xpu not installed correctly. Verify `torch.__version__` ends
in `+xpu` (not `+cpu`). Re-run the §3.2 install; if a non-XPU build is already present, add
`--force-reinstall` so pip replaces it (the version number is the same, so pip skips the swap
otherwise):
```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu --force-reinstall
```

**Triton package errors (wrong package installed)**: If you see Triton-related errors
(`ImportError`, `RuntimeError` on first kernel execution, or driver-conflict crashes) that
persist after clearing caches, the environment may have the wrong Triton package — e.g. `triton`
(NVIDIA-targeting) or `triton-windows` installed alongside `pytorch-triton-xpu` or `triton-xpu`.
Fix: uninstall all Triton variants, then reinstall only the correct one for your `torch+xpu`
version (Step 1 §1.7 table):
```bat
pip uninstall triton pytorch-triton pytorch-triton-xpu triton-xpu triton-windows -y
REM Then install the correct package, e.g. for torch 2.11.0+xpu:
pip install triton-xpu==3.7.*
```
After reinstalling, clear the Triton caches before the next run (the `rmdir` commands are in
Step 7's Troubleshooting under "Stale Triton cache").
