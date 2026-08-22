---
name: qwen-image-edit-aipc-finetune-01-preflight
description: >
  Step 1 of 8. Start here to fine-tune or personalize Qwen-Image-Edit with LoRA on an
  Intel AI PC (Core Ultra processor) Windows laptop — even if the user doesn't use those
  exact terms, whenever they want to train Qwen-Image-Edit on their own dataset, mention
  XPU training, or NF4 QLoRA on Windows. This step covers the one-time human prerequisites:
  Intel Arc driver, Windows registry tweaks, Miniforge/conda, Visual Studio C++ workload,
  Intel oneAPI Base Toolkit, model download, and cloning the training framework. Run this
  before every other step in the series.
---

# Step 1 — Pre-flight (Human Prerequisites)

**Series position**: Step 1 of 8 in the Qwen-Image-Edit AI PC fine-tuning walkthrough.
This step covers everything the agent cannot do unattended; the user works through each
item once.

**Next step**: when the checklist at the end is fully confirmed, proceed to skill
`qwen-image-edit-aipc-finetune-02-dataset`.

---

## Background & scope

This series fine-tunes **Qwen-Image-Edit** with LoRA on a single Intel AI PC (Core Ultra
processor) Windows laptop, written to be consumed by an agent acting for a non-expert user.

**Scope**: **NF4 QLoRA only.** Other recipes (bf16/fp16 LoRA, full fine-tuning) are out of
scope — 4-bit quantization of the transformer is what makes the workload fit on AI PC
unified memory.

**Recipe**: NF4-quantized transformer + LoRA + `bitsandbytes.optim.Adam8bit` + cache-first
workflow + mode-aware component loading. Single-card XPU training under `accelerate launch`
with `distributed_type: NO` and `mixed_precision: 'no'`.

**Minimum system requirement: 32 GB system RAM.** 16 GB is not supported — the NF4 DiT
alone needs ~10 GB of the unified budget, leaving too little for training activations.

**Trainer selection is model-driven, not machine-driven.** The series supports both
trainer classes; which one a given model needs is determined by that model's
`model_index.json` `_class_name` field, not by its version number:

| `model_index.json` `_class_name` | Trainer | Example models |
|---|---|---|
| `QwenImageEditPipeline` | `QwenImageEdit` | `Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | `QwenImageEditPlus` | `Qwen-Image-Edit-2509`, `Qwen-Image-Edit-2511` |

Selecting by `_class_name` (done automatically in Step 4) rather than by version string
means future checkpoints map correctly too: any model that declares one of these pipeline
classes resolves automatically, and a model declaring an unrecognized pipeline class fails
loudly rather than being guessed. Do not set the trainer by hand from the model name.

The two trainers differ in how control images reach the text encoder (the `QwenImageEdit`
trainer passes only the main control; `QwenImageEditPlus` passes all controls). They
otherwise use the same NF4 QLoRA recipe and the same XPU adaptations; the Plus trainer
needs its `load_model` adapted too (Step 5).

**Full workflow**: Step 1 human pre-flight → Step 2 dataset prep → Step 3 conda env →
Step 4 hardware probe + config → Step 5 framework adaptation → Step 6 conditional NF4
pre-quantization → Step 7 training → Step 8 validation.

---

## 1.1 Hardware sanity

- Intel AI PC with **Core Ultra** processor and Intel iGPU.
- Note your RAM size in GB (16 / 32 / 64 / 64+) — you'll pass it at Step 4 as
  `--ram-tier`. **32 GB is the minimum.**

> **Size convention (whole series).** All sizes here — disk, model files, RAM, VRAM — are
> given the way **Windows displays them**: binary GiB written as "GB" (e.g. Windows shows
> a 9.82 GiB file as "9.82 GB"; a 32 GB machine shows ~31 GB of RAM in Task Manager).
> `probe_hw.py` (Step 3) reports `ram_gb`/`vram_gb` in these same GiB units so you can
> check them against your machine.

## 1.2 Disk space

The estimates below cover **training-related components only** (model, data, outputs,
conda env). Prerequisite tools required by this step (Intel Arc driver, oneAPI, Visual
Studio, etc.) need additional space on top of this — consult their installers.

Estimate at minimum **~65 GB free** on the target drive for training components; **~80 GB**
if both NF4 pre-quantizations (Step 6) are performed.

| Item | Approx. size |
|---|---|
| Qwen-Image-Edit model repository (the downloaded pipeline, unmodified) | ~54 GB |
| NF4-quantized transformer output (Step 6, optional) | ~10 GB |
| NF4-quantized text encoder output (Step 6, optional) | ~5.5 GB |
| Dataset (depends on yours; character-composition reference is small) | ~1–5 GB |
| Embedding cache (proportional to dataset size) | ~0.5 MB/sample (e.g. ~18 MB for 35 samples) |
| Training checkpoints (LoRA adapter only — no base weights; the resumable `-last-` ones additionally carry optimizer/scheduler/RNG state and are several times larger) | tens of MB for an adapter-only checkpoint, × N. Scales with LoRA rank, and roughly doubles if you apply the Step 7 fp32-adapter fix. `train.checkpoints_total_limit` is declared in the config schema but never wired into accelerate's `ProjectConfiguration`, so nothing prunes them — delete old ones by hand (Step 7). |
| Conda env | ~6 GB |

## 1.3 Windows registry settings (admin Command Prompt)

Two registry tweaks make AI PC fine-tuning materially smoother. Open an **admin Command
Prompt** (right-click Start → "Terminal (Admin)" or search "cmd" and run as administrator):

**Long-path support** — model file paths can exceed Windows' 260-char limit:

```bat
powershell -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1"
```

**Shared GPU memory ceiling** — AI PC iGPUs use unified memory; raising this ceiling lets
the OS commit more memory to the GPU process during the NF4 cache phase. Open Registry
Editor (Win+R → `regedit`), navigate to:

```
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager
```

Change `SystemPartitionCommitLimitPercentage` to a higher value (e.g. **75** from the
default ~57). Reboot for both changes to take effect.

## 1.4 Intel GPU driver

Install the latest Intel Arc Graphics driver:

https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html

Verify after install: Windows Device Manager → Display adapters → "Intel Arc Graphics"
appears without warnings.

## 1.5 Conda / Python

Install Miniforge from conda-forge:

https://conda-forge.org/download/

Confirm `conda --version` works in CMD after install (may require new shell session).

## 1.6 Visual Studio Community (oneAPI prerequisite)

Required to provide the C++ build toolchain that oneAPI uses for Triton JIT kernel
compilation:

https://visualstudio.microsoft.com/vs/community/

During install, check the **"Desktop development with C++"** workload. Other workloads are
not required.

## 1.7 Intel oneAPI Base Toolkit

Required for bitsandbytes Triton-backed XPU kernels (NF4 quantize on first load,
`Adam8bit.step()` every step). Install **after** §1.6 (oneAPI's installer detects the C++
toolchain).

Match the oneAPI version to the `torch+xpu` version you plan to install in Step 3:

| `torch+xpu` | Triton package | oneAPI version |
|---|---|---|
| `2.9.0+xpu` | `pytorch-triton-xpu` 3.5 | 2025.2 |
| `2.9.1+xpu` | `pytorch-triton-xpu` 3.5 | 2025.2 |
| `2.10.0+xpu` | `triton-xpu` 3.6 | 2025.3 |
| `2.11.0+xpu` | `triton-xpu` 3.7 | 2025.3 |
| `2.12.0+xpu` | `triton-xpu` 3.7.1 | 2025.3 |
| `2.12.1+xpu` | `triton-xpu` 3.7.1 | 2025.3 |
| `2.13.0+xpu` | `triton-xpu` 3.7.2 | 2026.1 |

Download from:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Default install path: `C:\Program Files (x86)\Intel\oneAPI`. The training launcher (Step 7)
invokes `setvars.bat` from this path; if you install elsewhere, update the launcher.

> **If your `torch+xpu` version isn't listed above** (e.g. pip pulls a newer release than
> this series anticipates): infer the matching oneAPI from pip's installed Intel transitive
> dependencies. Complete §1.5 (conda) + Step 3 §3.1–§3.2 first (create env +
> `pip install torch --index-url https://download.pytorch.org/whl/xpu`), then in the
> activated env run:
>
> ```bat
> pip list | findstr /I "intel- onemkl-sycl-"
> ```
>
> The major version of the `intel-*` and `onemkl-sycl-*` packages tells you which oneAPI
> release to install (e.g. seeing `2025.3.x` → install Intel oneAPI Base Toolkit **2025.3**).
> After oneAPI installs, return to Step 3 §3.3 to install the remaining project dependencies.

**Already have oneAPI installed?** Consider installing the `torch+xpu` version that matches
your existing oneAPI rather than upgrading oneAPI. Check which oneAPI version you have
(below), then install the corresponding `torch+xpu` in Step 3 — often less work than
reinstalling oneAPI.

Verify after install (fresh CMD):

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
where icpx
icpx --version
```

`where icpx` must print a real path. To read the exact toolkit version (e.g. `2025.3`),
look at the `InstalledDir` line in the `icpx --version` output — the version directory in
the path is the toolkit version:

```
InstalledDir: C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\compiler
                                                           ^^^^^
                                             this is the toolkit version
```

(`icpx --version` also prints `Compiler 2025.3.3 ...` — the three-part compiler build
version may differ slightly from the two-part toolkit version `2025.3` in the path. Use the
path number to match against the §1.7 table.)

Alternatively, open `C:\Program Files (x86)\Intel\oneAPI\Installer\installer.exe` to see
the installed toolkit version in the UI.

If `setvars.bat` reports `'vars.bat' is not recognized`, see Troubleshooting below.

**Note the oneAPI version** — you will need it at Step 3 to pick the matching `torch+xpu`.

## 1.8 Download Qwen-Image-Edit model

Qwen-Image-Edit is an open model — no HuggingFace account or token is required. Download to
a local directory that the agent will use as `pretrained_model_name_or_path` (~54 GB):

```bat
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir <path\to\model>
```

(https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

Other Qwen-Image-Edit releases (e.g. `Qwen-Image-Edit-2509`, or the base `Qwen-Image-Edit`)
also work; replace the repo ID and local-dir path accordingly. The model you download
determines the trainer — Step 4 detects and sets it automatically from the model's
`model_index.json`, so no manual trainer choice is needed here.

## 1.9 Confirm the training framework is cloned

The `qwen-image-finetune` framework must be cloned to a local directory before starting — this
is a one-time human action done from the README's setup instructions (it clones the upstream
repo with a Windows-safe checkout). Confirm the clone exists:

- The project directory is on disk and non-empty.
- `git status` inside it runs without error (two paths may show as `deleted` — expected on
  Windows; it does not affect the workflow).

This is the unmodified upstream repo. The agent will apply all XPU adaptations (Step 5) during
setup — they are not auto-applied. If the clone is missing, direct the user to the README's
setup section before continuing.

## 1.10 Pre-flight checklist

Confirm each item before signaling the agent to proceed to Step 2:

- [ ] AI PC with Core Ultra processor confirmed
- [ ] RAM tier noted (16 / 32 / 64 / 64+ GB; ≥ 32 GB required) for `--ram-tier` at Step 4
- [ ] ≥ 65 GB free disk space on target drive (≥ 80 GB if pre-quantizing both transformer + text_encoder, Step 6)
- [ ] Long-path registry enabled; GPU shared memory ceiling adjusted (75); reboot done
- [ ] Intel Arc Graphics driver installed (Device Manager shows iGPU clean)
- [ ] conda installed (`conda --version` works)
- [ ] Visual Studio Community installed with "Desktop development with C++" workload
- [ ] Intel oneAPI Base Toolkit installed (`where icpx` works in a fresh CMD after `setvars.bat`); **oneAPI version noted**
- [ ] Qwen-Image-Edit model downloaded to a local directory; **model path noted**
- [ ] `qwen-image-finetune` project cloned to a local directory; **project path noted**
- [ ] Training dataset prepared (or ready to be prepared at Step 2)

## PASS signal

All checklist items confirmed by the user.

**Proceed to**: `qwen-image-edit-aipc-finetune-02-dataset`

---

## Troubleshooting

**`setvars.bat` reports `'vars.bat' is not recognized`**: the
`NoDefaultCurrentDirectoryInExePath` environment/registry setting blocks loading from the
current directory across the `setvars.bat` call. The training launcher (Step 7) handles
this automatically by clearing and restoring the variable. For manual verification in this
step, run from a CMD opened directly (not from inside another wrapper shell):

```bat
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
```

**Machine reboots unexpectedly (later, during training)**: the machine restarts without a
BSOD, especially under high memory or XPU load, or hangs then reboots. First thing to try:
update the Intel Arc Graphics driver (§1.4) and reboot. **After the reboot, re-verify
the shared GPU memory setting** — driver installation can reset `SystemPartitionCommitLimitPercentage`
back to the default (~57). Re-apply §1.3 if it reverted, and reboot again.
