---
name: qwen-image-edit-aipc-finetune-05-adaptation
description: >
  Step 5 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-04-config. Apply all required XPU/NF4 source patches to the
  qwen-image-finetune clone: device_utils.py, residual CUDA/flash-attention fixes, the
  accelerate XPU config, Pydantic schema patches, the NF4 QLoRA autocast/dtype patches,
  mode-aware loading, and (only for QwenImageEditPlus models) the Plus trainer load_model.
  Ends with an import test and an adaptation checker. Only use once step 4 config lint passes.
---

# Step 5 — Framework Adaptation (XPU + NF4 Patches)

**Series position**: Step 5 of 8.

> **Scope**: this step adapts a clone of `tsiendragon/qwen-image-finetune` for XPU. The
> framework is a custom training loop (not HF Trainer), so it hardcodes `torch.cuda.*` and
> CUDA-only paths. If your clone is already adapted (has `device_utils.py` and the patches
> below), some or all steps may already be done — run the §5.8 import test + checker to confirm
> before re-applying anything.

**Prerequisites**:
- `config.yaml` with `trainer:` set (Step 4) — determines whether the Plus section (§5.7) applies
- The `qwen-image-finetune` project directory (Step 1)

**Next step**: when `adaptation_check.py` exits 0, proceed to skill
`qwen-image-edit-aipc-finetune-06-quantize`.

---

## Which trainer files to adapt

§5.1–§5.6 are framework-wide and **always apply, including to the base trainer
(`qwen_image_edit_trainer.py`)** — even when your model uses the Plus trainer, because
`QwenImageEditPlusTrainer` *inherits* the base trainer's training loop (that's where the NF4-A2
autocast lives) and its device-wrapped helpers. So: always do §5.1–§5.6.

**Then, only if `detect_trainer.py` selected `QwenImageEditPlus`** (Step 4), also do §5.7 —
adapting the Plus trainer's own `load_model`. (The Plus trainer overrides only `load_model` and
the multi-image methods; everything else it inherits from the base trainer you already adapted.)

## 5.1 Device-agnostic helpers (`device_utils.py`)

Replace each `torch.cuda.*` call in the framework with a device-agnostic wrapper that checks XPU
first. This confines the XPU adaptation to the framework's own source files and leaves
third-party libraries (Triton, bitsandbytes, accelerate) untouched.

In qwen-image-finetune the wrappers live at `src/qflux/utils/device_utils.py`. Create this file
(before any other change if adapting a fresh clone) with exactly this content:

```python
# src/qflux/utils/device_utils.py
# enable XPU: device-agnostic helpers — call these instead of torch.cuda.* directly.
import contextlib
import torch


def device_empty_cache() -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_synchronize() -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def device_manual_seed_all(seed: int) -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)


def device_context(device):
    """Device context manager replacing torch.cuda.device().

    torch.xpu.device("cpu") raises; this helper returns nullcontext() for
    CPU and the correct accelerator context otherwise.
    """
    device_str = str(device) if device is not None else "cpu"
    if device_str == "cpu":
        return contextlib.nullcontext()
    if torch.xpu.is_available():           # enable XPU
        return torch.xpu.device(device)
    return torch.cuda.device(device)
```

Replace each direct `torch.cuda.*` call in the framework:

| Original | Replacement | Import |
|---|---|---|
| `torch.cuda.empty_cache()` | `device_empty_cache()` | `from qflux.utils.device_utils import device_empty_cache` |
| `torch.cuda.synchronize()` | `device_synchronize()` | `from qflux.utils.device_utils import device_synchronize` |
| `torch.cuda.manual_seed_all(s)` | `device_manual_seed_all(s)` | `from qflux.utils.device_utils import device_manual_seed_all` |
| `with torch.cuda.device(dev):` | `with device_context(dev):` | `from qflux.utils.device_utils import device_context` |

Import the helpers from the submodule directly (as in the table); there is no need to re-export
them from `qflux/utils/__init__.py`.

## 5.2 Find and fix residual patterns

The helpers cover functional calls. A separate scan is needed for string-based device comparisons
and CUDA-only library imports, which helpers cannot address:

```bash
grep -rn "torch\.cuda\.\|flash_attention_2\|transformer_engine\|device\.type.*cuda\|Qwen/Qwen-Image-Edit" src/qflux/
```

The scan covers **both** trainer files (`qwen_image_edit_trainer.py` and
`qwen_image_edit_plus_trainer.py`) — fix every hit in whichever one(s) your model uses.

| Pattern | Fix |
|---|---|
| `flash_attention_2` (in `from_pretrained` calls or `config.json`) | Replace with `"sdpa"`. Flash Attention 2 is CUDA-only |
| `transformer_engine` import / use | Comment out — CUDA-only library |
| `device.type == "cuda"` literal compare | Add an `xpu` branch (string compare can't be patched) |
| Hardcoded model id in `load_model` — `load_vae("Qwen/Qwen-Image-Edit", …)`, `load_qwenvl("Qwen/Qwen-Image-Edit", …)` (base) or `"Qwen/Qwen-Image-Edit-2509"` (Plus) | Replace with `self.config.model.pretrained_model_name_or_path`. The hardcoded id makes the trainer **fetch that repo from the network** instead of using the model you already downloaded locally (Step 1) — wasting time, and loading the wrong model's VAE/text_encoder if you trained on a different checkpoint. Point it at your local model dir. |
| `mp.set_start_method("spawn", force=True)` | Guard with `if not torch.xpu.is_available(): ...`. Windows defaults to `spawn` already; `force=True` raises `RuntimeError: context has already been set` |
| `torch.cuda.device(device)` context manager | Replace with `device_context(device)` from `device_utils.py`. `torch.xpu.device("cpu")` raises when `device="cpu"` (e.g. text_encoder placed on CPU for inference); `device_context()` returns `nullcontext()` for CPU |
| `train_epoch()` ignores `max_train_steps` — runs `num_epochs × dataset_size` steps instead | **Two edits, both needed.** (1) Guard the per-batch loop in `train_epoch()`: `if self.global_step >= self.config.train.max_train_steps: return`. (2) Break the epoch loop in `fit()` on the same condition: `if self.training_interrupted or self.global_step >= self.config.train.max_train_steps: break`. See the note below — edit (1) alone stops training but not the epoch loop. |

Example `device.type` fix in a config validator:

```python
if d.type == "cuda":
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA not available but got device={d}.")
# enable XPU: validate XPU device availability
if d.type == "xpu" and not torch.xpu.is_available():
    raise ValueError(f"XPU not available but got device={d}.")
```

> **Why the `max_train_steps` fix needs both edits.** `train_epoch()`'s guard saves a final
> checkpoint and `return`s — but it only returns from `train_epoch()`. `fit()`'s
> `for epoch in range(...)` loop keeps going, so every remaining epoch re-enters `train_epoch()`,
> trips the guard again, and writes **another** full resumable checkpoint before returning. With
> `num_epochs` set well above the epoch count `max_train_steps` actually reaches (the Step 4
> recommender's default pairing), that is one redundant checkpoint directory per remaining epoch,
> each carrying a complete copy of the optimizer state — enough to fill a disk on a long run. The
> symptom is a pile of `checkpoint-last-<epoch>-<step>-last` directories that all share the **same**
> step number and differ only by epoch. Nothing prunes them: `train.checkpoints_total_limit` is
> declared in the schema but never wired into `ProjectConfiguration`, so it has no effect.
>
> The copies are byte-identical, so a run that skipped edit (2) still trained correctly — the cost
> is disk and save time, not the model. `adaptation_check.py` reports the missing break as a WARN
> for that reason, not an ERROR.
>
> Breaking the epoch loop as well makes the guard write exactly one final checkpoint. `fit()`'s own
> closing `save_checkpoint(...)` reuses the same epoch and step, hence the same directory name, so it
> overwrites rather than adding another.

## 5.3 accelerate single-card XPU config

`qflux.main` launches via `accelerate launch --config_file <path>`. Create an XPU-specific config
`accelerate_config_xpu.yaml` in the project root:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
mixed_precision: 'no'
num_machines: 1
num_processes: 1
main_training_function: main
dynamo_backend: 'no'
use_cpu: false
deepspeed_config: {}
```

Key differences from a typical CUDA multi-GPU config:

| Field | CUDA multi-GPU | XPU single-card | Why |
|---|---|---|---|
| `distributed_type` | `MULTI_GPU` | `NO` | Single process; no collective ops |
| `mixed_precision` | `bf16` | `'no'` | Set `'no'` so this file matches what actually runs: `base_trainer` passes a hardcoded `mixed_precision="no"` to `Accelerator`, and an explicit argument overrides whatever the launch config says. Mixed precision is applied by the trainer's own `torch.autocast` (§5.5.3 NF4-A2) |
| `dynamo_backend` | `inductor` | `'no'` | This recipe does not use `torch.compile`; leave the wrapper out of the path |

> **YAML gotcha**: `dynamo_backend: no` unquoted parses as YAML boolean `False`; accelerate then
> calls `.upper()` on it and raises `AttributeError: 'bool' object has no attribute 'upper'`.
> Always quote `'no'` (and same for `mixed_precision: 'no'`).

## 5.4 Pydantic schema patches

Two edits to `src/qflux/data/config.py` so the NF4 config the recommender emits validates against
the upstream schema.

**a) Add the `quantize_type` field to `ModelConfig`.** Upstream `ModelConfig` has `quantize: bool`
but **no `quantize_type` field**, and it is declared with `ConfigDict(extra="forbid")` — so any
config carrying `model.quantize_type: nf4` fails immediately with `pydantic ... Extra inputs are
not permitted`. Add the field:

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...
    quantize: bool = False
    quantize_type: str | None = None   # enable XPU: "nf4" selects NF4 QLoRA loading
```

The NF4 loader (§5.5 / Step 6) reads `quantize_type == "nf4"` to pass `use_nf4=True`; without this
field the whole NF4 recipe is unreachable.

**b) Relax `num_workers` to `>= 0`.** Upstream declares `num_workers` with a `> 0` constraint. On
Windows under `spawn` start-method, classes registered via `trust_remote_code` are not picklable
across processes — workers crash on dataset iteration, and `num_workers = 0` (the fix) is rejected
by the schema. Relax the constraint to `>= 0` in the dataset config validator. The recommender
defaults to `1`, but `0` should be acceptable as an escape hatch. (This is one of two Windows
`spawn`-related fixes — the other is the `mp.set_start_method` guard in §5.2; they are independent.)

> **Cache key default (`prompt_empty_drop_keys`).** Upstream `CacheConfig` defaults this to
> `["prompt_embed", ...]` — note the missing `s`. The cached key is `prompt_embeds`, so
> caption-dropout raises `KeyError: 'prompt_embed'`. The recommender sidesteps this by **emitting
> `cache.prompt_empty_drop_keys: ["prompt_embeds", "prompt_embeds_mask"]`** in the config, so no
> source edit is required as long as you use the generated config. (If you hand-write a config,
> set this key yourself.)

## 5.5 BitsAndBytes XPU backend + NF4 QLoRA required patches

### 5.5.1 BitsAndBytes XPU backend (informational)

Knowing which bnb operations dispatch to which backend on XPU helps diagnose failures and explains
why Step 1 requires Visual Studio + oneAPI. For `bitsandbytes >= 0.48.2`:

| Operation | Backend | When it runs |
|---|---|---|
| `dequantize_4bit` | **SYCL** | Every NF4 forward and backward pass |
| `dequantize_blockwise` | **SYCL** | Used alongside NF4 dequant |
| `gemv_4bit` | **SYCL** | Inference-only single-token path |
| `quantize_4bit`, `quantize_blockwise` | **Triton** | Initial quantization (e.g. Step 6 pre-quantize) |
| `optimizer_update_8bit_blockwise`, `optimizer_update_32bit` | **Triton** | Every `bnb.optim.Adam8bit.step()` call |

**Key consequence**: NF4 forward/backward does not use a dedicated NF4 gemm kernel — bnb
dequantizes the 4-bit weight via SYCL and hands off to PyTorch's native XPU matmul. Only the
**optimizer step** and **initial quantization** actually go through Triton.

**Practical impact**: if a training run loads a pre-quantized NF4 checkpoint (Step 6 output) and
uses `Adam8bit`, Triton kernel JIT compile runs on the first optimizer step — this is why Step 1
oneAPI + Visual Studio C++ workload are required.

The dispatch above is empirical for `bitsandbytes 0.49.2`. Newer versions may shift; if odd
failures appear, check `bitsandbytes/backends/xpu/ops.py` in your installed bnb version for the
current registration.

### 5.5.2 NF4-A1 — the NF4 loading path in `load_model.py`

**Upstream has no NF4 loading path at all.** `load_model.py` never passes a `quantization_config`,
and `ModelConfig` has no `quantize_type` field (§5.4a). `src/qflux/models/quantize.py` contains
4-bit and fp8 helpers, but the loaders and trainers do not call them. So NF4 QLoRA is not a setting
you switch on — the loading path has to be added.

Add an NF4 branch to `load_transformer` (the DiT) and `load_qwenvl` (the text_encoder), selected by
`self.config.model.quantize_type == "nf4"` and passed down from the trainer as `use_nf4=`. Four
details are easy to get wrong:

- **Use the matching `BitsAndBytesConfig` per component.** The DiT is a diffusers `ModelMixin` and
  the text_encoder is a transformers model; each library ships its own `BitsAndBytesConfig` class and
  each `from_pretrained` expects its own. Import `diffusers.BitsAndBytesConfig` for the DiT and
  `transformers.BitsAndBytesConfig` for the text_encoder.
- **Do not pass `device_map` on the NF4 branch.** Upstream hardcodes `device_map="cpu"`; leave it off
  when `quantization_config` is set and let bitsandbytes place the weights. Keep `device_map` only on
  the non-quantized branch.
- **Keep `torch_dtype=weight_dtype`.** Upstream already passes it on the bf16 path; the risk is
  dropping it while restructuring the kwargs for the NF4 branch.
- **Pre-quantized directories load directly** — no `quantization_config` needed. A
  `transformer_path` / `text_encoder_path` produced in Step 6 already carries `quantization_config`
  in its own `config.json`, so `from_pretrained(<that dir>)` re-loads it as NF4 by itself. Branch on
  the path being set *before* the online-NF4 branch.

In qwen-image-finetune: `src/qflux/models/load_model.py` — see `_make_nf4_config` and the
`use_nf4` / `transformer_path` branches.

```python
# load_transformer, NF4 branch
kwargs = dict(
    subfolder="transformer",
    torch_dtype=weight_dtype,
    use_safetensors=True,
    attn_implementation="sdpa",          # see §5.2
)
if use_nf4:
    from diffusers import BitsAndBytesConfig   # diffusers' own class, for the DiT
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=weight_dtype,
        bnb_4bit_use_double_quant=True,
    )
else:
    kwargs["device_map"] = device_map     # non-quantized path only
```

### 5.5.3 NF4-A2 — `torch.autocast` wrap around the trainable forward

**What it is**: the only autocast on the training path. `base_trainer.py` constructs the
`Accelerator` with `mixed_precision="no"` hardcoded — the line reading it from
`train.mixed_precision` is commented out — so accelerate's own autocast hook never engages (it is
gated on a fp16/bf16/fp8 setting). Nothing else wraps the forward, so this is where mixed
precision comes from.

**Symptom if missing**: fit fails on step 1 with a dtype-mismatch `RuntimeError` from the attention
op.

**Fix**: wrap the trainable model's forward in `torch.autocast`. In qwen-image-finetune this lives
in `src/qflux/trainer/qwen_image_edit_trainer.py` — search for
`torch.autocast(_autocast_device, dtype=self.weight_dtype)`:

```python
_autocast_device = "xpu" if torch.xpu.is_available() else "cuda"
with torch.autocast(_autocast_device, dtype=weight_dtype):
    output = model(inputs, ...)
```

It coerces the ops inside to `weight_dtype` without changing effective training precision, so no
dtype needs to be set anywhere in the YAML.

## 5.6 Mode-aware loading (required)

This is **required for this skill's cache-first NF4 QLoRA recipe**. It skips the component the
current phase never uses (the DiT during cache, the text_encoder during fit), freeing that memory
for the work that does run. On a 32 GB machine it is routinely the difference between fitting and
an XPU OOM / `DEVICE_LOST`; on larger machines it prevents wasted loads and keeps behavior
deterministic.

**Required behavior**:
- Cache phase (`--cache`): `skip_dit` is required — DiT is never invoked during embedding
  pre-compute, so loading it only wastes memory.
- Fit phase: `skip_text_encoder` is required when `cache.use_cache == true`, cache files exist on
  disk, AND `validation.enabled == false`.
- Exception: if `validation.enabled == true`, the text_encoder must stay loaded for live
  validation samples. This is a deliberate opt-out and is generally not viable on 32 GB unless
  Step 6 NF4 text_encoder is set.

| Phase / mode | Skipped component | Why it helps | Why it's safe |
|---|---|---|---|
| `--cache` (TrMode.cache) | DiT (the NF4 transformer, ~10 GB — the Step 6 quantized size) | The cache phase only runs text_encoder + VAE; the DiT is never called, so loading it just parks ~10 GB NF4 DiT on the XPU for nothing (and, for online NF4, runs an unnecessary quantization pass). On a 32 GB machine that wasted ~10 GB is a common OOM / `DEVICE_LOST` trigger. | DiT is never invoked during cache, so skipping it cannot change cache output |
| `fit` + `cache.use_cache=true` + cache populated + validation disabled | text_encoder (Qwen2.5-VL bf16, ~15 GB; ~5.5 GB if NF4 Step 6) | With a populated cache, every training sample already has its embedding, so the text_encoder is dead weight during fit. Skipping it frees that ~15 GB (≈5.5 GB if NF4) for the DiT + optimizer. | All samples have pre-computed embeddings; the text_encoder is only needed for live encoding (e.g. validation samples) |

VAE is **always** loaded and is not subject to any skip logic. Unlike text_encoder (~15 GB), VAE
is only ~250 MB — unloading it saves negligible memory. Additionally, its config attributes
(`scale_factor`, `latent_mean`, `latent_std`, `z_dim`) are used throughout the trainer's forward
pass regardless of mode, so it must remain resident. The recommended cache device is `xpu:0`
because VAE is small and cache pass throughput benefits from accelerator execution.

Implementation pattern (in the selected trainer's `load_model`, after applying §5.1–§5.5 —
`qwen_image_edit_trainer.py` for the base trainer, `qwen_image_edit_plus_trainer.py` for Plus per §5.7):

```python
from qflux.data.config import TrMode

is_cache_mode = self.config.mode == TrMode.cache
cache_will_serve_all = (
    self.config.mode == TrMode.fit
    and self.use_cache
    and self.cache_exist
    and not self.config.validation.enabled
)
skip_dit = is_cache_mode
skip_text_encoder = cache_will_serve_all
```

Then use `skip_dit` / `skip_text_encoder` to conditionally skip those component loads.

Required config flags to enable the normal fit-skip path:

```yaml
cache:
  use_cache: true        # cache-first workflow (Step 7)
  cache_dir: <path>      # cache must exist before fit starts
validation:
  enabled: false         # or run validation in a separate pass after fit
```

If you need validation during fit (periodic sample generation), the text_encoder MUST stay loaded
— the ~15 GB cost is unavoidable. On 32 GB machines this typically forces NF4 text_encoder
(Step 6). Otherwise keep `validation.enabled: false` and run Step 8 post-training validation.

This applies to **whichever trainer your model selects** — the base and Plus trainers use the same
`load_model` skip logic (Plus gets it via §5.7). `adaptation_check.py` fails when the selected
trainer lacks `skip_dit` or `skip_text_encoder`.

## 5.7 Plus trainer: apply the same `load_model` adaptation

**Apply only if `detect_trainer.py` selected `QwenImageEditPlus`** (2509/2511 models). For the base
`QwenImageEdit` model, skip this — §5.1–§5.6 already covered its trainer.

The Plus trainer (`qwen_image_edit_plus_trainer.py`) inherits from the base trainer and overrides
`load_model`. There is **nothing new to learn here**: give its `load_model` the same adaptations the
base trainer's `load_model` got, so the two end up matching line-for-line **except** that Plus keeps
`QwenImageEditPlusPipeline`. Concretely, the same three things you already did to the base
`load_model` apply here:

- §5.2 — hardcoded model paths → `self.config.model.pretrained_model_name_or_path`;
- §5.6 — the `skip_dit` / `skip_text_encoder` mode-aware block (do not re-derive it here; copy the
  §5.6 pattern, which is identical for both trainers);
- the NF4 wiring — `use_nf4=`, `transformer_path`, `text_encoder_path`.

The multi-image methods (`prepare_embeddings`, `_get_qwen_prompt_embeds`) and the NF4-A2 autocast
(§5.5, in the inherited training step) need no change.

Two details where the upstream Plus `load_model` is *more* unadapted than the base one — easy to
miss, so confirm both:

- its `from_pretrained(...)` is missing `text_encoder=None` (add it — otherwise a redundant ~15 GB
  text_encoder copy loads via the pipe);
- its `load_transformer(...)` call omits `use_nf4=`, so online NF4 silently loads bf16 — add
  `use_nf4=(self.config.model.quantize_type == "nf4")`.

Verify by diffing your Plus `load_model` against the base trainer's, and re-run the §5.8 import test
with the Plus trainer line added.

## 5.8 Verify the adaptation before training

**Step 1 — import test.** Write a throwaway file that loads each module you touched:

```python
# _test_residual_cuda.py — delete after use
import os, sys
sys.path.insert(0, "src")
os.environ["QFLUX_DOTENV_LOADED"] = "1"  # skip HF login on package import
import torch

# Add one line per modified module — examples:
from qflux.utils.device_utils import device_empty_cache  # noqa: F401
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer  # noqa: F401
# If trainer is QwenImageEditPlus, also add:
# from qflux.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer  # noqa: F401
print("[OK] all modified modules imported cleanly")

# If you touched attn_implementation logic, also assert:
# from qflux.models.load_model import _attn_implementation
# assert _attn_implementation() == "sdpa"
```

A clean run = no syntax errors and no broken references introduced by your changes.

**Step 2 — adaptation checker.** It greps the source for the markers each patch leaves behind and
catches a *partial* adaptation before the cache phase. Save this as `adaptation_check.py` in the
project root:

```python
# adaptation_check.py
# Static verifier that the Step 5 XPU/NF4 adaptations were applied to the repo clone. Greps source
# for the markers each patch leaves behind; fails (exit 1) on any ERROR so an incomplete adaptation
# is caught before the cache phase.
# Usage:
#   python adaptation_check.py --src src
#   python adaptation_check.py --src src --config config.yaml
# With --config, only the trainer named in the YAML is checked for NF4 wiring / mode-aware loading.
# Without it, the base trainer is always checked and the Plus trainer too if present.
# Network policy: reads only local source files.
import argparse, os, re, sys

TRAINER_FILE = {"QwenImageEdit": "qwen_image_edit_trainer.py",
                "QwenImageEditPlus": "qwen_image_edit_plus_trainer.py"}


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _method_body(text, name):
    """Return the source of a method (def <name> -> next def at the same indent)."""
    m = re.search(rf"\n(\s*)def {re.escape(name)}\b", text)
    if not m:
        return ""
    indent = m.group(1)
    nxt = re.search(rf"\n{indent}def \w+", text[m.end():])
    end = m.end() + nxt.start() if nxt else len(text)
    return text[m.start():end]


def _load_model_body(text):
    return _method_body(text, "load_model")


def _trainer_from_config(config_path):
    text = _read(config_path)
    if text is None:
        return None
    m = re.search(r"^\s*trainer\s*:\s*([A-Za-z0-9_]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def check(src, config_path):
    errors, warnings = [], []
    qflux = os.path.join(src, "qflux")

    # §5.1 device_utils.py
    if not os.path.isfile(os.path.join(qflux, "utils", "device_utils.py")):
        errors.append("§5.1: src/qflux/utils/device_utils.py is missing — create it and route torch.cuda.* calls through it.")

    # §5.4a quantize_type field + prompt_embed typo
    cfg_text = _read(os.path.join(qflux, "data", "config.py"))
    if cfg_text is None:
        errors.append("could not read src/qflux/data/config.py")
    else:
        if "quantize_type" not in cfg_text:
            errors.append("§5.4a: ModelConfig has no quantize_type field — add `quantize_type: str | None = None` (else extra='forbid' rejects the NF4 config).")
        if re.search(r'prompt_empty_drop_keys[^\n]*"prompt_embed"', cfg_text):
            warnings.append('§5.4: CacheConfig default still uses the "prompt_embed" typo (missing \'s\'). Harmless if your config sets prompt_empty_drop_keys (the recommender does); otherwise caption-dropout raises KeyError: \'prompt_embed\'.')

    # §5.2 max_train_steps guard
    bt = _read(os.path.join(qflux, "trainer", "base_trainer.py"))
    if bt is None:
        errors.append("could not read src/qflux/trainer/base_trainer.py")
    else:
        STEP_GUARD = r"global_step\s*>=\s*self\.config\.train\.max_train_steps"
        if not re.search(STEP_GUARD, _method_body(bt, "train_epoch")):
            errors.append("§5.2: train_epoch has no max_train_steps guard — training will run num_epochs × dataset_size steps. Add `if self.global_step >= self.config.train.max_train_steps: return` in the per-batch loop.")
        # The guard only returns from train_epoch; fit()'s epoch loop keeps going and re-trips it
        # once per remaining epoch, each time writing another resumable checkpoint. WARN, not ERROR:
        # those copies are byte-identical, so the trained result is unaffected — it costs disk and
        # save time. WARN rather than silent because nothing else reports it.
        if not re.search(STEP_GUARD, _method_body(bt, "fit")):
            warnings.append("§5.2: fit()'s epoch loop does not break on max_train_steps, so every epoch past it re-trips the train_epoch guard and writes another (byte-identical) resumable checkpoint. Training is still correct; the cost is disk and save time. Add `if self.training_interrupted or self.global_step >= self.config.train.max_train_steps: break` to the epoch loop.")

    # §5.5 NF4-A2: torch.autocast wrap in the base trainer (Plus inherits). WARN not ERROR:
    # a missing wrap fails loudly at fit step 1 anyway.
    base_tr = _read(os.path.join(qflux, "trainer", "qwen_image_edit_trainer.py"))
    if base_tr is not None and "autocast" not in base_tr:
        warnings.append("§5.5 NF4-A2: no torch.autocast wrap found in qwen_image_edit_trainer.py — it is the only autocast on the XPU path, since the Accelerator runs with mixed_precision='no'. Without it, fit fails with a dtype mismatch on step 1.")

    # which trainer(s) to check for NF4 wiring / mode-aware loading
    selected = _trainer_from_config(config_path) if config_path else None
    targets = [selected] if selected in TRAINER_FILE else ["QwenImageEdit"]
    if selected not in TRAINER_FILE and os.path.isfile(os.path.join(qflux, "trainer", TRAINER_FILE["QwenImageEditPlus"])):
        targets.append("QwenImageEditPlus")

    for tname in targets:
        tpath = os.path.join(qflux, "trainer", TRAINER_FILE[tname])
        ttext = _read(tpath)
        if ttext is None:
            warnings.append(f"{tname}: {TRAINER_FILE[tname]} not found — skipped.")
            continue
        body = _load_model_body(ttext)
        if not body:
            warnings.append(f"{tname}: no load_model found in {TRAINER_FILE[tname]}.")
            continue
        if not any(marker in body for marker in ("use_nf4", "quantization_type", "quantize_type")):
            errors.append(f"§5.5/§5.7: {tname}.load_model does not wire NF4 loading — online NF4 may silently load bf16. Pass the config's NF4 selector to load_transformer, e.g. `use_nf4=(self.config.model.quantize_type == 'nf4')`.")
        if "skip_dit" not in body:
            errors.append(f"§5.6/§5.7: {tname}.load_model has no skip_dit/mode-aware block. This skill requires cache-first + mode-aware loading: skip DiT during cache and skip text_encoder during fit when cache is populated.")
        if "skip_text_encoder" not in body:
            errors.append(f"§5.6/§5.7: {tname}.load_model has no skip_text_encoder mode-aware fit path. This skill requires skipping the text_encoder during fit when cache is populated and validation is disabled.")
        if tname == "QwenImageEditPlus":
            m = re.search(r"QwenImageEditPlusPipeline\.from_pretrained\s*\(", body)
            if m:
                depth, start, args_block = 1, m.end(), None
                for i, ch in enumerate(body[start:], start):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            args_block = body[start:i]
                            break
                if args_block is not None and not re.search(r"\btext_encoder\s*=\s*None\b", args_block):
                    errors.append("§5.7: QwenImageEditPlusPipeline.from_pretrained is missing `text_encoder=None` — otherwise the pipeline loads a redundant ~15 GB text_encoder before the trainer loads its own copy.")

    # residual torch.cuda + flash_attention_2 in the files this recipe adapts
    scoped = [os.path.join("trainer", "base_trainer.py"),
              os.path.join("trainer", TRAINER_FILE["QwenImageEdit"]),
              os.path.join("trainer", TRAINER_FILE["QwenImageEditPlus"]),
              os.path.join("models", "load_model.py")]
    residual_cuda, flash = [], []
    for rel in scoped:
        ftext = _read(os.path.join(qflux, rel))
        if ftext is None:
            continue
        for i, line in enumerate(ftext.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "torch.cuda." in s and "is_available" not in s:
                residual_cuda.append(f"{rel}:{i}")
            if "flash_attention_2" in s and "return" not in s:
                flash.append(f"{rel}:{i}")
    if residual_cuda:
        warnings.append("§5.2: residual torch.cuda.<op> calls (review — some may be legitimate): " + ", ".join(residual_cuda[:8]) + (" …" if len(residual_cuda) > 8 else ""))
    if flash:
        warnings.append("§5.2: flash_attention_2 still referenced (use sdpa on XPU): " + ", ".join(flash[:8]) + (" …" if len(flash) > 8 else ""))

    return errors, warnings


def main():
    p = argparse.ArgumentParser(description="Verify the Step 5 XPU/NF4 adaptations in a repo clone.")
    p.add_argument("--src", default="src", help="Path to the repo's src/ directory")
    p.add_argument("--config", help="Optional training YAML; restricts the trainer check")
    args = p.parse_args()
    if not os.path.isdir(os.path.join(args.src, "qflux")):
        print(f"ERROR: {args.src}/qflux not found — pass the repo's src/ dir via --src.", file=sys.stderr)
        return 2
    errors, warnings = check(args.src, args.config)
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    if errors:
        print(f"\nFAIL: {len(errors)} error(s), {len(warnings)} warning(s). Apply the missing patches before the cache phase.")
        return 1
    print(f"\nPASS: 0 errors, {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run:

```bat
python adaptation_check.py --src src --config config.yaml
```

PASS (exit 0) = every required patch is present (§5.1 device_utils, §5.4a `quantize_type` field,
§5.2 `max_train_steps` guard, §5.5/§5.7 NF4 wiring, §5.6 mode-aware loading, and the Plus trainer's
`text_encoder=None`). It also WARNs on review items such as residual `torch.cuda`, the
`prompt_embed` typo, and a missing §5.5 NF4-A2 autocast fallback. Fix every ERROR; review the WARNs.
Pass `--config` so only the trainer your model uses is checked.

## PASS signal

- Import test prints `[OK] all modified modules imported cleanly`
- `adaptation_check.py --src src --config config.yaml` exits 0 (`PASS: 0 errors`) — mode-aware
  `skip_dit`/`skip_text_encoder` and (for Plus) `text_encoder=None` are required; review WARNs
  such as a missing §5.5 NF4-A2 autocast

**Proceed to**: `qwen-image-edit-aipc-finetune-06-quantize`

---

## Troubleshooting

**`Extra inputs are not permitted` on `quantize_type`**: §5.4a wasn't applied — `ModelConfig` is
missing the `quantize_type` field.

**`mp.set_start_method` raises `RuntimeError: context has already been set`**: Windows defaults to
`spawn`; the guard in §5.2 wasn't applied.

**`accelerator.device` is `cuda:0` instead of `xpu:0`**: `accelerate_config_xpu.yaml` (§5.3) is not
being passed to `accelerate launch`, or it has `distributed_type: MULTI_GPU` instead of `NO`.
Confirm the launcher passes `--config_file accelerate_config_xpu.yaml` and the config has
`distributed_type: NO` and `use_cpu: false`. **Fallback for older accelerate** (< 0.27) lacking
native XPU device detection:

```python
# After Accelerator() construction, force the device if detection failed:
if torch.xpu.is_available() and accelerator.device.type != "xpu":
    accelerator.state.device = torch.device("xpu:0")
```

**`RuntimeError: Expected all tensors to be on the same device`**: a `torch.cuda.current_device()`
call was not replaced with a device-agnostic wrapper. Run the §5.2 scan to locate it; replace with
the appropriate `device_utils.py` helper or an explicit `.to(self.accelerator.device)` call.
