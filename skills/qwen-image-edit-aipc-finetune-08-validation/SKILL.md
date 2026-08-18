---
name: qwen-image-edit-aipc-finetune-08-validation
description: >
  Step 8 of 8 (final) of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-07-training. Validate that the trained LoRA learned the intended
  edit: optional in-training TensorBoard sampling, post-training base-vs-LoRA visual comparison with
  inference_compare.py, quality/precision configs, and a three-dimension visual evaluation checklist.
  Only use once training has produced a checkpoint.
---

# Step 8 — Validation

**Series position**: Step 8 of 8 (final step).

Validation answers: **did the LoRA learn the intended edit?** The approach has three layers of
increasing depth — use as many as needed.

**Prerequisites**:
- Training complete; at least one checkpoint written (Step 7)
- Test samples set aside (Step 2)
- Conda env active; model paths known

---

## 8.1 In-training visual monitoring (TensorBoard sampling)

The framework can generate sample outputs during training at regular intervals and log them to
TensorBoard, giving a visual sense of progress without extra code.
*(Source: `docs/guide/validation_sampling.md`, qwen-image-finetune v3.1.0)*

**Configure in the training YAML:**

```yaml
validation:
  enabled: true
  steps: 100          # generate samples every 100 training steps
  max_samples: 2      # 2-5 is practical; more = slower validation pass
  seed: 42            # fixed seed for consistent comparison across checkpoints
  dataset:
    class_path: "qflux.data.dataset.ImageDataset"
    init_args:
      dataset_path:
        - split: test
          repo_id: <your-dataset>
      selected_control_indexes: [1]
      processor:
        class_path: "qflux.data.preprocess.ImageProcessor"
        init_args:
          process_type: center_crop
          target_size: [384, 672]
          controls_size: [[384, 672], [512, 512]]
```

**Start TensorBoard** (in a separate terminal):
```bash
tensorboard --logdir=outputs/<run_name>/logs
```
Then open the **Images** tab to compare outputs across checkpoints.

> ⚠️ **32 GB AI PC constraint**: `validation.enabled: true` is the explicit exception to the normal
> Step 5 fit-phase `skip_text_encoder` requirement, forcing `text_encoder` to stay in memory during
> fit. On 32 GB machines this causes OOM. Options:
> - **Recommended**: keep `validation.enabled: false` during training; run §8.2 post-training instead.
> - Train on a 64 GB+ machine where the extra ~15 GB fits.
> - Step 6 §6.2 NF4 text_encoder reduces text_encoder from ~15 GB to ~5.5 GB, which may make
>   in-training validation feasible on 32 GB — not tested with `validation.enabled: true`.

## 8.2 Post-training visual comparison (inference_compare.py)

After training completes, run inference twice on a held-out test set — once with the base model and
once with the trained LoRA — then compare the outputs.

**Getting a test split**

Use whichever source matches how you prepared your test set in Step 2 §2.8:
- **Image directory** (held-out folder with `training_images/` + `control_images/`): use `--test-dir`
- **Parquet** (directory containing `data/*.parquet`, or a direct `.parquet` file path): use `--test-parquet`
- **CSV** (same column format as the training CSV): use `--test-csv`

All three sources produce the same output; choose the one that matches your data.

**Save the script** as `inference_compare.py` in the project root:

```python
# inference_compare.py
# Run inference on a test split with and without LoRA for visual comparison.
# Supported test data sources (pass exactly one):
#   --test-parquet  parquet file or directory containing data/*.parquet files
#   --test-dir      local directory with training_images/ + control_images/ layout
#   --test-csv      CSV with columns: path_target, path_control[_N], prompt
# Usage:
#   python inference_compare.py --config config.yaml --test-dir <test-folder>
#       --output-dir outputs/compare/<run>/base --num-samples 5
#   python inference_compare.py --config config.yaml --test-dir <test-folder>
#       --output-dir outputs/compare/<run>/lora --num-samples 5
#       --lora-weight outputs/<run>/checkpoint-<step>/pytorch_lora_weights.safetensors
# Output per dir: sample_NN_<id>.png, sample_NN_<id>__target.png, manifest.json
# Prerequisite: run from the qwen-image-finetune project root with qflux importable (src/ on PYTHONPATH).
import argparse, csv, io, json, os, re, sys
from pathlib import Path
import PIL.Image
import torch
from qflux.data.config import Config, TrMode, load_config_from_yaml

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_IMG_DIRS = ["training_images", "images", "target_images", "target", "targets"]
_CTRL_DIRS = ["control_images", "control", "condition_images", "controls"]
_CTRL_EXTRA = re.compile(r"_control_(\d+)\.[a-zA-Z]+$")
_MASK_PAT = re.compile(r"_mask\.[a-zA-Z]+$")


def _open_image(path):
    return PIL.Image.open(path).convert("RGB")


def _read_parquet_samples(parquet_path, n):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is required for --test-parquet. Install: pip install pyarrow")
    if parquet_path.is_dir():
        candidates = sorted(parquet_path.glob("data/test-*.parquet"))
        if not candidates:
            candidates = sorted(parquet_path.glob("data/*.parquet"))
        if not candidates:
            raise SystemExit(f"No parquet files found under {parquet_path}/data/.")
        table = pq.read_table(candidates[0].as_posix())
    else:
        table = pq.read_table(parquet_path.as_posix())
    rows = table.slice(0, n).to_pylist()
    samples = []
    for i, row in enumerate(rows):
        ctrl_imgs = []
        for entry in row.get("control_images") or []:
            if isinstance(entry, dict) and "bytes" in entry and entry["bytes"]:
                ctrl_imgs.append(PIL.Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))
            elif isinstance(entry, (bytes, bytearray)):
                ctrl_imgs.append(PIL.Image.open(io.BytesIO(entry)).convert("RGB"))
        target_entry = row.get("target_image")
        tgt = None
        if isinstance(target_entry, dict) and target_entry.get("bytes"):
            tgt = PIL.Image.open(io.BytesIO(target_entry["bytes"])).convert("RGB")
        samples.append({"id": row.get("id", str(i)), "control_images": ctrl_imgs,
                        "target_image": tgt, "prompt": row.get("prompt", "")})
    return samples


def _read_local_dir_samples(dataset_dir, n):
    images_dir = next((dataset_dir / d for d in _IMG_DIRS if (dataset_dir / d).exists()), None)
    control_dir = next((dataset_dir / d for d in _CTRL_DIRS if (dataset_dir / d).exists()), None)
    if images_dir is None or control_dir is None:
        raise SystemExit(f"Could not find image/control subdirectories in {dataset_dir}.")
    stems = [f.stem for f in sorted(images_dir.iterdir())
             if f.suffix.lower() in _IMG_EXTS
             and not _MASK_PAT.search(f.name) and not _CTRL_EXTRA.search(f.name)]
    samples = []
    for stem in stems:
        if len(samples) >= n:
            break
        target_path = next((images_dir / f"{stem}{e}" for e in _IMG_EXTS
                             if (images_dir / f"{stem}{e}").exists()), None)
        main_ctrl = next((control_dir / f"{stem}{e}" for e in _IMG_EXTS
                          if (control_dir / f"{stem}{e}").exists()), None)
        if target_path is None or main_ctrl is None:
            continue
        extra_ctrls, idx = [], 1
        while True:
            ec = next((control_dir / f"{stem}_control_{idx}{e}" for e in _IMG_EXTS
                       if (control_dir / f"{stem}_control_{idx}{e}").exists()), None)
            if ec is None:
                break
            extra_ctrls.append(ec); idx += 1
        prompt_path = (images_dir / f"{stem}.txt") if (images_dir / f"{stem}.txt").exists() \
            else (control_dir / f"{stem}.txt") if (control_dir / f"{stem}.txt").exists() else None
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path else ""
        ctrl_imgs = [_open_image(str(main_ctrl))] + [_open_image(str(p)) for p in extra_ctrls]
        samples.append({"id": stem, "control_images": ctrl_imgs,
                        "target_image": _open_image(str(target_path)), "prompt": prompt})
    if not samples:
        raise SystemExit(f"No valid samples found in {dataset_dir}.")
    return samples


def _read_csv_samples(csv_path, n):
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"CSV {csv_path} is empty.")
    control_cols = sorted(c for c in rows[0] if c.startswith("path_control"))
    if not control_cols:
        raise SystemExit(f"CSV {csv_path} has no path_control* columns.")
    samples = []
    for i, row in enumerate(rows[:n]):
        ctrl_imgs = [_open_image(row[c]) for c in control_cols
                     if row.get(c) and os.path.exists(row[c])]
        if not ctrl_imgs:
            continue
        tgt_path = row.get("path_target", "")
        tgt = _open_image(tgt_path) if tgt_path and os.path.exists(tgt_path) else None
        samples.append({"id": row.get("id", str(i)), "control_images": ctrl_imgs,
                        "target_image": tgt, "prompt": row.get("prompt", "")})
    if not samples:
        raise SystemExit(f"No valid samples loaded from CSV {csv_path}.")
    return samples


def _load_test_samples(args, n):
    if args.test_parquet:
        return _read_parquet_samples(Path(args.test_parquet), n)
    if args.test_dir:
        return _read_local_dir_samples(Path(args.test_dir), n)
    return _read_csv_samples(Path(args.test_csv), n)


def _instantiate_trainer(config):
    if config.trainer_type == "QwenImageEdit":
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
        return QwenImageEditTrainer(config)
    if config.trainer_type == "QwenImageEditPlus":
        from qflux.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
        return QwenImageEditPlusTrainer(config)
    raise NotImplementedError(f"supports QwenImageEdit and QwenImageEditPlus, got {config.trainer_type}")


def main():
    p = argparse.ArgumentParser(description="Run inference on a test split for visual LoRA comparison.")
    p.add_argument("--config", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-parquet",
                     help="Parquet file or directory containing data/*.parquet files")
    src.add_argument("--test-dir",
                     help="Local directory with training_images/ + control_images/ layout")
    src.add_argument("--test-csv",
                     help="CSV with columns: path_target, path_control[_N], prompt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--lora-weight", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-encoder-device", choices=["cpu", "xpu:0"], default="cpu",
                   help="text_encoder placement; default 'cpu' (Config B, bf16 on CPU). Use 'xpu:0' "
                        "for Config A — requires --text-encoder-path and a raised Step 1 §1.3 XPU cap.")
    p.add_argument("--text-encoder-path", default=None,
                   help="Dir of a pre-quantized NF4 text_encoder (Step 6 §6.2); overrides "
                        "model.text_encoder_path. Required for Config A (--text-encoder-device xpu:0).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config_from_yaml(args.config)
    config.mode = TrMode.fit
    config.cache.use_cache = False
    config.data.init_args.use_cache = False
    config.validation.enabled = False
    # Text-encoder placement (§8.3 Config A vs B): default to CPU (Config B, bf16 on CPU) regardless
    # of the training YAML. The loader picks NF4 whenever model.text_encoder_path is set (independent
    # of device), so clear it for Config B to load the original bf16 encoder. Config A
    # (--text-encoder-device xpu:0) REQUIRES a pre-quantized NF4 text_encoder via
    # --text-encoder-path (or model.text_encoder_path).
    config.predict.devices.text_encoder = args.text_encoder_device
    if args.text_encoder_path is not None:
        config.model.text_encoder_path = args.text_encoder_path
    if args.text_encoder_device == "cpu":
        config.model.text_encoder_path = None
    elif not getattr(config.model, "text_encoder_path", None):
        raise SystemExit("Config A (--text-encoder-device xpu:0) needs a pre-quantized NF4 "
                         "text_encoder: pass --text-encoder-path <dir> or set model.text_encoder_path, "
                         "or use the default Config B (bf16 on CPU).")

    if args.lora_weight:
        config.model.lora.pretrained_weight = args.lora_weight
        print(f"[inference_compare] LoRA mode: {args.lora_weight}")
    else:
        config.model.lora.pretrained_weight = None
        print("[inference_compare] Base model mode (no LoRA)")

    samples = _load_test_samples(args, args.num_samples)
    print(f"[inference_compare] Loaded {len(samples)} test samples")
    torch.manual_seed(args.seed)
    label = "lora" if args.lora_weight else "base"

    def _run():
        trainer = _instantiate_trainer(config)
        trainer.setup_predict()
        manifest = {"mode": label, "lora_weight": args.lora_weight, "config": args.config,
                    "num_inference_steps": args.num_inference_steps, "seed": args.seed, "samples": []}
        for i, s in enumerate(samples):
            sid = s["id"]
            print(f"[inference_compare] {i + 1}/{len(samples)}: id={sid}")
            try:
                # Derive height/width from the first control image so the trainer's prepare_embeddings
                # path receives non-None values. (PIL.Image.size returns (width, height))
                ctrl_w, ctrl_h = s["control_images"][0].size
                out_imgs = trainer.predict(image=s["control_images"], prompt=s["prompt"],
                                           height=ctrl_h, width=ctrl_w,
                                           num_inference_steps=args.num_inference_steps)
                out_img = out_imgs[0] if isinstance(out_imgs, list) else out_imgs
                fname = f"sample_{i:02d}_{sid}.png"
                out_img.save(out_dir / fname)
                if s["target_image"] is not None:
                    s["target_image"].save(out_dir / f"sample_{i:02d}_{sid}__target.png")
                manifest["samples"].append({"id": sid, "prompt": s["prompt"], "output": fname,
                                            "has_target": s["target_image"] is not None})
            except Exception as e:
                import traceback
                print(f"[inference_compare] sample {i} FAILED: {e}")
                manifest["samples"].append({"id": sid, "error": f"{type(e).__name__}: {e}",
                                            "traceback": traceback.format_exc()})
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[inference_compare] Done. Output: {out_dir}")

    if MemoryProbe is not None:
        with MemoryProbe(label=f"inference-{label}"):
            _run()
    else:
        _run()


if __name__ == "__main__":
    main()
```

**Running** (from the qwen-image-finetune project root, with conda env active):

```bat
REM Activate environment first (same as training)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
call conda activate qwen-image-edit-xpu
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1

REM --- Using an image directory test set (--test-dir) ---
REM Step 1 — base model (no LoRA)
python inference_compare.py ^
    --config config.yaml ^
    --test-dir <path\to\test-folder> ^
    --output-dir outputs\compare\<run>\base ^
    --num-samples 5

REM Step 2 — LoRA-applied
python inference_compare.py ^
    --config config.yaml ^
    --test-dir <path\to\test-folder> ^
    --output-dir outputs\compare\<run>\lora ^
    --num-samples 5 ^
    --lora-weight outputs\<run>\checkpoint-<step>\pytorch_lora_weights.safetensors

REM --- Using a parquet file or directory (--test-parquet) ---
REM (replace --test-dir with --test-parquet <path\to\test.parquet>)

REM --- Using a CSV file (--test-csv) ---
REM (replace --test-dir with --test-csv <path\to\test.csv>)
```

Each output directory contains `sample_NN_<id>.png` (model output),
`sample_NN_<id>__target.png` (ground-truth if available), and `manifest.json` with prompts and metadata.

**Quality and speed trade-offs**

Two parameters worth tuning to improve output quality, independently of the LoRA weights themselves:

- **Model precision**: higher-precision DiT and text encoder consistently produce better results.
  Three configurations, ordered by quality:

  | Config | DiT | Text encoder | Required system RAM |
  |---|---|---|---|
  | A | NF4 on XPU | NF4 on XPU | 32 GB + XPU cap (Step 1 §1.3) |
  | B *(default)* | NF4 on XPU | bf16 on CPU | 32 GB |
  | C | bf16 on XPU | bf16 on CPU | ≥64 GB |

  `inference_compare.py` defaults to **Config B** (text_encoder on CPU) regardless of what the
  training YAML specifies. Select a different config:
  - **B** (default): nothing to do — the script forces the text_encoder to CPU.
  - **A**: pass `--text-encoder-device xpu:0` **together with** `--text-encoder-path <dir>`, the
    pre-quantized NF4 text_encoder from Step 6 §6.2 (or set `model.text_encoder_path` in the YAML).
    The script requires it for Config A and errors out otherwise.
    **On a 32 GB machine, first increase the iGPU shared-memory percentage (Step 1 §1.3)** —
    the default is prone to OOM that can crash the whole machine under Config A.
  - **C** (≥64 GB): set `model.quantize_type` to a non-NF4 value (or remove it) so the DiT loads in
    bf16; keep the text_encoder on CPU (default). The NF4-trained LoRA applies to the bf16 base
    unchanged. Note the bf16 DiT is ~38 GB, so it needs the Step 1 §1.3 shared-memory ceiling raised
    to fit on the XPU even on a 64 GB machine.

- **Inference steps** (`--num-inference-steps`, default 20): more steps improve quality at the cost
  of time. Start with the default and increase only if the output quality is insufficient.

> **32 GB AI PC — constraints by config**:
> - **Config B** (default): system RAM approaches 31–32 GB. On some machines, bf16 text_encoder
>   (~15 GB) may cause OOM alongside NF4 DiT. If this happens, fall back to Config A by setting
>   `model.text_encoder_path` (Step 6 §6.2) — trades some output quality for reliability.
> - **Config A** (both models on XPU): highest XPU demand of the three, rising with resolution. If
>   you see `UR_RESULT_ERROR_DEVICE_LOST` or a silent crash, raise the GPU shared memory ceiling
>   (Step 1 §1.3) — the default registry value is often insufficient for this configuration.
> - **Higher inference steps** (50+): XPU memory accumulates across the denoising loop; confirm
>   stability at the default 20 steps before increasing.
>
> Other options:
> - `--num-samples 3` for a quicker comparison with fewer samples.
> - Monitor with `python monitor_xpu_memory.py` (Step 7) from a second terminal.

## 8.3 Visual evaluation checklist

When inspecting the base vs. LoRA outputs, apply these three questions for each sample. They align
with established image-editing evaluation dimensions: ⁴

| Dimension | Question to ask |
|---|---|
| **Instruction adherence** | Does the LoRA output follow the edit prompt? (e.g., for "Add the character to the image" — is the character present in the right place?) |
| **Edit quality** | Does the edit look natural and seamless? (no obvious artefacts, blurring, or inconsistency at edit boundaries) |
| **Detail preservation** | Are unedited regions unchanged between base and LoRA outputs? (background, non-target areas should look identical) |

A successful LoRA should pass all three on most samples. Failure patterns:
- **No change**: base and LoRA outputs are identical → training had no effect; check loss trajectory and training steps.
- **Instruction ignored**: character not added, wrong edit → LoRA learned something unrelated; check prompt consistency in dataset.
- **Degraded quality**: artefacts, color shifts, garbled regions → possible overfitting or learning rate too high.

For a small dataset (~30 samples), use **smooth loss** rather than raw per-step loss to judge
convergence — raw loss is noisy on small datasets due to repeated batches.

---
⁴ ImgEdit (Ye et al., arXiv:2505.20275, May 2025), https://github.com/PKU-YuanGroup/ImgEdit. Introduces instruction adherence, editing quality, and detail preservation as the three evaluation dimensions for image editing; adopted here as a manual inspection framework. Evaluates image editing models generally, not specifically qwen-image-finetune.

## PASS signal

- Both `inference_compare.py` runs exit 0
- Output directories contain PNG files and `manifest.json` with no errors
- Visual inspection: LoRA outputs visibly differ from base outputs; §8.3 checklist applied to judge
  the direction and quality of the change

**This is the final step — the fine-tuning walkthrough is complete.**

---

## Troubleshooting

**`TypeError: unsupported operand type for //: 'NoneType' and 'int'`**: `height`/`width` not passed
to `trainer.predict()`, so `prepare_embeddings` receives `batch["height"] = None`. The bundled
`inference_compare.py` already applies the fix (derive from the first control image's PIL size). If
you see this in a custom script, apply the same pattern:
```python
ctrl_w, ctrl_h = s["control_images"][0].size  # PIL.size = (width, height)
out_imgs = trainer.predict(..., height=ctrl_h, width=ctrl_w, ...)
```

**`ValueError: offline mode — must specify weight_name`**: loading a LoRA checkpoint when
`HF_HUB_OFFLINE=1` is set, because `diffusers.load_lora_adapter()` calls `_best_guess_weight_name()`
which raises in offline mode even for local paths. **Fix** (already applied in
`qflux/trainer/base_trainer.py`): if `pretrained_weight` ends in `.safetensors`, extract the parent
directory and filename and pass `weight_name` explicitly to `load_lora_adapter()`. If your qflux
lacks this fix, either upgrade, or pass the checkpoint **directory** (not the full file path) with
the filename `pytorch_lora_weights.safetensors` (PEFT default).

**`UR_RESULT_ERROR_DEVICE_LOST` during inference**: XPU out-of-memory. Use Config B (text_encoder on
CPU), reduce `--num-samples`, or raise the registry GPU memory ceiling (Step 1 §1.3) and reboot.

**Inference hangs in the VAE (no error, no progress)**: do **not** debug by adding
`torch.xpu.synchronize()`, `self.vae.to("cpu")`, or extra `.to(...)` moves inside `decode_vae_latent`
— that may introduce new hangs and does not fix the cause. Instead, overwrite `decode_vae_latent` with
the original upstream implementation from `qflux/trainer/qwen_image_edit_trainer.py` (inherited
unchanged by `QwenImageEditPlusTrainer`) and re-run.
