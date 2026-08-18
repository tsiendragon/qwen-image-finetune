---
name: qwen-image-edit-aipc-finetune-04-config
description: >
  Step 4 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-03-env-setup. Auto-detect the trainer from the model's
  model_index.json, run the hardware probe, generate a recommended NF4 QLoRA training YAML
  matched to the machine, fill in machine-specific paths, and lint the config. Only use once
  step 3 environment setup passes.
---

# Step 4 — Hardware Probe → Config Recommendation

**Series position**: Step 4 of 8.

**Prerequisites**:
- Conda env active with torch+xpu (Step 3); `probe_hw.py` in the project root
- Model downloaded to a local directory (Step 1); **model path** needed here
- Dataset validated (Step 2); **dataset path** needed here
- RAM tier known (Step 1); **RAM size in GB** (16/32/64) needed here

**Next step**: when `config_check.py` exits 0 and `config.yaml` has its trainer set, proceed
to skill `qwen-image-edit-aipc-finetune-05-adaptation`.

Three scripts produce and check a training YAML matched to the user's machine:
`detect_trainer.py` picks the trainer, `recommend_config.py` emits the config,
`config_check.py` lints it.

---

## 4.1 Detect the trainer from the model

The trainer is determined by the model's `model_index.json` `_class_name` field — not by the
model name or version number. This is the single source of truth for the `trainer:` field.

Save this as `detect_trainer.py` in the project root:

```python
# detect_trainer.py
# Detect which qflux trainer a model needs, from its model_index.json _class_name:
#   QwenImageEditPipeline      -> "QwenImageEdit"      (Qwen-Image-Edit)
#   QwenImageEditPlusPipeline  -> "QwenImageEditPlus"  (Qwen-Image-Edit-2509 / 2511)
# Usage:
#   python detect_trainer.py <model_dir>
#   python detect_trainer.py --json <model_dir>
# Exit codes: 0 = detected; 2 = model_index.json missing; 3 = unknown pipeline.
# Network policy: reads only a local model_index.json. A HuggingFace repo id (not a local dir)
# has no local file to read and exits 2 — download the model first (Step 1) or set trainer manually.
import argparse, json, os, sys

# Map a diffusers pipeline `_class_name` to the qflux trainer kind. Extend this table if
# upstream adds new Qwen-Image-Edit pipeline variants.
PIPELINE_TO_TRAINER = {
    "QwenImageEditPipeline": {"trainer": "QwenImageEdit", "multi_image": False,
                               "note": "Qwen-Image-Edit, single-image editing"},
    "QwenImageEditPlusPipeline": {"trainer": "QwenImageEditPlus", "multi_image": True,
                                   "note": "Qwen-Image-Edit-2509 / 2511, multi-image editing"},
}


def detect(model_dir):
    """Return {pipeline_class, trainer, multi_image, note} for a local model dir.
    Raises FileNotFoundError if model_index.json is absent, KeyError if the pipeline is unknown."""
    index_path = os.path.join(model_dir, "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(index_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    pipeline_class = index.get("_class_name")
    if pipeline_class not in PIPELINE_TO_TRAINER:
        raise KeyError(pipeline_class)
    entry = PIPELINE_TO_TRAINER[pipeline_class]
    return {"pipeline_class": pipeline_class, "trainer": entry["trainer"],
            "multi_image": entry["multi_image"], "note": entry["note"]}


def main():
    p = argparse.ArgumentParser(description="Detect the qflux trainer for a model from its model_index.json.")
    p.add_argument("model_dir", help="Local directory of the downloaded model")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = p.parse_args()
    try:
        result = detect(args.model_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e} not found — pass a local model directory that contains model_index.json (Step 1).", file=sys.stderr)
        return 2
    except KeyError as e:
        known = ", ".join(sorted(PIPELINE_TO_TRAINER))
        print(f"ERROR: unrecognized pipeline _class_name={e!s} in {args.model_dir}/model_index.json. Known: {known}.", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['trainer']}   ({result['note']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run:

```bat
python detect_trainer.py <path\to\model>
```

**Note the trainer name** (`QwenImageEdit` or `QwenImageEditPlus`) — you need it in Step 5 to
know whether the Plus-trainer adaptation applies. A `QwenImageEditPlus` value means Step 5's
extra Plus section applies.

## 4.2 Run the hardware probe

```bat
python probe_hw.py > probe.json
```

(`probe_hw.py` was saved in Step 3. If it's missing, copy it from Step 3 §3.5.)

## 4.3 Generate the recommended training config

Save this as `recommend_config.py` in the project root (it imports `detect_trainer.py` from
the same directory):

```python
# recommend_config.py
# Read probe JSON (from probe_hw.py) and emit a complete NF4-QLoRA training YAML that fits the
# machine. Trainer selection is MODEL-driven: pass --model-path so model_index.json picks the
# trainer. Without --model-path, trainer is a TODO placeholder.
# Resolution: when --dataset-path is given, target_size and controls_size are scaled to
# preserve the dataset's real aspect ratio at a conservative pixel budget (≈ 258 k–262 k px²).
# Without a dataset path, shape-bucket defaults are used. These are starting points; adjust
# based on your machine and confirm with monitor_xpu_memory.py (Step 7).
# Usage:
#   python probe_hw.py | python recommend_config.py
#   python recommend_config.py --probe probe.json --out config.yaml
#   python recommend_config.py --probe probe.json --model-path <model-dir> --ram-tier 32 --dataset-path <path> --out config.yaml
# Selected patterns are printed to stderr. Network policy: no network access.
import argparse
import glob
import io
import json
import os
import re
import sys


# ── Dataset introspection ──────────────────────────────────────────────────────
# Supports image directory, parquet, and CSV datasets.
# Returns the real target shape, control count, and per-control shapes from the
# first sample — so target_size and controls_size are seeded from actual data.

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_IMG_DIRS = ["training_images", "images", "target_images", "target", "targets"]
_CTRL_DIRS = ["control_images", "control", "condition_images", "controls"]


def _detect_dataset_format(path):
    """Return 'image_directory' | 'parquet' | 'csv' | 'unknown'."""
    if not path:
        return "unknown"
    if isinstance(path, str) and path.lower().endswith(".csv"):
        return "csv"
    if os.path.isdir(path):
        if glob.glob(os.path.join(path, "data", "*.parquet")) or \
           glob.glob(os.path.join(path, "*.parquet")):
            return "parquet"
        return "image_directory"
    return "unknown"


def _img_hw_from_file(path):
    """(H, W) from an image file path using only stdlib struct (no PIL needed)."""
    try:
        import struct
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return (h, w)
        if head[:2] == b"\xff\xd8":
            with open(path, "rb") as f:
                f.read(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        break
                    seg_len = struct.unpack(">H", f.read(2))[0]
                    if marker[1] in (0xC0, 0xC2):
                        f.read(1)
                        h, w = struct.unpack(">HH", f.read(4))
                        return (h, w)
                    f.read(seg_len - 2)
    except Exception:
        pass
    return None


def _img_hw_from_cell(cell):
    """(H, W) from a parquet image cell (dict{bytes} / raw bytes). Requires PIL."""
    try:
        from PIL import Image
        if isinstance(cell, dict) and "bytes" in cell:
            cell = cell["bytes"]
        if isinstance(cell, (bytes, bytearray)):
            with Image.open(io.BytesIO(cell)) as im:
                w, h = im.size
            return (h, w)
        if hasattr(cell, "size"):
            w, h = cell.size
            return (h, w)
    except Exception:
        pass
    return None


def _introspect_local(path):
    img_dir = next((os.path.join(path, d) for d in _IMG_DIRS
                    if os.path.isdir(os.path.join(path, d))), None)
    ctrl_dir = next((os.path.join(path, d) for d in _CTRL_DIRS
                     if os.path.isdir(os.path.join(path, d))), None)
    if img_dir is None or ctrl_dir is None:
        return None
    ctrl_re = re.compile(r"_control_\d+\.(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)
    targets = [p for p in sorted(glob.glob(os.path.join(img_dir, "*.*")))
               if p.lower().endswith(_IMG_EXTS)
               and not os.path.basename(p).lower().endswith("_mask.png")
               and not ctrl_re.search(os.path.basename(p))]
    if not targets:
        return None
    stem = os.path.splitext(os.path.basename(targets[0]))[0]
    target_hw = _img_hw_from_file(targets[0])
    main_ctrl = next((os.path.join(ctrl_dir, stem + e) for e in _IMG_EXTS
                      if os.path.isfile(os.path.join(ctrl_dir, stem + e))), None)
    controls_hw = [_img_hw_from_file(main_ctrl)] if main_ctrl else []
    i = 1
    while True:
        ec = next((os.path.join(ctrl_dir, f"{stem}_control_{i}{e}") for e in _IMG_EXTS
                   if os.path.isfile(os.path.join(ctrl_dir, f"{stem}_control_{i}{e}"))), None)
        if ec is None:
            break
        controls_hw.append(_img_hw_from_file(ec))
        i += 1
    n = (1 if main_ctrl else 0) + (i - 1)
    return {"n_controls_total": n or None, "target_hw": target_hw, "controls_hw": controls_hw,
            "format": "image_directory"}


def _introspect_parquet(path):
    files = sorted(glob.glob(os.path.join(path, "data", "*.parquet")) or
                   glob.glob(os.path.join(path, "*.parquet")))
    train = [f for f in files if os.path.basename(f).startswith("train")] or files
    if not train:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(train[0])
    except Exception:
        return None
    if len(df) == 0:
        return None
    row = df.iloc[0]
    controls = row.get("control_images") if "control_images" in df.columns else None
    controls_hw = []
    n_total = 0
    if controls is not None:
        try:
            seq = list(controls)
        except TypeError:
            seq = [controls]
        n_total = len(seq)
        for c in seq:
            controls_hw.append(_img_hw_from_cell(c))
    target_hw = (_img_hw_from_cell(row.get("target_image"))
                 if "target_image" in df.columns else None)
    return {"n_controls_total": n_total or None, "target_hw": target_hw,
            "controls_hw": controls_hw, "format": "parquet"}


def _introspect_csv(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    ctrl_cols = sorted(c for c in df.columns if "path_control" in c)
    row = df.iloc[0]
    base = os.path.dirname(path)

    def _resolve(rel):
        if not isinstance(rel, str):
            return None
        return rel if os.path.isabs(rel) else os.path.join(base, rel)

    controls_hw = []
    for c in ctrl_cols:
        p = _resolve(row.get(c))
        controls_hw.append(_img_hw_from_file(p) if p and os.path.isfile(p) else None)
    target_hw = None
    if "path_target" in df.columns:
        tp = _resolve(row.get("path_target"))
        if tp and os.path.isfile(tp):
            target_hw = _img_hw_from_file(tp)
    return {"n_controls_total": len(ctrl_cols) or None, "target_hw": target_hw,
            "controls_hw": controls_hw, "format": "csv"}


def introspect_dataset(dataset_path):
    """Introspect a local dataset and return shape/control info.

    Returns dict with: detected (bool), format, n_controls_total, target_hw,
    controls_hw, shape ('square'|'landscape'|'portrait'), ratio.
    Returns detected=False for remote repo IDs or unrecognised paths.
    """
    if not dataset_path:
        return {"detected": False}
    fmt = _detect_dataset_format(dataset_path)
    if fmt == "image_directory":
        result = _introspect_local(dataset_path)
    elif fmt == "parquet":
        result = _introspect_parquet(dataset_path)
    elif fmt == "csv":
        result = _introspect_csv(dataset_path)
    else:
        return {"detected": False}
    if result is None:
        return {"detected": False}
    target_hw = result.get("target_hw")
    if target_hw:
        h, w = target_hw
        ratio = h / w if w else 1.0
        if 0.9 <= ratio <= 1.1:
            shape = "square"
        elif ratio < 0.9:
            shape = "landscape"
        else:
            shape = "portrait"
    else:
        ratio, shape = 1.0, "landscape"
    result.update({"detected": True, "shape": shape, "ratio": round(ratio, 3)})
    return result


def recommend(probe, dataset_path=None, trainer=None, ram_tier=None):
    """Return (config_dict, selected_patterns_list).

    This skill is NF4-QLoRA-only: NF4 transformer + bnb Adam8bit are
    always emitted. RAM tier gates only the optional NF4 text_encoder layer (Step 6 §6.2).

    When --dataset-path is given, the dataset is introspected to determine the
    actual image aspect ratio and control count. target_size and controls_size are
    scaled to preserve the real aspect ratio at the same pixel budget as the
    corresponding documented defaults (landscape [384,672]=258 048 px²,
    square [512,512]=262 144 px², portrait [672,384]=258 048 px²). This avoids
    center_crop discarding content while keeping edge lengths in the same range
    as the original defaults. Without a dataset path the three-bucket defaults
    are emitted directly.
    """
    ram_gb = probe.get("ram_gb", -1.0)

    # NF4 transformer + Adam8bit: always-on for this skill's NF4 QLoRA recipe.
    use_nf4_transformer = True
    use_adam8bit = True
    if ram_tier is not None:
        use_nf4_text_encoder_recommended = ram_tier <= 32
    else:
        use_nf4_text_encoder_recommended = (ram_gb > 0 and ram_gb < 36)
    is_32gb_tier = use_nf4_text_encoder_recommended

    # Introspect the dataset to get the actual aspect ratio and control info.
    info = introspect_dataset(dataset_path) if dataset_path else {"detected": False}
    shape = info.get("shape", "landscape")
    controls_hw = info.get("controls_hw") or []

    # Reference pixel areas — these are the areas of the existing documented
    # default sizes for each shape class. Using them as the scaling reference
    # means edge lengths stay in the same range as the documented defaults.
    _REF_AREA = {"landscape": 384 * 672,   # 258 048
                 "square":    512 * 512,   # 262 144
                 "portrait":  672 * 384}   # 258 048

    def _scale_hw(h, w, ref_area):
        """Scale (H, W) to ref_area while preserving ratio; snap to 16× grid."""
        if h <= 0 or w <= 0:
            return None
        scale = (ref_area / (h * w)) ** 0.5
        sh = max(16, round(h * scale / 16) * 16)
        sw = max(16, round(w * scale / 16) * 16)
        return [sh, sw]

    # target_size: scale actual H/W to the reference area for this shape class.
    # Fall back to the three-bucket default when no introspection data is available.
    target_hw = info.get("target_hw")
    ref_area = _REF_AREA[shape]
    if target_hw:
        scaled = _scale_hw(target_hw[0], target_hw[1], ref_area)
        target_size = scaled if scaled else ([512, 512] if shape == "square"
                                             else [672, 384] if shape == "portrait"
                                             else [384, 672])
    elif shape == "square":
        target_size = [512, 512]
    elif shape == "portrait":
        target_size = [672, 384]
    else:
        target_size = [384, 672]

    # controls_size: scale each detected control's actual H/W to ref_area.
    # Fall back to shape-matched defaults when no introspection data is available.
    def _ctrl_size(hw):
        if not hw or hw[0] <= 0 or hw[1] <= 0:
            return list(target_size)
        ch, cw = hw
        r = ch / cw
        cs = "square" if 0.9 <= r <= 1.1 else ("landscape" if r < 0.9 else "portrait")
        scaled = _scale_hw(ch, cw, _REF_AREA[cs])
        return scaled if scaled else list(target_size)

    if controls_hw:
        controls_size = [_ctrl_size(hw) for hw in controls_hw[:2]]
        if len(controls_size) < 2:
            controls_size.append([512, 512])
    elif shape == "square":
        controls_size = [[512, 512], [512, 512]]
    elif shape == "portrait":
        controls_size = [[672, 384], [512, 512]]
    else:
        controls_size = [[384, 672], [512, 512]]

    cache_devices = {"vae": "xpu:0", "text_encoder": "xpu:0"}

    patterns = ["Step 6.1 Pre-quantize transformer (NF4)",
                "Step 5.5 NF4 QLoRA patches (A1 + A2; apply before training)"]
    if use_nf4_text_encoder_recommended:
        patterns.append("Step 6.2 Pre-quantize text_encoder (recommended for 32 GB machines)")
    patterns += ["Step 5.6 Mode-aware loading", "Step 7 Cache-first workflow",
                 "Step 4.5 YAML config patterns (this script's output applies them)"]

    trainer_value = trainer or ("<TODO: detect via detect_trainer.py — "
                                "QwenImageEdit (original) or QwenImageEditPlus (2509/2511)>")

    cfg = {
        "trainer": trainer_value,
        "model": {
            "pretrained_model_name_or_path": "<TODO: downloaded Qwen-Image-Edit pipeline dir>",
            "quantize": False,
            "lora": {"r": 16, "lora_alpha": 16, "init_lora_weights": "gaussian",
                     "target_modules": ["to_k", "to_q", "to_v", "to_out.0"], "pretrained_weight": None},
        },
        "data": {
            "class_path": "qflux.data.dataset.ImageDataset",
            "init_args": {
                "dataset_path": dataset_path if dataset_path else "<TODO: dataset path; see Step 2>",
                "caption_dropout_rate": 0.05, "prompt_image_dropout_rate": 0.05,
                "selected_control_indexes": [1],
                "cache_dir": "${cache.cache_dir}", "use_cache": "${cache.use_cache}",
                "processor": {"class_path": "qflux.data.preprocess.ImageProcessor",
                              "init_args": {"process_type": "center_crop",
                                            "target_size": target_size, "controls_size": controls_size}},
            },
            "batch_size": 1, "num_workers": 1, "shuffle": True,
        },
        "logging": {"output_dir": "<TODO: outputs/<run_name>>", "report_to": "tensorboard",
                    "tracker_project_name": "<TODO: run name>"},
        "lr_scheduler": {"scheduler_type": "cosine", "warmup_steps": 50, "num_cycles": 0.5, "power": 1.0},
        "train": {"gradient_accumulation_steps": 2, "max_train_steps": 1000, "num_epochs": 100,
                  # checkpoints_total_limit is declared in the schema but never wired into
                  # accelerate's ProjectConfiguration, so it prunes nothing — see Step 7.
                  "checkpointing_steps": 100, "checkpoints_total_limit": 10, "max_grad_norm": 1.0,
                  "mixed_precision": "no", "gradient_checkpointing": True, "low_memory": True},
        "cache": {"devices": cache_devices, "cache_dir": "<TODO: outputs/<run_name>/cache>",
                  "use_cache": True,
                  # Emit explicitly: the upstream CacheConfig default is ["prompt_embed", ...]
                  # (missing 's'), which does not match the cached key "prompt_embeds" and raises
                  # KeyError: 'prompt_embed' during caption-dropout.
                  "prompt_empty_drop_keys": ["prompt_embeds", "prompt_embeds_mask"]},
        "predict": {"devices": {"vae": "xpu:0", "text_encoder": "xpu:0", "dit": "xpu:0"}},
        "resume": None,
        "validation": {"enabled": False},
    }

    # NF4 + Adam8bit always emitted (skill scope is NF4 QLoRA only).
    cfg["model"]["transformer_path"] = "<TODO: pre-quantized NF4 dir from Step 6>"
    if use_nf4_text_encoder_recommended:
        cfg["model"]["text_encoder_path"] = "<TODO: pre-quantized NF4 text_encoder dir from Step 6>"
    cfg["model"]["quantize_type"] = "nf4"
    cfg["optimizer"] = {"class_path": "bitsandbytes.optim.Adam8bit",
                        "init_args": {"lr": 0.0001, "betas": [0.9, 0.999]}}
    return cfg, patterns


def _format_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if (v.startswith("${") or v.startswith("<") or "/" in v or " " in v
                or ":" in v or v == "no" or v == "yes" or v == "null"):
            return f'"{v}"'
        return v
    return str(v)


def _to_yaml(obj, indent=0):
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return f"{sp}{{}}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                lines.append(f"{sp}{k}:"); lines.append(_to_yaml(v, indent + 1))
            elif isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                lines.append(f"{sp}{k}:"); lines.append(_to_yaml(v, indent + 1))
            elif isinstance(v, list):
                if all(isinstance(x, list) for x in v):
                    inner = ", ".join("[" + ", ".join(_format_scalar(y) for y in x) + "]" for x in v)
                    lines.append(f"{sp}{k}: [{inner}]")
                else:
                    lines.append(f"{sp}{k}: [{', '.join(_format_scalar(x) for x in v)}]")
            else:
                lines.append(f"{sp}{k}: {_format_scalar(v)}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for x in obj:
            if isinstance(x, dict):
                lines.append(f"{sp}-"); lines.append(_to_yaml(x, indent + 1))
            else:
                lines.append(f"{sp}- {_format_scalar(x)}")
        return "\n".join(lines)
    else:
        return f"{sp}{_format_scalar(obj)}"


def to_yaml(obj):
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False, default_flow_style=None, allow_unicode=True).rstrip() + "\n"
    except ImportError:
        return _to_yaml(obj) + "\n"


def main():
    p = argparse.ArgumentParser(description="Recommend a training config given hardware probe output.")
    p.add_argument("--probe", help="Path to probe JSON file (default: read stdin)")
    p.add_argument("--out", help="Path to write YAML (default: stdout)")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--model-path", default=None,
                   help="Local model dir; its model_index.json picks the trainer. Omit to emit a TODO.")
    p.add_argument("--ram-tier", type=int, default=None,
                   help="User-confirmed RAM size in GB (16/32/64/64+). Authoritative for the NF4 "
                        "text_encoder recommendation; without it the probe's ram_gb is used.")
    args = p.parse_args()
    probe = json.load(open(args.probe) if args.probe else sys.stdin)
    trainer, trainer_comment = None, None
    if args.model_path:
        import os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        try:
            import detect_trainer
            det = detect_trainer.detect(args.model_path)
            trainer = det["trainer"]
            trainer_comment = f"auto-detected from model_index.json (_class_name={det['pipeline_class']})"
        except FileNotFoundError:
            print(f"# WARNING: no model_index.json under {args.model_path}; leaving trainer as TODO.", file=sys.stderr)
        except Exception as e:
            print(f"# WARNING: trainer detection failed ({e}); leaving trainer as TODO.", file=sys.stderr)
    cfg, patterns = recommend(probe, dataset_path=args.dataset_path, trainer=trainer, ram_tier=args.ram_tier)
    yaml_text = to_yaml(cfg)
    if trainer_comment:
        yaml_text = re.sub(r"^(trainer:.*)$", lambda m: f"{m.group(1)}  # {trainer_comment}",
                           yaml_text, count=1, flags=re.MULTILINE)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(yaml_text)
    else:
        sys.stdout.write(yaml_text)

    # Print dataset introspection result and size guidance to stderr.
    if args.dataset_path:
        info = introspect_dataset(args.dataset_path)
        if info.get("detected"):
            fmt = info.get("format", "?")
            sh = info["shape"]
            thw = info.get("target_hw")
            chws = info.get("controls_hw") or []
            nc = info.get("n_controls_total")
            print(f"\n# Dataset introspected ({fmt} format):", file=sys.stderr)
            print(f"#   shape:            {sh} (ratio {info['ratio']})", file=sys.stderr)
            print(f"#   target (H×W):     {thw}", file=sys.stderr)
            print(f"#   controls/sample:  {nc}  per-control (H×W): {chws}", file=sys.stderr)
            print(f"# → target_size and controls_size seeded from actual dataset dimensions.", file=sys.stderr)
            print(f"# IMPORTANT: edge lengths are a conservative starting point.", file=sys.stderr)
            print(f"# XPU memory scales with total token count (target + controls + text).", file=sys.stderr)
            print(f"# Verify headroom with monitor_xpu_memory.py (Step 7) before", file=sys.stderr)
            print(f"# increasing edge lengths beyond the emitted values.", file=sys.stderr)
        else:
            print("\n# Dataset introspection not possible for this path.", file=sys.stderr)
            print("# (Remote repo IDs and unrecognised paths are not introspected locally.)", file=sys.stderr)
            print("# target_size defaults to landscape [384,672]. Adjust shape and edge", file=sys.stderr)
            print("# lengths to match your dataset — see Step 2 §2.5 for guidance.", file=sys.stderr)

    print("\n# Selected patterns to apply in later steps:", file=sys.stderr)
    for pat in patterns:
        print(f"#  - {pat}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

Run:

```bat
python recommend_config.py ^
    --probe probe.json ^
    --model-path <path\to\model> ^
    --ram-tier <16|32|64> ^
    --dataset-path <path\to\dataset> ^
    --out config.yaml
```

- **Pass `--ram-tier`** = the machine's RAM size in GB (16/32/64/64+), confirmed at Step 1.
  This is the authoritative input for the one RAM-tier-dependent choice (NF4 text_encoder). If
  omitted, the recommender falls back to the probe's `ram_gb` (GiB), treating `< 36` as the 32 GB tier.
- **Pass `--model-path`** so the trainer is set automatically. Without it, `trainer:` is a TODO
  you must fill via `detect_trainer.py` — never guess it from the model name.
- **Pass `--dataset-path`** to fill the dataset_path placeholder inline. When `--dataset-path`
  is given, the recommender also introspects the dataset to read the actual target image shape,
  number of controls, and each control's dimensions — and uses these to seed `target_size` and
  `controls_size` from real data. Works for image directory, parquet, and CSV formats. The stderr
  output explains what was detected and what to adjust.

The recommender prints selected patterns and shape guidance to stderr — **note both for Steps 5–7**.

> **`target_size` requires agent review after the recommender runs.** The recommender sets the
> *shape* (aspect ratio) of `target_size` to match your dataset, but the *edge lengths* it emits
> are only a conservative starting point — the right size depends on your XPU memory, the number
> of control images per sample, and their sizes. After generating `config.yaml`:
> 1. Check the stderr shape note — confirm the detected shape matches your data.
> 2. Consider whether the edge lengths are appropriate: if your images are significantly larger or
>    smaller than the starting value, adjust them (keeping the same ratio). Both dimensions must
>    be multiples of 16.
> 3. After training starts (Step 7), monitor XPU memory with `monitor_xpu_memory.py` and reduce
>    `target_size` if memory is tight, or increase if there is headroom and quality matters.
> The recommender introspects the dataset to determine actual target/control image dimensions and
> control count. This works for all three supported formats: **image directory** (reads image file
> headers — no PIL needed), **parquet** (reads the first row's embedded image bytes — requires
> `pandas`), and **CSV** (reads the first row's image files). For remote repo IDs that have not
> been downloaded locally, introspection is not possible and the recommender falls back to the
> landscape default with a stderr note.

## 4.4 Fill in the TODO placeholders in config.yaml

Replace all `<TODO: ...>` placeholders (machine-specific paths the recommender cannot know):
- `model.pretrained_model_name_or_path` → your downloaded model directory
- `model.transformer_path` → leave as-is for now (filled in Step 6 — optional; omit to use online NF4)
- `model.text_encoder_path` → leave as-is for now (filled in Step 6; emitted at the 32 GB tier)
- `data.init_args.dataset_path` → validated dataset (auto-filled if you passed `--dataset-path`)
- `logging.output_dir` → e.g. `outputs/my_run`
- `logging.tracker_project_name` → e.g. `my_run`
- `cache.cache_dir` → e.g. `outputs/my_run/cache`

> **Parquet training config path**: `--dataset-path` fills the `data.init_args.dataset_path`
> field as a plain string. For **image directory** datasets (Step 2 §2.2 layout with
> `training_images/` and `control_images/`), a plain string path works. For **parquet** datasets
> (a directory containing `data/*.parquet`), the qflux loader cannot parse a plain string path
> — edit the generated YAML's `data.init_args.dataset_path` to use the dict format before training:
>
> ```yaml
> data:
>   init_args:
>     dataset_path:
>       - repo_id: <path/to/local-parquet-dir>
>         split: train
> ```

## 4.5 Recommended settings (reference)

> **Minimum required: 32 GB system RAM.** Tiers are the machine's RAM size in GB (16/32/64/64+),
> confirmed at Step 1 and passed as `--ram-tier`. On a Core Ultra iGPU that RAM is shared with the
> GPU as unified memory. `probe_hw.py` reports `ram_gb` (GiB) as a cross-check — it reads a few
> GiB *below* the GB tier number. **Don't read the tier from a GPU-memory figure**: a tool
> reporting an "Intel … GPU (16 GB)" is showing the shared-memory carve-out (`vram_gb`, ~half the
> RAM), not total RAM — that laptop is typically a **32 GB machine and is supported**. A **16 GB**
> machine is below minimum: the NF4 DiT alone needs ~10 GB.

The recommender emits the following. Each row includes the reason and test evidence where available.

| Setting | Value | Why |
|---|---|---|
| `model.quantize_type` | `nf4` | Scope of this skill; NF4 QLoRA only |
| `model.transformer_path` | `<pre-quant dir or null>` | **Optional.** Online NF4 uses shard-by-shard streaming so memory impact is negligible. Value of pre-quantizing is startup time (saves one quantization pass per run). Recommended for multiple training iterations. See Step 6. |
| Pre-quantize text_encoder (`model.text_encoder_path`) | **strongly recommended on 32 GB RAM**; optional on 64 GB+ | NF4 shrinks the text_encoder from ~15 GB to ~5.5 GB, small enough to sit on the XPU (`cache.devices.text_encoder: xpu:0`) comfortably during the cache phase instead of crowding the unified budget — and it enables the both-on-XPU inference config (Step 8 Config A). The recommender emits the `text_encoder_path` slot at the 32 GB tier (`--ram-tier <= 32`, or fallback `ram_gb < 36`); fill it after Step 6. On 32 GB, do **not** put the bf16 text_encoder on XPU; if `text_encoder_path` is not ready, set `cache.devices.text_encoder: cpu` as a slower fallback. See Step 6. |
| `cache.use_cache` | `true` | **Required.** Without caching, the trainer keeps text_encoder active during fit to encode prompts per batch while the NF4 DiT is also loaded — combined memory pressure crashes on 32 GB machines and can stall even when it fits. Cache-first runs text_encoder and VAE at cache time, writes embeddings + latents to disk, then frees text_encoder before the NF4 DiT loads for fit. See Step 7. |
| Mode-aware loading | default on (**required for this recipe**) | Skips the component the current phase doesn't use (DiT in cache, text_encoder in fit). Without this, the cache phase wastes ~10 GB on an unused DiT and the fit phase keeps an unused text_encoder resident; on 32 GB this commonly causes XPU OOM or `DEVICE_LOST`. See Step 5. |
| `data.processor.target_size` | scaled to dataset shape | When `--dataset-path` is given, the recommender reads the actual image dimensions and scales `target_size` to preserve the real aspect ratio at a conservative pixel budget (≈ 258 k–262 k px²). Without a dataset path it falls back to three-bucket defaults. The emitted value is a starting point — XPU memory scales with total token count across target, controls, and text (image token counts scale with pixel area). Adjust after generation: see Step 2 §2.5 for shape guidance and Step 7 `monitor_xpu_memory.py` for memory confirmation. |
| ↳ `[256, 448]` / `[384, 384]` | Lower XPU demand | Significantly faster per step; fast iteration and prototyping |
| ↳ `[384, 672]` / `[512, 512]` | Middle | Moderate XPU demand; these are the reference sizes used as the pixel budget for landscape / square datasets |
| ↳ `[512, 768]` / `[640, 640]` | Higher XPU demand | Significantly slower per step; larger per-step memory footprint — confirm fit for your machine + control configuration by measuring |
| `data.processor.controls_size` | scaled to each control's shape | List of `[H, W]` sizes — one entry per control image type. The first entry sets the size for the **main control image** (always present); subsequent entries set the size for additional control images (`_control_1`, `_control_2`, …) if your dataset provides them. When `--dataset-path` is given, each slot is scaled to preserve the actual control image's aspect ratio at the same pixel budget. If `controls_size` is omitted entirely, the framework defaults each control to `target_size`. Changes require rebuilding the cache (Step 7). |
| `train.gradient_checkpointing` | `true` | **Recommended at all resolutions; required at ≥[384,672]-scale.** Without it, activation memory causes XPU OOM during attention at the starting resolution and larger. Reducing LoRA rank does not resolve this. |
| `train.max_train_steps` | `1000` | Practical starting point for a narrow task on a ~35-sample dataset. Reduce to **50–200** for a quick pipeline sanity check; increase for larger datasets or more complex edits. Monitor smooth loss — stop when it plateaus across consecutive checkpoints. **When you lower this, lower `checkpointing_steps` to match**, and on the first run confirm training actually stops at `max_train_steps` (Step 7) — if it overshoots, the Step 5 `train_epoch` guard wasn't applied. |
| `train.checkpointing_steps` | `100` | Must be **≤ `max_train_steps`**, otherwise the run finishes before any checkpoint is written and you have nothing to validate. Keep them in sync — e.g. for a 5-step sanity run set `checkpointing_steps: 5`. `config_check.py` warns if this is violated. |
| `train.gradient_accumulation_steps` | `2` | Effective batch-size > 1 without extra per-step memory |
| `optimizer.class_path` | `bitsandbytes.optim.Adam8bit` | Memory difference vs `torch.optim.Adam` is negligible at this LoRA scale. Uses Triton-backed kernels; the recommender's default choice — swap in `torch.optim.Adam` if you prefer the stock optimizer. |
| `lora.r` | `16` | Memory-neutral — activation memory dominates, not LoRA parameters. Quality choice; adjust freely between 8–32. |
| `data.batch_size` | `1` | Single GPU, minimum |
| `data.num_workers` | `1` | Windows spawn-mode; qflux pydantic schema rejects 0 |
| `train.mixed_precision` | `"no"` | **Inert** — `base_trainer` constructs the `Accelerator` with a hardcoded `mixed_precision="no"` and never reads this field (the line that would read it is commented out). Keep it `"no"` so the config states what actually runs. Mixed precision comes from the trainer's own `torch.autocast` (Step 5 §5.5.3 NF4-A2). |

### Manual override

If the probe is incorrect (e.g. `oneapi_version = null` because `setvars.bat` hasn't run) or
different settings are desired: edit the generated YAML directly (the table above documents what
each key does), or hand-edit `probe.json` before piping into the recommender. The recommender is
**conservative** — it picks the safest config that fits. Loosen settings (larger batch, larger
LoRA rank, lower grad-accum) only after the conservative config trains successfully and you
verify XPU memory has headroom (via the monitor script in Step 7).

## 4.6 Lint the config

Save this as `config_check.py` in the project root (imports `detect_trainer.py` from the same dir):

```python
# config_check.py
# Pre-flight linter for a Qwen-Image-Edit AI PC training config. Run AFTER recommend_config.py
# and BEFORE the cache phase. Catches config mistakes that have broken runs: wrong trainer for
# the model, missing NF4, an empty-cache "fix" that disables caching, bad text_encoder placement.
# Usage:
#   python config_check.py <config.yaml>
#   python config_check.py <config.yaml> --probe probe.json
#   python config_check.py <config.yaml> --ram-tier 32
# Exit 0 = PASS (warnings ok); Exit 1 = FAIL (errors present). Network policy: local files only.
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import detect_trainer
except Exception:  # pragma: no cover
    detect_trainer = None


def _load_yaml(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}
        except Exception as e:
            raise SystemExit(f"ERROR: cannot parse YAML (install pyyaml or omegaconf): {e}")


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_todo_or_empty(value):
    # recommend_config.py emits "<TODO: ...>" placeholders; treat them, None, and "" as missing.
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or stripped.startswith("<TODO")
    return False


def _device_type(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("xpu"):
        return "xpu"
    if text.startswith("cpu"):
        return "cpu"
    return text.split(":", 1)[0] if text else None


def check(cfg, probe, ram_tier=None):
    errors, warnings = [], []
    trainer = cfg.get("trainer")
    model_path = _get(cfg, "model", "pretrained_model_name_or_path")
    quantize_type = _get(cfg, "model", "quantize_type")
    quantize = _get(cfg, "model", "quantize", default=False)
    text_encoder_path = _get(cfg, "model", "text_encoder_path")
    use_cache = _get(cfg, "cache", "use_cache", default=True)
    cache_text_encoder_device = _get(cfg, "cache", "devices", "text_encoder")
    optimizer = _get(cfg, "optimizer", "class_path")
    max_train_steps = _get(cfg, "train", "max_train_steps")
    checkpointing_steps = _get(cfg, "train", "checkpointing_steps")

    valid_trainers = {"QwenImageEdit", "QwenImageEditPlus"}
    if trainer not in valid_trainers:
        errors.append(f"trainer={trainer!r} is not supported. Use one of {sorted(valid_trainers)} (auto-detect with detect_trainer.py).")

    # Double quantization: quantize:true triggers FP8 online quant; with quantize_type:nf4 the
    # model is quantized twice and OOMs the XPU.
    if quantize is True and quantize_type == "nf4":
        errors.append("model.quantize: true AND model.quantize_type: nf4 — this double-quantizes (FP8 on top of NF4) and OOMs the XPU. Set model.quantize: false and keep model.quantize_type: nf4.")

    if quantize_type != "nf4":
        errors.append(f"model.quantize_type={quantize_type!r} — this skill is NF4 QLoRA only. Set model.quantize_type: nf4 (without it the DiT loads in bf16 and over-commits the iGPU; observed as cache-phase machine shutdowns on 32 GB).")

    detected = None
    if detect_trainer is not None and isinstance(model_path, str):
        try:
            detected = detect_trainer.detect(model_path)
        except FileNotFoundError:
            warnings.append(f"could not read {model_path}/model_index.json — cannot verify the trainer matches the model (is the path local and downloaded?).")
        except KeyError as e:
            warnings.append(f"unrecognized pipeline _class_name={e!s} in model_index.json.")
    if detected is not None and trainer in valid_trainers:
        want = detected["trainer"]
        if want == "QwenImageEditPlus" and trainer == "QwenImageEdit":
            warnings.append(f"model is {detected['pipeline_class']} (its native trainer is QwenImageEditPlus), but trainer=QwenImageEdit. This RUNS, but only the main control image reaches the text encoder (single-image conditioning). Switch to QwenImageEditPlus unless you intend the single-image path.")
        elif want == "QwenImageEdit" and trainer == "QwenImageEditPlus":
            errors.append(f"model is {detected['pipeline_class']} (single-image), but trainer=QwenImageEditPlus. The Plus multi-image pipeline does not match this model — use trainer=QwenImageEdit.")

    if use_cache is False:
        errors.append("cache.use_cache is false. Caching is required on AI PC tiers. If the cache dir is empty, RUN the cache phase — do NOT disable caching. Live text_encoder encoding during fit is outside this skill's supported recipe and commonly crashes or stalls on AI PCs.")

    # NF4 text_encoder is recommended at the 32 GB tier. Tier from --ram-tier (GB) when given,
    # else probe ram_gb (GiB; the 32 GB tier is < 36).
    is_32gb_tier = None
    if ram_tier is not None:
        is_32gb_tier = ram_tier <= 32
        if ram_tier < 32:
            warnings.append(f"--ram-tier {ram_tier} is below this skill's 32 GB minimum (16 GB is not supported) — the NF4 DiT alone needs ~10 GB.")
    elif probe is not None:
        ram_gb = probe.get("ram_gb", -1.0)
        if ram_gb > 0:
            is_32gb_tier = ram_gb < 36
    if is_32gb_tier:
        te_missing = _is_todo_or_empty(text_encoder_path)
        if te_missing and _device_type(cache_text_encoder_device) == "xpu":
            errors.append("32 GB tier has cache.devices.text_encoder on XPU but model.text_encoder_path is unset/TODO. This would load the bf16 text_encoder on XPU during cache and can OOM or crash. Either run Step 6 NF4 text_encoder pre-quantization and set model.text_encoder_path, or set cache.devices.text_encoder: cpu as a slower fallback.")
        elif te_missing:
            warnings.append("32 GB tier and model.text_encoder_path is unset. NF4 text_encoder (Step 6) is recommended to relieve cache-phase RAM; CPU text_encoder cache is a slower fallback.")

    if (isinstance(max_train_steps, int) and isinstance(checkpointing_steps, int)
            and checkpointing_steps > max_train_steps):
        warnings.append(f"train.checkpointing_steps={checkpointing_steps} > train.max_train_steps={max_train_steps} — no checkpoint will be saved. Set checkpointing_steps <= max_train_steps so at least one checkpoint lands.")

    if optimizer and optimizer != "bitsandbytes.optim.Adam8bit":
        warnings.append(f"optimizer.class_path={optimizer!r} — recommend bitsandbytes.optim.Adam8bit (Triton-backed kernels).")

    return errors, warnings


def main():
    p = argparse.ArgumentParser(description="Pre-flight linter for an AI PC training config.")
    p.add_argument("config", help="Path to the training YAML")
    p.add_argument("--probe", help="Optional probe.json (enables the 32 GB text_encoder check)")
    p.add_argument("--ram-tier", type=int, default=None,
                   help="User-confirmed RAM size in GB (16/32/64/64+); authoritative over probe ram_gb.")
    args = p.parse_args()
    cfg = _load_yaml(args.config)
    probe = json.load(open(args.probe, encoding="utf-8")) if args.probe else None
    errors, warnings = check(cfg, probe, ram_tier=args.ram_tier)
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    if errors:
        print(f"\nFAIL: {len(errors)} error(s), {len(warnings)} warning(s). Fix the errors before the cache phase.")
        return 1
    print(f"\nPASS: 0 errors, {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run:

```bat
python config_check.py config.yaml --ram-tier <16|32|64>
```

> At this stage, `config.yaml` still has `<TODO: ...>` placeholders for `transformer_path` and
> `text_encoder_path` (filled in Step 6). The linter treats these as "unset" and WARNs rather
> than ERRORs for them. It ERRORs on `quantize_type`, `cache.use_cache`, and `trainer` mismatches
> — fix those before proceeding.

## PASS signal

- `probe.json` produced by `probe_hw.py`
- `config.yaml` produced by `recommend_config.py`; trainer is set (not a TODO)
- `config_check.py config.yaml --ram-tier <N>` exits 0 (`PASS: 0 errors`)
- **Trainer name noted** (`QwenImageEdit` or `QwenImageEditPlus`) — needed in Step 5

**Proceed to**: `qwen-image-edit-aipc-finetune-05-adaptation`
