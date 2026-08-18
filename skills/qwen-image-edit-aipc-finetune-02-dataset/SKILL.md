---
name: qwen-image-edit-aipc-finetune-02-dataset
description: >
  Step 2 of 8 of the Qwen-Image-Edit AI PC fine-tuning walkthrough. Use after
  qwen-image-edit-aipc-finetune-01-preflight. Prepare, structure, and validate a training
  dataset (image directory, parquet, or CSV), understand the control/target
  edit-pair paradigm, follow sample-size and image-quality and prompt guidance, run the
  dataset validator, and set aside a held-out test set. Only use once step 1 pre-flight is done.
---

# Step 2 — Dataset Preparation

**Series position**: Step 2 of 8.

**Next step**: when `dataset_validate.py` exits 0 and the test set is set aside, proceed to
skill `qwen-image-edit-aipc-finetune-03-env-setup`.

The qwen-image-finetune framework (`qflux.data.dataset.ImageDataset`) supports three dataset
formats. Pick whichever fits the user's workflow; all three are validated by the script in §2.7.

---

## 2.1 Format options

| Format | When to use | Schema |
|---|---|---|
| **Image directory** | Quick iteration on a custom dataset | Stem-paired files: `<stem>.png` (target) + `<stem>.png` (control) + `<stem>.txt` (prompt), under `training_images/` and `control_images/` subdirs |
| **Parquet** | Datasets distributed in parquet format | Directory containing `data/*.parquet` files; columns: `target_image`, `control_images` (list), `prompt`, optional `control_mask` |
| **CSV** | Migrating from another pipeline | Columns: `path_target`, `path_control` (one or more `path_control_N`), `prompt`, optional `path_mask` |

Where the data comes from (HuggingFace Hub, ModelScope, locally created, etc.) is a matter of **source**, not format. Download the dataset to a local path first, then point the training config at the local path in the matching format above.

Detailed schemas live in `src/qflux/data/dataset.py` (`ImageDataset.__init__` docstring).
The Step 4 recommender emits a `data:` block matching whichever format the user has.

## 2.2 Image directory layout

```
dataset_root/
├── training_images/
│   ├── sampleA.png              # target image — the desired OUTPUT after the edit
│   └── sampleA.txt              # required: prompt text (the edit instruction)
└── control_images/
    ├── sampleA.png              # main control — INPUT condition (e.g. background scene)
    ├── sampleA_control_1.png    # optional: 2nd control (e.g. character reference image)
    └── sampleA_mask.png         # optional: mask of the edited region (character silhouette)
```

**Naming rules** (from `qflux.data.dataset.ImageDataset` documentation):
- Extra control images: append `_control_1`, `_control_2`, … to the base name
- Mask: `<base>_mask.png` in either `control_images/` or `training_images/`
- Prompt: `.txt` file with the same base name; if present in both dirs, `training_images/` wins

**Alternative directory names** accepted: `images` / `target_images` / `target` for targets;
`control` / `condition_images` / `controls` for controls.

**Multi-control example** (character composition task): `sampleA.png` = background scene,
`sampleA_control_1.png` = character on white background, `sampleA_mask.png` = character
silhouette in the final composition, prompt = "Add the character to the image". This is the
pattern used by the `TsienDragon/character-composition` reference dataset.

**Mask usage**: when a mask is present, `edit_mask_loss` applies higher training weight to
the masked region, focusing the LoRA on the area that changed between control and target.
Useful when only part of the image is edited.

## 2.3 Sample size guidance

| Sample count | Expected outcome |
|---|---|
| < 5 | Hard floor — `dataset_validate.py` rejects; too few to cache meaningfully |
| 5–19 | Smoke test only; rapid overfitting expected within the first tens of steps |
| 20–50 | **Practical starting point for a narrow task** (single subject, single edit type, single viewpoint set). Community experience confirms this range works for specific character/product LoRAs. ² |
| 50–100 | Better generalization; recommended if the edit should work across varying scenes or lighting. A 50-image rendered dataset has been validated for complex spatial transformations. ³ |
| 100–200+ | For more general edit styles that should work across many different scenes and subjects. ² |

**Quality beats quantity.** Adding low-quality or inconsistent pairs actively harms the LoRA
— they introduce noise the model cannot learn a clean pattern from. ¹ ²

**Single-task LoRA preferred.** If you want to train multiple distinct edits (e.g., style +
object replacement), train separate LoRAs rather than one mixed dataset. Multi-task LoRA
often causes the tasks to interfere with each other. ²

The validator (§2.7) enforces the floor of 5 samples and warns below 30. The framework caches
embeddings per-sample (Step 7), so larger datasets cost more upfront cache time but no extra
per-step cost during fit.

---
¹ FlyMyAI LoRA Trainer, https://github.com/FlyMyAI/flymyai-lora-trainer, Aug 2025. Uses a different training framework; cited for model-level image quality guidance applicable to Qwen-Image-Edit.  
² HuggingFace Forums: "Question about lora fine tune qwen-image-edit" (John6666, Nov 2025), https://discuss.huggingface.co/t/question-about-lora-fine-tune-qwen-image-edit/170633. Qwen-Image-Edit-specific community guidance; cited for dataset size ranges, quality advice, and single-task LoRA recommendation.  
³ とりにく, "vast.AIでQwen image Edit 2509のLoRA学習", https://note.com/tori29umai/n/n256f30d51669, Sept 2025. Uses Musubi Tuner (a separate training framework); dataset construction and spatial transformation advice is model-level and framework-independent.

## 2.4 Control image and target image — the edit pair

Qwen-Image-Edit is an **image editing** model, not an image generation model. Every training
sample teaches the model one specific edit:

| Part | Role | Example |
|---|---|---|
| **Control image(s)** | The input condition(s) — what the user provides | Background scene, character reference, source style |
| **Target image** | The desired output — what the model should produce | The same scene after the edit is applied |
| **Prompt** | The edit instruction | "Add the character to the image" |

The key difference from caption-based LoRA training (e.g. for image generation): the prompt
describes **what to do**, not **what the result looks like**.

**Multi-control paradigm** — the framework natively supports multiple control images per
sample, which is useful when the edit requires more than one reference:
- Main control (`sampleA.png`): the primary input scene or reference
- Additional controls (`sampleA_control_1.png`, `_control_2.png`, …): supplementary
  references (e.g. a character sheet, a style reference)
- Mask (`sampleA_mask.png`): region of interest — tells the model which part of the image the
  edit targets

**Verified example** (character-composition task): the `TsienDragon/character-composition`
dataset is one verified instance of this setup — it uses two controls (a background image and
a character-on-white-background image) plus a mask of the character's silhouette, with a fixed
prompt "Add the character to the image". The target is the character correctly composited into
the background. User-created datasets following the same format work equally well.

**General task examples** (for reference, not exhaustive):
- *Viewpoint change*: control = source angle; target = desired angle; prompt = describe the viewpoint transformation ³
- *Object addition*: control = scene; target = scene with object added; prompt = "Add [object] to the image"
- *Style change*: control = original; target = restyled; prompt = describe the style change

**Important**: control and target must share the same subject/context. Unrelated images in a
pair prevent the model from learning a coherent edit mapping.

**Practical tip**: synthetic or rendered control images (e.g. 3D, game engine) produce very
consistent results because they eliminate photographic variation that the model might
otherwise try to replicate. ³

---
³ とりにく, "vast.AIでQwen image Edit 2509のLoRA学習", https://note.com/tori29umai/n/n256f30d51669, Sept 2025. Uses Musubi Tuner (a separate training framework); dataset construction advice is model-level and framework-independent.

## 2.5 Image quality guidelines

- **Resolution**: source images should be **at least as large as your `target_size`** in each
  dimension. The training preprocessor crops/resizes source images down to `target_size` — if
  the source is smaller, it gets upscaled first (quality loss).
  - Source resolution larger than `target_size` is always fine; the crop just has more to choose from.
  - A practical floor: source images should be at least as large as your chosen `target_size` in
    both H and W. If your images are already at or above the `target_size` the recommender emits
    (matched to your dataset's shape), no additional action is needed.
  *(Recommended range from `docs/guide/data-preparation.md`: 512×512 to 1024×1024 — appropriate
  for moderate `target_size` configurations used with this series.)*
- **Aspect ratio — set `target_size`'s shape to match your dataset.** `target_size` is
  `[height, width]`. With the default `process_type: center_crop`, the preprocessor scales each
  image to fill `target_size` and crops whatever does not fit — so if the target's proportions
  differ from your images, part of every image is cropped away. Match the **shape** (ratio) to
  your data: a **square** dataset needs a square (H = W) target; a **landscape** dataset a wide
  (W > H) target; a **portrait** dataset a tall (H > W) target. To keep the **whole frame**
  instead of cropping, use `process_type: center_padding` (pads with borders rather than cropping).
  - **Edge lengths are a memory question — and `target_size` is not the only input.** The DiT
    processes target, all control images, and the text prompt together in joint attention; the
    total token count across all of them drives memory. Each image's token count scales with its
    pixel area (larger `target_size` or `controls_size` = more tokens = more memory). The text
    prompt token count is roughly fixed and not affected by image size settings. The number of
    control images comes from your dataset, not the config: the main control always counts, and
    additional `_control_N` images add to it only when your samples provide them (`controls_size`
    merely gives the size to use for each — its 2nd entry applies only if a 2nd control exists).
    More or larger control images cost more memory.
  - **Pick edge lengths for your own dataset by measuring, not from a fixed number.** The Step 4
    §4.5 resolution tiers are the recommender's starting points; because the total cost depends on
    your control count and sizes too, keep the shape matched to your data, start conservative, and
    confirm headroom with `monitor_xpu_memory.py` (Step 7) before increasing. Both dimensions must
    be multiples of 16. Changing `target_size` or `controls_size` requires rebuilding the cache
    (Step 7).
- **Format**: JPG, JPEG, PNG, WebP — all supported by the framework. RGB (3-channel) required.
- **Consistency within dataset**: maintain consistent image quality, lighting style, and
  subject framing across pairs. Inconsistency is a common cause of blurry or incoherent model
  outputs. ¹
- **Quality over quantity**: more low-quality pairs actively harm the LoRA — they introduce
  noise the model cannot learn a clean pattern from. ¹
- **Avoid**: watermarks or text overlays in control/target images; heavy JPEG artefacts;
  mixing very different visual styles (e.g., anime and photorealistic) in the same dataset
  without a compelling reason.

---
¹ FlyMyAI LoRA Trainer, https://github.com/FlyMyAI/flymyai-lora-trainer, Aug 2025. Uses a different training framework; cited for model-level image quality guidance applicable to Qwen-Image-Edit.

## 2.6 Prompt strategy

The framework expects prompts to be **descriptive editing instructions** — text that tells
the model what to do, not what the result looks like.  
*(From `docs/guide/data-preparation.md`: "Descriptive editing instructions"; recommended length 10–200 words)*

| Prompt type | Example | Use |
|---|---|---|
| **Edit instruction** (correct) | "Add the character to the image" | For image editing LoRA training |
| **Target image caption** (incorrect for editing) | "A character standing in a room" | Trains the model to generate, not edit |

**Key guidelines:**

- **Describe the transformation**: "Add the character to the image" teaches *what to do*
  (placement); "A character in a room" only describes *what exists* in the result — the model
  cannot learn the editing operation from it.
- **Consistency within a dataset**: if all samples share the same edit type (e.g., character
  composition), a fixed or near-fixed prompt like "Add the character to the image" is valid and
  has been shown to work well. The model learns the edit from the image pairs; the prompt
  anchors what operation is being requested.
- **Specificity when edits vary**: if your dataset covers multiple edit types or subjects,
  prompts should distinguish them — e.g. "Add the character to the outdoor scene" vs "Add the
  character to the interior scene" — so the model learns to condition on the instruction, not
  just pattern-match visually.
- **Length**: 10–200 words accepted by the framework. Short, clear instructions work well for
  specific tasks; longer prompts are appropriate when the edit is complex or context-dependent.
- **Avoid ambiguity**: "Edit the image" teaches nothing — a prompt must specify what kind of edit.

## 2.7 Validation step

Before committing to a training run, validate the dataset. Save the script below as
`dataset_validate.py` (in the project root or anywhere convenient) and run it:

```python
# dataset_validate.py
# Validate a dataset for qwen-image-edit fine-tuning.
# Detects format (image directory / parquet / CSV) and runs format-appropriate checks.
# Usage:
#   python dataset_validate.py <path/to/image-directory-dataset>
#   python dataset_validate.py <path/to/dataset.csv>
#   parquet dir: python dataset_validate.py <path/to/parquet-dir>
# Output (stdout): JSON verdict. Exit code: 0 if valid, 1 otherwise.
# Network policy: does NOT fetch remote repos; for a remote owner/repo argument it
# only sanity-checks the format string and warns to verify via max_train_steps=1.
import argparse, csv, json, os, re, sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGE_DIR_NAMES = ["training_images", "images", "target_images", "target", "targets"]
CONTROL_DIR_NAMES = ["control_images", "control", "condition_images", "controls"]
CONTROL_EXTRA_PATTERN = re.compile(r"_control_(\d+)\.(?:png|jpe?g|webp|bmp)$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"_mask\.png$", re.IGNORECASE)


def detect_format(dataset_path):
    p = Path(dataset_path)
    if str(dataset_path).lower().endswith(".csv"):
        return "csv"
    if p.is_dir():
        if (p / "data").exists() and any((p / "data").glob("*.parquet")):
            return "parquet"
        return "image_directory"
    if "/" in str(dataset_path) and not p.exists():
        return "parquet"
    return "unknown"


def _size_assessment(count):
    issues, warnings = [], []
    if count < 5:
        issues.append(f"Sample count {count} below the hard floor of 5; pipeline cannot meaningfully train.")
    elif count < 10:
        warnings.append(f"Sample count {count} is in smoke-test territory (5-9); expect rapid overfitting within ~100 steps.")
    elif count < 30:
        warnings.append(f"Sample count {count} below recommended baseline of 30; expect mild quality variance for single-concept LoRA.")
    return issues, warnings


def validate_local(path):
    issues, warnings = [], []
    images_dir = next((path / n for n in IMAGE_DIR_NAMES if (path / n).exists()), None)
    if images_dir is None:
        return {"valid": False, "format": "image_directory", "sample_count": 0,
                "issues": [f"No images directory found. Expected one of {IMAGE_DIR_NAMES} under {path}"], "warnings": []}
    control_dir = next((path / n for n in CONTROL_DIR_NAMES if (path / n).exists()), None)
    if control_dir is None:
        return {"valid": False, "format": "image_directory", "sample_count": 0,
                "issues": [f"No control directory found. Expected one of {CONTROL_DIR_NAMES} under {path}"], "warnings": []}
    targets = []
    for f in images_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
            continue
        if MASK_PATTERN.search(f.name) or CONTROL_EXTRA_PATTERN.search(f.name):
            continue
        targets.append(f)
    if not targets:
        return {"valid": False, "format": "image_directory", "sample_count": 0,
                "issues": [f"No target images found in {images_dir}"], "warnings": warnings}
    paired, no_control, no_prompt = 0, [], []
    for t in targets:
        stem = t.stem
        has_control = any((control_dir / f"{stem}{ext}").exists() for ext in IMG_EXTS)
        if not has_control:
            no_control.append(stem)
            continue
        has_prompt = (images_dir / f"{stem}.txt").exists() or (control_dir / f"{stem}.txt").exists()
        if not has_prompt:
            no_prompt.append(stem)
            continue
        paired += 1
    if no_control:
        issues.append(f"{len(no_control)} target images have no matching control: {no_control[:5]}{' ...' if len(no_control) > 5 else ''}")
    if no_prompt:
        issues.append(f"{len(no_prompt)} samples have no prompt .txt file: {no_prompt[:5]}{' ...' if len(no_prompt) > 5 else ''}")
    si, sw = _size_assessment(paired)
    issues.extend(si); warnings.extend(sw)
    return {"valid": len(issues) == 0, "format": "image_directory", "sample_count": paired, "issues": issues, "warnings": warnings}


def validate_csv(path):
    issues, warnings = [], []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as e:
        return {"valid": False, "format": "csv", "sample_count": 0, "issues": [f"Failed to read CSV: {e}"], "warnings": []}
    missing = [c for c in ["path_target", "prompt"] if c not in cols]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    if not [c for c in cols if "path_control" in c]:
        issues.append("No path_control* columns found")
    si, sw = _size_assessment(len(rows))
    issues.extend(si); warnings.extend(sw)
    return {"valid": len(issues) == 0, "format": "csv", "sample_count": len(rows), "issues": issues, "warnings": warnings}


def validate_parquet(repo_or_path):
    issues, warnings = [], []
    sample_count = -1
    p = Path(repo_or_path)
    if p.is_dir():
        parquets = list((p / "data").glob("*.parquet"))
        if not parquets:
            return {"valid": False, "format": "parquet", "sample_count": 0,
                    "issues": [f"Local dir {p} has no data/*.parquet files"], "warnings": []}
        try:
            import pandas as pd
            df = pd.read_parquet(parquets[0])
            cols = df.columns.tolist()
            missing = [c for c in ["target_image", "control_images", "prompt"] if c not in cols]
            if missing:
                issues.append(f"Missing required columns in parquet: {missing} (need ['target_image', 'control_images', 'prompt'])")
            sample_count = len(df)
            si, sw = _size_assessment(sample_count)
            issues.extend(si); warnings.extend(sw)
        except ImportError:
            warnings.append("pandas not available; could not inspect parquet schema")
            sample_count = len(parquets)
    else:
        if "/" not in repo_or_path:
            issues.append(f"Not a valid HF repo id (need 'owner/name' format): {repo_or_path}")
        else:
            warnings.append(f"Remote HF repo '{repo_or_path}' — schema validation requires network access and cannot be done from file headers alone. Verify by running training once with max_train_steps=1.")
    return {"valid": len(issues) == 0, "format": "parquet", "sample_count": sample_count, "issues": issues, "warnings": warnings}


def main():
    p = argparse.ArgumentParser(description="Validate a dataset for qwen-image-edit fine-tuning.")
    p.add_argument("dataset_path", help="Local dir, parquet dir, remote HuggingFace repo id, or .csv file path")
    args = p.parse_args()
    fmt = detect_format(args.dataset_path)
    if fmt == "image_directory":
        result = validate_local(Path(args.dataset_path))
    elif fmt == "parquet":
        result = validate_parquet(args.dataset_path)
    elif fmt == "csv":
        result = validate_csv(Path(args.dataset_path))
    else:
        result = {"valid": False, "format": "unknown", "sample_count": 0,
                  "issues": [f"Could not detect format for {args.dataset_path}"], "warnings": []}
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
```

Run:

```bash
python dataset_validate.py <path>
# Image directory:    python dataset_validate.py <path/to/dataset>
#   parquet dir: python dataset_validate.py <path/to/parquet-dir>
# CSV:          python dataset_validate.py <path/to/dataset.csv>
```

JSON verdict format:
```json
{
  "valid": true,
  "format": "image_directory",
  "sample_count": 35,
  "issues": [],
  "warnings": []
}
```

Exit code 0 = valid (proceed to Step 3); 1 = invalid (fix before proceeding). Issues are
blockers; warnings are advisory (e.g., small sample count fine for smoke test, problematic
for real training).

Common issues caught: missing `training_images/` or `control_images/` subdir; target images
without matching control or prompt; sample count too low.

**Manual check after the script passes** — the validator only checks format and structure; it
cannot verify semantic correctness. Before starting a full training run, inspect 3–5 random
pairs by eye:

- Does the target image look like a plausible result of applying the prompt to the control image(s)?
- Is it clear what changed between control and target, and does the prompt describe that change?
- Is the subject/scene the same in control and target (they are before/after, not unrelated images)?

If any pair fails this check, the data has a consistency problem that will reduce LoRA quality.
Finding and fixing a few bad pairs early is much cheaper than diagnosing a poorly-trained model later.

## 2.8 Hold out a test set

Before finalising your dataset, **set aside at least 3 samples that will not be used for
training**. These become your held-out test set for Step 8 post-training visual comparison
(base model vs. trained LoRA).

Choosing which samples to hold out:
- Pick samples that are **representative** of the edit you are teaching — the comparison is
  only meaningful if the test images resemble the training distribution.
- Do **not** reuse training samples as test samples. The model has seen those images during
  training, so any apparent quality difference is unreliable.

**For datasets with a parquet test split** (directory with `data/test-*.parquet`): pass the
directory or parquet file path with `--test-parquet` in Step 8.

**For image directory datasets**: physically move the held-out pairs to a separate folder
(e.g. `test/`) before running `dataset_validate.py`. Pass that folder with `--test-dir`
in Step 8 (the inference script supports the same `training_images/` + `control_images/`
layout directly, no conversion needed).

If you are working from a small dataset where holding out samples would leave too few for
training (see §2.3 thresholds), collect a few additional pairs specifically for testing rather
than reducing the training set.

## PASS signal

- `dataset_validate.py` exits 0; `valid: true` in JSON output
- At least 3 test samples set aside in a separate folder (or confirmed in the parquet test split)
- **Dataset path noted** for use at Step 4

**Proceed to**: `qwen-image-edit-aipc-finetune-03-env-setup`
