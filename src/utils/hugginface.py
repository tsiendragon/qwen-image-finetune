# pip install -U datasets huggingface_hub hf-transfer
# export HUGGINGFACE_HUB_TOKEN=hf_xxx   # or HF_TOKEN

from pathlib import Path
from typing import Dict, List, Optional
import os
import re

from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence
from huggingface_hub import create_repo

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

FEATURES = Features(
    {
        "id": Value("string"),
        "control_images": Sequence(Image()),  # list of control images (>=1)
        "control_mask": Image(),  # None if absent
        "target_image": Image(),  # None in test if absent
        "prompt": Value("string"),
    }
)

_digit_suffix_re = re.compile(r"_(\d+)$")  # matches ..._<n>


def _pick_first_existing(base: Path) -> Optional[Path]:
    """Return first existing file among known image extensions for a base path."""
    stem = base.with_suffix("")  # drop any ext
    for ext in _IMG_EXTS:
        p = Path(stem.as_posix() + ext)
        if p.exists():
            return p
        pU = Path(stem.as_posix() + ext.upper())
        if pU.exists():
            return pU
    return None


def _find_control_images(ctrl_dir: Path, base: str) -> List[Path]:
    """
    Collect control images for a basename:
      <base>.* (primary), <base>_1.*, <base>_2.*, ...
      Excludes <base>_mask.*
      Returns sorted: primary first, then _1, _2, ...
    """
    results: List[Path] = []
    # primary
    primary = _pick_first_existing(ctrl_dir / base)
    if primary:
        results.append(primary)

    # numbered variants by glob then filter
    for ext in _IMG_EXTS:
        for p in sorted(ctrl_dir.glob(f"{base}_*{ext}")):
            stem = p.stem
            if stem.endswith("_mask"):  # exclude mask
                continue
            m = _digit_suffix_re.search(stem[len(base):])
            if m:  # only accept pure numeric suffixes
                results.append(p)

    # natural sort: primary first, then by numeric suffix
    def _order_key(p: Path):
        s = p.stem
        if s == base:
            return (0, 0)
        m = _digit_suffix_re.search(s[len(base):])
        n = int(m.group(1)) if m else 1_000_000
        return (1, n)

    results = sorted(list(dict.fromkeys(results)), key=_order_key)
    return results


def _find_mask(ctrl_dir: Path, base: str) -> Optional[Path]:
    """Return <base>_mask.* if exists, else None."""
    for ext in _IMG_EXTS:
        p = ctrl_dir / f"{base}_mask{ext}"
        if p.exists():
            return p
    return None


def _collect_split(root: Path, split: str) -> Dataset:
    """
    Build a HF Dataset (schema = FEATURES) from:
      root/split/control_images/
      root/split/training_images/
    Requires a prompt file <base>.txt in training_images.
    Target image <base>.* optional (None if missing).
    """
    split_dir = root / split
    ctrl_dir = split_dir / "control_images"
    tgt_dir = split_dir / "training_images"

    if not ctrl_dir.is_dir() or not tgt_dir.is_dir():
        raise FileNotFoundError(
            f"[{split}] missing subfolders: {ctrl_dir} and/or {tgt_dir}"
        )

    txt_files = sorted(tgt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"[{split}] no *.txt prompts in {tgt_dir}")

    rows: List[dict] = []
    skipped_no_ctrl = 0

    for txt in txt_files:
        base = txt.stem
        control_list = _find_control_images(ctrl_dir, base)
        if not control_list:
            skipped_no_ctrl += 1
            continue

        mask = _find_mask(ctrl_dir, base)
        tgt = _pick_first_existing(tgt_dir / base)  # optional

        rows.append(
            {
                "id": base,
                "control_images": [str(p) for p in control_list],  # list of paths
                "control_mask": str(mask) if mask is not None else None,
                "target_image": str(tgt) if tgt is not None else None,
                "prompt": txt.read_text(encoding="utf-8").strip(),
            }
        )

    if not rows:
        raise RuntimeError(
            f"[{split}] collected 0 valid items (missing control images?)."
        )
    if skipped_no_ctrl:
        print(f"[WARN][{split}] samples skipped (no control image): {skipped_no_ctrl}")

    return Dataset.from_list(rows, features=FEATURES)


def upload_editing_dataset(root_dir: str, repo_id: str, private: bool = True) -> None:
    """
    Build {train,test} with unified schema and push to HF Hub (dataset repo).
    Token read from env: HUGGINGFACE_HUB_TOKEN or HF_TOKEN.
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    splits: Dict[str, Dataset] = {}
    for split in ("train", "test"):
        if (root / split).exists():
            splits[split] = _collect_split(root, split)

    if not splits:
        raise RuntimeError(
            "No splits found. Expect ./train and/or ./test under root_dir."
        )

    dsd = DatasetDict(splits)

    create_repo(
        repo_id, repo_type="dataset", private=private, exist_ok=True, token=token
    )
    dsd.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload dataset (multi-control, optional mask, unified schema)",
    )
    print(
        f"✅ Pushed {repo_id} | splits={list(dsd.keys())} | "
        + f"sizes={{ {', '.join([k+':'+str(len(v)) for k, v in dsd.items()])} }}"
    )


def load_editing_dataset(repo_id: str, split: Optional[str] = None):
    """
    Load back from Hub. Works for private repos via env token.
    Returns DatasetDict if split=None, else a single Dataset.
    """
    from datasets import load_dataset

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    kwargs = {"token": token} if token else {}
    if split is None:
        return load_dataset(repo_id, **kwargs)
    return load_dataset(repo_id, split=split, **kwargs)
