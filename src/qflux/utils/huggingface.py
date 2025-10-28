# pip install -U datasets huggingface_hub hf-transfer
# export HUGGINGFACE_HUB_TOKEN=hf_xxx   # or HF_TOKEN
import hashlib
import os
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import HfApi, create_repo, hf_hub_download

from qflux.trainer.constants import LORA_FILE_BASE_NAME


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

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


def _pick_first_existing(base: Path) -> Path | None:
    """Return first existing file among known image extensions for a base path."""
    stem = base.with_suffix("")  # drop any ext
    for ext in _IMG_EXTS:
        p = Path(stem.as_posix() + ext)
        if p.exists():
            return p
        pu = Path(stem.as_posix() + ext.upper())
        if pu.exists():
            return pu
    return None


def _find_control_images(ctrl_dir: Path, base: str) -> list[Path]:
    """
    Collect control images for a basename:
      <base>.* (primary), <base>_control_1.*, <base>_control_2.*, ...
      Excludes <base>_mask.*
      Returns sorted: primary first, then _1, _2, ...
    """
    results: list[Path] = []
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
            m = _digit_suffix_re.search(stem[len(base) :])
            if m:  # only accept pure numeric suffixes
                results.append(p)

    # natural sort: primary first, then by numeric suffix
    def _order_key(p: Path):
        s = p.stem
        if s == base:
            return (0, 0)
        m = _digit_suffix_re.search(s[len(base) :])
        n = int(m.group(1)) if m else 1_000_000
        return (1, n)

    results = sorted(dict.fromkeys(results), key=_order_key)
    return results


def _find_mask(ctrl_dir: Path, base: str) -> Path | None:
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
        raise FileNotFoundError(f"[{split}] missing subfolders: {ctrl_dir} and/or {tgt_dir}")

    txt_files = sorted(tgt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"[{split}] no *.txt prompts in {tgt_dir}")

    rows: list[dict] = []
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
        raise RuntimeError(f"[{split}] collected 0 valid items (missing control images?).")
    if skipped_no_ctrl:
        print(f"[WARN][{split}] samples skipped (no control image): {skipped_no_ctrl}")

    return Dataset.from_list(rows, features=FEATURES)


def _find_image_with_any_format(root: str | Path, _csv_path: str) -> Path | None:
    """
    根据 CSV 中的路径查找实际存在的图像文件，自动尝试所有支持的格式。

    Args:
        root: 数据集根目录
        csv_path: CSV 中记录的相对路径（可能格式不正确）

    Returns:
        实际存在的图像文件路径，如果未找到则返回 None
    """
    _csv_path_obj: Path = Path(_csv_path)
    _root: Path = Path(root)

    # 首先尝试 CSV 中记录的确切路径
    exact_path = _root / _csv_path_obj
    if exact_path.exists():
        return exact_path

    # 如果确切路径不存在，尝试所有支持的格式
    stem_path = exact_path.with_suffix("")  # 去掉扩展名

    for ext in _IMG_EXTS:
        # 尝试小写扩展名
        candidate = Path(str(stem_path) + ext)
        if candidate.exists():
            return candidate

        # 尝试大写扩展名
        candidate = Path(str(stem_path) + ext.upper())
        if candidate.exists():
            return candidate

    return None


def _collect_split_from_csv(root: str, split: str) -> Dataset:
    """
    Build a HF Dataset from CSV metadata file.

    Expected structure:
      root/split.csv - CSV with columns: image,control,control_1,prompt (train)
                       or control,control_1,prompt (test)
      root/ - all image files referenced in CSV

    CSV columns:
    - image: target image path (optional, e.g., train/target/xxx.png)
    - control: main control image path (e.g., train/control/xxx.png)
    - control_1, control_2, ...: additional control images
    - prompt: text prompt
    """
    import pandas as pd

    _root = Path(root)

    csv_file = _root / f"{split}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file.as_posix()}")

    df = pd.read_csv(csv_file.as_posix())

    # 检测CSV列结构
    required_cols = ["control", "prompt"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file.as_posix()}")

    rows: list[dict] = []

    for idx, row in df.iterrows():
        # 生成ID
        if "image" in df.columns and pd.notna(row["image"]):
            # 从target图像路径生成ID
            base_id = Path(row["image"]).stem
        else:
            # 从control图像路径生成ID
            base_id = Path(row["control"]).stem

        # 收集控制图像 - 自动查找正确格式
        control_images = []

        # 主控制图像
        main_control_path = _find_image_with_any_format(root, row["control"])
        if main_control_path:
            control_images.append(str(main_control_path))

        # 额外控制图像 (control_1, control_2, ...)
        control_idx = 1
        while f"control_{control_idx}" in df.columns:
            if pd.notna(row[f"control_{control_idx}"]):
                extra_control_path = _find_image_with_any_format(root, row[f"control_{control_idx}"])
                if extra_control_path:
                    control_images.append(str(extra_control_path))
            control_idx += 1

        if not control_images:
            print(f"[WARN] Skipping row {idx}: no control images found")
            continue

        # 目标图像 (可选) - 自动查找正确格式
        target_image = None
        if "image" in df.columns and pd.notna(row["image"]):
            target_path = _find_image_with_any_format(root, row["image"])
            if target_path:
                target_image = str(target_path)

        # 查找mask (可选) - 可能在 control 或 target 目录中
        control_mask = None
        control_parent = Path(row["control"]).parent
        control_stem = Path(row["control"]).stem

        # 构建可能的 mask 路径 - 支持多种图像格式
        potential_masks = []

        # 在 control 目录中查找所有支持的格式
        for ext in _IMG_EXTS:
            potential_masks.extend(
                [
                    root / control_parent / f"{control_stem}_mask{ext}",
                    root / control_parent / f"{control_stem}_mask{ext.upper()}",
                ]
            )

        # 如果有 target 图像，也在 target 目录中查找
        if "image" in df.columns and pd.notna(row["image"]):
            target_parent = Path(row["image"]).parent
            target_stem = Path(row["image"]).stem
            for ext in _IMG_EXTS:
                potential_masks.extend(
                    [
                        root / target_parent / f"{target_stem}_mask{ext}",
                        root / target_parent / f"{target_stem}_mask{ext.upper()}",
                        # 也尝试用 control 的 stem 在 target 目录中查找
                        root / target_parent / f"{control_stem}_mask{ext}",
                        root / target_parent / f"{control_stem}_mask{ext.upper()}",
                    ]
                )

        for potential_mask in potential_masks:
            if potential_mask.exists():
                control_mask = str(potential_mask)
                break

        rows.append(
            {
                "id": base_id,
                "control_images": control_images,
                "control_mask": control_mask,
                "target_image": target_image,
                "prompt": str(row["prompt"]).strip(),
            }
        )

    if not rows:
        raise RuntimeError(f"[{split}] No valid samples found in CSV")

    print(f"[{split}] Loaded {len(rows)} samples from CSV")
    return Dataset.from_list(rows, features=FEATURES)


def upload_editing_dataset_from_csv(root_dir: str, repo_id: str, private: bool = True) -> None:
    """
    Build dataset from CSV metadata files and push to HF Hub.

    Expected structure:
      root_dir/
        train.csv - CSV with image,control,control_1,prompt columns
        test.csv  - CSV with control,control_1,prompt columns
        train/target/xxx.png - target images
        train/control/xxx.png - control images
        test/control/xxx.png - control images

    Args:
        root_dir: Directory containing CSV files and images
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        private: Whether to create a private repository
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        msg = "Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment."
        raise OSError(msg)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    splits: dict[str, Dataset] = {}

    # 检查并加载CSV文件
    for split in ("train", "test"):
        csv_file = root / f"{split}.csv"
        if csv_file.exists():
            try:
                splits[split] = _collect_split_from_csv(str(root), split)
            except Exception as e:
                print(f"[WARN] Failed to load {split} split: {e}")
                continue

    if not splits:
        msg = "No valid splits found. Expected train.csv and/or test.csv"
        raise RuntimeError(msg)

    dsd = DatasetDict(splits)

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    dsd.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload dataset from CSV (multi-control, unified schema)",
    )
    size_info = ", ".join([k + ":" + str(len(v)) for k, v in dsd.items()])
    print(f"✅ Pushed {repo_id} | splits={list(dsd.keys())} | sizes={{{size_info}}}")


def upload_editing_dataset(root_dir: str, repo_id: str, private: bool = True) -> None:
    """
    Build {train,test} with unified schema and push to HF Hub (dataset repo).
    Token read from env: HUGGINGFACE_HUB_TOKEN or HF_TOKEN.
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise OSError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    splits: dict[str, Dataset] = {}
    for split in ("train", "test"):
        if (root / split).exists():
            splits[split] = _collect_split(root, split)

    if not splits:
        raise RuntimeError("No splits found. Expect ./train and/or ./test under root_dir.")

    dsd = DatasetDict(splits)

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    dsd.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload dataset (multi-control, optional mask, unified schema)",
    )
    print(
        f"✅ Pushed {repo_id} | splits={list(dsd.keys())} | "
        + f"sizes={{ {', '.join([k + ':' + str(len(v)) for k, v in dsd.items()])} }}"
    )


def load_editing_dataset(repo_id: str, split: str | None = None):
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


def is_huggingface_repo(path: str) -> bool:
    """
    Detect if the path is a Hugging Face repository ID.

    HF repo patterns:
    - username/dataset-name
    - organization/dataset-name
    - dataset-name (for datasets without namespace)

    Local path patterns:
    - /absolute/path/to/dataset
    - ./relative/path
    - ../relative/path
    - path/without/leading/slash (treated as relative)
    """
    # Check if it's an absolute path
    if os.path.isabs(path):
        return False

    # Check if it's a relative path with ./ or ../
    if path.startswith("./") or path.startswith("../"):
        return False

    # Check if the path exists locally
    if os.path.exists(path):
        return False

    # Check HF repo pattern: should contain at most one '/'
    # and not contain invalid characters for repo names
    parts = path.split("/")
    if len(parts) <= 2 and all(part.replace("-", "").replace("_", "").isalnum() for part in parts):
        from datasets import get_dataset_config_names

        try:
            a = get_dataset_config_names(path)
            if a:
                return True
        except Exception as e:
            print(e)
            return False
    return False


def download_lora(
    repo_id="TsienDragon/qwen-image-edit-lora-face-segmentation", filename="pytorch_lora_weights.safetensors"
):
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_sha_index(api: HfApi, repo_id: str) -> dict[str, str]:
    """返回 {path_in_repo: sha256_hex}（自动处理 LFS或非LFS）"""
    idx = {}
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    for s in getattr(info, "siblings", []) or []:
        if getattr(s, "lfs", None) and "oid" in s.lfs and str(s.lfs["oid"]).startswith("sha256:"):
            idx[s.rfilename] = s.lfs["oid"].split(":", 1)[1]
        elif getattr(s, "sha", None):  # 非LFS少见
            idx[s.rfilename] = s.sha
    return idx


def upload_lora_safetensors(
    src_path: str,  # 本地 .safetensors 文件路径，或包含它的目录
    repo_id: str,  # "user_or_org/repo-name"
    *,
    token: str | None = None,  # 不传则读 HF_TOKEN 或 HUGGINGFACE_HUB_TOKEN
    private: bool = True,
    remote_name: str | None = None,  # 远端文件名；默认仍叫 pytorch_lora_weights.safetensors
    commit_message: str = "Upload LoRA weights",
    extra_files: dict[str, str] | None = None,  # {本地路径: 远端路径}
) -> str:
    """
    只上传 LoRA 权重文件到 Hugging Face Hub（仓库类型为 model）。
    若远端已有同名且内容一致则跳过。返回仓库的 Web URL。
    """
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing token. Set HF_TOKEN or pass token=...")

    # 1) 解析本地文件路径：支持传目录或文件
    if os.path.isdir(src_path):
        local_file = os.path.join(src_path, LORA_FILE_BASE_NAME)
    else:
        local_file = src_path

    if not os.path.isfile(local_file):
        raise FileNotFoundError(f"LoRA file not found: {local_file}")
    if not local_file.endswith(".safetensors"):
        raise ValueError("Expect a .safetensors file")

    # 2) 远端文件名
    remote_file = remote_name or LORA_FILE_BASE_NAME

    # 3) 创建仓库（存在即复用）
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    api = HfApi(token=token)

    # 4) 构建远端文件→sha 索引，用于跳过相同文件
    try:
        remote_idx = _remote_sha_index(api, repo_id)
    except Exception:
        remote_idx = {}

    # 5) 主 LoRA 文件：仅在不存在或内容不同时上传
    local_sha = _sha256(local_file)
    if remote_idx.get(remote_file) != local_sha:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=remote_file,  # 可用 "lora/xxx.safetensors" 放子目录
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

    # 6) 可选：额外文件（README / train_config.yaml / adapter_config.json 等）
    if extra_files:
        for lp, rp in extra_files.items():
            if not os.path.isfile(lp):
                raise FileNotFoundError(f"Extra file not found: {lp}")
            sha = _sha256(lp)
            # 若远端已有且内容一致，则跳过
            if remote_idx.get(rp) == sha:
                continue
            api.upload_file(
                path_or_fileobj=lp,
                path_in_repo=rp,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add/Update {os.path.basename(rp)}",
            )

    # 7) 返回网页地址（不要再取 .html_url）
    return f"https://huggingface.co/{repo_id}"


if __name__ == "__main__":
    from datasets import get_dataset_config_names

    a = get_dataset_config_names("TsienDragon/face_segmentation_20")
    # # returns ['default']
    print(a, type(a))
    for x in a:
        print(x, type(x))
    # print(is_huggingface_repo("TsienDragon/face_segmentation0"))

    # dataset = load_editing_dataset("TsienDragon/face_segmentation_20", split="train")
    # print(dataset)
    # data = dataset[0]
    # print(data)
    # print('len', len(dataset))

    # lora_path = download_lora()
    # print(lora_path)
    # Test LoRA upload
    # lora_weight = '/tmp/image_edit_lora/character_composition_fp16/'
    # lora_weight += 'characterCompositionQwenImageEditFp16/v1/'
    # lora_weight += 'pytorch_lora_weights.safetensors'
    # train_config = '/mnt/nas/public2/lilong/repos/qwen-image-finetune/'
    # train_config += 'tests/test_configs/'
    # train_config += 'test_example_qwen_image_edit_fp16_character_composition.yaml'
    # upload_lora_safetensors(
    #     lora_weight, 'TsienDragon/qwen-image-edit-character-composition',
    #     extra_files={train_config: 'train_config.yaml'}
    # )

    # lora_weight = '/tmp/image_edit_lora/character_composition_fp16/'
    # lora_weight += 'characterCompositionFluxKontextFp16/pytorch_lora_weights.safetensors'
    # train_config = '/mnt/nas/public2/lilong/repos/qwen-image-finetune/'
    # train_config += 'tests/test_configs/'
    # train_config += 'test_example_fluxkontext_fp16_character_composition.yaml'
    # url = upload_lora_safetensors(
    #     lora_weight, 'TsienDragon/flux-kontext-character-composition',
    #     extra_files={train_config: 'train_config.yaml'}
    # )
    # print(url)

    # 测试 CSV 格式数据集上传
    # upload_editing_dataset_from_csv(
    #     '/mnt/nas/public2/lilong/data/openimages/character_composition',
    #     'TsienDragon/character-composition'
    # )
    dd = load_editing_dataset("TsienDragon/character-composition")
    print(dd)
    print(dd["train"][0])
    print(dd["test"][0])
    print(len(dd["train"]))
    print(len(dd["test"]))
    print(dd["train"].features)
    print(dd["test"].features)
    print(dd["train"].column_names)
    print(dd["test"].column_names)
