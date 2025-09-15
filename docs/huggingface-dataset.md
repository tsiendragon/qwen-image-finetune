### Hugging Face 数据集上传与下载指南

本文档介绍如何将项目中的图像编辑数据集按统一规范上传到 Hugging Face Hub，以及后续如何下载/加载使用。配套实现见 `src/utils/hugginface.py` 与示例脚本 `upload_dataset.py`。

---

## 目录结构规范

数据根目录需包含 `train/` 与/或 `test/` 两个子目录，每个子目录下包含：

- `control_images/`：控制图像（至少 1 张）。支持多控制图像，命名规则：
  - 主图：`<base>.*`（任意扩展名）
  - 追加序号：`<base>_control_1.*`, `<base>_control_2.*`, ...（使用 _control_N 格式）
  - 可选掩码：`<base>_mask.*`（若存在将被用作 `control_mask`）
- `training_images/`：目标图与文本提示对：
  - 必需：`<base>.txt`（prompt）
  - 可选：`<base>.*`（target image，测试集可缺省）

支持的图片扩展名：`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`（大小写均可）。

示例：

```
face_seg/
  train/
    control_images/
      sampleA.jpg
      sampleA_control_1.jpg
      sampleA_mask.png
    training_images/
      sampleA.txt
      sampleA.png
  test/
    control_images/
      sampleB.png
    training_images/
      sampleB.txt
```

---

## 环境准备

安装依赖：

```bash
pip install -U datasets huggingface_hub hf-transfer
```

配置访问令牌（两者其一）：

```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
# 或
export HF_TOKEN=hf_xxx
```

为加速上传，建议启用：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 数据集字段规范（Features）

内部统一的 HF Dataset schema：

- `id: string`（样本基名）
- `control_images: Sequence[Image]`（≥1 张）
- `control_mask: Image | None`（可选）
- `target_image: Image | None`（训练集建议提供，测试可为空）
- `prompt: string`（从 `<base>.txt` 读取）

---

## 上传到 Hugging Face Hub

使用脚本 `upload_dataset.py`（内部调用 `upload_editing_dataset`）：

```python
from src.utils.hugginface import upload_editing_dataset

upload_editing_dataset(
    root_dir="face_seg",                 # 含 train/ 与/或 test/
    repo_id="<org_or_user>/<dataset>",  # 例如: "TsienDragon/face_segmentation_20"
    private=False                         # 公开或私有
)
```

或直接运行示例脚本（按需修改）：

```bash
python upload_dataset.py
```

注意：

- 首次会自动创建 HF 数据集仓库（若不存在）。
- 缺少 `train/` 与 `test/` 时会报错；至少提供其一。
- 若某个样本缺少控制图像，该样本将被跳过并在日志给出警告。

---

## 从 Hub 下载/加载

统一加载接口：

```python
from src.utils.hugginface import load_editing_dataset

dsd = load_editing_dataset("<org_or_user>/<dataset>")  # 返回 DatasetDict
print(dsd)

# 取训练集一个样本
sample = dsd["train"][0]
print(type(sample["control_images"]))
print(type(sample["control_mask"]))
print(type(sample["target_image"]))
print(type(sample["prompt"]))
```

如为私有数据集，需提前设置 `HUGGINGFACE_HUB_TOKEN` 或 `HF_TOKEN` 环境变量。

按需仅加载单个切分：

```python
train_ds = load_editing_dataset("<org_or_user>/<dataset>", split="train")
```

---

## 常见问题与排查

- 缺少 token：报错提示设置 `HUGGINGFACE_HUB_TOKEN` 或 `HF_TOKEN`。
- 未找到切分：确保根目录下存在 `train/` 或 `test/` 子目录。
- 无任何有效样本：
  - 检查 `training_images/` 是否存在 `<base>.txt`（prompt 必需）。
  - 检查 `control_images/` 是否存在 `<base>.*` 或 `<base>_control_1.*` 等控制图像。
  - 确认额外控制图像使用正确的命名格式（如 `_control_1`, `_control_2`）。
- 掩码未生效：确认命名为 `<base>_mask.<ext>` 且位于 `control_images/` 下。

---

## 参考实现

- 上传与加载实现：`src/utils/hugginface.py`
- 上传示例：`upload_dataset.py`


