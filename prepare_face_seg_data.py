#!/usr/bin/env python3
"""
人脸分割数据整理脚本
将 /mnt/nas/public2/lilong/data/openimages/face_seg 数据整理成 dataset.py 可用的格式

数据结构：
- training_images/: 分割图（训练目标）
- control_images/: 人脸图（控制条件）
"""

import os
import shutil
from pathlib import Path
import argparse


def prepare_face_seg_data(source_dir, target_dir):
    """
    整理人脸分割数据

    Args:
        source_dir: 原始数据目录 (/mnt/nas/public2/lilong/data/openimages/face_seg)
        target_dir: 目标数据目录
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 检查源目录
    img_dir = source_path / "img"
    masks_dir = source_path / "masks"

    if not img_dir.exists():
        raise ValueError(f"源图片目录不存在: {img_dir}")
    if not masks_dir.exists():
        raise ValueError(f"源mask目录不存在: {masks_dir}")

    # 创建目标目录结构
    target_images_dir = target_path / "training_images"
    target_control_dir = target_path / "control_images"

    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_control_dir.mkdir(parents=True, exist_ok=True)

    print(f"创建目标目录: {target_images_dir}")
    print(f"创建目标目录: {target_control_dir}")

    # 扫描所有图片文件
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))

    caption_text = "change the image from the face to the face segmentation mask"

    processed_count = 0
    skipped_count = 0

    for img_file in img_files:
        # 获取文件名（不含扩展名）
        base_name = img_file.stem

        # 查找对应的mask文件
        mask_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_mask = masks_dir / f"{base_name}{ext}"
            if potential_mask.exists():
                mask_file = potential_mask
                break

        if mask_file is None:
            print(f"警告: 找不到对应的mask文件: {base_name}")
            skipped_count += 1
            continue

                # 复制mask文件到 training_images 目录（分割图作为训练目标）
        target_mask_file = target_images_dir / mask_file.name
        shutil.copy2(mask_file, target_mask_file)

        # 复制原始图片文件到 control_images 目录（人脸图作为控制条件）
        target_img_file = target_control_dir / img_file.name
        shutil.copy2(img_file, target_img_file)

        # 创建caption文件（与训练图片同目录）
        caption_file = target_images_dir / f"{base_name}.txt"
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption_text)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"已处理 {processed_count} 个文件...")

    print(f"\n数据整理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过文件: {skipped_count} 个文件")
    print(f"目标目录: {target_path}")
    print(f"训练图片目录（分割图）: {target_images_dir}")
    print(f"控制图片目录（人脸图）: {target_control_dir}")

    # 验证数据完整性
    verify_data_integrity(target_path)


def verify_data_integrity(target_dir):
    """验证整理后的数据完整性"""
    target_path = Path(target_dir)
    training_images_dir = target_path / "training_images"
    control_images_dir = target_path / "control_images"

    print("\n验证数据完整性...")

    # 获取所有图片文件
    img_files = list(training_images_dir.glob("*.jpg")) + \
                list(training_images_dir.glob("*.jpeg")) + \
                list(training_images_dir.glob("*.png"))

    missing_files = []

    for img_file in img_files:
        base_name = img_file.stem

        # 检查caption文件
        caption_file = training_images_dir / f"{base_name}.txt"
        if not caption_file.exists():
            missing_files.append(f"缺少caption文件: {caption_file}")

        # 检查对应的control文件
        control_found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            control_file = control_images_dir / f"{base_name}{ext}"
            if control_file.exists():
                control_found = True
                break

        if not control_found:
            missing_files.append(f"缺少control文件: {base_name}")

    if missing_files:
        print("发现问题:")
        for issue in missing_files:
            print(f"  - {issue}")
    else:
        print(f"✓ 数据完整性验证通过，共 {len(img_files)} 对有效数据")


def main():
    parser = argparse.ArgumentParser(description="整理人脸分割数据")
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/nas/public2/lilong/data/openimages/face_seg",
        help="源数据目录路径"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/mnt/nas/public2/lilong/data/openimages/face_seg_processed",
        help="目标数据目录路径"
    )

    args = parser.parse_args()

    print(f"源目录: {args.source}")
    print(f"目标目录: {args.target}")
    print("开始整理数据...")

    try:
        prepare_face_seg_data(args.source, args.target)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
