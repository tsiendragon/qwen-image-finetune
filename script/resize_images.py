#!/usr/bin/env python3
"""
图片大小调整脚本
如果图片宽度大于1024像素，则调整为1024像素，保持宽高比不变
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm




def resize_image_if_needed(input_path, output_path, max_width=1024):
    """
    如果图片宽度大于max_width，则调整大小并保持宽高比

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        max_width: 最大宽度，默认1024

    Returns:
        bool: 是否进行了调整
    """
    try:
        with Image.open(input_path) as img:
            original_width, original_height = img.size

            # 如果宽度小于等于max_width，直接复制
            if original_width <= max_width:
                if input_path != output_path:
                    img.save(output_path, quality=95, optimize=True)
                return False

            # 计算新的尺寸，保持宽高比
            new_width = max_width
            new_height = int((original_height * max_width) / original_width)

            # 调整图片大小
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存调整后的图片
            resized_img.save(output_path, quality=95, optimize=True)

            print(f"调整 {input_path}: {original_width}x{original_height} -> {new_width}x{new_height}")
            return True

    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {e}")
        return False


def process_directory(input_dir, output_dir=None, max_width=1024, in_place=False):
    """
    处理目录中的所有图片

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则使用输入目录
        max_width: 最大宽度
        in_place: 是否原地修改
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return

    if in_place:
        output_path = input_path
    else:
        output_path = Path(output_dir) if output_dir else input_path.parent / f"{input_path.name}_resized"

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 收集所有图片文件
    image_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)

    if not image_files:
        print(f"在目录 {input_dir} 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    # 创建输出目录结构
    if not in_place:
        output_path.mkdir(parents=True, exist_ok=True)

    resized_count = 0
    total_count = len(image_files)

    # 处理每个图片文件
    for img_file in tqdm(image_files, desc="处理图片"):
        # 计算相对路径
        rel_path = img_file.relative_to(input_path)

        if in_place:
            output_file = img_file
        else:
            output_file = output_path / rel_path
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # 调整图片大小
        if resize_image_if_needed(img_file, output_file, max_width):
            resized_count += 1

    print(f"\n处理完成!")
    print(f"总图片数: {total_count}")
    print(f"调整大小的图片数: {resized_count}")
    print(f"未调整的图片数: {total_count - resized_count}")

    if not in_place:
        print(f"输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='调整图片大小脚本')
    parser.add_argument('input_dir', help='输入图片目录路径')
    parser.add_argument('-o', '--output_dir', help='输出目录路径（可选）')
    parser.add_argument('-w', '--max_width', type=int, default=1024,
                       help='最大宽度（默认: 1024）')
    parser.add_argument('--in-place', action='store_true',
                       help='原地修改图片（覆盖原文件）')

    args = parser.parse_args()

    # 确认原地修改操作
    if args.in_place:
        response = input("警告: 将原地修改图片文件，这将覆盖原始文件。是否继续？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return

    process_directory(
        args.input_dir,
        args.output_dir,
        args.max_width,
        args.in_place
    )


if __name__ == "__main__":
    main()
