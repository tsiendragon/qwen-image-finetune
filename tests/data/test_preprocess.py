"""
测试配置模块的功能，确保配置文件正确加载和解析
"""

import numpy as np


class TestImageProcessor:
    """配置加载和验证的测试类"""

    def test_select_pixels_candidate(self):
        """测试选择最佳像素数"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs
        from qflux.utils.images import calculate_best_resolution

        config = ImageProcessorInitArgs(
            multi_resolutions=["512*512", "640*640", "768*512", "832*576"],
            max_aspect_ratio=3.0
        )
        processor = ImageProcessor(config)
        assert processor.multi_resolutions == [512*512, 640*640, 768*512, 832*576]
        assert processor.max_aspect_ratio == 3.0

        # Test pixel candidate selection and resolution calculation
        test_cases = [
            (1024, 768, 832*576),
            (300, 900, 256*1024),
            (400, 400, 512*512),
            (600, 600, 768*512),
            (800, 400, 512*512),
            (900, 500, 832*576),
        ]

        for w, h, expected_pixels in test_cases:
            selected_pixels = processor._select_pixels_candidate(w, h)
            assert selected_pixels == expected_pixels, \
                f"Failed for {w}x{h}: got {selected_pixels}, expected {expected_pixels}"

            # Verify calculate_best_resolution produces valid dimensions
            new_w, new_h = calculate_best_resolution(w, h, selected_pixels)
            assert new_w % 32 == 0 and new_h % 32 == 0, \
                f"Resolution not 32-divisible: {new_w}x{new_h}"
            assert abs(new_w * new_h - selected_pixels) < selected_pixels * 0.1, \
                f"Area mismatch: {new_w}*{new_h} vs {selected_pixels}"

    def test_process_image_multi_resolution(self):
        """测试 multi_resolutions 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        config = ImageProcessorInitArgs(
            multi_resolutions=["512*512", "640*640", "768*512", "832*576"],
            max_aspect_ratio=3.0
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (800x600 RGB)
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, None, None)

        # 验证输出尺寸
        assert processed.shape[2] == 3, "Image should have 3 channels"
        assert processed.shape[0] % 16 == 0, "Height should be divisible by 16"
        assert processed.shape[1] % 16 == 0, "Width should be divisible by 16"

        # 验证选择的像素数接近某个候选值
        actual_pixels = processed.shape[0] * processed.shape[1]
        assert any(abs(actual_pixels - candidate) < candidate * 0.1
                   for candidate in processor.multi_resolutions), \
               f"Processed pixels {actual_pixels} not close to any candidate"

    def test_process_image_resize(self):
        """测试 resize 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (512, 768)  # (height, width)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="resize"
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (800x600 RGB)
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, target_size, None)

        # 验证输出尺寸精确匹配目标尺寸
        assert processed.shape == (target_size[0], target_size[1], 3), \
               f"Expected shape {(target_size[0], target_size[1], 3)}, got {processed.shape}"

    def test_process_image_center_crop(self):
        """测试 center_crop 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (512, 512)  # (height, width)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="center_crop"
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (800x600 RGB)
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, target_size, None)

        # 验证输出尺寸
        assert processed.shape == (target_size[0], target_size[1], 3), \
               f"Expected shape {(target_size[0], target_size[1], 3)}, got {processed.shape}"

    def test_process_image_center_padding(self):
        """测试 center_padding 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (1024, 1024)  # (height, width)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="center_padding"
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (600x800 RGB)
        test_image = np.random.randint(100, 200, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, target_size, None)

        # 验证输出尺寸
        assert processed.shape == (target_size[0], target_size[1], 3), \
               f"Expected shape {(target_size[0], target_size[1], 3)}, got {processed.shape}"

        # 验证填充区域（应该是黑色，即0）
        # 由于图像被缩放并居中，四周应该有黑色填充
        # 检查四个角落是否有填充（黑色像素）
        assert np.all(processed[0, 0, :] == 0), "Top-left corner should be padded (black)"
        assert np.all(processed[0, -1, :] == 0), "Top-right corner should be padded (black)"

    def test_process_image_right_padding(self):
        """测试 right_padding 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (1024, 1024)  # (height, width)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="right_padding"
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (600x800 RGB)
        test_image = np.random.randint(100, 200, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, target_size, None)

        # 验证输出尺寸
        assert processed.shape == (target_size[0], target_size[1], 3), \
               f"Expected shape {(target_size[0], target_size[1], 3)}, got {processed.shape}"

        # 验证左侧开始有内容（非黑色），右侧有填充（黑色）
        # 左上角应该有内容
        assert not np.all(processed[300, 0, :] == 0), "Left side should have content"
        # 右侧应该有填充
        assert np.all(processed[0, -1, :] == 0), "Right side should be padded (black)"

    def test_process_image_fixed_pixels(self):
        """测试 fixed_pixels 模式的图像处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_pixels = 512 * 512
        config = ImageProcessorInitArgs(
            target_pixels=target_pixels,
            process_type="fixed_pixels"
        )
        processor = ImageProcessor(config)

        # 创建测试图像 (800x600 RGB)
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        processed = processor._process_image(test_image, None, target_pixels)

        # 验证输出像素数接近目标像素数
        actual_pixels = processed.shape[0] * processed.shape[1]
        # best_area_near 可能会调整目标像素数，所以允许一定误差
        assert abs(actual_pixels - target_pixels) < target_pixels * 0.3, \
               f"Processed pixels {actual_pixels} too far from target {target_pixels}"

        # 验证尺寸能被16整除（因为 step=16）
        assert processed.shape[0] % 16 == 0, "Height should be divisible by 16"
        assert processed.shape[1] % 16 == 0, "Width should be divisible by 16"

    def test_process_image_grayscale(self):
        """测试灰度图像的处理"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (512, 512)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="resize"
        )
        processor = ImageProcessor(config)

        # 创建灰度测试图像 (600x800)
        test_image = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        processed = processor._process_image(test_image, target_size, None)

        # 验证输出尺寸
        assert processed.shape[:2] == (target_size[0], target_size[1]), \
               f"Expected shape {target_size}, got {processed.shape[:2]}"

    def test_resize_controls_mask_to_image_disabled(self):
        """测试禁用 resize_controls_mask_to_image 时的行为"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (512, 512)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="resize",
            resize_controls_mask_to_image=False  # 明确禁用
        )
        processor = ImageProcessor(config)

        # 创建不同尺寸的测试数据
        test_data = {
            'image': np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),
            'mask': np.random.randint(0, 255, (300, 400), dtype=np.uint8),
            'control': np.random.randint(0, 255, (750, 1000, 3), dtype=np.uint8),
        }

        processed = processor.preprocess(test_data)

        # 验证所有输出都是目标尺寸（各自独立处理）
        assert processed['image'].shape == (3, target_size[0], target_size[1]), \
               f"Image shape mismatch: {processed['image'].shape}"
        assert processed['mask'].shape == (target_size[0], target_size[1]), \
               f"Mask shape mismatch: {processed['mask'].shape}"
        assert processed['control'].shape == (3, target_size[0], target_size[1]), \
               f"Control shape mismatch: {processed['control'].shape}"

    def test_resize_controls_mask_with_same_size(self):
        """测试当 control 和 mask 已经与 image 同尺寸时的行为"""
        from qflux.data.preprocess import ImageProcessor
        from qflux.data.config import ImageProcessorInitArgs

        target_size = (512, 512)
        config = ImageProcessorInitArgs(
            target_size=target_size,
            process_type="resize",
            resize_controls_mask_to_image=True
        )
        processor = ImageProcessor(config)

        # 创建相同尺寸的测试数据
        test_data = {
            'image': np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),
            'mask': np.random.randint(0, 255, (600, 800), dtype=np.uint8),
            'control': np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),
        }

        # 应该不会报错，正常处理
        processed = processor.preprocess(test_data)

        assert processed['image'].shape == (3, target_size[0], target_size[1])
        assert processed['mask'].shape == (target_size[0], target_size[1])
        assert processed['control'].shape == (3, target_size[0], target_size[1])
