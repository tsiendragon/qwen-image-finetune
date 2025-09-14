import cv2
import numpy as np
import PIL
import torch
import logging
from src.data.config import ImageProcessorInitArgs


class ImageProcessor:
    def __init__(self, processor_config: ImageProcessorInitArgs):
        self.processor_config = processor_config
        self.devisible_by = 16
        self.resize_mode = self.processor_config.resize_mode
        if isinstance(self.resize_mode, str):
            interpolation_map = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "lanczos": cv2.INTER_LANCZOS4,
                "area": cv2.INTER_AREA
            }
            self.interpolation = interpolation_map.get(self.resize_mode.lower(), cv2.INTER_LINEAR)
        else:
            self.interpolation = self.resize_mode
        self.target_size = self.processor_config.target_size

        if self.processor_config.controls_size is None:
            self.controls_size = [self.target_size]
        else:
            controls_size = self.processor_config.controls_size
            if isinstance(controls_size, list) and isinstance(controls_size[0], (int, float)):
                self.controls_size = [controls_size]
            else:
                self.controls_size = controls_size

        self.target_size = self.make_divisible(self.target_size)
        self.controls_size = [self.make_divisible(size) for size in self.controls_size]
        logging.info(f"ImageProcessor initialized with target_size: {self.target_size}"
                     f"controls_size: {self.controls_size}")

    def make_divisible(self, target_size):
        h, w = target_size
        h = h // self.devisible_by * self.devisible_by
        w = w // self.devisible_by * self.devisible_by
        return h, w

    def read_image(self, image_path):
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]  # BGR to RGB
        return img

    def any2numpy(self, any):
        if isinstance(any, str):
            return self.read_image(any)
        elif isinstance(any, torch.Tensor):
            return any.numpy()
        elif isinstance(any, np.ndarray):
            return any
        elif isinstance(any, PIL.Image.Image):
            # 检查图像模式
            if any.mode == 'L':  # 灰度图像
                # 对于灰度图像，转换为3通道灰度图
                gray_array = np.array(any)
                return gray_array
            else:  # RGB或其他模式
                return np.array(any.convert('RGB'))
        else:
            raise ValueError(f"Unsupported type: {type(any)}")

    def preprocess(self, data, target_size=None, controls_size=None):
        """处理图像、掩码和控制图像，支持多种处理模式：resize、center_crop和*_padding"""
        # 将图像转换为numpy数组
        target_h, target_w = target_size if target_size is not None else self.target_size
        controls_size = controls_size if controls_size is not None else self.controls_size

        if 'image' in data:
            image = self.any2numpy(data['image'])
            processed_image = self._process_image(image, (target_h, target_w))
            data['image'] = self._to_tensor(processed_image)

        # 处理mask（如果存在）
        if 'mask' in data:
            mask = self._process_image(data['mask'], (target_h, target_w))
            mask = mask / 255.0
            mask = torch.from_numpy(mask).to(torch.float32)
            data['mask'] = mask

        # 处理控制图像（如果存在）
        if 'control' in data:
            control = self.any2numpy(data['control'])
            processed_control = self._process_image(control, controls_size[0])
            data['control'] = self._to_tensor(processed_control)

        if 'controls' in data:  # extrol
            controls = [self.any2numpy(x) for x in data['controls']]
            if len(self.controls_size) == 1:
                controls = [self._process_image(control, controls_size[0]) for control in controls]
            else:
                assert len(controls_size) == len(controls)+1, "the number of controls_size should be same of controls" # NOQA
                controls = [self._process_image(controls[i], controls_size[i+1]) for i in range(len(controls))]  # NOQA
            data['controls'] = [self._to_tensor(control) for control in controls]
        return data

    def _to_tensor(self, image):
        """将图像转换为范围[0, 1]的张量"""
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def _process_image(self, image, target_size):
        """根据处理类型处理图像"""
        target_h, target_w = target_size

        if self.processor_config.process_type == 'resize':
            # 直接调整大小到目标尺寸
            return cv2.resize(image, (target_w, target_h), interpolation=self.interpolation)

        elif self.processor_config.process_type == 'center_crop':
            return self._center_crop(image, target_size)

        elif self.processor_config.process_type.endswith('_padding'):
            return self._padding(image, target_size)

        else:
            # 默认使用居中裁剪
            return self._center_crop(image, target_size)

    def _center_crop(self, image, target_size):
        """居中裁剪图像"""
        h, w = image.shape[:2]  # 2,2
        target_h, target_w = target_size  # 0.4,0.2

        # 计算缩放比例，保持原始宽高比
        scale = min(w / target_w, h / target_h)  # min(2/0.2, 2/0.4) = 5
        new_w, new_h = int(target_w * scale), int(target_h * scale)  # 0.2*5, 0.4*5 = 1, 2
        new_bb = [int((w - new_w) // 2), int((h - new_h) // 2), int((w + new_w) // 2), int((h + new_h) // 2)]
        crop_image = image[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]]
        crop_image = cv2.resize(crop_image, (target_w, target_h), interpolation=self.interpolation)
        return crop_image

    def _padding(self, image, target_size):
        """根据填充类型调整图像大小并填充"""
        h, w = image.shape[:2]  # 2,2
        target_h, target_w = target_size  # 0.4,0.2

        # 计算缩放比例，保持原始宽高比，但不超过目标尺寸
        scale = min(target_w / w, target_h / h)  # min(0.2/2, 0.4/2) = 0.1
        new_w, new_h = int(w * scale), int(h * scale)  # 2*0.1, 2*0.1 = 0.2, 0.2

        # 调整大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

        # 创建目标尺寸的空白画布
        if len(image.shape) == 2:
            result = np.zeros((target_h, target_w), dtype=np.uint8)
        else:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 根据填充类型确定位置
        if self.processor_config.process_type == 'center_padding':
            # 居中填充
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
        elif self.processor_config.process_type == 'right_padding':
            # 右侧填充
            start_x = 0
            start_y = (target_h - new_h) // 2
        else:  # 默认居中填充
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2

        # 将调整大小后的图像放置在画布上
        result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return result
