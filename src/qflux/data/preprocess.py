import logging
import math

import cv2
import numpy as np
import PIL
import torch

from qflux.data.config import ImageProcessorInitArgs


def _count_pairs_and_examples(area, min_side=256, max_side=2048, step=16, max_examples=12):
    """
    对指定 area，统计满足：
      H=step*a, W=step*b, a,b ∈ [min_side/step, max_side/step] 且 a*b = area/(step*step)
    的 (H,W) 有序对数量，并返回若干示例。
    """
    if area % (step * step) != 0:
        return 0, []
    N = area // (step * step)
    amin, amax = min_side // step, max_side // step

    count = 0
    examples = []
    for a in range(amin, amax + 1):
        if N % a != 0:
            continue
        b = N // a
        if amin <= b <= amax:
            H, W = step * a, step * b
            count += 1
            # 收集少量示例，避免返回过大
            if len(examples) < max_examples:
                examples.append((H, W))
    return count, examples


def best_area_near(A, tol=0.20, min_side=256, max_side=2048, step=16, max_examples=12):
    """
    输入面积 A，返回：
      - best_area: 在 [A*(1-tol), A*(1+tol)] 内、可分解 (H,W)（满足16整除与边界）最多的面积
      - count: 该 best_area 下的 (H,W) 组合数（有序对）
      - examples: 若干满足条件的 (H,W) 示例（不超过 max_examples）
    并列时，优先与 A 相对误差更小者；再并列取面积更小者。
    """
    if A <= 0:
        raise ValueError("A must be positive.")
    lo = math.ceil(A * (1 - tol))
    hi = math.floor(A * (1 + tol))

    # 仅枚举能被 (step*step) 整除的候选面积
    base = step * step  # 256
    start = ((lo + base - 1) // base) * base
    if start > hi:
        return None  # 窗口内没有满足 16*16 可整除的候选

    best = None  # (count, rel_err, area, examples)
    area = start
    while area <= hi:
        cnt, exs = _count_pairs_and_examples(area, min_side, max_side, step, max_examples)
        if cnt > 0:
            rel_err = abs(area - A) / A
            item = (cnt, rel_err, area, exs)
            if best is None:
                best = item
            else:
                # 先比组合数，再比与A的相对误差，再比面积大小
                if item[0] > best[0] or (
                    item[0] == best[0] and (item[1] < best[1] or (item[1] == best[1] and item[2] < best[2]))
                ):
                    best = item
        area += base

    if best is None:
        return None

    cnt, rel_err, area_star, exs = best
    return {"best_area": area_star, "count": cnt, "relative_error": rel_err, "examples": exs}


def best_hw_given_area(
    A: int,
    w: int,
    h: int,
    step: int = 16,
    min_side: int | None = None,
    max_side: int | None = None,
):
    """
    在固定面积 A 下，枚举所有 (new_w, new_h)，要求：
      1) new_w % step == 0, new_h % step == 0
      2) new_w * new_h == A
      3) （若提供）min_side <= new_w,new_h <= max_side
    从中选使 new_w/new_h 最接近 w/h 的解（比例距离用对数距离以保证对称性）。
    并列时用 (|new_w - w| + |new_h - h|) 打破；再并列选较小的 max(new_w,new_h)。

    返回: (new_w, new_h)；若无解则返回 None。
    """
    base = step * step
    if A % base != 0:
        # 面积必须能被 (step*step) 整除，否则无法分成 step 的倍数
        return None

    target_ratio = w / h
    N = A // base  # 令 new_w = step*b, new_h = step*a，则 a*b = N

    # a,b 为正整数因子对
    amin = 1 if min_side is None else math.ceil(min_side / step)
    bmin = amin
    amax = float("inf") if max_side is None else math.floor(max_side / step)
    bmax = amax

    best = None  # (ratio_dist, l1_dist_to_wh, max_side_len, new_w, new_h)

    # 为了不漏解，枚举 a（对应 new_h），由 N%a==0 推出 b（对应 new_w）
    # 注意：这是有序对 (new_w,new_h)，横纵比不同算不同解
    a_low = max(1, amin)
    a_high = min(N, amax) if amax != float("inf") else N
    for a in range(a_low, int(a_high) + 1):
        if N % a != 0:
            continue
        b = N // a
        if b < bmin or (bmax != float("inf") and b > bmax):
            continue

        new_h = step * a
        new_w = step * b
        if (min_side is not None and (new_w < min_side or new_h < min_side)) or (
            max_side is not None and (new_w > max_side or new_h > max_side)
        ):
            continue

        # 比例距离（对数对称度量，避免偏向 >1 或 <1）
        ratio = new_w / new_h
        ratio_dist = abs(math.log(ratio / target_ratio))

        # 次级打分：距离原尺寸的 L1
        l1_dist = abs(new_w - w) + abs(new_h - h)

        # 再次级：更小的最长边
        max_len = max(new_w, new_h)

        score = (ratio_dist, l1_dist, max_len, new_w, new_h)
        if best is None or score < best:
            best = score

    if best is None:
        return None
    return best[3], best[4]


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
                "area": cv2.INTER_AREA,
            }
            self.interpolation = interpolation_map.get(self.resize_mode.lower(), cv2.INTER_LINEAR)
        else:
            self.interpolation = self.resize_mode

        self.target_size = self.processor_config.target_size
        self.target_pixels = self.processor_config.target_pixels
        self.controls_pixels = self.processor_config.controls_pixels
        self.controls_size = self.processor_config.controls_size
        # Multi-resolution support
        self.multi_resolutions = self.processor_config.multi_resolutions
        self.max_aspect_ratio = self.processor_config.max_aspect_ratio

        # Parse multi_resolutions format
        self._parse_multi_resolution_config()
        # Resize controls and mask to image size before processing
        self.resize_controls_mask_to_image = self.processor_config.resize_controls_mask_to_image

        # 如果target_pixels 和 如果target_size 都为空的话，则表示生成的图片尺寸和第一个 control 图片的尺寸相同
        if self.target_size is None and self.target_pixels is None and self.multi_resolutions is None:
            if self.controls_size is not None:
                self.target_size = self.controls_size[0]
            elif self.controls_pixels is not None:
                self.target_pixels = self.controls_pixels[0]
            else:
                print("target_size and target_pixels and controls_size and controls_pixels are all None")

        # 如果controls_size 为空, 则使用 target size 或者 target_pixels
        if self.controls_pixels is None and self.controls_size is None:
            if self.target_size is not None:
                self.controls_size = [self.target_size]
            elif self.target_pixels is not None:
                self.controls_pixels = [self.target_pixels]
            elif self.multi_resolutions is None:
                # Only raise error if multi_resolutions is also not configured
                print("target_size and target_pixels and controls_size and controls_pixels are all None")

        if self.controls_size is not None:
            if isinstance(self.controls_size, list) and isinstance(self.controls_size[0], (int, float)):
                self.controls_size = [self.controls_size]
        if self.controls_pixels is not None:
            if isinstance(self.controls_pixels, int):
                self.controls_pixels = [self.controls_pixels]

        # make it devisible by 16 for shapes and pixels
        if self.target_size is not None:
            self.target_size = self.make_divisible(self.target_size)
        if self.controls_size is not None:
            self.controls_size = [self.make_divisible(size) for size in self.controls_size]
        if self.target_pixels is not None:
            self.target_pixels = best_area_near(self.target_pixels)["best_area"]
            logging.info(f"target_pixels after best_hw_given_area {self.target_pixels}")
            # self.target_pixels = 32*32*(self.target_pixels//(32*32))
        if self.controls_pixels is not None:
            self.controls_pixels = [best_area_near(pixel)["best_area"] for pixel in self.controls_pixels]
            # self.controls_pixels = [32*32*(size//(32*32)) for size in self.controls_pixels]
            logging.info(f"controls_pixels after best_hw_given_area {self.controls_pixels}")

        logging.info(
            f"ImageProcessor initialized with target_size: {self.target_size}"
            f"controls_size: {self.controls_size}"
            f"target_pixels: {self.target_pixels}"
            f"controls_pixels: {self.controls_pixels}"
        )

    def make_divisible(self, target_size):
        h, w = target_size
        h = h // self.devisible_by * self.devisible_by
        w = w // self.devisible_by * self.devisible_by
        return h, w

    def _parse_multi_resolution_config(self):
        """Parse multi_resolutions config into separate lists for each image type

        After parsing, sets:
        - self.multi_res_target: List of pixel candidates for target image
        - self.multi_res_controls: List of lists, one per control image
        - self.multi_res_mode: 'simple' or 'advanced'
        """
        if self.multi_resolutions is None:
            self.multi_res_mode = None
            self.multi_res_target = None
            self.multi_res_controls = None
            return

        # Format 1: Simple list - applies to all images
        if isinstance(self.multi_resolutions, list):
            self.multi_res_mode = "simple"
            self.multi_res_target = self.multi_resolutions
            self.multi_res_controls = [self.multi_resolutions]  # Same for all controls
            logging.info(f"Multi-resolution mode: simple (shared candidates: {self.multi_resolutions})")

        # Format 2: Advanced dict - separate configs per image type
        elif isinstance(self.multi_resolutions, dict):
            self.multi_res_mode = "advanced"

            # Get target candidates
            self.multi_res_target = self.multi_resolutions.get(
                "target", self.multi_resolutions.get("controls", [[]])[0]
            )

            # Get controls candidates
            if "controls" in self.multi_resolutions:
                self.multi_res_controls = self.multi_resolutions["controls"]
            else:
                # Fallback: use target candidates for all controls
                self.multi_res_controls = [self.multi_res_target]

            logging.info(
                f"Multi-resolution mode: advanced\n"
                f"  Target candidates: {self.multi_res_target}\n"
                f"  Controls candidates: {self.multi_res_controls}"
            )
        else:
            raise ValueError(f"multi_resolutions must be list or dict, got {type(self.multi_resolutions)}")

    def _select_pixels_candidate(self, orig_w: int, orig_h: int, candidates: list = None) -> int:
        """Select best resolution from candidates

        Args:
            orig_w: Original image width
            orig_h: Original image height
            candidates: List of pixel candidates. If None, uses self.multi_resolutions

        Returns:
            Best pixel count from candidates
        """
        if candidates is None:
            candidates = self.multi_resolutions  # type: ignore

        if candidates is None or len(candidates) == 0:
            raise ValueError("No resolution candidates provided")

        orig_area = orig_w * orig_h
        orig_ratio = orig_w / orig_h

        # Check aspect ratio limit
        if self.max_aspect_ratio is not None:
            if orig_ratio > self.max_aspect_ratio or orig_ratio < 1.0 / self.max_aspect_ratio:
                logging.warning(
                    f"Image aspect ratio {orig_ratio:.2f} exceeds max_aspect_ratio {self.max_aspect_ratio:.2f}"
                )
                raise ValueError(
                    f"Image aspect ratio {orig_ratio:.2f} exceeds max_aspect_ratio {self.max_aspect_ratio:.2f}"
                )

        best_candidate = None
        relative_error = [abs(candidate_pixels - orig_area) / orig_area for candidate_pixels in candidates]
        best_candidate = candidates[np.argmin(relative_error)]
        return best_candidate

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
            if any.mode == "L":  # 灰度图像
                # 对于灰度图像，转换为3通道灰度图
                gray_array = np.array(any)
                return gray_array
            else:  # RGB或其他模式
                return np.array(any.convert("RGB"))
        else:
            raise ValueError(f"Unsupported type: {type(any)}")

    def get_multi_res_cand(self, multi_res_target=None, multi_res_controls=None, input_date: str = "target"):
        if input_date == "target":
            if multi_res_target is not None:
                return multi_res_target
            elif self.multi_res_target is not None:
                return self.multi_res_target
            else:
                return None
        elif input_date.startswith("control"):
            control_idx = int(input_date.split("_")[1])
            if multi_res_controls is None:
                if self.multi_res_controls is not None:
                    multi_res_controls = self.multi_res_controls
                else:
                    return None
            if len(multi_res_controls) == 0:
                return None
            return multi_res_controls[control_idx % len(multi_res_controls)]

    def preprocess(
        self,
        data,
        target_size=None,
        controls_size=None,
        target_pixels=None,
        controls_pixels=None,
        multi_res_target=None,
        multi_res_controls=None,
    ):
        """处理图像、掩码和控制图像，支持多种处理模式：resize、center_crop和*_padding"""
        # 将图像转换为numpy数组

        target_size = target_size if target_size is not None else self.target_size
        controls_size = controls_size if controls_size is not None else self.controls_size
        target_pixels = target_pixels if target_pixels is not None else self.target_pixels
        controls_pixels = controls_pixels if controls_pixels is not None else self.controls_pixels

        # If resize_controls_mask_to_image is enabled, resize control and mask to match image size first
        if self.resize_controls_mask_to_image and "image" in data:
            image = self.any2numpy(data["image"])
            image_h, image_w = image.shape[:2]

            # Resize mask to image size
            if "mask" in data:
                mask = self.any2numpy(data["mask"])
                if mask.shape[:2] != (image_h, image_w):
                    mask = cv2.resize(mask, (image_w, image_h), interpolation=self.interpolation)
                data["mask"] = mask

            # Resize control to image size
            if "control" in data:
                control = self.any2numpy(data["control"])
                if control.shape[:2] != (image_h, image_w):
                    control = cv2.resize(control, (image_w, image_h), interpolation=self.interpolation)
                data["control"] = control

        if "image" in data:
            image = self.any2numpy(data["image"])
            # For target image, use multi_res_target candidates
            multi_res_cand = self.get_multi_res_cand(multi_res_target=multi_res_target, input_date="target")

            processed_image = self._process_image(
                image, target_size, target_pixels, multi_res_candidates=multi_res_cand
            )
            data["image"] = self._to_tensor(processed_image)

        # 处理mask（如果存在）- mask follows target image resolution
        if "mask" in data:
            multi_res_cand = self.get_multi_res_cand(multi_res_target=multi_res_target, input_date="target")
            mask = self._process_image(data["mask"], target_size, target_pixels, multi_res_candidates=multi_res_cand)
            mask = mask / 255.0
            mask = torch.from_numpy(mask).to(torch.float32)
            data["mask"] = mask

        # 处理控制图像（如果存在）
        if "control" in data:
            control = self.any2numpy(data["control"])
            if controls_size is not None:
                controls_size_0 = controls_size[0]
            else:
                controls_size_0 = None
            if controls_pixels is not None:
                controls_pixels_0 = controls_pixels[0]
            else:
                controls_pixels_0 = None
            # For first control (controls[0]), use multi_res_controls[0]
            multi_res_cand = self.get_multi_res_cand(multi_res_controls=multi_res_controls, input_date="control_0")
            processed_control = self._process_image(
                control, controls_size_0, controls_pixels_0, multi_res_candidates=multi_res_cand
            )
            data["control"] = self._to_tensor(processed_control)

        if "controls" in data:  # extra controls
            controls = [self.any2numpy(x) for x in data["controls"]]
            new_controls = []
            for i in range(len(controls)):
                if controls_size is not None:
                    if i + 1 >= len(controls_size):
                        print("controls_size in preprocess", controls_size, "i", i)
                    controls_size_i = controls_size[i + 1]  # index starting from 1,
                else:
                    controls_size_i = None
                if controls_pixels is not None:
                    controls_pixels_i = controls_pixels[i + 1]
                else:
                    controls_pixels_i = None
                # For additional controls, use multi_res_controls[i+1] if available
                multi_res_cand = self.get_multi_res_cand(
                    multi_res_controls=multi_res_controls, input_date=f"control_{i + 1}"
                )
                processed_control = self._process_image(
                    controls[i],
                    controls_size_i,
                    controls_pixels_i,
                    multi_res_candidates=multi_res_cand,
                )
                new_controls.append(processed_control)
            data["controls"] = [self._to_tensor(control) for control in new_controls]
        return data

    def _to_tensor(self, image):
        """将图像转换为范围[0, 1]的张量"""
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def _process_image(self, image, target_size, target_pixels, multi_res_candidates=None):
        """根据处理类型处理图像

        Args:
            image: Image to process
            target_size: Target size (H, W)
            target_pixels: Target pixel count
            multi_res_candidates: List of pixel candidates for multi-resolution mode
        """

        # Multi-resolution mode: select best candidate
        if multi_res_candidates is not None:
            from qflux.utils.images import calculate_best_resolution

            h, w = image.shape[:2]
            best_pixels = self._select_pixels_candidate(w, h, candidates=multi_res_candidates)
            new_w, new_h = calculate_best_resolution(w, h, best_pixels)
            return cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

        if self.processor_config.process_type == "resize":
            # 直接调整大小到目标尺寸
            target_h, target_w = target_size
            return cv2.resize(image, (target_w, target_h), interpolation=self.interpolation)

        elif self.processor_config.process_type == "center_crop":
            return self._center_crop(image, target_size)

        elif self.processor_config.process_type.endswith("_padding"):
            return self._padding(image, target_size)
        elif self.processor_config.process_type == "fixed_pixels":
            return self._fixed_pixels(image, target_pixels)

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
        crop_image = image[new_bb[1] : new_bb[3], new_bb[0] : new_bb[2]]
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
        if self.processor_config.process_type == "center_padding":
            # 居中填充
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
        elif self.processor_config.process_type == "right_padding":
            # 右侧填充
            start_x = 0
            start_y = (target_h - new_h) // 2
        else:  # 默认居中填充
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2

        # 将调整大小后的图像放置在画布上
        result[start_y : start_y + new_h, start_x : start_x + new_w] = resized
        return result

    def _fixed_pixels(self, image, target_pixels):
        """根据固定像素调整图像大小"""
        h, w = image.shape[:2]
        target_pixels = int(target_pixels / (32 * 32)) * (32 * 32)
        print("original w, h", w, h)
        new_w, new_h = best_hw_given_area(target_pixels, w, h)
        print("new shape", new_w, new_h, "target_pixels", target_pixels)
        assert new_w * new_h == target_pixels, f"new_w * new_h {new_w * new_h} != target_pixels {target_pixels}"
        return cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)
