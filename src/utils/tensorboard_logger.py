"""
TensorBoard Logger Class for logging scalars, text, images, and image pairs.
"""

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms


class TensorBoardLogger:
    """
    TensorBoard logger for logging various data types including scalars, text, images, and image pairs.
    """

    def __init__(self, log_dir: str, comment: str = ""):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir (str): Directory to save TensorBoard logs
            comment (str): Optional comment to add to log directory name
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        self.step_counter = 0

    def log_scalar(self, tag: str, value: Union[float, int, torch.Tensor], step: Optional[int] = None):
        """
        Log scalar value to TensorBoard.

        Args:
            tag (str): Name of the scalar
            value (Union[float, int, torch.Tensor]): Scalar value to log
            step (int, optional): Global step. If None, uses internal counter.
        """
        if step is None:
            step = self.step_counter

        if isinstance(value, torch.Tensor):
            value = value.item()

        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: Optional[int] = None):
        """
        Log multiple scalars with a common main tag.

        Args:
            main_tag (str): Main tag for grouping scalars
            tag_scalar_dict (dict): Dictionary of {tag: scalar_value}
            step (int, optional): Global step. If None, uses internal counter.
        """
        if step is None:
            step = self.step_counter

        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text to TensorBoard.

        Args:
            tag (str): Name of the text entry
            text (str): Text content to log
            step (int, optional): Global step. If None, uses internal counter.
        """
        if step is None:
            step = self.step_counter

        self.writer.add_text(tag, text, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray, Image.Image],
                  step: Optional[int] = None, dataformats: str = 'CHW'):
        """
        Log image to TensorBoard.

        Args:
            tag (str): Name of the image
            image (Union[torch.Tensor, np.ndarray, Image.Image]): Image to log
            step (int, optional): Global step. If None, uses internal counter.
            dataformats (str): Format of image data ('CHW', 'HWC', etc.)
        """
        if step is None:
            step = self.step_counter

        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
            dataformats = 'CHW'
        # Convert numpy array to tensor
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = torch.from_numpy(image)
            if len(image.shape) == 3 and dataformats == 'HWC':
                image = image.permute(2, 0, 1)  # HWC -> CHW
                dataformats = 'CHW'

        # Ensure tensor is in the right format
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).to(torch.uint8)
            elif image.dtype != torch.uint8:
                image = image.to(torch.uint8)

        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(self, tag: str, images: Union[torch.Tensor, List],
                   step: Optional[int] = None, dataformats: str = 'NCHW'):
        """
        Log multiple images to TensorBoard.

        Args:
            tag (str): Name of the image set
            images (Union[torch.Tensor, List]): Batch of images to log
            step (int, optional): Global step. If None, uses internal counter.
            dataformats (str): Format of image data ('NCHW', 'NHWC', etc.)
        """
        if step is None:
            step = self.step_counter

        # Convert list of images to tensor
        if isinstance(images, list):
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = transforms.ToTensor()(img)
                elif isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                    if len(img.shape) == 3:
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                processed_images.append(img)
            images = torch.stack(processed_images)
            dataformats = 'NCHW'

        # Ensure tensor is in the right format
        if isinstance(images, torch.Tensor):
            if images.max() <= 1.0:
                images = (images * 255).to(torch.uint8)
            elif images.dtype != torch.uint8:
                images = images.to(torch.uint8)

        self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_image_pair(self, tag: str, image_pair: Tuple[Union[torch.Tensor, np.ndarray, Image.Image],
                                                          Union[torch.Tensor, np.ndarray, Image.Image]],
                       step: Optional[int] = None, labels: Tuple[str, str] = ("Original", "Modified")):
        """
        Log a pair of images (e.g., before/after, input/output) to TensorBoard.

        Args:
            tag (str): Name of the image pair
            image_pair (Tuple): Tuple of two images (original, modified)
            step (int, optional): Global step. If None, uses internal counter.
            labels (Tuple[str, str]): Labels for the two images
        """
        if step is None:
            step = self.step_counter

        image1, image2 = image_pair

        # Log individual images with labels
        self.log_image(f"{tag}/{labels[0]}", image1, step)
        self.log_image(f"{tag}/{labels[1]}", image2, step)

        # Create side-by-side comparison
        try:
            # Convert images to the same format
            def prepare_image(img):
                if isinstance(img, Image.Image):
                    return transforms.ToTensor()(img)
                elif isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                    if len(img.shape) == 3 and img.shape[2] == 3:  # HWC
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                return img.float()

            img1_tensor = prepare_image(image1)
            img2_tensor = prepare_image(image2)

            # Resize to same dimensions if needed
            if img1_tensor.shape != img2_tensor.shape:
                h, w = max(img1_tensor.shape[1], img2_tensor.shape[1]), max(img1_tensor.shape[2], img2_tensor.shape[2])
                img1_tensor = torch.nn.functional.interpolate(img1_tensor.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
                img2_tensor = torch.nn.functional.interpolate(img2_tensor.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)

            # Concatenate horizontally
            comparison = torch.cat([img1_tensor, img2_tensor], dim=2)  # Concatenate along width
            self.log_image(f"{tag}/Comparison", comparison, step)

        except Exception as e:
            print(f"Warning: Failed to create image comparison for {tag}: {e}")

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray],
                      step: Optional[int] = None, bins: str = 'tensorflow'):
        """
        Log histogram to TensorBoard.

        Args:
            tag (str): Name of the histogram
            values (Union[torch.Tensor, np.ndarray]): Values to create histogram from
            step (int, optional): Global step. If None, uses internal counter.
            bins (str): Binning method for histogram
        """
        if step is None:
            step = self.step_counter

        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_figure(self, tag: str, figure, step: Optional[int] = None, close: bool = True):
        """
        Log matplotlib figure to TensorBoard.

        Args:
            tag (str): Name of the figure
            figure: Matplotlib figure object
            step (int, optional): Global step. If None, uses internal counter.
            close (bool): Whether to close the figure after logging
        """
        if step is None:
            step = self.step_counter

        self.writer.add_figure(tag, figure, step, close=close)

    def increment_step(self):
        """Increment the internal step counter."""
        self.step_counter += 1

    def set_step(self, step: int):
        """Set the internal step counter to a specific value."""
        self.step_counter = step

    def flush(self):
        """Flush all pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer and flush all data."""
        self.writer.flush()
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
