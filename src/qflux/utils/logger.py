import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import PIL
import swanlab
import torch
import torchvision
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from tensorboardX import SummaryWriter

from qflux.data.config import Config


def load_logger(name, log_level="INFO"):
    logger = get_logger(name, log_level=log_level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    return logger


class BaseLogger(ABC):
    """所有日志工具的基类，定义统一接口"""

    @classmethod
    def create(
        cls,
        config,
        versioned_dir: str,
        experiment_name: str,
    ):
        """
        工厂方法：根据配置创建合适的logger实例

        Args:
            config: 配置对象，包含logging配置
            accelerator: 可选的Accelerator实例

        Returns:
            BaseLogger的子类实例
        """
        if not hasattr(config, "logging") or not hasattr(config.logging, "report_to"):
            raise ValueError("Config must have logging.report_to attribute")
        from datetime import datetime

        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")  # 或者 "%y%m%d_%H%M"
        experiment_name = f"{experiment_name}_{timestamp}"

        report_to = config.logging.report_to
        hyperparams = {
            "learning_rate": float(config.optimizer.init_args.get("lr", 0.0001)),
            "batch_size": int(config.data.batch_size),
            "max_train_steps": int(config.train.max_train_steps),
            "num_epochs": int(config.train.num_epochs),
            "gradient_accumulation_steps": int(config.train.gradient_accumulation_steps),
            "mixed_precision": str(config.train.mixed_precision),
            "lora_r": int(config.model.lora.r),
            "lora_alpha": int(config.model.lora.lora_alpha),
            "model_name": str(config.model.pretrained_model_name_or_path),
            "optimizer": str(config.optimizer.class_path),
            "loss": str(config.loss.class_path),
        }

        # 创建TensorBoard logger
        project_name = config.logging.tracker_project_name
        os.makedirs(versioned_dir, exist_ok=True)
        if report_to == "tensorboard":
            # from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=versioned_dir)
            return TensorBoardLogger(writer)

        # 创建Weights & Biases logger
        elif report_to == "wandb":
            # 初始化wandb
            run = wandb.init(
                entity=os.environ.get("WANDB_ENTITY"),
                project=project_name,
                dir=versioned_dir,
                name=experiment_name,
                config=hyperparams,
                tags=config.logging.tags,
                notes=config.logging.notes,
                reinit=True,  # 允许在同一进程中多次初始化
                settings=wandb.Settings(
                    x_disable_stats=False,  # 禁用 system metrics
                    x_disable_meta=False,  # 禁用 metadata
                    start_method="thread",  # 使用线程启动
                ),
            )
            return WandbLogger(run)

        # 创建SwanLab logger
        elif report_to == "swanlab":
            workspace = os.environ.get("SWANLAB_WORKSPACE")
            if not os.path.exists(versioned_dir):
                logging.warning(
                    f"versioned_dir does not exist: {versioned_dir}. "
                    "This may indicate an earlier failure. Creating it now."
                )
                os.makedirs(versioned_dir)
            # 使用SwanLab官方推荐的初始化方式
            swan = swanlab.init(
                project=project_name,
                workspace=workspace,
                experiment_name=experiment_name,
                description=config.logging.notes,
                tags=config.logging.tags,
                logdir=versioned_dir,
                config=hyperparams,
            )
            return SwanLabLogger(swan)

        # 不支持的日志工具
        else:
            raise ValueError(f"Unsupported logger type: {report_to}")

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """记录标量值"""
        pass

    @abstractmethod
    def log_scalars(self, scalars_dict: dict[str, float], step: int) -> None:
        """记录多个标量值"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """记录训练指标"""
        pass

    @abstractmethod
    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
    ) -> None:
        """记录单张图像"""
        pass

    @abstractmethod
    def log_images(
        self,
        name: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
        nrow: int = 4,
    ) -> None:
        """记录多张图像，自动拼接为网格"""
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int) -> None:
        """记录文本"""
        pass

    @abstractmethod
    def log_table(self, name: str, rows: list[dict[str, Any]], columns: list[str], step: int) -> None:
        """记录表格数据"""
        pass

    @abstractmethod
    def flush(self) -> None:
        """刷新日志，确保写入"""
        pass


class TensorBoardLogger(BaseLogger):
    """TensorBoard日志工具封装"""

    def __init__(self, writer):
        self.writer = writer

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, scalars_dict: dict[str, float], step: int) -> None:
        for name, value in scalars_dict.items():
            self.writer.add_scalar(name, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
    ) -> None:
        if isinstance(image, np.ndarray):
            # 如果是HWC格式，转换为CHW
            if image.ndim == 3 and image.shape[2] in (1, 3, 4):
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        self.writer.add_image(name, image, step, dataformats="CHW")

    def log_images(
        self,
        name: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
        nrow: int = 4,
        commit=False,
    ) -> None:
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # 确保是[B,C,H,W]格式
        if images.dim() == 3:
            images = images.unsqueeze(0)

        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=True, scale_each=True)
        self.writer.add_image(name, grid, step, dataformats="CHW")
        self.flush()

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(name, text, step)

    def log_table(self, name, rows, columns, step: int) -> None:
        # TensorBoard没有原生表格支持
        return

    def flush(self) -> None:
        self.writer.flush()


class WandbLogger(BaseLogger):
    """Weights & Biases日志工具封装"""

    def __init__(self, run):
        self.run = run

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.run.log({name: value}, step=step)

    def log_scalars(self, scalars_dict: dict[str, float], step: int) -> None:
        self.run.log(scalars_dict, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
    ) -> None:
        import wandb

        # input range is [0,1]

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # 确保是HWC格式
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1 and image.min() >= 0:
            image = (image * 255).astype(np.uint8)
        image = image.astype(np.uint8)
        self.run.log({name: wandb.Image(image, caption=caption)}, step=step, commit=True)

    def log_images(
        self,
        name: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
        nrow: int = 4,
        commit=False,
    ) -> None:
        import wandb

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()

            # 确保是[B,C,H,W]格式
            if images.dim() == 3:
                images = images.unsqueeze(0)

            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
            npimg = grid.permute(1, 2, 0).numpy()  # CHW -> HWC
        else:
            # 假设numpy数组已经是正确格式
            npimg = images
        # convert to PIL image
        if npimg.min() >= 0 and npimg.max() <= 1:
            npimg = npimg * 255
        npimg = npimg.astype(np.uint8)
        npimg = PIL.Image.fromarray(npimg)

        self.run.log({name: wandb.Image(npimg, caption=caption)}, step=step, commit=commit)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.run.log({name: text}, step=step)

    def log_table(self, name: str, rows: list[dict[str, Any]], columns: list[str], step: int) -> None:
        if not rows:
            return
        table = wandb.Table(columns=columns, data=rows)
        self.run.log({name: table})

    def flush(self) -> None:
        # wandb会自动刷新
        pass


class SwanLabLogger(BaseLogger):
    """SwanLab日志工具封装"""

    def __init__(self, swan: swanlab):
        self.swan = swan

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.swan.log({name: value}, step=step)

    def log_scalars(self, scalars_dict: dict[str, float], step: int) -> None:
        self.swan.log(scalars_dict, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.swan.log(metrics, step=step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
    ) -> None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # 确保是HWC格式
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        self.swan.log({name: swanlab.Image(image, caption=caption)}, step=step)

    def log_images(
        self,
        name: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
        nrow: int = 4,
        commit=False,
    ) -> None:
        import swanlab

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()

            # 确保是[B,C,H,W]格式
            if images.dim() == 3:
                images = images.unsqueeze(0)

            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
            npimg = grid.permute(1, 2, 0).numpy()  # CHW -> HWC
        else:
            # 假设numpy数组已经是正确格式
            npimg = images
        # convert to PIL image
        if npimg.min() >= 0 and npimg.max() <= 1:
            npimg = npimg * 255
        npimg = npimg.astype(np.uint8)
        npimg = PIL.Image.fromarray(npimg)

        self.swan.log({name: swanlab.Image(npimg, caption=caption)}, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        pass
        # self.swan.log({name: swanlab.Text(text)}, step=step)

    def log_table(self, name: str, rows: list[dict[str, Any]], columns: list[str], step: int) -> None:
        # SwanLab支持直接记录表格数据
        table = swanlab.echarts.Table()
        table.add(columns, rows)
        self.swan.log({name: table})

    def flush(self) -> None:
        # SwanLab会自动刷新
        pass


class LoggerManager:
    """日志管理器，统一管理日志工具"""

    def __init__(
        self,
        accelerator: Accelerator,
        config: Config,
        versioned_dir: str,
        experiment_name: str,
    ):
        """
        初始化LoggerManager

        Args:
            accelerator: 可选的Accelerator实例，用于分布式训练
            config: 配置对象，包含logging配置
        """
        self.accelerator = accelerator
        self.config = config
        """初始化配置的日志工具"""
        # 如果在分布式环境中，只有主进程初始化logger
        if self.accelerator and not self.accelerator.is_main_process:
            logging.info(f"[{self.accelerator.process_index}] not initialized")
            return
        else:
            # 使用BaseLogger的工厂方法创建logger实例
            self.logger = BaseLogger.create(self.config, versioned_dir, experiment_name)
            logging.info(f"[{self.accelerator.process_index}] {self.logger} initialized")

    def should_do_logging(self):
        return bool(not self.accelerator or self.accelerator.is_main_process)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """记录标量值"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_scalar(name, value, step)

    def log_scalars(self, scalars_dict: dict[str, float], step: int) -> None:
        """记录多个标量值"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_scalars(scalars_dict, step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """记录训练指标"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_metrics(metrics, step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
    ) -> None:
        """记录单张图像"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_image(name, image, step, caption)

    def log_images(
        self,
        name: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        caption: str | None = None,
        nrow: int = 4,
        max_images: int = 16,
        commit=False,
    ) -> None:
        """记录多张图像，自动拼接为网格"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_images(name, images, step, caption, nrow, commit=commit)

    def log_text(self, name: str, text: str, step: int) -> None:
        """记录文本"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            self.logger.log_text(name, text, step)

    def log_table(
        self,
        name: str,
        rows: list[dict[str, Any]],
        columns: list[str],
        step: int,
        max_rows: int = 64,
    ) -> None:
        """记录表格数据"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.should_do_logging():
            rows_clip = rows[:max_rows]
            self.logger.log_table(name, rows_clip, columns, step)

    def flush(self) -> None:
        """刷新日志工具"""
        # 如果在分布式环境中，只有主进程刷新日志
        if self.should_do_logging():
            self.logger.flush()
