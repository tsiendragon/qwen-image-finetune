# 多日志工具集成计划

**最近更新**: 2025-10-28
**负责人**: lilong

## 1. 概述

本文档描述了在Qwen-Image-Finetune项目中集成多种日志记录工具的计划，包括现有的TensorBoard和Weights & Biases (wandb)，以及新增的SwanLab支持。

## 2. 背景与需求

当前系统已经支持TensorBoard和Weights & Biases作为日志记录工具，但随着机器学习生态系统的发展，SwanLab作为一个新兴的、开源的实验跟踪工具，提供了现代化的设计和更好的用户体验，值得集成到我们的训练框架中。

### 2.1 现有功能

目前系统支持以下日志记录功能：

- 通过Accelerator框架集成TensorBoard和Weights & Biases
- 记录训练指标（损失值、学习率等）
- 记录图像样本（训练样本、验证结果等）
- 记录文本数据（如提示词、评估结果等）

### 2.2 新增需求

- 集成SwanLab作为第三种日志记录选项
- 保持与现有日志工具API的一致性
- 确保所有日志工具能够记录相同类型的数据（标量、图像、文本等）
- 提供简单的配置选项，允许用户选择使用哪种日志工具（每次只使用一个）
- 通过统一的Logger Wrapper接口，简化不同日志工具的使用

## 3. 技术方案

### 3.1 架构设计

我们将在现有的日志记录系统基础上进行扩展，主要修改以下几个方面：

1. **配置系统**：扩展`LoggingConfig`，增加SwanLab作为可选项
2. **日志工具封装**：扩展`logger.py`，增加对SwanLab的支持
3. **训练器集成**：更新`BaseTrainer`中的日志初始化逻辑

### 3.2 配置系统修改

在`src/qflux/data/config.py`中的`LoggingConfig`类中：

```python
class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./output"
    report_to: str = "tensorboard"  # tensorboard, wandb, swanlab, none
    tracker_project_name: str | None = None  # will get the value from trainer

    @field_validator("report_to")
    @classmethod
    def _check_report_to(cls, v: str) -> str:
        allowed = {"tensorboard", "wandb", "swanlab", "none"}
        if v not in allowed:
            raise ValueError(f"report_to must be one of {allowed}")
        return v
```

### 3.3 日志工具封装修改

为了实现统一的日志接口，我们将创建一个`LoggerWrapper`类，封装不同的日志工具，提供统一的API。这样可以简化代码，并使未来添加新的日志工具更加容易。

#### 3.3.1 Logger Wrapper设计

在`src/qflux/utils/logger.py`中，我们将添加以下代码：

```python
import logging
import abc
from typing import Any, Dict, List, Optional, Union
import numpy as np

import torch
import torchvision

class BaseLogger(abc.ABC):
    """所有日志工具的基类，定义统一接口"""

    @classmethod
    def create(cls, config, accelerator=None):
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

        report_to = config.logging.report_to

        # 创建TensorBoard logger
        if report_to == "tensorboard":
            if accelerator:
                tb = accelerator.get_tracker("tensorboard")
                if hasattr(tb, "writer"):
                    return TensorBoardLogger(tb.writer)

            # 如果没有accelerator或获取失败，尝试直接创建
            from torch.utils.tensorboard import SummaryWriter
            output_dir = getattr(config.logging, "output_dir", "./output/tensorboard")
            writer = SummaryWriter(log_dir=output_dir)
            return TensorBoardLogger(writer)

        # 创建Weights & Biases logger
        elif report_to == "wandb":
            # 初始化wandb
            import wandb
            from dotenv import load_dotenv
            import os

            # 加载.env文件中的环境变量
            load_dotenv()

            # 当使用Accelerator时，不需要手动初始化wandb
            # Accelerator会自动处理wandb的初始化
            if accelerator:
                run = accelerator.get_tracker("wandb", unwrap=True)
                if run is not None:
                    return WandbLogger(run)

            # 如果没有accelerator或获取失败，尝试直接创建
            # 获取项目名称和配置
            project_name = getattr(config.logging, "tracker_project_name", None) or os.environ.get("WANDB_PROJECT") or "qwen-image-finetune"

            # 获取配置字典
            config_dict = {}
            if hasattr(config, "get_dict"):
                config_dict = config.get_dict()
            elif hasattr(config, "__dict__"):
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}

            # 初始化wandb
            run = wandb.init(
                project=project_name,
                config=config_dict,
                dir=getattr(config.logging, "output_dir", None),
                reinit=True,  # 允许在同一进程中多次初始化
                settings=wandb.Settings(
                    _disable_stats=True,  # 禁用 system metrics
                    _disable_meta=True,   # 禁用 metadata
                )
            )
            return WandbLogger(run)

        # 创建SwanLab logger
        elif report_to == "swanlab":
            # 初始化swanlab
            import swanlab
            from dotenv import load_dotenv
            import os

            # 加载.env文件中的环境变量
            load_dotenv()

            # 当使用Accelerator时，尝试获取已初始化的SwanLab实例
            if accelerator:
                swan = accelerator.get_tracker("swanlab", unwrap=True)
                if swan is not None:
                    return SwanLabLogger(swan)

            # 如果没有accelerator或获取失败，尝试直接创建
            # 获取配置信息
            project_name = getattr(config.logging, "tracker_project_name", None) or os.environ.get("SWANLAB_PROJECT") or "qwen-image-finetune"
            workspace = os.environ.get("SWANLAB_WORKSPACE")
            output_dir = getattr(config.logging, "output_dir", "./output/swanlab")

            # 获取配置字典，用于记录超参数
            config_dict = {}
            if hasattr(config, "get_dict"):
                config_dict = config.get_dict()
            elif hasattr(config, "__dict__"):
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}

            # 使用SwanLab官方推荐的初始化方式
            swan = swanlab.init(
                project=project_name,
                workspace=workspace,
                logdir=output_dir,
                config=config_dict
            )
            return SwanLabLogger(swan)

        # 不支持的日志工具
        else:
            raise ValueError(f"Unsupported logger type: {report_to}")

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """记录标量值"""
        pass

    @abc.abstractmethod
    def log_scalars(self, scalars_dict: Dict[str, float], step: int) -> None:
        """记录多个标量值"""
        pass

    @abc.abstractmethod
    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None) -> None:
        """记录单张图像"""
        pass

    @abc.abstractmethod
    def log_images(self, name: str, images: Union[torch.Tensor, np.ndarray], step: int,
                  caption: Optional[str] = None, nrow: int = 4) -> None:
        """记录多张图像，自动拼接为网格"""
        pass

    @abc.abstractmethod
    def log_text(self, name: str, text: str, step: int) -> None:
        """记录文本"""
        pass

    @abc.abstractmethod
    def log_table(self, name: str, rows: List[Dict[str, Any]], step: int) -> None:
        """记录表格数据"""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """刷新日志，确保写入"""
        pass


class TensorBoardLogger(BaseLogger):
    """TensorBoard日志工具封装"""

    def __init__(self, writer):
        self.writer = writer

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, scalars_dict: Dict[str, float], step: int) -> None:
        for name, value in scalars_dict.items():
            self.writer.add_scalar(name, value, step)

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None) -> None:
        if isinstance(image, np.ndarray):
            # 如果是HWC格式，转换为CHW
            if image.ndim == 3 and image.shape[2] in (1, 3, 4):
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        self.writer.add_image(name, image, step, dataformats="CHW")

    def log_images(self, name: str, images: Union[torch.Tensor, np.ndarray], step: int,
                  caption: Optional[str] = None, nrow: int = 4) -> None:
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # 确保是[B,C,H,W]格式
        if images.dim() == 3:
            images = images.unsqueeze(0)

        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
        self.writer.add_image(name, grid, step, dataformats="CHW")

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(name, text, step)

    def log_table(self, name: str, rows: List[Dict[str, Any]], step: int) -> None:
        # TensorBoard没有原生表格支持，转换为文本
        text = "\n".join(
            f"{i}. " + " | ".join(f"{k}: {v}" for k, v in r.items())
            for i, r in enumerate(rows)
        )
        self.writer.add_text(name, text, step)

    def flush(self) -> None:
        self.writer.flush()


class WandbLogger(BaseLogger):
    """Weights & Biases日志工具封装"""

    def __init__(self, run):
        self.run = run

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.run.log({name: value}, step=step)

    def log_scalars(self, scalars_dict: Dict[str, float], step: int) -> None:
        self.run.log(scalars_dict, step=step)

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None) -> None:
        import wandb

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # 确保是HWC格式
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        self.run.log({name: wandb.Image(image, caption=caption)}, step=step)

    def log_images(self, name: str, images: Union[torch.Tensor, np.ndarray], step: int,
                  caption: Optional[str] = None, nrow: int = 4) -> None:
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

        self.run.log({name: wandb.Image(npimg, caption=caption)}, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.run.log({name: text}, step=step)

    def log_table(self, name: str, rows: List[Dict[str, Any]], step: int) -> None:
        import wandb

        if not rows:
            return

        cols = list(rows[0].keys())
        table = wandb.Table(columns=cols)
        for r in rows:
            table.add_data(*[str(r[c]) for c in cols])
        self.run.log({name: table}, step=step)

    def flush(self) -> None:
        # wandb会自动刷新
        pass


class SwanLabLogger(BaseLogger):
    """SwanLab日志工具封装"""

    def __init__(self, swan):
        self.swan = swan

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.swan.log({name: value}, step=step)

    def log_scalars(self, scalars_dict: Dict[str, float], step: int) -> None:
        self.swan.log(scalars_dict, step=step)

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None) -> None:
        import swanlab

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # 确保是HWC格式
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        self.swan.log({name: swanlab.Image(image, caption=caption)}, step=step)

    def log_images(self, name: str, images: Union[torch.Tensor, np.ndarray], step: int,
                  caption: Optional[str] = None, nrow: int = 4) -> None:
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

        self.swan.log({name: swanlab.Image(npimg, caption=caption)}, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.swan.log({name: text}, step=step)

    def log_table(self, name: str, rows: List[Dict[str, Any]], step: int) -> None:
        # SwanLab支持直接记录表格数据
        self.swan.log({name: rows}, step=step)

    def flush(self) -> None:
        # SwanLab会自动刷新
        pass


class LoggerManager:
    """日志管理器，统一管理日志工具"""

    def __init__(self, accelerator=None, config=None):
        """
        初始化LoggerManager

        Args:
            accelerator: 可选的Accelerator实例，用于分布式训练
            config: 配置对象，包含logging配置
        """
        self.accelerator = accelerator
        self.config = config
        self.logger = None

        # 如果提供了配置，尝试初始化logger
        if config is not None:
            self._initialize_logger()

    @classmethod
    def create(cls, config, accelerator=None):
        """
        工厂方法：根据配置创建LoggerManager实例

        Args:
            config: 配置对象，包含logging配置
            accelerator: 可选的Accelerator实例

        Returns:
            LoggerManager实例
        """
        return cls(accelerator, config)

    def _initialize_logger(self):
        """初始化配置的日志工具"""
        # 如果在分布式环境中，只有主进程初始化logger
        if self.accelerator and not self.accelerator.is_main_process:
            return

        try:
            # 使用BaseLogger的工厂方法创建logger实例
            self.logger = BaseLogger.create(self.config, self.accelerator)
            logging.info(f"{self.logger.__class__.__name__} initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize logger: {e}")

            # 如果初始化失败，尝试使用accelerator的默认日志功能
            if self.accelerator:
                logging.info("Falling back to accelerator's default logging")
            else:
                logging.warning("No logger available, logs will be printed to console only")
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """记录标量值"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        if self.logger:
            try:
                self.logger.log_scalar(name, value, step)
            except Exception as e:
                logging.warning(f"Failed to log scalar with {self.logger.__class__.__name__}: {e}")
                # 如果日志工具失败且有accelerator，使用accelerator的log方法
                if self.accelerator:
                    self.accelerator.log({name: value}, step=step)
        else:
            # 如果没有可用的日志工具，使用accelerator的log方法或打印
            if self.accelerator:
                self.accelerator.log({name: value}, step=step)
            else:
                logging.info(f"[{name}] {value} (step {step})")

    def log_scalars(self, scalars_dict: Dict[str, float], step: int) -> None:
        """记录多个标量值"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        if self.logger:
            try:
                self.logger.log_scalars(scalars_dict, step)
            except Exception as e:
                logging.warning(f"Failed to log scalars with {self.logger.__class__.__name__}: {e}")
                # 如果日志工具失败且有accelerator，使用accelerator的log方法
                if self.accelerator:
                    self.accelerator.log(scalars_dict, step=step)
        else:
            # 如果没有可用的日志工具，使用accelerator的log方法或打印
            if self.accelerator:
                self.accelerator.log(scalars_dict, step=step)
            else:
                logging.info(f"Metrics (step {step}): {scalars_dict}")

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None) -> None:
        """记录单张图像"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        logged = False
        if self.logger:
            try:
                self.logger.log_image(name, image, step, caption)
                logged = True
            except Exception as e:
                logging.warning(f"Failed to log image with {self.logger.__class__.__name__}: {e}")

        # 如果没有可用的日志工具或记录失败，记录一个标量占位符
        if not logged:
            if self.accelerator:
                self.accelerator.log({f"{name}/logged": 1}, step=step)
            else:
                logging.info(f"Image logged: {name} (step {step})")

    def log_images(self, name: str, images: Union[torch.Tensor, np.ndarray], step: int,
                  caption: Optional[str] = None, nrow: int = 4, max_images: int = 16) -> None:
        """记录多张图像，自动拼接为网格"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        # 预处理：裁样、归一化到[0,1]
        if isinstance(images, torch.Tensor):
            t = images.detach().float()[:max_images]
            if t.min() < 0 or t.max() > 1:
                t = (t + 1) / 2  # 假设输入范围是[-1,1]
            t = t.clamp(0, 1)
            images = t

        logged = False
        if self.logger:
            try:
                self.logger.log_images(name, images, step, caption, nrow)
                logged = True
            except Exception as e:
                logging.warning(f"Failed to log images with {self.logger.__class__.__name__}: {e}")

        # 如果没有可用的日志工具或记录失败，记录一个标量占位符
        if not logged:
            num_images = len(images) if isinstance(images, list) else images.shape[0]
            if self.accelerator:
                self.accelerator.log({f"{name}/num_images": num_images}, step=step)
            else:
                logging.info(f"Images logged: {name}, count: {num_images} (step {step})")

    def log_text(self, name: str, text: str, step: int) -> None:
        """记录文本"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        logged = False
        if self.logger:
            try:
                self.logger.log_text(name, text, step)
                logged = True
            except Exception as e:
                logging.warning(f"Failed to log text with {self.logger.__class__.__name__}: {e}")

        # 如果没有可用的日志工具或记录失败，打印到控制台
        if not logged:
            if self.accelerator:
                self.accelerator.print(f"[{name}] {text[:100]}...")
            else:
                logging.info(f"[{name}] {text[:100]}...")

    def log_table(self, name: str, rows: List[Dict[str, Any]], step: int, max_rows: int = 64) -> None:
        """记录表格数据"""
        # 如果在分布式环境中，只有主进程记录日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        rows_clip = rows[:max_rows]

        logged = False
        if self.logger:
            try:
                self.logger.log_table(name, rows_clip, step)
                logged = True
            except Exception as e:
                logging.warning(f"Failed to log table with {self.logger.__class__.__name__}: {e}")

        # 如果没有可用的日志工具或记录失败，打印到控制台
        if not logged:
            if self.accelerator:
                self.accelerator.print(f"[{name}] (no tracker) sample: {rows_clip[:3]}")
            else:
                logging.info(f"[{name}] (no tracker) sample: {rows_clip[:3]}")

    def flush(self) -> None:
        """刷新日志工具"""
        # 如果在分布式环境中，只有主进程刷新日志
        if self.accelerator and not self.accelerator.is_main_process:
            return

        if not self.logger:
            return

        try:
            self.logger.flush()
        except Exception as e:
            logging.warning(f"Failed to flush {self.logger.__class__.__name__}: {e}")
```

#### 3.3.2 简化现有的日志函数

有了`LoggerManager`后，我们可以简化现有的`log_images_auto`和`log_text_auto`函数：

```python
def get_logger_manager(accelerator):
    """获取或创建LoggerManager实例"""
    if not hasattr(accelerator, "_logger_manager"):
        accelerator._logger_manager = LoggerManager(accelerator)
    return accelerator._logger_manager

def log_images_auto(accelerator, tag, images, step, caption=None, nrow=4, max_images=16):
    """images: [B,C,H,W] in [-1,1]"""
    logger_manager = get_logger_manager(accelerator)
    logger_manager.log_images(tag, images, step, caption, nrow, max_images)

def log_text_auto(accelerator, tag, rows, step, max_rows=64):
    """
    rows: list[dict] 或 list[str]
    """
    logger_manager = get_logger_manager(accelerator)

    if rows and isinstance(rows[0], str):
        # 将字符串列表转换为字典列表
        rows = [{"text": s} for s in rows]

    logger_manager.log_table(tag, rows, step, max_rows)
```

### 3.4 训练器集成

在`BaseTrainer`中，我们需要确保在初始化Accelerator时正确处理SwanLab选项，并使用新的LoggerManager：

```python
# 在BaseTrainer.__init__方法中
self.accelerator = Accelerator(
    gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
    mixed_precision=self.config.train.mixed_precision,
    log_with=self.config.logging.report_to,
    project_dir=self.config.logging.output_dir,
    # 其他参数...
)

# 初始化日志追踪器
if self.config.logging.report_to != "none":
    project_name = self.config.logging.tracker_project_name or self.experiment_name
    self.accelerator.init_trackers(
        project_name,
        config=simple_config,  # 已有的配置字典
    )

    # 初始化LoggerManager
    from qflux.utils.logger import get_logger_manager
    self.logger_manager = get_logger_manager(self.accelerator)
```

在`BaseTrainer`的`log_metrics`方法中，我们可以使用LoggerManager替代直接调用accelerator.log：

```python
def log_metrics(self, metrics, step=None):
    """使用LoggerManager记录指标"""
    step = step if step is not None else self.global_step

    # 使用LoggerManager记录指标
    from qflux.utils.logger import get_logger_manager
    logger_manager = get_logger_manager(self.accelerator)
    logger_manager.log_scalars(metrics, step)
```

### 3.5 配置示例

以下是使用不同日志工具的配置示例：

#### 3.5.1 使用TensorBoard（默认）

```yaml
logging:
  output_dir: "./output"
  report_to: "tensorboard"
  tracker_project_name: "my-project"
```

#### 3.5.2 使用Weights & Biases

```yaml
logging:
  output_dir: "./output"
  report_to: "wandb"
  tracker_project_name: "my-project"
```

#### 3.5.3 使用SwanLab

```yaml
logging:
  output_dir: "./output"
  report_to: "swanlab"
  tracker_project_name: "my-project"
```



## 4. 实现计划

### 4.1 任务分解

1. 更新`requirements.txt`添加SwanLab依赖
2. 更新`LoggingConfig`以支持SwanLab选项
3. 创建统一的Logger Wrapper类，实现对各种日志工具的封装
4. 更新`BaseTrainer`中的日志初始化逻辑，使用Logger Wrapper
5. 添加SwanLab集成测试
6. 更新文档说明新的日志功能

### 4.2 测试计划

#### 4.2.1 单元测试

在`tests/src/utils/test_logger.py`中添加以下测试：

```python
def test_logger_manager_initialization():
    """测试LoggerManager初始化"""
    # 模拟Accelerator
    accelerator = MagicMock()

    # 测试不同的日志工具配置
    logger_manager = LoggerManager(accelerator)
    assert isinstance(logger_manager, LoggerManager)

def test_tensorboard_logger():
    """测试TensorBoardLogger功能"""
    # 模拟TensorBoard writer
    writer = MagicMock()
    logger = TensorBoardLogger(writer)

    # 测试各种日志方法
    logger.log_scalar("test", 1.0, 0)
    writer.add_scalar.assert_called_once_with("test", 1.0, 0)

    # 测试图像日志
    image = torch.zeros(3, 32, 32)
    logger.log_image("test_img", image, 0)
    writer.add_image.assert_called_once()

def test_wandb_logger():
    """测试WandbLogger功能"""
    # 需要mock wandb模块
    with patch("wandb.Image") as mock_image:
        run = MagicMock()
        logger = WandbLogger(run)

        # 测试各种日志方法
        logger.log_scalar("test", 1.0, 0)
        run.log.assert_called_once_with({"test": 1.0}, step=0)

        # 重置mock
        run.reset_mock()

        # 测试图像日志
        image = torch.zeros(3, 32, 32)
        logger.log_image("test_img", image, 0)
        run.log.assert_called_once()

def test_swanlab_logger():
    """测试SwanLabLogger功能"""
    # 需要mock swanlab模块
    with patch("swanlab.Image") as mock_image:
        swan = MagicMock()
        logger = SwanLabLogger(swan)

        # 测试各种日志方法
        logger.log_scalar("test", 1.0, 0)
        swan.log.assert_called_once_with({"test": 1.0}, step=0)

        # 重置mock
        swan.reset_mock()

        # 测试图像日志
        image = torch.zeros(3, 32, 32)
        logger.log_image("test_img", image, 0)
        swan.log.assert_called_once()
```

#### 4.2.2 集成测试

创建一个简单的训练脚本，测试不同日志工具的集成：

```python
def test_logger_integration_with_trainer():
    """测试日志工具与训练器的集成"""
    # 创建配置
    config = Config()
    config.logging.report_to = "tensorboard"  # 或 "wandb", "swanlab", "all"

    # 初始化训练器
    trainer = BaseTrainer(config)

    # 记录一些指标
    metrics = {"loss": 0.5, "accuracy": 0.8}
    trainer.log_metrics(metrics)

    # 记录一些图像
    images = torch.rand(4, 3, 64, 64)
    log_images_auto(trainer.accelerator, "test/images", images, trainer.global_step)

    # 验证日志是否正确记录（需要手动检查日志文件或UI）
```



### 4.3 文档计划

1. 更新`docs/guide/training.md`，添加关于日志工具支持的使用说明：
   - 如何配置不同的日志工具（TensorBoard、Weights & Biases、SwanLab）

2. 在`docs/changelog`中记录此次更新，包括：
   - SwanLab支持
   - 统一的Logger Wrapper设计

3. 更新`README.md`，提及新增的日志功能

## 5. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| SwanLab API变更 | 低 | 中 | 封装SwanLab API，使其与我们的系统解耦 |
| 与现有代码不兼容 | 低 | 高 | 保持API一致性，确保向后兼容 |
| 日志工具导入失败 | 中 | 低 | 使用try-except处理导入错误，确保训练不会因为日志工具问题而中断 |

## 6. 时间线

| 阶段 | 任务 | 预计完成时间 |
|------|------|--------------|
| 1 | 更新requirements.txt | 0.5天 |
| 1 | 更新LoggingConfig | 0.5天 |
| 2 | 设计并实现Logger Wrapper | 2天 |
| 2 | 更新BaseTrainer | 1天 |
| 3 | 编写单元测试 | 1天 |
| 3 | 进行集成测试 | 1天 |
| 4 | 更新文档 | 1天 |
| | **总计** | **7天** |

## 7. 参考资料

- [SwanLab GitHub仓库](https://github.com/SwanHubX/SwanLab)
- [SwanLab官方文档](https://swanlab.cn)
- [Accelerate文档](https://huggingface.co/docs/accelerate/index)
- [TensorBoard文档](https://www.tensorflow.org/tensorboard)
- [Weights & Biases文档](https://docs.wandb.ai/)

## 8. 实现细节

### 8.1 代码修改细节

#### 8.1.1 现有代码与新设计的对比

**现有代码**:
- 直接使用accelerator.log记录标量指标
- 使用log_images_auto和log_text_auto函数，内部尝试使用不同的日志工具
- 没有统一的日志接口，不同类型的数据使用不同的函数

**新设计**:
- 创建统一的Logger接口和实现
- 使用LoggerManager管理单个日志工具实例
- 提供统一的API记录不同类型的数据

#### 8.1.2 需要修改的文件

1. **src/qflux/utils/logger.py**:
   - 添加BaseLogger抽象类及其实现类
   - 添加LoggerManager类
   - 更新log_images_auto和log_text_auto函数，使用LoggerManager

2. **src/qflux/data/config.py**:
   - 更新LoggingConfig类，添加swanlab选项

3. **src/qflux/trainer/base_trainer.py**:
   - 更新初始化日志工具的代码
   - 修改log_metrics方法，使用LoggerManager

### 8.2 分布式训练中的Logger处理

在分布式训练环境中，需要确保只有主进程记录日志，避免重复记录和资源竞争。

#### 8.2.1 主进程检查

在LoggerManager的所有方法中，都需要首先检查是否是主进程：

```python
def log_scalar(self, name: str, value: float, step: int) -> None:
    """记录标量值"""
    if not self.accelerator.is_main_process:
        return

    # 日志记录代码...
```

#### 8.2.2 同步日志数据

在某些情况下，需要从所有进程收集数据后再记录日志：

```python
# 在训练循环中
# 计算平均损失
loss = accelerator.gather(loss).mean().item()

# 只在主进程记录
if accelerator.is_main_process:
    logger_manager.log_scalar("loss", loss, global_step)
```

#### 8.2.3 处理分布式环境中的图像记录

当记录生成的图像时，确保只在主进程上进行：

```python
# 生成图像
with torch.no_grad():
    generated_images = model(input_ids)

# 收集所有进程的图像
gathered_images = accelerator.gather(generated_images)

# 只在主进程记录
if accelerator.is_main_process:
    logger_manager.log_images("samples", gathered_images[:16], global_step)
```

### 8.3 不同Logger的初始化与登录

#### 8.3.1 TensorBoard

TensorBoard不需要登录，只需要指定输出目录：

```python
# 在Accelerator初始化时自动处理
accelerator = Accelerator(
    log_with="tensorboard",
    project_dir=config.logging.output_dir
)
```

#### 8.3.2 Weights & Biases (wandb)

wandb需要API密钥进行登录，可以通过环境变量或.env文件提供：

```python
# 在.env文件中设置
# WANDB_API_KEY=your_api_key
# WANDB_ENTITY=your_entity_name
# WANDB_PROJECT=your_project_name

# 在代码中初始化wandb
def initialize_wandb(config):
    """初始化wandb，如果未登录则尝试使用环境变量登录"""
    try:
        import wandb
        from dotenv import load_dotenv

        # 加载.env文件中的环境变量
        load_dotenv()

        # 检查是否已登录
        if not wandb.api.api_key:
            # 如果未登录，尝试使用环境变量中的API密钥
            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                logging.warning("WANDB_API_KEY not found in environment variables")
                return False

        # 设置默认项目和实体
        os.environ.setdefault("WANDB_PROJECT", config.logging.tracker_project_name or "qwen-image-finetune")

        return True
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}")
        return False
```

#### 8.3.3 SwanLab

SwanLab也需要API密钥进行登录，同样可以通过环境变量或.env文件提供：

```python
# 在.env文件中设置
# SWANLAB_API_KEY=your_api_key
# SWANLAB_ENTITY=your_entity_name
# SWANLAB_PROJECT=your_project_name

# 在代码中初始化swanlab
def initialize_swanlab(config):
    """初始化swanlab，如果未登录则尝试使用环境变量登录"""
    try:
        import swanlab
        from dotenv import load_dotenv

        # 加载.env文件中的环境变量
        load_dotenv()

        # 检查是否已登录或尝试登录
        api_key = os.environ.get("SWANLAB_API_KEY")
        if not api_key:
            logging.warning("SWANLAB_API_KEY not found in environment variables")
            return False

        # 设置默认项目和实体
        entity = os.environ.get("SWANLAB_ENTITY")
        project = os.environ.get("SWANLAB_PROJECT") or config.logging.tracker_project_name or "qwen-image-finetune"

        # 登录SwanLab
        swanlab.login(api_key=api_key)

        return True
    except Exception as e:
        logging.warning(f"Failed to initialize swanlab: {e}")
        return False
```

#### 8.3.4 在BaseTrainer中集成LoggerManager

在BaseTrainer的初始化方法中，按照Hugging Face Accelerate的官方推荐方式初始化日志追踪器，并使用LoggerManager：

```python
def __init__(self, config):
    # 其他初始化代码...

    # 准备日志配置
    log_with = config.logging.report_to if config.logging.report_to != "none" else None

    # 初始化Accelerator
    self.accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision,
        log_with=log_with,  # 传入日志工具类型
        project_dir=config.logging.output_dir,
        # 其他参数...
    )

    # 初始化日志追踪器
    if log_with is not None:
        # 准备配置字典，用于记录超参数
        config_dict = self.get_config_dict()

        # 设置项目名称
        project_name = config.logging.tracker_project_name or self.experiment_name

        # 初始化追踪器
        self.accelerator.init_trackers(
            project_name,
            config=config_dict,
        )

        # 使用LoggerManager.create工厂方法创建实例
        from qflux.utils.logger import LoggerManager
        self.logger_manager = LoggerManager.create(config, self.accelerator)

        self.accelerator.print(f"Initialized {log_with} logger with project name: {project_name}")
```

### 8.4 示例代码

#### 8.4.1 使用LoggerManager记录指标

```python
# 在训练循环中
def train_step(self):
    # 训练代码...

    # 记录指标
    metrics = {"loss": loss.item(), "lr": lr}
    self.logger_manager.log_scalars(metrics, self.global_step)

    # 每N步记录图像
    if self.global_step % self.log_interval == 0:
        self.logger_manager.log_images("samples/generated", generated_images, self.global_step)
```

#### 8.4.2 配置文件示例

```yaml
# 使用TensorBoard配置
logging:
  output_dir: "./output"
  report_to: "tensorboard"
  tracker_project_name: "my-image-generation"
```

```yaml
# 使用Weights & Biases配置
logging:
  output_dir: "./output"
  report_to: "wandb"
  tracker_project_name: "my-image-generation"
```

```yaml
# 使用SwanLab配置
logging:
  output_dir: "./output"
  report_to: "swanlab"
  tracker_project_name: "my-image-generation"
```

#### 8.4.3 .env文件示例

```
# Weights & Biases配置
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_project_name

# SwanLab配置
SWANLAB_WORKSPACE=your_workspace_name
# SWANLAB_PROJECT是可选的，如果不设置，会使用config.logging.tracker_project_name
```
