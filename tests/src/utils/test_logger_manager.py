import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from qflux.utils.logger import (
    BaseLogger,
    LoggerManager,
    SwanLabLogger,
    TensorBoardLogger,
    WandbLogger,
)


class TestBaseLogger:
    """测试BaseLogger工厂方法"""

    def test_create_tensorboard_logger(self, tmp_path):
        """测试创建TensorBoard logger"""
        # 创建一个简单的配置
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        logger = BaseLogger.create(config)
        assert isinstance(logger, TensorBoardLogger)
        assert logger.writer is not None

    @pytest.mark.integration
    def test_create_wandb_logger_without_accelerator(self, tmp_path):
        """测试不使用accelerator创建wandb logger"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "wandb"
        config.logging.output_dir = str(tmp_path)
        config.logging.tracker_project_name = "test-project"
        config.get_dict = Mock(return_value={"test": "config"})

        # 设置环境变量以避免实际登录
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            with patch("qflux.utils.logger.wandb") as mock_wandb:
                mock_run = Mock()
                mock_wandb.init.return_value = mock_run

                logger = BaseLogger.create(config)
                assert isinstance(logger, WandbLogger)
                assert logger.run is not None

    @pytest.mark.integration
    def test_create_swanlab_logger_without_accelerator(self, tmp_path):
        """测试不使用accelerator创建swanlab logger"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "swanlab"
        config.logging.output_dir = str(tmp_path)
        config.logging.tracker_project_name = "test-project"
        config.get_dict = Mock(return_value={"test": "config"})

        with patch("qflux.utils.logger.swanlab") as mock_swanlab:
            mock_swan = Mock()
            mock_swanlab.init.return_value = mock_swan

            logger = BaseLogger.create(config)
            assert isinstance(logger, SwanLabLogger)
            assert logger.swan is not None

    def test_create_unsupported_logger(self):
        """测试创建不支持的logger类型"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "unsupported"

        with pytest.raises(ValueError, match="Unsupported logger type"):
            BaseLogger.create(config)


class TestTensorBoardLogger:
    """测试TensorBoard logger实现"""

    @pytest.fixture
    def tb_logger(self, tmp_path):
        """创建TensorBoard logger fixture"""
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tmp_path))
        yield TensorBoardLogger(writer)
        writer.close()

    def test_log_scalar(self, tb_logger):
        """测试记录标量"""
        tb_logger.log_scalar("test/loss", 0.5, step=0)
        tb_logger.flush()

    def test_log_scalars(self, tb_logger):
        """测试记录多个标量"""
        scalars = {"loss": 0.5, "accuracy": 0.9}
        tb_logger.log_scalars(scalars, step=0)
        tb_logger.flush()

    def test_log_image_tensor(self, tb_logger):
        """测试记录图像（tensor格式）"""
        image = torch.rand(3, 64, 64)
        tb_logger.log_image("test/image", image, step=0)
        tb_logger.flush()

    def test_log_image_numpy(self, tb_logger):
        """测试记录图像（numpy格式）"""
        image = np.random.rand(64, 64, 3)
        tb_logger.log_image("test/image", image, step=0)
        tb_logger.flush()

    def test_log_images(self, tb_logger):
        """测试记录多张图像"""
        images = torch.rand(4, 3, 64, 64)
        tb_logger.log_images("test/images", images, step=0, nrow=2)
        tb_logger.flush()

    def test_log_text(self, tb_logger):
        """测试记录文本"""
        tb_logger.log_text("test/text", "Hello, world!", step=0)
        tb_logger.flush()

    def test_log_table(self, tb_logger):
        """测试记录表格"""
        rows = [{"id": 0, "value": "A"}, {"id": 1, "value": "B"}]
        tb_logger.log_table("test/table", rows, step=0)
        tb_logger.flush()


class TestWandbLogger:
    """测试Wandb logger实现"""

    @pytest.fixture
    def wandb_logger(self):
        """创建Wandb logger fixture"""
        mock_run = Mock()
        logger = WandbLogger(mock_run)
        return logger

    def test_log_scalar(self, wandb_logger):
        """测试记录标量"""
        wandb_logger.log_scalar("test/loss", 0.5, step=0)
        wandb_logger.run.log.assert_called_once()

    def test_log_scalars(self, wandb_logger):
        """测试记录多个标量"""
        scalars = {"loss": 0.5, "accuracy": 0.9}
        wandb_logger.log_scalars(scalars, step=0)
        wandb_logger.run.log.assert_called_once()

    def test_log_image_tensor(self, wandb_logger):
        """测试记录图像（tensor格式）"""
        with patch("qflux.utils.logger.wandb") as mock_wandb:
            mock_wandb.Image = Mock(return_value="image")
            image = torch.rand(3, 64, 64)
            wandb_logger.log_image("test/image", image, step=0)
            wandb_logger.run.log.assert_called_once()

    def test_log_images(self, wandb_logger):
        """测试记录多张图像"""
        with patch("qflux.utils.logger.wandb") as mock_wandb:
            mock_wandb.Image = Mock(return_value="image")
            images = torch.rand(4, 3, 64, 64)
            wandb_logger.log_images("test/images", images, step=0, nrow=2)
            wandb_logger.run.log.assert_called_once()

    def test_log_text(self, wandb_logger):
        """测试记录文本"""
        wandb_logger.log_text("test/text", "Hello, world!", step=0)
        wandb_logger.run.log.assert_called_once()

    def test_log_table(self, wandb_logger):
        """测试记录表格"""
        with patch("qflux.utils.logger.wandb") as mock_wandb:
            mock_table = Mock()
            mock_wandb.Table = Mock(return_value=mock_table)
            rows = [{"id": 0, "value": "A"}, {"id": 1, "value": "B"}]
            wandb_logger.log_table("test/table", rows, step=0)
            wandb_logger.run.log.assert_called_once()


class TestSwanLabLogger:
    """测试SwanLab logger实现"""

    @pytest.fixture
    def swanlab_logger(self):
        """创建SwanLab logger fixture"""
        mock_swan = Mock()
        logger = SwanLabLogger(mock_swan)
        return logger

    def test_log_scalar(self, swanlab_logger):
        """测试记录标量"""
        swanlab_logger.log_scalar("test/loss", 0.5, step=0)
        swanlab_logger.swan.log.assert_called_once()

    def test_log_scalars(self, swanlab_logger):
        """测试记录多个标量"""
        scalars = {"loss": 0.5, "accuracy": 0.9}
        swanlab_logger.log_scalars(scalars, step=0)
        swanlab_logger.swan.log.assert_called_once()

    def test_log_image_tensor(self, swanlab_logger):
        """测试记录图像（tensor格式）"""
        with patch("qflux.utils.logger.swanlab") as mock_swanlab:
            mock_swanlab.Image = Mock(return_value="image")
            image = torch.rand(3, 64, 64)
            swanlab_logger.log_image("test/image", image, step=0)
            swanlab_logger.swan.log.assert_called_once()

    def test_log_images(self, swanlab_logger):
        """测试记录多张图像"""
        with patch("qflux.utils.logger.swanlab") as mock_swanlab:
            mock_swanlab.Image = Mock(return_value="image")
            images = torch.rand(4, 3, 64, 64)
            swanlab_logger.log_images("test/images", images, step=0, nrow=2)
            swanlab_logger.swan.log.assert_called_once()

    def test_log_text(self, swanlab_logger):
        """测试记录文本"""
        swanlab_logger.log_text("test/text", "Hello, world!", step=0)
        swanlab_logger.swan.log.assert_called_once()

    def test_log_table(self, swanlab_logger):
        """测试记录表格"""
        rows = [{"id": 0, "value": "A"}, {"id": 1, "value": "B"}]
        swanlab_logger.log_table("test/table", rows, step=0)
        swanlab_logger.swan.log.assert_called_once()


class TestLoggerManager:
    """测试LoggerManager"""

    @pytest.fixture
    def mock_accelerator(self):
        """创建mock accelerator"""
        accelerator = Mock()
        accelerator.is_main_process = True
        accelerator.log = Mock()
        accelerator.print = Mock()
        return accelerator

    def test_create_with_tensorboard(self, tmp_path, mock_accelerator):
        """测试使用TensorBoard创建LoggerManager"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        assert manager.logger is not None
        assert isinstance(manager.logger, TensorBoardLogger)

    def test_create_without_accelerator(self, tmp_path):
        """测试不使用accelerator创建LoggerManager"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, accelerator=None)
        assert manager.logger is not None

    def test_log_scalar_main_process(self, tmp_path, mock_accelerator):
        """测试主进程记录标量"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        manager.log_scalar("test/loss", 0.5, step=0)

    def test_log_scalar_non_main_process(self, tmp_path):
        """测试非主进程不记录标量"""
        accelerator = Mock()
        accelerator.is_main_process = False

        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, accelerator)
        manager.log_scalar("test/loss", 0.5, step=0)
        # 非主进程应该直接返回，不记录

    def test_log_scalars(self, tmp_path, mock_accelerator):
        """测试记录多个标量"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        scalars = {"loss": 0.5, "accuracy": 0.9}
        manager.log_scalars(scalars, step=0)

    def test_log_image(self, tmp_path, mock_accelerator):
        """测试记录图像"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        image = torch.rand(3, 64, 64)
        manager.log_image("test/image", image, step=0)

    def test_log_images(self, tmp_path, mock_accelerator):
        """测试记录多张图像"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        images = torch.rand(4, 3, 64, 64) * 2 - 1  # [-1, 1] range
        manager.log_images("test/images", images, step=0, nrow=2)

    def test_log_text(self, tmp_path, mock_accelerator):
        """测试记录文本"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        manager.log_text("test/text", "Hello, world!", step=0)

    def test_log_table(self, tmp_path, mock_accelerator):
        """测试记录表格"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        rows = [{"id": 0, "value": "A"}, {"id": 1, "value": "B"}]
        manager.log_table("test/table", rows, step=0)

    def test_flush(self, tmp_path, mock_accelerator):
        """测试刷新日志"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config, mock_accelerator)
        manager.flush()

    def test_fallback_to_accelerator_log(self, mock_accelerator):
        """测试当logger失败时回退到accelerator.log"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = "/invalid/path"

        # 创建一个会失败的logger
        manager = LoggerManager(mock_accelerator, config)
        manager.logger = None  # 模拟logger初始化失败

        # 应该使用accelerator的log方法
        manager.log_scalar("test/loss", 0.5, step=0)
        mock_accelerator.log.assert_called_once()

    def test_without_logger_without_accelerator(self):
        """测试没有logger和accelerator时的行为"""
        manager = LoggerManager(accelerator=None, config=None)
        # 应该不会抛出异常
        manager.log_scalar("test/loss", 0.5, step=0)
        manager.log_scalars({"loss": 0.5}, step=0)
        manager.log_image("test/image", torch.rand(3, 64, 64), step=0)
        manager.log_images("test/images", torch.rand(4, 3, 64, 64), step=0)
        manager.log_text("test/text", "Hello", step=0)
        manager.log_table("test/table", [{"id": 0}], step=0)
        manager.flush()


@pytest.mark.integration
class TestLoggerIntegration:
    """集成测试：测试不同logger的实际使用"""

    def test_tensorboard_integration(self, tmp_path):
        """测试TensorBoard的完整工作流"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "tensorboard"
        config.logging.output_dir = str(tmp_path)

        manager = LoggerManager.create(config)

        # 记录各种类型的数据
        manager.log_scalar("train/loss", 0.5, step=0)
        manager.log_scalars({"train/loss": 0.4, "train/acc": 0.9}, step=1)
        manager.log_image("train/sample", torch.rand(3, 64, 64), step=0)
        manager.log_images("train/batch", torch.rand(4, 3, 64, 64), step=0)
        manager.log_text("train/info", "Training started", step=0)
        manager.log_table("train/metrics", [{"epoch": 0, "loss": 0.5}], step=0)
        manager.flush()

        # 验证日志文件已创建
        log_files = list(Path(tmp_path).glob("events.out.tfevents.*"))
        assert log_files

    @pytest.mark.skip(reason="需要实际的wandb环境")
    def test_wandb_integration(self, tmp_path):
        """测试Wandb的完整工作流（需要实际环境）"""
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            config = Mock()
            config.logging = Mock()
            config.logging.report_to = "wandb"
            config.logging.output_dir = str(tmp_path)
            config.logging.tracker_project_name = "test-project"
            config.get_dict = Mock(return_value={})

            manager = LoggerManager.create(config)

            manager.log_scalar("train/loss", 0.5, step=0)
            manager.log_scalars({"train/loss": 0.4, "train/acc": 0.9}, step=1)
            manager.flush()

    @pytest.mark.skip(reason="需要实际的swanlab环境")
    def test_swanlab_integration(self, tmp_path):
        """测试SwanLab的完整工作流（需要实际环境）"""
        config = Mock()
        config.logging = Mock()
        config.logging.report_to = "swanlab"
        config.logging.output_dir = str(tmp_path)
        config.logging.tracker_project_name = "test-project"
        config.get_dict = Mock(return_value={})

        manager = LoggerManager.create(config)

        manager.log_scalar("train/loss", 0.5, step=0)
        manager.log_scalars({"train/loss": 0.4, "train/acc": 0.9}, step=1)
        manager.flush()
