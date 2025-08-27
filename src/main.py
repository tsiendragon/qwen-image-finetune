import diffusers
import transformers
from src.data.dataset import loader
from src.utils.options import parse_args
from src.utils.logger import get_logger

logger = get_logger(__name__, log_level="INFO")


def main():
    """使用模块化的 Trainer 类进行训练"""
    # 解析配置
    config = parse_args()

    # 设置日志级别
    if hasattr(config, 'local_rank') and config.local_rank == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 创建训练器
    from src.qwen_image_edit_trainer import QwenImageEditTrainer

    trainer = QwenImageEditTrainer(config)

    # 加载数据
    train_dataloader = loader(
        config.data.class_path,
        config.data.init_args,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=config.data.shuffle,
    )

    if config.mode == 'cache':
        trainer.cache(train_dataloader)
    else:
        # 开始训练
        trainer.fit(train_dataloader)


if __name__ == "__main__":
    main()
