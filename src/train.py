import datasets
import diffusers
import transformers
from data.dataset import loader
from utils.options import parse_args
from utils.logger import get_logger
from trainer import Trainer

logger = get_logger(__name__, log_level="INFO")






def main():
    """使用模块化的 Trainer 类进行训练"""
    # 解析配置
    config = parse_args()

    # 设置日志级别
    if hasattr(config, 'local_rank') and config.local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 创建训练器
    trainer = Trainer(config)

    # 设置训练器
    trainer.setup()
    trainer.setup_models()
    trainer.configure_optimizers()

    # 加载数据
    train_dataloader = loader(**config.data.init_args)

    # 开始训练
    trainer.fit(train_dataloader)


if __name__ == "__main__":
    main()