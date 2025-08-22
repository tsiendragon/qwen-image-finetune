import logging
from accelerate.logging import get_logger

def load_logger(name, log_level="INFO"):
    logger = get_logger(name, log_level=log_level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    return logger