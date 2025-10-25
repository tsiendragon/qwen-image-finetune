import argparse

from qflux.data.config import TrMode, load_config_from_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        required=True,
        help="path to config",
    )
    parser.add_argument("--resume", type=str, default="", help="path to resume checkpoint")
    parser.add_argument("--cache", action="store_true", help="cache the dataset")
    parser.add_argument("--fit-no-cache", action="store_true", help="force non-cache mode for training")
    args = parser.parse_args()

    if args.config == "":
        raise ValueError("config file is required")

    config = load_config_from_yaml(args.config)

    if args.resume:
        config.resume = args.resume

    if args.cache:
        config.mode = TrMode.cache
    elif args.fit_no_cache:
        config.mode = TrMode.fit
        config.cache.use_cache = False
        config.data.init_args.use_cache = False
    else:
        config.mode = TrMode.fit
    return config
