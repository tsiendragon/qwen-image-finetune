from src.data.config import load_config_from_yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    parser.add_argument("--resume", type='str', default=None, help="path to resume checkpoint")
    args = parser.parse_args()
    config = load_config_from_yaml(args.config)
    if args.resume:
        config.resume = args.resume
    return config
