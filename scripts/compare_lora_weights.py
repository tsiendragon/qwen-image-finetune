#!/usr/bin/env python
"""
Script to compare two LoRA weight files.
"""
import argparse
import sys
import os

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qflux.qflux.utils.lora_compare import compare_lora_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two LoRA weight files")
    parser.add_argument(
        "--lora1",
        type=str,
        required=True,
        help="Path to the first LoRA weights file"
    )
    parser.add_argument(
        "--lora2",
        type=str,
        required=True,
        help="Path to the second LoRA weights file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        all_match, result = compare_lora_weights(
            args.lora1,
            args.lora2,
            verbose=not args.quiet
        )

        # If in quiet mode, just print the summary
        if args.quiet:
            print(f"All keys match and have same shapes: {all_match}")
            print(f"Common keys: {result['common_keys_count']}")
            print(f"Keys only in first file: {len(result['keys_only_in_first'])}")
            print(f"Keys only in second file: {len(result['keys_only_in_second'])}")
            print(f"Shape mismatches: {len(result['shape_mismatches'])}")

        # Return success (0) if all match, otherwise return 1
        return 0 if all_match else 1

    except Exception as e:
        print(f"Error comparing LoRA weights: {str(e)}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
