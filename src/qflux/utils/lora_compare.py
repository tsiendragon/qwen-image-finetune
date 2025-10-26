"""
Utility functions for comparing LoRA weights.
"""

import os

from safetensors.torch import load_file


def compare_lora_weights(lora_path1: str, lora_path2: str, verbose: bool = True) -> tuple[bool, dict]:
    """
    Compare two LoRA weight files to check if their keys match and if the shapes are the same.

    Args:
        lora_path1: Path to the first LoRA weights file
        lora_path2: Path to the second LoRA weights file
        verbose: Whether to print detailed information

    Returns:
        Tuple containing:
            - Boolean indicating if all keys match and have the same shapes
            - Dictionary with comparison results
    """
    # Check if files exist
    if not os.path.exists(lora_path1):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path1}")
    if not os.path.exists(lora_path2):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path2}")

    # Load the LoRA weights
    if verbose:
        print(f"Loading LoRA weights from {lora_path1}")
    lora_weights1 = load_file(lora_path1)

    if verbose:
        print(f"Loading LoRA weights from {lora_path2}")
    lora_weights2 = load_file(lora_path2)

    # Get the keys
    keys1 = set(lora_weights1.keys())
    keys2 = set(lora_weights2.keys())

    # Check if keys match
    keys_only_in_1 = keys1 - keys2
    keys_only_in_2 = keys2 - keys1
    common_keys = keys1.intersection(keys2)

    # Check shapes for common keys
    shape_mismatches = {}
    for key in common_keys:
        shape1 = lora_weights1[key].shape
        shape2 = lora_weights2[key].shape
        if shape1 != shape2:
            shape_mismatches[key] = (shape1, shape2)

    # Prepare result
    all_match = len(keys_only_in_1) == 0 and len(keys_only_in_2) == 0 and len(shape_mismatches) == 0

    result = {
        "all_match": all_match,
        "keys_only_in_first": list(keys_only_in_1),
        "keys_only_in_second": list(keys_only_in_2),
        "common_keys_count": len(common_keys),
        "shape_mismatches": shape_mismatches,
        "first_file_keys_count": len(keys1),
        "second_file_keys_count": len(keys2),
    }

    if verbose:
        print("\nComparison Results:")
        print(f"All keys match and have same shapes: {all_match}")
        print(f"Total keys in first file: {len(keys1)}")
        print(f"Total keys in second file: {len(keys2)}")
        print(f"Common keys: {len(common_keys)}")
        print(f"Keys only in first file: {len(keys_only_in_1)}")
        print(f"Keys only in second file: {len(keys_only_in_2)}")
        print(f"Shape mismatches: {len(shape_mismatches)}")

        if len(keys_only_in_1) > 0 and verbose:
            print("\nSample keys only in first file:")
            for key in list(keys_only_in_1)[:5]:  # Show only first 5 keys
                print(f"  - {key} (shape: {lora_weights1[key].shape})")
            if len(keys_only_in_1) > 5:
                print(f"  ... and {len(keys_only_in_1) - 5} more")

        if len(keys_only_in_2) > 0 and verbose:
            print("\nSample keys only in second file:")
            for key in list(keys_only_in_2)[:5]:  # Show only first 5 keys
                print(f"  - {key} (shape: {lora_weights2[key].shape})")
            if len(keys_only_in_2) > 5:
                print(f"  ... and {len(keys_only_in_2) - 5} more")

        if len(shape_mismatches) > 0 and verbose:
            print("\nShape mismatches:")
            for key, (shape1, shape2) in list(shape_mismatches.items())[:5]:
                print(f"  - {key}: {shape1} vs {shape2}")
            if len(shape_mismatches) > 5:
                print(f"  ... and {len(shape_mismatches) - 5} more")

    return all_match, result
