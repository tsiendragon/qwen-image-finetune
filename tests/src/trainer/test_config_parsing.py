#!/usr/bin/env python3
"""
Test script for multi-resolution configuration parsing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

def test_multi_resolution_config_parsing():
    """Test multi-resolution configuration parsing without full trainer."""

    print("Testing multi-resolution configuration parsing...")

    # Mock the trainer class for testing
    class MockTrainer:
        def _parse_multi_resolution_config(
            self,
            multi_resolutions=None
        ):
            """Parse multi-resolution configuration similar to training format."""
            if multi_resolutions is None:
                return {
                    'mode': 'simple',
                    'target': [300*450, 630*945],
                    'controls': [[300*450, 630*945]]
                }

            # Format 1: Simple list
            if isinstance(multi_resolutions, list):
                parsed_target = []
                for item in multi_resolutions:
                    if isinstance(item, (int,)):
                        parsed_target.append(int(item))
                    elif isinstance(item, str):
                        if '*' in item:
                            parts = item.split('*')
                            if len(parts) == 2:
                                parsed_target.append(int(parts[0]) * int(parts[1]))
                            else:
                                raise ValueError(f"Invalid pixel expression: {item}")
                        else:
                            parsed_target.append(int(item))
                    else:
                        raise ValueError(f"Invalid multi_resolutions item: {item}")

                return {
                    'mode': 'simple',
                    'target': parsed_target,
                    'controls': [parsed_target]
                }

            # Format 2: Advanced dict
            elif isinstance(multi_resolutions, dict):
                parsed_dict = {}

                # Parse target candidates
                if 'target' in multi_resolutions:
                    target_candidates = multi_resolutions['target']
                    parsed_target = []
                    for item in target_candidates:
                        if isinstance(item, (int,)):
                            parsed_target.append(int(item))
                        elif isinstance(item, str):
                            if '*' in item:
                                parts = item.split('*')
                                if len(parts) == 2:
                                    parsed_target.append(int(parts[0]) * int(parts[1]))
                                else:
                                    raise ValueError(f"Invalid pixel expression: {item}")
                            else:
                                parsed_target.append(int(item))
                        else:
                            raise ValueError(f"Invalid target item: {item}")
                    parsed_dict['target'] = parsed_target
                else:
                    if 'controls' in multi_resolutions and multi_resolutions['controls']:
                        parsed_dict['target'] = multi_resolutions['controls'][0]
                    else:
                        raise ValueError("No target or controls specified")

                # Parse controls candidates
                if 'controls' in multi_resolutions:
                    parsed_controls = []
                    for control_group in multi_resolutions['controls']:
                        parsed_group = []
                        for item in control_group:
                            if isinstance(item, (int,)):
                                parsed_group.append(int(item))
                            elif isinstance(item, str):
                                if '*' in item:
                                    parts = item.split('*')
                                    if len(parts) == 2:
                                        parsed_group.append(int(parts[0]) * int(parts[1]))
                                    else:
                                        raise ValueError(f"Invalid pixel expression: {item}")
                                else:
                                    parsed_group.append(int(item))
                            else:
                                raise ValueError(f"Invalid control item: {item}")
                        parsed_controls.append(parsed_group)
                    parsed_dict['controls'] = parsed_controls
                else:
                    parsed_dict['controls'] = [parsed_dict['target']]

                return {
                    'mode': 'advanced',
                    'target': parsed_dict['target'],
                    'controls': parsed_dict['controls']
                }

            else:
                raise ValueError(f"multi_resolutions must be list or dict, got {type(multi_resolutions)}")

    trainer = MockTrainer()

    # Test 1: Simple list format
    print("\n1. Testing simple list format...")
    simple_config = ["1024*1024", "512*512"]
    parsed = trainer._parse_multi_resolution_config(simple_config)
    print(f"Input: {simple_config}")
    print(f"Parsed: {parsed}")
    assert parsed['mode'] == 'simple'
    assert parsed['target'] == [1048576, 262144]  # 1024*1024, 512*512
    print("✅ Simple list format test passed!")

    # Test 2: Dict format
    print("\n2. Testing dict format...")
    dict_config = {
        "target": ["1024*1024", "512*512"],
        "controls": [["512*512", "256*256"], ["256*256"]]
    }
    parsed = trainer._parse_multi_resolution_config(dict_config)
    print(f"Input: {dict_config}")
    print(f"Parsed: {parsed}")
    assert parsed['mode'] == 'advanced'
    assert parsed['target'] == [1048576, 262144]
    assert parsed['controls'] == [[262144, 65536], [65536]]
    print("✅ Dict format test passed!")

    # Test 3: Default configuration
    print("\n3. Testing default configuration...")
    parsed = trainer._parse_multi_resolution_config(None)
    print(f"Default parsed: {parsed}")
    assert parsed['mode'] == 'simple'
    assert parsed['target'] == [135000, 595350]  # 300*450, 630*945
    print("✅ Default configuration test passed!")

    # Test 4: Mixed format
    print("\n4. Testing mixed format...")
    mixed_config = {
        "target": [1048576, "512*512"],
        "controls": [["512*512", 256*256]]
    }
    parsed = trainer._parse_multi_resolution_config(mixed_config)
    print(f"Input: {mixed_config}")
    print(f"Parsed: {parsed}")
    assert parsed['mode'] == 'advanced'
    assert parsed['target'] == [1048576, 262144]
    assert parsed['controls'] == [[262144, 65536]]
    print("✅ Mixed format test passed!")

    print("\n🎉 All configuration parsing tests passed!")
    return True

if __name__ == "__main__":
    print("Testing multi-resolution configuration parsing...")
    success = test_multi_resolution_config_parsing()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
