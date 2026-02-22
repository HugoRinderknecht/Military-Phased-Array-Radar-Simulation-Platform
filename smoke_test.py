#!/usr/bin/env python3
"""
Smoke Test - Basic System Validation
Tests basic functionality of the phased array radar simulation platform
"""

import sys
from pathlib import Path


def test_basic_imports():
    """Test basic module imports"""
    print("="*70)
    print("SMOKE TEST - Phased Array Radar Simulation Platform")
    print("="*70)
    print()

    success_count = 0
    fail_count = 0
    total_count = 0

    # Test numpy and dependencies
    print("Testing Python dependencies...")
    total_count += 1
    try:
        import numpy as np
        import scipy
        import numba
        print("  [PASS] Core numerical libraries (numpy, scipy, numba)")
        success_count += 1
    except ImportError as e:
        print(f"  [FAIL] Core numerical libraries: {e}")
        fail_count += 1

    # Test web framework
    total_count += 1
    try:
        import fastapi
        import uvicorn
        import websockets
        print("  [PASS] Web framework (fastapi, uvicorn, websockets)")
        success_count += 1
    except ImportError as e:
        print(f"  [FAIL] Web framework: {e}")
        fail_count += 1

    # Test basic radar types
    total_count += 1
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from radar.common.types import (
            TargetType, MotionModel, TaskType, TrackState,
            Position3D, Velocity3D, Plot, Track
        )
        print("  [PASS] Radar type definitions")
        success_count += 1

        # Test enum values
        assert TargetType.AIRCRAFT.value == 'aircraft'
        assert MotionModel.CONSTANT_VELOCITY.value == 'cv'
        print("  [PASS] Enum values correct")
    except Exception as e:
        print(f"  [FAIL] Radar type definitions: {e}")
        fail_count += 1

    # Test constants
    total_count += 1
    try:
        from radar.common.constants import (
            PhysicsConstants, MathConstants
        )
        assert PhysicsConstants.C == 299792458.0
        print("  [PASS] Physical constants")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Physical constants: {e}")
        fail_count += 1

    # Test math utils
    total_count += 1
    try:
        from radar.common.utils.math_utils import (
            db_to_linear, linear_to_db, deg_to_rad, rad_to_deg
        )
        assert abs(db_to_linear(10.0) - 10.0) < 0.01
        assert abs(deg_to_rad(180.0) - 3.14159) < 0.001
        print("  [PASS] Math utilities")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Math utilities: {e}")
        fail_count += 1

    # Test coordinate transforms
    total_count += 1
    try:
        from radar.common.utils.coord_transform import (
            enu_to_azel, azel_to_enu
        )
        # Test round-trip conversion
        r, az, el = enu_to_azel(100.0, 0.0, 0.0)
        x, y, z = azel_to_enu(r, az, el)
        assert abs(x - 100.0) < 0.01
        print("  [PASS] Coordinate transformations")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Coordinate transformations: {e}")
        fail_count += 1

    # Test signal utilities
    total_count += 1
    try:
        from radar.common.utils.signal_utils import (
            generate_lfm_pulse, pulse_compression, next_power_of_2
        )
        signal = generate_lfm_pulse(20e6, 10e-6, 10e6)
        assert len(signal) == 200
        assert next_power_of_2(100) == 128
        print("  [PASS] Signal utilities")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Signal utilities: {e}")
        fail_count += 1

    # Test containers
    total_count += 1
    try:
        from radar.common.containers.ring_buffer import RingBuffer
        rb = RingBuffer(capacity=10)
        assert rb.is_empty()
        rb.write(1)
        rb.write(2)
        assert rb.read() == 1
        print("  [PASS] Container classes")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Container classes: {e}")
        fail_count += 1

    # Test logger
    total_count += 1
    try:
        from radar.common.logger import get_logger
        logger = get_logger('test')
        logger.info('Test log message')
        print("  [PASS] Logging system")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Logging system: {e}")
        fail_count += 1

    # Test protocol messages
    total_count += 1
    try:
        from radar.protocol.messages import (
            StartCommand, StopCommand, PlotDataMessage
        )
        cmd = StartCommand(time_scale=1.0)
        assert cmd.type == "cmd_start"
        print("  [PASS] Protocol messages")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Protocol messages: {e}")
        fail_count += 1

    print()
    print("="*70)
    print(f"SMOKE TEST COMPLETE")
    print(f"  Passed: {success_count}/{total_count}")
    print(f"  Failed: {fail_count}/{total_count}")
    print(f"  Success Rate: {success_count/total_count*100:.1f}%")
    print("="*70)

    return fail_count == 0


def test_file_structure():
    """Test that critical files exist"""
    print()
    print("Checking file structure...")

    critical_files = [
        "radar/__init__.py",
        "radar/main.py",
        "radar/common/__init__.py",
        "radar/common/types.py",
        "radar/common/constants.py",
        "radar/protocol/messages.py",
        "radar_config.toml",
        "requirements.txt",
        "README.md",
    ]

    missing = []
    for f in critical_files:
        if not Path(f).exists():
            missing.append(f)

    if missing:
        print(f"  [FAIL] Missing {len(missing)} critical files")
        for f in missing:
            print(f"    - {f}")
        return False
    else:
        print(f"  [PASS] All {len(critical_files)} critical files present")
        return True


def test_config():
    """Test configuration loading"""
    print()
    print("Testing configuration...")

    try:
        from radar.common.config import SystemConfig
        config = SystemConfig()
        print("  [PASS] Configuration loaded")
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration loading: {e}")
        return False


def show_summary(all_passed):
    """Show test summary"""
    print()
    print("="*70)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
        print("The system is ready for use.")
    else:
        print("RESULT: SOME TESTS FAILED")
        print("Please check the errors above.")
    print("="*70)


if __name__ == "__main__":
    import_pass = test_basic_imports()
    file_pass = test_file_structure()
    config_pass = test_config()

    all_passed = import_pass and file_pass and config_pass
    show_summary(all_passed)

    sys.exit(0 if all_passed else 1)
