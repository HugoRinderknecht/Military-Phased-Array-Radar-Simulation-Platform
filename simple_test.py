#!/usr/bin/env python3
"""
Simple System Test - Test core functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all major modules can be imported"""
    print("="*70)
    print("SYSTEM TEST - Phased Array Radar Simulation Platform")
    print("="*70)
    print()

    passed = 0
    failed = 0

    # Test basic types
    print("[1/10] Testing type definitions...")
    try:
        from radar.common.types import (
            TargetType, MotionModel, TaskType, TrackState,
            Position3D, Velocity3D, Plot, Track
        )
        # Test enum values
        assert TargetType.AIRCRAFT.value == 'aircraft'
        assert MotionModel.CONSTANT_VELOCITY.value == 'cv'
        # Test dataclasses
        pos = Position3D(100.0, 200.0, 300.0)
        vel = Velocity3D(10.0, 20.0, 30.0)
        assert pos.x == 100.0
        assert vel.magnitude() > 0
        print("  [PASS] Type definitions working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Type definitions: {e}")
        failed += 1

    # Test constants
    print("[2/10] Testing physical constants...")
    try:
        from radar.common.constants import PhysicsConstants, MathConstants
        assert PhysicsConstants.C == 299792458.0
        assert MathConstants.PI > 0
        print("  [PASS] Physical constants working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Physical constants: {e}")
        failed += 1

    # Test math utilities
    print("[3/10] Testing math utilities...")
    try:
        from radar.common.utils.math_utils import (
            db_to_linear, linear_to_db, deg_to_rad, rad_to_deg,
            normalize_angle, wrap_angle
        )
        assert abs(db_to_linear(10.0) - 10.0) < 0.01
        assert abs(deg_to_rad(180.0) - 3.14159) < 0.001
        assert abs(rad_to_deg(3.14159) - 180.0) < 0.1
        print("  [PASS] Math utilities working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Math utilities: {e}")
        failed += 1

    # Test coordinate transforms
    print("[4/10] Testing coordinate transforms...")
    try:
        from radar.common.utils.coord_transform import (
            enu_to_radar, radar_to_enu
        )
        # Test round-trip conversion
        pos = radar_to_enu(100.0, 0.0, 0.0)
        r, az, el = enu_to_radar(pos)
        assert abs(r - 100.0) < 0.01
        print("  [PASS] Coordinate transforms working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Coordinate transforms: {e}")
        failed += 1

    # Test signal utilities
    print("[5/10] Testing signal utilities...")
    try:
        from radar.common.utils.signal_utils import (
            generate_lfm_pulse, next_power_of_2
        )
        signal = generate_lfm_pulse(20e6, 10e-6, 10e6)
        assert len(signal) == 200
        assert next_power_of_2(100) == 128
        print("  [PASS] Signal utilities working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Signal utilities: {e}")
        failed += 1

    # Test containers
    print("[6/10] Testing container classes...")
    try:
        from radar.common.containers.ring_buffer import RingBuffer
        rb = RingBuffer(capacity=10)
        assert rb.is_empty()
        rb.write(1)
        rb.write(2)
        assert rb.read() == 1
        print("  [PASS] Container classes working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Container classes: {e}")
        failed += 1

    # Test logger
    print("[7/10] Testing logging system...")
    try:
        from radar.common.logger import get_logger
        logger = get_logger('test')
        logger.info('Test log message')
        print("  [PASS] Logging system working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Logging system: {e}")
        failed += 1

    # Test protocol
    print("[8/10] Testing protocol messages...")
    try:
        from radar.protocol.messages import (
            StartCommand, StopCommand, PlotDataMessage
        )
        cmd = StartCommand(time_scale=1.0)
        assert cmd.type == "cmd_start"
        print("  [PASS] Protocol messages working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Protocol messages: {e}")
        failed += 1

    # Test config
    print("[9/10] Testing configuration...")
    try:
        from radar.common.config import SystemConfig
        config = SystemConfig()
        print("  [PASS] Configuration loading working")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Configuration: {e}")
        failed += 1

    # Test file structure
    print("[10/10] Testing file structure...")
    critical_files = [
        "radar/__init__.py",
        "radar/main.py",
        "radar/common/types.py",
        "radar_config.toml",
    ]
    missing = [f for f in critical_files if not Path(f).exists()]
    if not missing:
        print(f"  [PASS] All critical files present")
        passed += 1
    else:
        print(f"  [FAIL] Missing files: {missing}")
        failed += 1

    # Summary
    print()
    print("="*70)
    print(f"RESULTS: {passed}/10 tests passed")
    if failed == 0:
        print("STATUS: ALL TESTS PASSED")
        print("="*70)
        return True
    else:
        print(f"STATUS: {failed} TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
