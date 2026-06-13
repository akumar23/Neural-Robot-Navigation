"""
Verification script for enhanced feature engineering implementation.

This script tests that the 20D feature implementation is working correctly
without requiring the full simulation environment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from collections import deque

# Import feature engineering functions
from src.robot_navigation.feature_engineering import (
    engineer_features,
    compute_goal_relative_features,
    compute_temporal_features,
    compute_spatial_goal_features,
    compute_derived_features,
    get_feature_names
)

def test_legacy_mode():
    """Test that legacy 12D mode still works."""
    print("=" * 60)
    print("TEST 1: Legacy Mode (12D)")
    print("=" * 60)

    sensor_readings = [100.0, 120.0, 150.0, 110.0, 95.0]
    action = 2

    features_12d = engineer_features(sensor_readings, action)

    print(f"✓ Legacy mode feature vector length: {len(features_12d)}")
    print(f"✓ Expected: 12, Got: {len(features_12d)}")
    assert len(features_12d) == 12, "Legacy mode should produce 12 features"

    print(f"✓ Feature names count: {len(get_feature_names(enhanced=False))}")
    assert len(get_feature_names(enhanced=False)) == 12, "Should have 12 feature names"

    print(f"✓ No NaN values: {not np.any(np.isnan(features_12d))}")
    print(f"✓ No Inf values: {not np.any(np.isinf(features_12d))}")

    print(f"\nFeature vector (12D): {features_12d}")
    print("✓ Legacy mode test PASSED\n")


def test_enhanced_mode():
    """Test that enhanced 20D mode works."""
    print("=" * 60)
    print("TEST 2: Enhanced Mode (20D)")
    print("=" * 60)

    # Setup test data
    sensor_readings = [100.0, 120.0, 150.0, 110.0, 95.0]
    action = 2
    robot_pos = (400.0, 300.0)
    robot_angle = 0.5  # radians
    goal_pos = (600.0, 400.0)
    velocity = (5.0, 2.0)
    action_history = deque([1, 2, 1, 0, -1], maxlen=5)

    features_20d = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos,
        robot_angle=robot_angle,
        goal_pos=goal_pos,
        velocity=velocity,
        action_history=action_history
    )

    print(f"✓ Enhanced mode feature vector length: {len(features_20d)}")
    print(f"✓ Expected: 20, Got: {len(features_20d)}")
    assert len(features_20d) == 20, "Enhanced mode should produce 20 features"

    print(f"✓ Feature names count: {len(get_feature_names(enhanced=True))}")
    assert len(get_feature_names(enhanced=True)) == 20, "Should have 20 feature names"

    print(f"✓ No NaN values: {not np.any(np.isnan(features_20d))}")
    print(f"✓ No Inf values: {not np.any(np.isinf(features_20d))}")

    print(f"\nFeature vector (20D):")
    print(f"  Raw sensors [0-4]: {features_20d[0:5]}")
    print(f"  Spatial derived [5-10]: {features_20d[5:11]}")
    print(f"  Goal relative [11-12]: {features_20d[11:13]}")
    print(f"  Temporal [13-16]: {features_20d[13:17]}")
    print(f"  Spatial-goal [17-18]: {features_20d[17:19]}")
    print(f"  Action [19]: {features_20d[19]}")

    print("\n✓ Enhanced mode test PASSED\n")


def test_empty_action_history():
    """Test handling of empty action history."""
    print("=" * 60)
    print("TEST 3: Empty Action History Edge Case")
    print("=" * 60)

    sensor_readings = [100.0, 120.0, 150.0, 110.0, 95.0]
    action = 2
    robot_pos = (400.0, 300.0)
    robot_angle = 0.5
    goal_pos = (600.0, 400.0)
    velocity = (5.0, 2.0)
    action_history = deque(maxlen=5)  # Empty

    features = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos,
        robot_angle=robot_angle,
        goal_pos=goal_pos,
        velocity=velocity,
        action_history=action_history
    )

    print(f"✓ Feature vector length: {len(features)}")
    print(f"✓ No NaN values: {not np.any(np.isnan(features))}")
    print(f"✓ No Inf values: {not np.any(np.isinf(features))}")

    # Check temporal features are zeros
    print(f"✓ Action momentum (should be 0.0): {features[15]}")
    print(f"✓ Action variance (should be 0.0): {features[16]}")
    assert features[15] == 0.0, "Empty history should give zero momentum"
    assert features[16] == 0.0, "Empty history should give zero variance"

    print("✓ Empty action history test PASSED\n")


def test_goal_relative_features():
    """Test goal-relative feature computation."""
    print("=" * 60)
    print("TEST 4: Goal-Relative Features")
    print("=" * 60)

    # Robot at origin facing east (0 radians)
    robot_pos = (0.0, 0.0)
    robot_angle = 0.0

    # Test 1: Goal directly ahead
    goal_pos = (100.0, 0.0)
    features = compute_goal_relative_features(robot_pos, robot_angle, goal_pos, max_range=150.0)
    print(f"Goal ahead: direction={features[0]:.4f} (should be ~0.0), distance={features[1]:.4f}")
    assert abs(features[0]) < 0.1, "Goal ahead should have direction near 0"

    # Test 2: Goal to the left
    goal_pos = (0.0, 100.0)
    features = compute_goal_relative_features(robot_pos, robot_angle, goal_pos, max_range=150.0)
    print(f"Goal left: direction={features[0]:.4f} (should be ~0.64), distance={features[1]:.4f}")
    assert features[0] > 0.5, "Goal to left should have positive direction"

    # Test 3: Goal to the right
    goal_pos = (0.0, -100.0)
    features = compute_goal_relative_features(robot_pos, robot_angle, goal_pos, max_range=150.0)
    print(f"Goal right: direction={features[0]:.4f} (should be ~-0.64), distance={features[1]:.4f}")
    assert features[0] < -0.5, "Goal to right should have negative direction"

    print("✓ Goal-relative features test PASSED\n")


def test_temporal_features():
    """Test temporal feature computation."""
    print("=" * 60)
    print("TEST 5: Temporal Features")
    print("=" * 60)

    # Test with full history
    velocity = (10.0, 5.0)
    action_history = deque([2, 2, 2, 2, 2], maxlen=5)  # Consistent right turn

    features = compute_temporal_features(velocity, action_history, max_velocity=10.0)
    print(f"Velocity features: vx={features[0]:.4f}, vy={features[1]:.4f}")
    print(f"Action momentum: {features[2]:.4f} (should be 0.4 for action=2)")
    print(f"Action variance: {features[3]:.4f} (should be 0.0 for consistent actions)")

    assert abs(features[2] - 0.4) < 0.01, "Momentum should be 2.0/5.0 = 0.4"
    assert abs(features[3]) < 0.01, "Variance should be 0 for consistent actions"

    # Test with varying actions
    action_history = deque([-2, 2, -2, 2, -2], maxlen=5)  # Oscillating
    features = compute_temporal_features(velocity, action_history, max_velocity=10.0)
    print(f"\nOscillating actions: momentum={features[2]:.4f}, variance={features[3]:.4f}")
    assert features[3] > 0.5, "Variance should be high for oscillating actions"

    print("✓ Temporal features test PASSED\n")


def test_spatial_goal_features():
    """Test spatial-goal feature computation."""
    print("=" * 60)
    print("TEST 6: Spatial-Goal Features")
    print("=" * 60)

    sensor_readings = [100.0, 120.0, 150.0, 110.0, 95.0]  # Front sensor clear

    # Test 1: Goal aligned with front sensor
    goal_direction = 0.0  # Straight ahead
    features = compute_spatial_goal_features(sensor_readings, goal_direction, max_range=150.0)
    print(f"Goal aligned: alignment={features[0]:.4f}, urgency={features[1]:.4f}")
    assert features[0] > 0.8, "Alignment should be high when goal is ahead and path clear"

    # Test 2: Goal perpendicular
    goal_direction = 1.0  # 180 degrees (behind)
    features = compute_spatial_goal_features(sensor_readings, goal_direction, max_range=150.0)
    print(f"Goal perpendicular: alignment={features[0]:.4f}, urgency={features[1]:.4f}")
    assert features[0] < 0.2, "Alignment should be low when goal is perpendicular"

    # Test 3: Close obstacle
    close_sensors = [10.0, 15.0, 20.0, 18.0, 12.0]
    features = compute_spatial_goal_features(close_sensors, 0.0, max_range=150.0)
    print(f"Close obstacle: urgency={features[1]:.4f} (should be high)")
    assert features[1] > 0.05, "Urgency should be high for close obstacles"

    print("✓ Spatial-goal features test PASSED\n")


def test_feature_consistency():
    """Test that features are consistent across calls."""
    print("=" * 60)
    print("TEST 7: Feature Consistency")
    print("=" * 60)

    sensor_readings = [100.0, 120.0, 150.0, 110.0, 95.0]
    action = 2
    robot_pos = (400.0, 300.0)
    robot_angle = 0.5
    goal_pos = (600.0, 400.0)
    velocity = (5.0, 2.0)
    action_history = deque([1, 2, 1, 0, -1], maxlen=5)

    # Call twice with same inputs
    features1 = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos, robot_angle=robot_angle,
        goal_pos=goal_pos, velocity=velocity,
        action_history=action_history
    )

    features2 = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos, robot_angle=robot_angle,
        goal_pos=goal_pos, velocity=velocity,
        action_history=action_history
    )

    print(f"✓ Features identical: {np.allclose(features1, features2)}")
    assert np.allclose(features1, features2), "Same inputs should produce identical features"

    print("✓ Feature consistency test PASSED\n")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE ENGINEERING VERIFICATION")
    print("=" * 60 + "\n")

    try:
        test_legacy_mode()
        test_enhanced_mode()
        test_empty_action_history()
        test_goal_relative_features()
        test_temporal_features()
        test_spatial_goal_features()
        test_feature_consistency()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nEnhanced feature engineering implementation is working correctly.")
        print("Ready to proceed with data collection and training.")
        print("\nNext steps:")
        print("  1. python scripts/collect_data.py  # Collect 100k samples")
        print("  2. python scripts/train.py         # Train model with 20D features")
        print("  3. python tests/test_robot_movement.py --tests 5  # Test inference")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
