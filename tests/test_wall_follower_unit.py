"""
Unit test for WallFollower class to verify behavior.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.robot_navigation.wall_follower import WallFollower
import numpy as np


def test_activation():
    """Test activation conditions."""
    wf = WallFollower()

    # Test 1: Should activate when stuck_counter > 5
    sensor_readings = [150, 150, 150, 150, 150]  # All sensors clear
    assert wf.should_activate(sensor_readings, stuck_counter=6) == True
    print("✓ Test 1 passed: Activates when stuck_counter > 5")

    # Test 2: Should activate when 3+ sensors < 80px
    sensor_readings = [70, 60, 50, 75, 150]  # 4 sensors close
    assert wf.should_activate(sensor_readings, stuck_counter=0) == True
    print("✓ Test 2 passed: Activates when 3+ sensors detect walls < 80px")

    # Test 3: Should NOT activate when conditions not met
    sensor_readings = [150, 90, 150, 90, 150]  # Only 2 sensors close
    assert wf.should_activate(sensor_readings, stuck_counter=2) == False
    print("✓ Test 3 passed: Does not activate when conditions not met")


def test_wall_following_right():
    """Test right wall following logic."""
    wf = WallFollower(target_distance=70)
    wf.active = True
    wf.preferred_side = 'right'

    # Test 1: Front blocked - should turn left sharply
    sensor_readings = [150, 150, 50, 70, 70]  # Front blocked
    action = wf.get_wall_following_action(sensor_readings)
    assert action == 5, f"Expected 5, got {action}"
    print(f"✓ Test 1 passed: Front blocked -> sharp left turn (action={action})")

    # Test 2: Too close to right wall - should turn left
    sensor_readings = [150, 150, 150, 40, 45]  # Right wall too close
    action = wf.get_wall_following_action(sensor_readings)
    assert action == 3, f"Expected 3, got {action}"
    print(f"✓ Test 2 passed: Too close to wall -> turn left (action={action})")

    # Test 3: Too far from right wall - should turn right
    sensor_readings = [150, 150, 150, 100, 110]  # Right wall too far
    action = wf.get_wall_following_action(sensor_readings)
    assert action < 0, f"Expected negative action, got {action}"
    print(f"✓ Test 3 passed: Too far from wall -> turn right (action={action})")

    # Test 4: Good distance - maintain course
    sensor_readings = [150, 150, 150, 70, 70]  # Right wall at target distance
    action = wf.get_wall_following_action(sensor_readings)
    print(f"✓ Test 4 passed: Good distance -> maintain (action={action})")


def test_wall_following_left():
    """Test left wall following logic."""
    wf = WallFollower(target_distance=70)
    wf.active = True
    wf.preferred_side = 'left'

    # Test 1: Front blocked - should turn right sharply
    sensor_readings = [70, 70, 50, 150, 150]  # Front blocked
    action = wf.get_wall_following_action(sensor_readings)
    assert action == -5, f"Expected -5, got {action}"
    print(f"✓ Test 1 passed: Front blocked -> sharp right turn (action={action})")

    # Test 2: Too close to left wall - should turn right
    sensor_readings = [45, 40, 150, 150, 150]  # Left wall too close
    action = wf.get_wall_following_action(sensor_readings)
    assert action == -3, f"Expected -3, got {action}"
    print(f"✓ Test 2 passed: Too close to wall -> turn right (action={action})")

    # Test 3: Too far from left wall - should turn left
    sensor_readings = [110, 100, 150, 150, 150]  # Left wall too far
    action = wf.get_wall_following_action(sensor_readings)
    assert action > 0, f"Expected positive action, got {action}"
    print(f"✓ Test 3 passed: Too far from wall -> turn left (action={action})")


def test_update_and_deactivation():
    """Test update and deactivation logic."""
    wf = WallFollower(max_follow_steps=10)

    # Test 1: Activation
    sensor_readings = [70, 60, 50, 75, 150]
    wf.update(sensor_readings, stuck_counter=0)
    assert wf.active == True
    assert wf.preferred_side in ['left', 'right']
    print(f"✓ Test 1 passed: Activated, chose {wf.preferred_side} wall")

    # Test 2: Deactivation on open space
    sensor_readings = [130, 130, 130, 130, 130]  # All clear
    wf.update(sensor_readings, stuck_counter=0)
    assert wf.active == False
    print("✓ Test 2 passed: Deactivated on open space")

    # Test 3: Deactivation on max steps
    wf = WallFollower(max_follow_steps=5)
    sensor_readings = [70, 60, 50, 75, 150]
    wf.update(sensor_readings, stuck_counter=7)  # Activate
    assert wf.active == True

    # Simulate 5 steps
    for i in range(5):
        wf.update([70, 60, 50, 75, 150], stuck_counter=0)

    assert wf.active == False
    print("✓ Test 3 passed: Deactivated after max_follow_steps")


def test_reset():
    """Test reset functionality."""
    wf = WallFollower()
    wf.active = True
    wf.preferred_side = 'right'
    wf.steps_following = 50

    wf.reset()

    assert wf.active == False
    assert wf.preferred_side is None
    assert wf.steps_following == 0
    print("✓ Test passed: Reset clears all state")


if __name__ == '__main__':
    print("Testing WallFollower class...")
    print("=" * 60)

    print("\n1. Testing activation conditions:")
    test_activation()

    print("\n2. Testing right wall following:")
    test_wall_following_right()

    print("\n3. Testing left wall following:")
    test_wall_following_left()

    print("\n4. Testing update and deactivation:")
    test_update_and_deactivation()

    print("\n5. Testing reset:")
    test_reset()

    print("\n" + "=" * 60)
    print("All tests passed!")
