"""
Unit test for WaypointPlanner to verify basic functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.robot_navigation.waypoint_planner import WaypointPlanner


def test_waypoint_planner():
    """Test basic waypoint planner functionality."""
    print("Testing WaypointPlanner...")

    # Initialize planner
    planner = WaypointPlanner(waypoint_distance=120, waypoint_reached_threshold=50)
    print("  ✓ WaypointPlanner initialized")

    # Test 1: Should not use waypoint when goal is close
    robot_pos = (100, 100)
    robot_angle = 0.0
    goal_pos = (200, 100)  # 100px away
    sensor_readings = np.array([150, 150, 150, 150, 150])  # All sensors clear

    target = planner.get_target(robot_pos, robot_angle, goal_pos, sensor_readings)
    assert planner.current_waypoint is None, "Should not create waypoint for close goal"
    assert target == (200, 100), "Should return goal directly"
    print("  ✓ Test 1 passed: No waypoint for close goal")

    # Test 2: Should use waypoint when goal is far and path is blocked
    robot_pos = (100, 100)
    robot_angle = 0.0
    goal_pos = (500, 100)  # 400px away
    sensor_readings = np.array([150, 150, 50, 150, 150])  # Front sensor blocked

    target = planner.get_target(robot_pos, robot_angle, goal_pos, sensor_readings)
    assert planner.current_waypoint is not None, "Should create waypoint for far goal with blocked path"
    assert target != (500, 100), "Should return waypoint, not goal"
    print(f"  ✓ Test 2 passed: Waypoint created at {target}")

    # Test 3: Waypoint should be cleared when reached
    planner.reset()
    robot_pos = (100, 100)
    robot_angle = 0.0
    goal_pos = (500, 100)
    sensor_readings = np.array([150, 150, 50, 150, 150])

    # Create waypoint
    target = planner.get_target(robot_pos, robot_angle, goal_pos, sensor_readings)
    waypoint = planner.current_waypoint
    assert waypoint is not None, "Waypoint should be created"

    # Move robot near waypoint
    robot_pos_near = (waypoint[0] + 20, waypoint[1] + 20)
    sensor_readings_clear = np.array([150, 150, 150, 150, 150])
    target2 = planner.get_target(robot_pos_near, robot_angle, goal_pos, sensor_readings_clear)

    assert planner.current_waypoint is None, "Waypoint should be cleared when reached"
    assert target2 == (500, 100), "Should return goal after waypoint reached"
    print("  ✓ Test 3 passed: Waypoint cleared when reached")

    # Test 4: Test should_use_waypoint logic
    planner.reset()

    # Far goal, clear path - should not use waypoint
    should_use = planner.should_use_waypoint(
        (100, 100),
        (600, 100),
        np.array([150, 150, 150, 150, 150])
    )
    assert not should_use, "Should not use waypoint with clear path"

    # Far goal, blocked path - should use waypoint
    should_use = planner.should_use_waypoint(
        (100, 100),
        (600, 100),
        np.array([150, 150, 80, 150, 150])
    )
    assert should_use, "Should use waypoint with blocked path"

    # Close goal, blocked path - should not use waypoint
    should_use = planner.should_use_waypoint(
        (100, 100),
        (250, 100),
        np.array([150, 150, 50, 150, 150])
    )
    assert not should_use, "Should not use waypoint for close goal even if blocked"
    print("  ✓ Test 4 passed: should_use_waypoint logic correct")

    # Test 5: Test generate_waypoint creates reasonable waypoints
    planner.reset()
    robot_pos = (100, 100)
    robot_angle = 0.0  # Facing right
    goal_pos = (500, 100)

    # Front blocked, left sensor open
    sensor_readings = np.array([150, 150, 50, 50, 50])  # Left sensors more open
    waypoint = planner.generate_waypoint(robot_pos, robot_angle, goal_pos, sensor_readings)

    # Waypoint should be roughly in the direction of open sensors
    assert isinstance(waypoint, tuple), "Waypoint should be tuple"
    assert len(waypoint) == 2, "Waypoint should have x,y coordinates"

    # Distance from robot should be approximately waypoint_distance
    waypoint_arr = np.array(waypoint)
    robot_arr = np.array(robot_pos)
    distance = np.linalg.norm(waypoint_arr - robot_arr)
    assert 100 < distance < 140, f"Waypoint distance {distance} should be near waypoint_distance (120)"
    print(f"  ✓ Test 5 passed: Waypoint generated at {waypoint}, distance={distance:.1f}")

    print("\nAll tests passed! ✓")


if __name__ == '__main__':
    test_waypoint_planner()
