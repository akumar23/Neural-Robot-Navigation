"""
Quick test script to verify SpatialMemory class functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.robot_navigation.spatial_memory import SpatialMemory
import numpy as np


def test_spatial_memory():
    """Test the SpatialMemory class."""
    print("Testing SpatialMemory class...")
    print("=" * 60)

    # Initialize spatial memory
    sm = SpatialMemory(grid_size=30, decay_rate=0.95, max_history=50)
    print("✓ Initialization successful")

    # Test 1: Add positions
    print("\nTest 1: Adding positions")
    sm.add_position((100, 100))
    sm.add_position((105, 102))  # Should be in same grid cell
    sm.add_position((140, 140))  # Different grid cell
    print(f"  Visit counts: {sm.visit_counts}")
    print(f"  Position history length: {len(sm.position_history)}")
    assert len(sm.position_history) == 3, "Should have 3 positions in history"
    print("✓ Adding positions works")

    # Test 2: Grid conversion
    print("\nTest 2: Grid cell conversion")
    grid1 = sm._position_to_grid((100, 100))
    grid2 = sm._position_to_grid((105, 102))
    grid3 = sm._position_to_grid((140, 140))
    print(f"  (100, 100) -> {grid1}")
    print(f"  (105, 102) -> {grid2}")
    print(f"  (140, 140) -> {grid3}")
    assert grid1 == grid2, "Close positions should map to same grid cell"
    assert grid1 != grid3, "Far positions should map to different grid cells"
    print("✓ Grid conversion works")

    # Test 3: Repulsion scores
    print("\nTest 3: Repulsion scores")
    score1 = sm.get_repulsion_score((100, 100))
    score2 = sm.get_repulsion_score((140, 140))
    score3 = sm.get_repulsion_score((500, 500))  # Never visited
    print(f"  Repulsion at (100, 100): {score1}")
    print(f"  Repulsion at (140, 140): {score2}")
    print(f"  Repulsion at (500, 500): {score3}")
    assert score1 > score2, "More visited cell should have higher repulsion"
    assert score3 == 0.0, "Never visited cell should have 0 repulsion"
    print("✓ Repulsion scores work")

    # Test 4: Decay
    print("\nTest 4: Visit decay")
    initial_counts = dict(sm.visit_counts)
    print(f"  Before decay: {initial_counts}")
    sm.decay_visits()
    print(f"  After decay: {sm.visit_counts}")
    for cell in sm.visit_counts:
        assert sm.visit_counts[cell] < initial_counts[cell], "Counts should decrease after decay"
    print("✓ Decay works")

    # Test 5: Oscillation detection
    print("\nTest 5: Oscillation detection")
    # Add many positions in a small area
    for i in range(20):
        sm.add_position((100 + np.random.uniform(-10, 10), 100 + np.random.uniform(-10, 10)))

    is_oscillating = sm.detect_oscillation((105, 105), window=15, threshold=30.0)
    print(f"  Oscillation detected: {is_oscillating}")
    assert is_oscillating, "Should detect oscillation in small area"

    # Test with position far away
    is_oscillating_far = sm.detect_oscillation((500, 500), window=15, threshold=30.0)
    print(f"  Oscillation at far position: {is_oscillating_far}")
    assert not is_oscillating_far, "Should not detect oscillation at far position"
    print("✓ Oscillation detection works")

    # Test 6: Reset
    print("\nTest 6: Reset")
    sm.reset()
    print(f"  Visit counts after reset: {sm.visit_counts}")
    print(f"  Position history after reset: {len(sm.position_history)}")
    assert len(sm.visit_counts) == 0, "Visit counts should be empty after reset"
    assert len(sm.position_history) == 0, "Position history should be empty after reset"
    print("✓ Reset works")

    # Test 7: Handle Vec2d-like objects
    print("\nTest 7: Vec2d-like object handling")

    class FakeVec2d:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    vec_pos = FakeVec2d(200, 200)
    sm.add_position(vec_pos)
    print(f"  Added Vec2d position: ({vec_pos.x}, {vec_pos.y})")
    print(f"  Position history: {sm.position_history}")
    assert len(sm.position_history) == 1, "Should handle Vec2d objects"
    print("✓ Vec2d handling works")

    # Test 8: Max history limit
    print("\nTest 8: Max history limit")
    sm_small = SpatialMemory(grid_size=30, decay_rate=0.95, max_history=10)
    for i in range(20):
        sm_small.add_position((i * 10, i * 10))

    print(f"  Added 20 positions with max_history=10")
    print(f"  Position history length: {len(sm_small.position_history)}")
    assert len(sm_small.position_history) == 10, "History should be limited to max_history"
    print("✓ Max history limit works")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == '__main__':
    test_spatial_memory()
