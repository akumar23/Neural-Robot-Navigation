"""
Test script to demonstrate OpennessScorer functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.robot_navigation.openness_scorer import OpennessScorer
import numpy as np


def test_openness_scorer():
    """Test the OpennessScorer with various scenarios."""

    scorer = OpennessScorer(max_sensor_range=150)

    print("=" * 70)
    print("OPENNESS SCORER TEST")
    print("=" * 70)
    print("\nSensor Layout:")
    print("  [0]: 66° left")
    print("  [1]: 33° left")
    print("  [2]: 0° (front)")
    print("  [3]: -33° right")
    print("  [4]: -66° right")
    print()

    # Test 1: Open corridor ahead
    print("-" * 70)
    print("Test 1: Open corridor ahead")
    print("-" * 70)
    sensor_readings = [50, 70, 150, 70, 50]  # Most open straight ahead
    print(f"Sensor readings: {sensor_readings}")

    actions = list(range(-5, 6))
    scores = scorer.score_all_actions(actions, sensor_readings)

    print("\nAction scores:")
    for action in sorted(scores.keys()):
        print(f"  Action {action:2d}: {scores[action]:.3f}")

    best_action = scorer.get_best_open_action(actions, sensor_readings)
    print(f"\nBest action: {best_action} (score: {scores[best_action]:.3f})")
    print("Expected: 0 (straight) should have highest score")

    # Test 2: Open space to the left
    print("\n" + "-" * 70)
    print("Test 2: Open space to the left")
    print("-" * 70)
    sensor_readings = [150, 140, 80, 40, 30]  # More open on left
    print(f"Sensor readings: {sensor_readings}")

    scores = scorer.score_all_actions(actions, sensor_readings)

    print("\nAction scores:")
    for action in sorted(scores.keys()):
        print(f"  Action {action:2d}: {scores[action]:.3f}")

    best_action = scorer.get_best_open_action(actions, sensor_readings)
    print(f"\nBest action: {best_action} (score: {scores[best_action]:.3f})")
    print("Expected: Positive action (left turn) should have highest score")

    # Test 3: Open space to the right
    print("\n" + "-" * 70)
    print("Test 3: Open space to the right")
    print("-" * 70)
    sensor_readings = [30, 40, 80, 140, 150]  # More open on right
    print(f"Sensor readings: {sensor_readings}")

    scores = scorer.score_all_actions(actions, sensor_readings)

    print("\nAction scores:")
    for action in sorted(scores.keys()):
        print(f"  Action {action:2d}: {scores[action]:.3f}")

    best_action = scorer.get_best_open_action(actions, sensor_readings)
    print(f"\nBest action: {best_action} (score: {scores[best_action]:.3f})")
    print("Expected: Negative action (right turn) should have highest score")

    # Test 4: Very confined space (all sensors close)
    print("\n" + "-" * 70)
    print("Test 4: Very confined space")
    print("-" * 70)
    sensor_readings = [25, 30, 35, 30, 25]  # All sensors detecting close walls
    print(f"Sensor readings: {sensor_readings}")

    scores = scorer.score_all_actions(actions, sensor_readings)

    print("\nAction scores:")
    for action in sorted(scores.keys()):
        print(f"  Action {action:2d}: {scores[action]:.3f}")

    best_action = scorer.get_best_open_action(actions, sensor_readings)
    print(f"\nBest action: {best_action} (score: {scores[best_action]:.3f})")
    print("Expected: Action 0 should be slightly preferred (most open directly ahead)")

    # Test 5: Partially blocked left, open right
    print("\n" + "-" * 70)
    print("Test 5: Comparing strong vs weak turns")
    print("-" * 70)
    sensor_readings = [40, 60, 100, 130, 140]  # Right is more open
    print(f"Sensor readings: {sensor_readings}")

    # Compare action -2 vs -5 (weak vs strong right turn)
    score_weak = scorer.score_action(-2, sensor_readings)
    score_strong = scorer.score_action(-5, sensor_readings)

    print(f"\nAction -2 (weak right): {score_weak:.3f}")
    print(f"Action -5 (strong right): {score_strong:.3f}")
    print("Expected: Strong turn should score higher when far-right sensor is most open")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_openness_scorer()
