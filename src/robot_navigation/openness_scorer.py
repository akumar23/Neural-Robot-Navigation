"""
Openness-Based Action Scoring Module

Evaluates actions based on how much open space they lead toward.
Uses weighted sensor readings based on action direction.
"""

import numpy as np


class OpennessScorer:
    """Scores actions based on how much open space they lead toward."""

    def __init__(self, max_sensor_range=150):
        """
        Initialize the openness scorer.

        Args:
            max_sensor_range: Maximum sensor reading in pixels (default: 150)
        """
        self.max_sensor_range = max_sensor_range

        # Sensor angles: [66°, 33°, 0°, -33°, -66°]
        # Indices: [0=left-far, 1=left-near, 2=front, 3=right-near, 4=right-far]

    def _get_weights_for_action(self, action):
        """
        Get sensor weights based on action direction.

        For left turns (positive actions): weight left sensors more
        For right turns (negative actions): weight right sensors more
        For straight (action 0): weight front sensor most
        Stronger turns get stronger side sensor weighting

        Args:
            action: Integer from -5 to +5

        Returns:
            np.array: Weights for sensors [0, 1, 2, 3, 4]
        """
        from .navigation_config import NavigationConfig
        config = NavigationConfig()

        # Base weights (all sensors contribute)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        if action == 0:
            # Straight: heavily weight front sensor
            weights = np.array([0.3, 0.5, 2.0, 0.5, 0.3])

        elif action > 0:
            # Turning left (positive actions)
            # Weight left sensors (indices 0, 1) more heavily
            # Stronger the turn, more we weight the far-left sensor
            turn_strength = action / 5.0  # Normalize to 0-1

            weights[0] = 1.5 + turn_strength * 1.0  # Left-far sensor (66°)
            weights[1] = 1.3 + turn_strength * 0.5  # Left-near sensor (33°)
            weights[2] = 1.0 - turn_strength * 0.3  # Front sensor
            weights[3] = 0.7 - turn_strength * 0.2  # Right-near sensor
            weights[4] = 0.5 - turn_strength * 0.3  # Right-far sensor

        else:  # action < 0
            # Turning right (negative actions)
            # Weight right sensors (indices 3, 4) more heavily
            turn_strength = abs(action) / 5.0  # Normalize to 0-1

            weights[0] = 0.5 - turn_strength * 0.3  # Left-far sensor
            weights[1] = 0.7 - turn_strength * 0.2  # Left-near sensor
            weights[2] = 1.0 - turn_strength * 0.3  # Front sensor
            weights[3] = 1.3 + turn_strength * 0.5  # Right-near sensor (33°)
            weights[4] = 1.5 + turn_strength * 1.0  # Right-far sensor (66°)

        # Ensure no negative weights
        weights = np.maximum(weights, config.openness_min_weight)

        return weights

    def score_action(self, action, sensor_readings):
        """
        Score a single action based on sensor readings.

        Higher sensor readings = more open space = higher score
        Weights are applied based on action direction

        Args:
            action: Integer from -5 to +5
            sensor_readings: Array of 5 sensor readings [66°, 33°, 0°, -33°, -66°]

        Returns:
            float: Openness score (0.0 to 1.0, higher = more open)
        """
        # Validate inputs
        if len(sensor_readings) != 5:
            raise ValueError(f"Expected 5 sensor readings, got {len(sensor_readings)}")

        # Get weights for this action
        weights = self._get_weights_for_action(action)

        # Normalize sensor readings to 0-1 range
        normalized_sensors = np.array(sensor_readings) / self.max_sensor_range
        normalized_sensors = np.clip(normalized_sensors, 0.0, 1.0)

        # Calculate weighted score
        weighted_scores = normalized_sensors * weights
        total_score = np.sum(weighted_scores)
        total_weight = np.sum(weights)

        # Normalize to 0-1 range
        openness_score = total_score / total_weight

        return float(openness_score)

    def score_all_actions(self, available_actions, sensor_readings):
        """
        Score all available actions.

        Args:
            available_actions: List of safe actions
            sensor_readings: Array of 5 sensor readings

        Returns:
            dict: {action: openness_score}
        """
        scores = {}
        for action in available_actions:
            scores[action] = self.score_action(action, sensor_readings)
        return scores

    def get_best_open_action(self, available_actions, sensor_readings):
        """
        Get the action leading to most open space.

        Args:
            available_actions: List of safe actions
            sensor_readings: Array of 5 sensor readings

        Returns:
            int: Best action, or 0 if no actions available
        """
        if not available_actions:
            return 0

        scores = self.score_all_actions(available_actions, sensor_readings)
        best_action = max(scores.items(), key=lambda x: x[1])[0]

        return best_action
