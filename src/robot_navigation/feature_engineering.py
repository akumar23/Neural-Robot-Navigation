"""
Feature Engineering for Collision Detection.

Derives additional features from raw sensor readings to provide the neural network
with more informative spatial context about the environment geometry.

Raw sensors: 5 raycasting distances at angles [66°, 33°, 0°, -33°, -66°]
Derived features capture relationships between sensors that help identify:
- Corners vs open spaces
- Left/right asymmetry
- Proximity to closest obstacle
- Environment complexity
"""

import numpy as np


def compute_derived_features(sensor_readings):
    """
    Compute derived features from 5 raw sensor readings.

    Args:
        sensor_readings: Array of 5 sensor distances [left_far, left_near, front, right_near, right_far]
                        Sensors at angles [66°, 33°, 0°, -33°, -66°]

    Returns:
        Array of derived features (6 additional features):
        - min_sensor: Closest obstacle distance (critical for safety)
        - sensor_variance: Environment complexity (low=corridor, high=clutter)
        - front_to_side_ratio: Detect corners vs open space ahead
        - left_right_asymmetry: Spatial bias (positive=more space on left)
        - front_clearance: Inverse front distance (emphasizes close obstacles)
        - side_gradient: Rate of change across sensors (detect walls)
    """
    sensors = np.array(sensor_readings, dtype=np.float32)
    epsilon = 1e-6  # Prevent division by zero

    # 1. Minimum sensor reading - closest obstacle distance
    min_sensor = np.min(sensors)

    # 2. Sensor variance - environment complexity
    # Low variance = corridor/uniform, High variance = cluttered/complex
    sensor_variance = np.std(sensors)

    # 3. Front-to-side ratio - detect corners vs open space
    # High ratio = open ahead, Low ratio = blocked ahead but sides open
    front = sensors[2]  # Center sensor (0°)
    avg_sides = (sensors[0] + sensors[4]) / 2  # Average of far sides (±66°)
    front_to_side_ratio = front / (avg_sides + epsilon)

    # 4. Left-right asymmetry - spatial bias
    # Positive = more space on left, Negative = more space on right
    left_avg = np.mean(sensors[0:2])   # sensors at 66°, 33°
    right_avg = np.mean(sensors[3:5])  # sensors at -33°, -66°
    left_right_asymmetry = left_avg - right_avg

    # 5. Front clearance - inverse distance emphasizing close obstacles
    # Higher value = closer obstacle ahead
    front_clearance = 1.0 / (front + epsilon)

    # 6. Side gradient - rate of change across sensors
    # Detects walls and angular obstacles
    side_gradient = np.mean(np.abs(np.diff(sensors)))

    return np.array([
        min_sensor,
        sensor_variance,
        front_to_side_ratio,
        left_right_asymmetry,
        front_clearance,
        side_gradient
    ], dtype=np.float32)


def engineer_features(sensor_readings, action):
    """
    Create full feature vector with raw sensors + derived features + action.

    Args:
        sensor_readings: Array of 5 raw sensor readings
        action: Steering action (-5 to +5)

    Returns:
        Feature vector of length 12:
        [5 raw sensors] + [6 derived features] + [action]
    """
    derived = compute_derived_features(sensor_readings)

    # Concatenate: raw sensors (5) + derived features (6) + action (1) = 12
    features = np.concatenate([
        sensor_readings,
        derived,
        [action]
    ])

    return features.astype(np.float32)


def get_feature_names():
    """Return names of all features for documentation/debugging."""
    return [
        # Raw sensors
        'sensor_left_far',      # 66°
        'sensor_left_near',     # 33°
        'sensor_front',         # 0°
        'sensor_right_near',    # -33°
        'sensor_right_far',     # -66°
        # Derived features
        'min_sensor',
        'sensor_variance',
        'front_to_side_ratio',
        'left_right_asymmetry',
        'front_clearance',
        'side_gradient',
        # Action
        'action'
    ]
