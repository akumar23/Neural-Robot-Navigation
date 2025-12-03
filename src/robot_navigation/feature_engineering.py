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

Enhanced features (20D total):
- 5 raw sensors
- 6 spatial derived features
- 2 goal-relative features (direction, distance)
- 4 temporal features (velocity, action momentum/variance)
- 2 spatial-goal features (alignment, urgency)
- 1 action
"""

import numpy as np
from collections import deque


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


def compute_goal_relative_features(robot_pos, robot_angle, goal_pos, max_range=150.0):
    """
    Compute goal-relative features from robot and goal positions.

    Args:
        robot_pos: Robot position (Vec2d or tuple (x, y))
        robot_angle: Robot orientation angle in radians
        goal_pos: Goal position (Vec2d or tuple (x, y))
        max_range: Maximum sensor range for normalization (default: 150.0)

    Returns:
        Array of goal-relative features (2 features):
        - goal_direction: Normalized angle to goal in robot's reference frame [-1, 1]
        - goal_distance: Normalized Euclidean distance to goal [0, inf)
    """
    # Handle both Vec2d and tuple inputs
    if hasattr(robot_pos, 'x'):
        robot_x, robot_y = robot_pos.x, robot_pos.y
    else:
        robot_x, robot_y = robot_pos[0], robot_pos[1]

    if hasattr(goal_pos, 'x'):
        goal_x, goal_y = goal_pos.x, goal_pos.y
    else:
        goal_x, goal_y = goal_pos[0], goal_pos[1]

    # Calculate angle to goal in world frame
    angle_to_goal = np.arctan2(goal_y - robot_y, goal_x - robot_x)

    # Convert to robot's reference frame and normalize to [-1, 1]
    relative_angle = angle_to_goal - robot_angle
    # Normalize to [-pi, pi]
    relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    # Normalize to [-1, 1]
    goal_direction = relative_angle / np.pi

    # Calculate normalized distance to goal
    distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
    goal_distance = distance / max_range

    return np.array([goal_direction, goal_distance], dtype=np.float32)


def compute_temporal_features(velocity, action_history, max_velocity=10.0):
    """
    Compute temporal features from velocity and action history.

    Args:
        velocity: Robot velocity (Vec2d or tuple (vx, vy))
        action_history: Deque or list of recent actions (last 5)
        max_velocity: Maximum velocity for normalization (default: 10.0)

    Returns:
        Array of temporal features (4 features):
        - velocity_x: Normalized x-component of velocity
        - velocity_y: Normalized y-component of velocity
        - action_momentum: Mean of recent actions (normalized)
        - action_variance: Standard deviation of recent actions (normalized)
    """
    # Handle both Vec2d and tuple inputs
    if hasattr(velocity, 'x'):
        vx, vy = velocity.x, velocity.y
    else:
        vx, vy = velocity[0], velocity[1]

    # Normalize velocity components
    velocity_x = vx / max_velocity
    velocity_y = vy / max_velocity

    # Handle action history
    if len(action_history) == 0:
        # No history yet - use zeros
        action_momentum = 0.0
        action_variance = 0.0
    else:
        actions_array = np.array(list(action_history), dtype=np.float32)
        # Normalize by max action value (5.0)
        action_momentum = np.mean(actions_array) / 5.0
        action_variance = np.std(actions_array) / 5.0

    return np.array([velocity_x, velocity_y, action_momentum, action_variance], dtype=np.float32)


def compute_spatial_goal_features(sensor_readings, goal_direction, front_idx=2, max_range=150.0):
    """
    Compute spatial-goal features combining sensor readings with goal information.

    Args:
        sensor_readings: Array of 5 sensor distances
        goal_direction: Normalized angle to goal in robot's reference frame [-1, 1]
        front_idx: Index of front sensor (default: 2 for center sensor)
        max_range: Maximum sensor range for normalization (default: 150.0)

    Returns:
        Array of spatial-goal features (2 features):
        - front_goal_alignment: How well front sensor aligns with goal direction
        - escape_urgency: Urgency to escape based on closest obstacle
    """
    sensors = np.array(sensor_readings, dtype=np.float32)
    epsilon = 1e-6  # Prevent division by zero

    # Front-goal alignment: higher when front sensor is clear AND aligned with goal
    # goal_direction is in [-1, 1], where 0 means goal is straight ahead
    alignment_factor = 1.0 - abs(goal_direction)  # 1.0 when aligned, 0.0 when perpendicular
    front_clearance_normalized = sensors[front_idx] / max_range
    front_goal_alignment = alignment_factor * front_clearance_normalized

    # Escape urgency: inverse of minimum sensor (closer obstacles = higher urgency)
    min_sensor = np.min(sensors)
    escape_urgency = 1.0 / (min_sensor + epsilon)

    return np.array([front_goal_alignment, escape_urgency], dtype=np.float32)


def engineer_features(sensor_readings, action, robot_pos=None, robot_angle=None,
                     goal_pos=None, velocity=None, action_history=None,
                     max_sensor_range=150.0, max_velocity=10.0):
    """
    Create full feature vector with raw sensors + derived features + action.

    Supports both legacy mode (12D) and enhanced mode (20D):
    - Legacy (12D): [5 sensors] + [6 spatial derived] + [action]
    - Enhanced (20D): [5 sensors] + [6 spatial] + [2 goal] + [4 temporal] + [2 spatial-goal] + [action]

    Args:
        sensor_readings: Array of 5 raw sensor readings
        action: Steering action (-5 to +5)
        robot_pos: Robot position (Vec2d or tuple) - optional for enhanced mode
        robot_angle: Robot orientation in radians - optional for enhanced mode
        goal_pos: Goal position (Vec2d or tuple) - optional for enhanced mode
        velocity: Robot velocity (Vec2d or tuple) - optional for enhanced mode
        action_history: Deque/list of recent actions - optional for enhanced mode
        max_sensor_range: Maximum sensor range for normalization (default: 150.0)
        max_velocity: Maximum velocity for normalization (default: 10.0)

    Returns:
        Feature vector of length 12 (legacy) or 20 (enhanced):
        Legacy: [5 raw sensors] + [6 derived features] + [action]
        Enhanced: [5 sensors] + [6 spatial] + [2 goal] + [4 temporal] + [2 spatial-goal] + [action]
    """
    # Always compute basic spatial derived features
    spatial_derived = compute_derived_features(sensor_readings)

    # Check if enhanced features are requested
    enhanced_mode = (robot_pos is not None and robot_angle is not None and
                    goal_pos is not None and velocity is not None and
                    action_history is not None)

    if enhanced_mode:
        # Compute goal-relative features
        goal_features = compute_goal_relative_features(
            robot_pos, robot_angle, goal_pos, max_sensor_range
        )

        # Compute temporal features
        temporal_features = compute_temporal_features(
            velocity, action_history, max_velocity
        )

        # Compute spatial-goal features
        # Extract goal_direction from goal_features for spatial-goal computation
        goal_direction = goal_features[0]
        spatial_goal_features = compute_spatial_goal_features(
            sensor_readings, goal_direction, front_idx=2, max_range=max_sensor_range
        )

        # Concatenate all features: [5 sensors] + [6 spatial] + [2 goal] + [4 temporal] + [2 spatial-goal] + [action] = 20
        features = np.concatenate([
            sensor_readings,        # 5 features
            spatial_derived,        # 6 features
            goal_features,          # 2 features
            temporal_features,      # 4 features
            spatial_goal_features,  # 2 features
            [action]                # 1 feature
        ])
    else:
        # Legacy mode: [5 sensors] + [6 spatial derived] + [action] = 12
        features = np.concatenate([
            sensor_readings,
            spatial_derived,
            [action]
        ])

    return features.astype(np.float32)


def get_feature_names(enhanced=False):
    """
    Return names of all features for documentation/debugging.

    Args:
        enhanced: If True, return names for 20D enhanced features,
                 otherwise return names for 12D legacy features

    Returns:
        List of feature names
    """
    base_names = [
        # Raw sensors (5)
        'sensor_left_far',      # 66°
        'sensor_left_near',     # 33°
        'sensor_front',         # 0°
        'sensor_right_near',    # -33°
        'sensor_right_far',     # -66°
        # Spatial derived features (6)
        'min_sensor',
        'sensor_variance',
        'front_to_side_ratio',
        'left_right_asymmetry',
        'front_clearance',
        'side_gradient',
    ]

    if enhanced:
        # Add enhanced features
        enhanced_names = base_names + [
            # Goal-relative features (2)
            'goal_direction',
            'goal_distance',
            # Temporal features (4)
            'velocity_x',
            'velocity_y',
            'action_momentum',
            'action_variance',
            # Spatial-goal features (2)
            'front_goal_alignment',
            'escape_urgency',
            # Action (1)
            'action'
        ]
        return enhanced_names
    else:
        # Legacy mode - just add action
        return base_names + ['action']
