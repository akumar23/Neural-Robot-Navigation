"""
Wall-following behavior for robot navigation.

Provides systematic wall-following escape behavior when the robot gets stuck in corners
or tight spaces. The robot follows either the left or right wall at a target distance,
using sensor readings to maintain wall distance and navigate around obstacles.
"""

import numpy as np


class WallFollower:
    """
    Wall-following behavior for escaping stuck situations.

    The robot can follow either left or right wall at a target distance,
    using sensor readings to maintain wall distance and navigate around obstacles.

    Robot Sensor Configuration:
        - 5 sensors at angles: [66°, 33°, 0°, -33°, -66°] (indices 0, 1, 2, 3, 4)
        - Index 0, 1 = left side sensors
        - Index 2 = front sensor
        - Index 3, 4 = right side sensors
        - Max range: 150px

    Action Space:
        - Actions are integers -5 to +5
        - Positive actions = turn left
        - Negative actions = turn right
        - Action 0 = straight
    """

    def __init__(self, target_distance=70, max_follow_steps=50):
        """
        Initialize the wall follower.

        Args:
            target_distance: Desired distance from wall in pixels (default: 70)
            max_follow_steps: Maximum steps to follow wall before deactivating (default: 50)
        """
        self.target_distance = target_distance
        self.max_follow_steps = max_follow_steps

        # State variables
        self.active = False
        self.preferred_side = None  # 'left' or 'right'
        self.steps_following = 0
        self.last_action = 0  # Track last action to reduce oscillation

    def should_activate(self, sensor_readings, stuck_counter):
        """
        Determine if wall-following should engage.

        Activate if:
        - stuck_counter exceeds threshold (robot is stuck)
        - OR multiple sensors detect walls below threshold (robot is in tight space)

        Args:
            sensor_readings: Array of 5 sensor readings in pixels
            stuck_counter: Number of consecutive iterations robot hasn't moved

        Returns:
            bool: True if wall-following should activate
        """
        from .navigation_config import NavigationConfig
        config = NavigationConfig()

        # Check if robot is stuck
        if stuck_counter > config.stuck_counter_wall_follow:
            return True

        # Check if multiple sensors detect close walls (tight space)
        close_walls = np.sum(np.array(sensor_readings) < config.wall_follower_close_wall_threshold)
        if close_walls >= config.wall_follower_close_wall_count:
            return True

        return False

    def get_wall_following_action(self, sensor_readings):
        """
        Calculate action to follow wall based on sensor readings.

        For right wall following:
        - Use sensor index 3 (-33°) as primary wall sensor
        - Use sensor index 4 (-66°) as secondary wall sensor
        - If front blocked (sensor 2 < 100): turn left sharply
        - If too close to wall (< target - 30): turn left
        - If too far from wall (> target + 30): turn right
        - Otherwise: go straight to reduce oscillation

        For left wall following: mirror the logic using sensors 1 and 0

        Args:
            sensor_readings: Array of 5 sensor readings in pixels

        Returns:
            int: Action from -5 to +5
        """
        # Get individual sensor readings
        left_far = sensor_readings[0]   # 66° left
        left_near = sensor_readings[1]  # 33° left
        front = sensor_readings[2]      # 0° straight
        right_near = sensor_readings[3] # -33° right
        right_far = sensor_readings[4]  # -66° right

        from .navigation_config import NavigationConfig
        config = NavigationConfig()

        action = 0  # Default to straight

        if self.preferred_side == 'right':
            # Following right wall - use right sensors
            wall_sensor = right_near  # Primary wall sensor

            # Front obstacle check - turn left sharply
            if front < config.wall_follower_front_blocked_threshold:
                action = 5  # Sharp left turn
            elif front < config.wall_follower_front_moderate_threshold:
                action = 3  # Moderate left turn
            # Wall distance control - use wider band to reduce oscillation
            elif wall_sensor < self.target_distance - config.wall_follower_distance_band:
                # Too close to wall - turn left (away from wall)
                action = 3
            elif wall_sensor > self.target_distance + config.wall_follower_distance_band:
                # Too far from wall - turn right (toward wall)
                if wall_sensor > config.wall_follower_far_threshold:
                    action = -3  # Turn more sharply toward wall
                else:
                    action = -2  # Gentle turn toward wall
            else:
                # In good range - prefer straight with slight bias
                # Only adjust if significantly off-center
                if wall_sensor < self.target_distance - config.wall_follower_distance_fine_adjust:
                    action = 1  # Slight left
                elif wall_sensor > self.target_distance + config.wall_follower_distance_fine_adjust:
                    action = -1  # Slight right
                else:
                    action = 0  # Go straight - reduces oscillation

        else:  # preferred_side == 'left'
            # Following left wall - use left sensors (mirror of right wall logic)
            wall_sensor = left_near  # Primary wall sensor

            # Front obstacle check - turn right sharply
            if front < config.wall_follower_front_blocked_threshold:
                action = -5  # Sharp right turn
            elif front < config.wall_follower_front_moderate_threshold:
                action = -3  # Moderate right turn
            # Wall distance control - use wider band to reduce oscillation
            elif wall_sensor < self.target_distance - config.wall_follower_distance_band:
                # Too close to wall - turn right (away from wall)
                action = -3
            elif wall_sensor > self.target_distance + config.wall_follower_distance_band:
                # Too far from wall - turn left (toward wall)
                if wall_sensor > config.wall_follower_far_threshold:
                    action = 3  # Turn more sharply toward wall
                else:
                    action = 2  # Gentle turn toward wall
            else:
                # In good range - prefer straight with slight bias
                # Only adjust if significantly off-center
                if wall_sensor < self.target_distance - config.wall_follower_distance_fine_adjust:
                    action = -1  # Slight right
                elif wall_sensor > self.target_distance + config.wall_follower_distance_fine_adjust:
                    action = 1  # Slight left
                else:
                    action = 0  # Go straight - reduces oscillation

        # Anti-oscillation: avoid rapid sign changes unless necessary
        if self.last_action != 0 and action != 0:
            # If switching direction, only allow if it's a strong correction
            if (self.last_action > 0 and action < 0) or (self.last_action < 0 and action > 0):
                if abs(action) < 3:
                    # Weak correction in opposite direction - go straight instead
                    action = 0

        self.last_action = action
        return action

    def update(self, sensor_readings, stuck_counter):
        """
        Update wall-following state each iteration.

        - If not active: check should_activate(), if True activate and randomly choose side
        - If active: increment steps, check deactivation conditions (open space or max steps)

        Deactivation conditions:
        - All sensors > 100px (open space found - lowered threshold)
        - steps_following >= max_follow_steps (timeout)

        Args:
            sensor_readings: Array of 5 sensor readings in pixels
            stuck_counter: Number of consecutive iterations robot hasn't moved
        """
        if not self.active:
            # Check if we should activate
            if self.should_activate(sensor_readings, stuck_counter):
                self.active = True
                self.steps_following = 0
                self.last_action = 0  # Reset action tracking
                # Randomly choose left or right wall to follow
                self.preferred_side = np.random.choice(['left', 'right'])
        else:
            # Already active - increment counter and check deactivation
            from .navigation_config import NavigationConfig
            config = NavigationConfig()

            self.steps_following += 1

            # Check for open space
            all_sensors_clear = np.all(np.array(sensor_readings) > config.wall_follower_deactivate_threshold)
            if all_sensors_clear:
                self.active = False
                self.preferred_side = None
                self.steps_following = 0
                self.last_action = 0

            # Check for timeout
            elif self.steps_following >= self.max_follow_steps:
                self.active = False
                self.preferred_side = None
                self.steps_following = 0
                self.last_action = 0

    def reset(self):
        """
        Reset wall following state - call on goal reach.
        """
        self.active = False
        self.preferred_side = None
        self.steps_following = 0
        self.last_action = 0
