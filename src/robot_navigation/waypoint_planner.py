"""
Waypoint Planning module for robot navigation.

Breaks down long-distance goals into intermediate targets to navigate around obstacles.
"""

import numpy as np
import numpy.linalg as la
from .helper import vector, PI


class WaypointPlanner:
    """Generate intermediate waypoints to navigate around obstacles."""

    def __init__(self, waypoint_distance=120, waypoint_reached_threshold=50,
                 screen_width=1080, screen_height=900, boundary_margin=50):
        """
        Args:
            waypoint_distance: How far ahead to place waypoints (default: 120px)
            waypoint_reached_threshold: Distance to consider waypoint reached (default: 50px)
            screen_width: Simulation screen width (default: 1080px)
            screen_height: Simulation screen height (default: 900px)
            boundary_margin: Minimum distance from walls (default: 50px)
        """
        self.waypoint_distance = waypoint_distance
        self.waypoint_reached_threshold = waypoint_reached_threshold
        self.current_waypoint = None

        # Simulation bounds
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.boundary_margin = boundary_margin

        # Sensor configuration (matching robot's setup)
        self.sensor_angles = [66, 33, 0, -33, -66]  # degrees
        self.sensor_max_range = 150

    def _clamp_to_bounds(self, x, y):
        """
        Clamp waypoint coordinates to stay within simulation bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            tuple (x, y) clamped to valid bounds
        """
        clamped_x = np.clip(x, self.boundary_margin, self.screen_width - self.boundary_margin)
        clamped_y = np.clip(y, self.boundary_margin, self.screen_height - self.boundary_margin)
        return (float(clamped_x), float(clamped_y))

    def _is_valid_waypoint(self, waypoint):
        """
        Check if waypoint is within valid simulation bounds.

        Args:
            waypoint: tuple (x, y)

        Returns:
            bool: True if waypoint is within bounds
        """
        x, y = waypoint
        return (self.boundary_margin <= x <= self.screen_width - self.boundary_margin and
                self.boundary_margin <= y <= self.screen_height - self.boundary_margin)

    def should_use_waypoint(self, robot_pos, goal_pos, sensor_readings):
        """
        Determine if we need an intermediate waypoint.

        Use waypoint if:
        - Goal is far away AND
        - Direct path seems blocked

        Args:
            robot_pos: tuple or Vec2d (x, y)
            goal_pos: tuple or Vec2d (x, y)
            sensor_readings: numpy array of 5 sensor values

        Returns: bool
        """
        from .navigation_config import NavigationConfig
        config = NavigationConfig()

        # Convert to numpy arrays for distance calculation
        robot_pos_arr = np.array([robot_pos[0], robot_pos[1]])
        goal_pos_arr = np.array([goal_pos[0], goal_pos[1]])

        # Calculate distance to goal
        distance_to_goal = la.norm(goal_pos_arr - robot_pos_arr)

        # Check if goal is far away
        if distance_to_goal <= config.waypoint_min_goal_distance:
            return False

        # Check if direct path seems blocked
        # sensor_readings[2] is the front sensor (0 degrees)
        front_sensor = sensor_readings[2]
        left_sensor = sensor_readings[1]   # 33 degrees
        right_sensor = sensor_readings[3]  # -33 degrees

        # Path is blocked if:
        # 1. Front sensor detects obstacle close
        # 2. OR front and one side are both detecting obstacles
        front_blocked = front_sensor < config.waypoint_front_blocked_threshold
        sides_blocked = ((front_sensor + left_sensor < config.waypoint_combined_sensor_threshold) or
                        (front_sensor + right_sensor < config.waypoint_combined_sensor_threshold))

        return front_blocked or sides_blocked

    def generate_waypoint(self, robot_pos, robot_angle, goal_pos, sensor_readings):
        """
        Generate a waypoint toward the most open direction that's roughly toward goal.

        Strategy:
        - Find the sensor with most open space
        - Calculate waypoint position in that direction
        - Bias toward sensors that are closer to goal direction

        Args:
            robot_pos: tuple or Vec2d (x, y)
            robot_angle: float (radians)
            goal_pos: tuple or Vec2d (x, y)
            sensor_readings: numpy array of 5 sensor values

        Returns: tuple (x, y) waypoint position
        """
        # Convert positions to numpy arrays
        robot_pos_arr = np.array([robot_pos[0], robot_pos[1]])
        goal_pos_arr = np.array([goal_pos[0], goal_pos[1]])

        # Calculate direction to goal
        goal_vector = goal_pos_arr - robot_pos_arr
        goal_angle = np.arctan2(goal_vector[1], goal_vector[0])

        # Calculate angle difference for each sensor relative to goal
        best_score = -np.inf
        best_sensor_idx = 2  # Default to front sensor

        for i, sensor_angle_deg in enumerate(self.sensor_angles):
            # Calculate absolute angle of this sensor
            sensor_angle_rad = robot_angle + np.radians(sensor_angle_deg)

            # Calculate how much this sensor is aligned with goal
            angle_diff = goal_angle - sensor_angle_rad
            # Normalize to [-pi, pi]
            angle_diff = (angle_diff + PI) % (2 * PI) - PI

            # Score = openness + alignment bonus
            # Openness: higher sensor reading is better
            from .navigation_config import NavigationConfig
            config = NavigationConfig()
            openness_score = sensor_readings[i] / self.sensor_max_range

            # Alignment: prefer sensors pointing toward goal
            # cos(angle_diff) gives 1 when aligned, -1 when opposite
            alignment_score = np.cos(angle_diff)

            # Combined score (weight openness more heavily)
            score = (openness_score * config.waypoint_openness_weight +
                    alignment_score * config.waypoint_alignment_weight)

            if score > best_score:
                best_score = score
                best_sensor_idx = i

        # Generate waypoint in the direction of the best sensor
        best_sensor_angle_deg = self.sensor_angles[best_sensor_idx]
        waypoint_angle = robot_angle + np.radians(best_sensor_angle_deg)

        # Place waypoint at waypoint_distance in that direction
        waypoint_x = robot_pos_arr[0] + self.waypoint_distance * np.cos(waypoint_angle)
        waypoint_y = robot_pos_arr[1] + self.waypoint_distance * np.sin(waypoint_angle)

        # Clamp waypoint to stay within simulation bounds
        waypoint_x, waypoint_y = self._clamp_to_bounds(waypoint_x, waypoint_y)

        return (waypoint_x, waypoint_y)

    def get_target(self, robot_pos, robot_angle, goal_pos, sensor_readings):
        """
        Get the current navigation target (either waypoint or final goal).

        - If current_waypoint exists and not reached: return waypoint
        - If current_waypoint reached: clear it
        - If should_use_waypoint: generate new waypoint
        - Otherwise: return goal

        Args:
            robot_pos: tuple or Vec2d (x, y)
            robot_angle: float (radians)
            goal_pos: tuple or Vec2d (x, y)
            sensor_readings: numpy array of 5 sensor values

        Returns: tuple (x, y) target position
        """
        robot_pos_arr = np.array([robot_pos[0], robot_pos[1]])

        # If we have a current waypoint, validate and check if it's reached
        if self.current_waypoint is not None:
            # First check if waypoint is still valid (within bounds)
            if not self._is_valid_waypoint(self.current_waypoint):
                # Invalid waypoint, clear it and generate a new one
                self.current_waypoint = None
            else:
                waypoint_arr = np.array([self.current_waypoint[0], self.current_waypoint[1]])
                distance_to_waypoint = la.norm(waypoint_arr - robot_pos_arr)

                if distance_to_waypoint < self.waypoint_reached_threshold:
                    # Waypoint reached, clear it
                    self.current_waypoint = None
                else:
                    # Still heading toward waypoint
                    return self.current_waypoint

        # No current waypoint, check if we need one
        if self.should_use_waypoint(robot_pos, goal_pos, sensor_readings):
            # Generate new waypoint
            self.current_waypoint = self.generate_waypoint(
                robot_pos, robot_angle, goal_pos, sensor_readings
            )
            return self.current_waypoint

        # No waypoint needed, return goal directly
        return (goal_pos[0], goal_pos[1])

    def reset(self):
        """Clear current waypoint - call on goal reach."""
        self.current_waypoint = None
