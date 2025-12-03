"""
Navigation Configuration Module

Centralizes all configuration parameters for robot navigation system.
This module provides a dataclass that contains all tunable thresholds,
distances, timeouts, and other magic numbers used throughout the codebase.
"""

from dataclasses import dataclass


@dataclass
class NavigationConfig:
    """
    Centralized configuration for robot navigation system.

    All magic numbers and tunable parameters are defined here with clear names
    and documentation. This allows easy experimentation and prevents inconsistencies.
    """

    # ========== Collision Detection Thresholds ==========
    collision_threshold_initial: float = 0.3
    """Initial collision probability threshold for safe actions (conservative)"""

    collision_threshold_max: float = 0.85
    """Maximum collision threshold when robot is stuck (permissive)"""

    collision_threshold_after_collision: float = 0.6
    """Collision threshold after a collision occurs (slightly permissive)"""

    collision_threshold_after_turn: float = 0.5
    """Collision threshold after turning around (moderate)"""

    collision_threshold_increase_step: float = 0.15
    """Amount to increase threshold when no safe actions available"""

    collision_threshold_decrease_step: float = 0.03
    """Amount to decrease threshold when safe actions are available"""

    collision_threshold_increase_no_progress: float = 0.1
    """Amount to increase threshold when no progress toward goal"""

    collision_threshold_max_no_progress: float = 0.8
    """Max threshold when not making progress"""

    collision_threshold_increase_after_collision: float = 0.05
    """Amount to increase threshold after collision"""

    # ========== Stuck Detection Parameters ==========
    stuck_position_change_threshold: float = 3.0
    """Minimum position change (pixels) to not be considered stuck"""

    stuck_counter_max: int = 8
    """Maximum stuck counter before forcing turn around"""

    stuck_counter_wall_follow: int = 5
    """Stuck counter threshold to activate wall following"""

    no_safe_actions_turn_threshold: int = 15
    """Number of iterations with no safe actions before turning"""

    consecutive_no_actions_threshold: int = 3
    """Consecutive no-action iterations before increasing threshold"""

    # ========== Progress Tracking ==========
    progress_distance_threshold: float = 5.0
    """Distance improvement (pixels) to be considered making progress"""

    no_progress_counter_max: int = 30
    """Max iterations without progress before increasing threshold"""

    # ========== Goal Detection ==========
    goal_reach_distance: float = 50.0
    """Distance (pixels) at which goal is considered reached"""

    # ========== Action Smoothing Parameters ==========
    action_smoother_history_length: int = 5
    """Number of recent actions to track for smoothing"""

    action_smoother_momentum_weight: float = 0.4
    """Weight for momentum in action selection (0-1)"""

    action_smoother_thrashing_threshold: int = 3
    """Number of sign changes to detect thrashing"""

    action_smoother_min_history_for_smoothing: int = 2
    """Minimum history length before applying smoothing"""

    # ========== Spatial Memory Parameters ==========
    spatial_memory_grid_size: int = 30
    """Size of spatial grid cells in pixels"""

    spatial_memory_decay_rate: float = 0.97
    """Decay rate for visit counts (0-1)"""

    spatial_memory_max_history: int = 100
    """Maximum positions to track in history"""

    spatial_memory_decay_interval: int = 10
    """Iterations between decay applications"""

    spatial_memory_decay_min_count: float = 0.1
    """Minimum visit count before removal"""

    spatial_memory_oscillation_window: int = 20
    """Number of recent positions for oscillation detection"""

    spatial_memory_oscillation_threshold: float = 60.0
    """Max average distance (pixels) to detect oscillation"""

    spatial_memory_oscillation_check_interval: int = 5
    """Iterations between oscillation checks"""

    spatial_memory_repulsion_weight: float = 2.0
    """Weight for spatial memory repulsion in action scoring"""

    # ========== Oscillation Detection (Legacy) ==========
    oscillation_position_window: int = 15
    """Window size for oscillation position tracking"""

    oscillation_distance_threshold: float = 30.0
    """Distance threshold for spatial oscillation detection"""

    oscillation_check_window: int = 20
    """Window of positions to check for oscillation"""

    oscillation_return_distance: float = 20.0
    """Distance for returning to similar positions"""

    # ========== Wall Following Parameters ==========
    wall_follower_target_distance: float = 70.0
    """Desired distance from wall (pixels)"""

    wall_follower_max_steps: int = 100
    """Maximum steps to follow wall before deactivating"""

    wall_follower_close_wall_threshold: float = 80.0
    """Distance to consider wall close (tight space detection)"""

    wall_follower_close_wall_count: int = 3
    """Number of close walls to activate wall following"""

    wall_follower_front_blocked_threshold: float = 80.0
    """Front sensor distance to consider blocked"""

    wall_follower_front_moderate_threshold: float = 110.0
    """Front sensor distance for moderate turn"""

    wall_follower_distance_band: float = 30.0
    """Band around target distance for oscillation reduction"""

    wall_follower_distance_fine_adjust: float = 15.0
    """Distance for fine adjustment while following"""

    wall_follower_far_threshold: float = 130.0
    """Distance to consider far from wall"""

    wall_follower_deactivate_threshold: float = 100.0
    """Minimum sensor reading to deactivate (open space)"""

    wall_follower_max_steps_default: int = 50
    """Default max follow steps (used in some contexts)"""

    # ========== Waypoint Planning Parameters ==========
    waypoint_distance: float = 120.0
    """Distance ahead to place waypoints (pixels)"""

    waypoint_reached_threshold: float = 50.0
    """Distance to consider waypoint reached (pixels)"""

    waypoint_min_goal_distance: float = 200.0
    """Minimum distance to goal before using waypoints"""

    waypoint_front_blocked_threshold: float = 100.0
    """Front sensor threshold for blocked path detection"""

    waypoint_combined_sensor_threshold: float = 200.0
    """Combined front+side sensor threshold for blocking"""

    waypoint_openness_weight: float = 2.0
    """Weight for openness in waypoint direction selection"""

    waypoint_alignment_weight: float = 0.5
    """Weight for goal alignment in waypoint direction"""

    # ========== Openness Scoring Parameters ==========
    openness_scorer_max_sensor_range: float = 150.0
    """Maximum sensor reading in pixels"""

    openness_weight_normal: float = 0.8
    """Normal weight for openness in action scoring"""

    openness_weight_stuck: float = 1.5
    """Increased weight for openness when stuck"""

    openness_weight_stuck_threshold: int = 3
    """Stuck counter threshold to increase openness weight"""

    openness_min_weight: float = 0.1
    """Minimum weight for any sensor"""

    # ========== Strong Action Thresholds ==========
    strong_action_threshold: int = 3
    """Minimum absolute action value for strong actions"""

    strong_action_threshold_oscillation: int = 3
    """Minimum action for oscillation breaking"""

    strong_action_threshold_thrashing: int = 2
    """Minimum action for thrashing recovery"""

    # ========== Action Selection Parameters ==========
    action_future_position_estimate: float = 30.0
    """Distance (pixels) to estimate future position"""

    action_space_min: int = -5
    """Minimum action value"""

    action_space_max: int = 5
    """Maximum action value"""

    safest_actions_count: int = 5
    """Number of safest actions to consider when none available"""

    # ========== Simulation Environment Parameters ==========
    simulation_action_repeat: int = 20
    """Number of physics steps per action decision"""

    simulation_screen_width: int = 1080
    """Simulation screen width (pixels)"""

    simulation_screen_height: int = 900
    """Simulation screen height (pixels)"""

    simulation_boundary_margin: int = 50
    """Minimum distance from simulation boundaries (pixels)"""

    simulation_wall_unit: int = 180
    """Unit size for wall construction (pixels)"""

    simulation_goal_offset: int = 60
    """Offset for goal placement from boundaries (pixels)"""

    simulation_goal_radius: int = 40
    """Goal detection radius (pixels)"""

    simulation_collision_reset_threshold: int = 5
    """Consecutive collisions before considering stuck"""

    simulation_reset_distance: int = 25
    """Distance to back up after collision (pixels)"""

    simulation_turn_around_short: int = 180
    """Short turn around duration (steps)"""

    simulation_turn_around_long: int = 250
    """Long turn around duration (steps)"""

    # ========== Robot Physics Parameters ==========
    robot_mass: int = 20
    """Robot mass for physics simulation"""

    robot_speed: int = 20
    """Robot movement speed"""

    robot_max_steering_force: float = 1.0
    """Maximum steering force"""

    robot_friction: float = 0.05
    """Friction coefficient"""

    robot_sensor_range: float = 150.0
    """Maximum sensor range (pixels)"""

    robot_length: int = 20
    """Robot length for collision shape (pixels)"""

    robot_width: int = 30
    """Robot width for collision shape (pixels)"""

    # ========== Steering Behavior Parameters ==========
    steering_wander_range: float = 0.1
    """Wander behavior angle range (fraction of PI)"""

    steering_max_scaler: int = 5
    """Maximum action magnitude for steering"""

    steering_perlin_scale_0: int = 250
    """Perlin noise scale parameter 0"""

    steering_perlin_scale_1: int = 2000
    """Perlin noise scale parameter 1"""

    steering_perlin_max_samples: int = 50
    """Max samples when searching for unchecked action"""

    # ========== Training Parameters ==========
    training_batch_size: int = 32
    """Batch size for neural network training"""

    training_learning_rate: float = 0.01
    """Initial learning rate for training"""

    training_scheduler_factor: float = 0.5
    """Factor to reduce learning rate"""

    training_scheduler_patience: int = 10
    """Epochs to wait before reducing learning rate"""

    training_min_lr: float = 1e-6
    """Minimum learning rate"""

    training_focal_gamma: float = 2.0
    """Focal loss gamma parameter"""

    training_focal_alpha_min: float = 0.1
    """Minimum alpha for focal loss"""

    training_epochs_default: int = 100
    """Default number of training epochs"""

    # ========== Data Collection Parameters ==========
    data_collection_action_repeat: int = 100
    """Action repeat for data collection"""

    data_collection_collision_window: float = 0.3
    """Time window to attribute collision to previous action"""

    data_collection_default_samples: int = 10000
    """Default number of samples to collect"""

    # ========== Testing Parameters ==========
    test_progress_report_interval: int = 50
    """Iterations between progress reports"""

    test_oscillation_check_count: int = 50
    """Oscillation score threshold for warning"""

    test_max_positions_saved: int = 1000
    """Maximum positions to save in test results"""


# Create a default singleton instance
default_config = NavigationConfig()
