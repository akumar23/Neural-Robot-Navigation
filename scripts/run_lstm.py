"""
LSTM-based Goal-Seeking Navigation with Stateful Inference.

Uses temporal memory (LSTM hidden state) to improve navigation by:
- Remembering consequences of previous actions
- Detecting oscillation/thrashing patterns
- Learning trajectory momentum
- Context-aware decision making

Key difference from feedforward: Hidden state persists across timesteps
and is updated after each action selection.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Seek
from src.robot_navigation import simulation as sim
from src.robot_navigation.networks import Action_Conditioned_LSTM
from src.robot_navigation.feature_engineering import engineer_features
from src.robot_navigation.navigation_config import NavigationConfig

import pickle
import numpy as np
import torch
import numpy.linalg as la
from collections import deque


def get_network_param(sim_env, action, scaler, action_history):
    """
    Get network input with enhanced feature engineering.

    Creates 20D feature vector with sensors + derived + goal + temporal + spatial-goal + action,
    then normalizes using the saved scaler.

    Args:
        sim_env: Simulation environment
        action: Steering action (-5 to +5)
        scaler: StandardScaler for normalization
        action_history: Deque of recent actions (last 5)

    Returns:
        Normalized network input tensor (1, 1, 20) for LSTM
    """
    sensor_readings = sim_env.raycasting()

    # Get robot state for enhanced features
    robot_pos = sim_env.robot.body.position
    robot_angle = sim_env.robot.body.angle
    goal_pos = sim_env.goal_body.position
    velocity = sim_env.robot.body.velocity

    # Create 20D feature vector
    features = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos,
        robot_angle=robot_angle,
        goal_pos=goal_pos,
        velocity=velocity,
        action_history=action_history
    )

    # Normalize
    normalized = scaler.transform(features.reshape(1, -1))

    # Convert to tensor with shape (batch=1, seq_len=1, features=20) for LSTM
    network_param = torch.FloatTensor(normalized).unsqueeze(1)

    return network_param


def goal_seeking_lstm(goals_to_reach, config=None, verbose=True):
    """
    LSTM-based goal-seeking navigation with stateful inference.

    Maintains hidden state across timesteps for temporal reasoning.
    Resets hidden state when reaching goal or after collision.

    Args:
        goals_to_reach: Number of goals to reach before stopping
        config: NavigationConfig instance (uses default if None)
        verbose: Whether to print progress
    """
    if config is None:
        config = NavigationConfig()

    sim_env = sim.SimulationEnvironment()
    action_repeat = config.simulation_action_repeat
    steering_behavior = Seek(sim_env.goal_body.position)

    # Load LSTM model
    models_path = Path(__file__).parent.parent / "models"
    model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(models_path / "saved_model_lstm.pkl"))
    model.eval()

    # Load normalization scaler
    scaler = pickle.load(open(models_path / "scaler_lstm.pkl", "rb"))

    # Initialize action history for temporal features
    action_history = deque(maxlen=5)

    # Initialize LSTM hidden state
    hidden = None  # Will be initialized to zeros on first forward pass

    # Navigation state
    goals_reached = 0
    last_position = None
    stuck_counter = 0
    collision_threshold = config.collision_threshold_initial
    consecutive_no_actions = 0
    last_distance_to_goal = float('inf')
    no_progress_counter = 0

    # Statistics
    total_iterations = 0
    total_collisions = 0
    total_stuck_events = 0

    if verbose:
        print("="*60)
        print("LSTM-BASED GOAL SEEKING NAVIGATION")
        print("="*60)
        print(f"Model: LSTM (2 layers, 64 hidden units)")
        print(f"Goals to reach: {goals_to_reach}")
        print(f"Initial collision threshold: {collision_threshold:.2f}")
        print("="*60 + "\n")

    while goals_reached < goals_to_reach:
        total_iterations += 1

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)

        # Check progress
        if distance_to_goal < last_distance_to_goal - config.progress_distance_threshold:
            no_progress_counter = 0
        else:
            no_progress_counter += 1

        last_distance_to_goal = distance_to_goal

        # Increase threshold if no progress
        if no_progress_counter > config.no_progress_counter_max:
            collision_threshold = min(
                config.collision_threshold_max_no_progress,
                collision_threshold + config.collision_threshold_increase_no_progress
            )
            no_progress_counter = 0

        # Check if goal reached
        if distance_to_goal < config.goal_reach_distance:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            goals_reached += 1
            if verbose:
                print(f"Goal {goals_reached} reached! (Iteration {total_iterations})")

            # Reset for new episode
            collision_threshold = config.collision_threshold_initial
            no_progress_counter = 0
            last_distance_to_goal = float('inf')
            hidden = None  # Reset hidden state for new episode
            action_history.clear()
            continue

        # Predict collision probabilities for all actions with LSTM
        action_space = np.arange(config.action_space_min, config.action_space_max + 1)
        action_predictions = {}

        # IMPORTANT: Use a temporary hidden state for action evaluation
        # We'll only update the real hidden state after selecting an action
        temp_hidden = hidden

        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler, action_history)

            # Forward pass with temporary hidden state
            with torch.no_grad():
                logit, temp_hidden_out = model(network_param, temp_hidden)
                prediction_value = torch.sigmoid(logit).item()

                # Handle NaN
                if np.isnan(prediction_value):
                    prediction_value = 1.0

            action_predictions[action] = prediction_value

            # Don't update hidden state yet - wait until we select action

        # Find safe actions
        actions_available = [a for a in action_space if action_predictions[a] < collision_threshold]

        # Get desired action from steering behavior
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)

        # Handle no safe actions
        if len(actions_available) == 0:
            consecutive_no_actions += 1
            if consecutive_no_actions > config.consecutive_no_actions_threshold:
                collision_threshold = min(
                    config.collision_threshold_max,
                    collision_threshold + config.collision_threshold_increase_step
                )
                consecutive_no_actions = 0
            else:
                # Take safest actions
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                if len(valid_predictions) == 0:
                    valid_predictions = action_predictions
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                safest_actions = [a[0] for a in sorted_actions[:config.safest_actions_count]]
                actions_available = safest_actions

            stuck_counter += 1
            if stuck_counter > config.no_safe_actions_turn_threshold:
                sim_env.turn_robot_around()
                stuck_counter = 0
                last_position = None
                collision_threshold = config.collision_threshold_after_turn
                hidden = None  # Reset hidden state after forced turn
                action_history.clear()
                total_stuck_events += 1
                if verbose:
                    print(f"  Stuck! Forced turn-around (Iteration {total_iterations})")
                continue
        else:
            consecutive_no_actions = 0
            # Gradually lower threshold
            if collision_threshold > config.collision_threshold_initial:
                collision_threshold = max(
                    config.collision_threshold_initial,
                    collision_threshold - config.collision_threshold_decrease_step
                )

        # Select action closest to desired
        if len(actions_available) == 0:
            closest_action = 0
        else:
            min_diff, closest_action = 9999, actions_available[0]
            for a in actions_available:
                diff = abs(desired_action - a)
                if diff < min_diff:
                    min_diff = diff
                    closest_action = a

        # CRITICAL: Update hidden state with the SELECTED action
        network_param = get_network_param(sim_env, closest_action, scaler, action_history)
        with torch.no_grad():
            _, hidden = model(network_param, hidden)

        # Update action history
        action_history.append(closest_action)

        # Get steering force and execute
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

        # Check if stuck (position-based)
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < config.stuck_position_change_threshold:
                stuck_counter += 1
                if stuck_counter > config.stuck_counter_max:
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    last_position = None
                    collision_threshold = config.collision_threshold_after_turn
                    hidden = None  # Reset hidden state
                    action_history.clear()
                    total_stuck_events += 1
                    if verbose:
                        print(f"  Stuck! Position-based turn (Iteration {total_iterations})")
                    continue
            else:
                stuck_counter = 0
        last_position = current_position

        # Execute action
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                steering_behavior.reset_action()
                stuck_counter = 0
                last_position = None
                total_collisions += 1

                # Increase threshold after collision
                collision_threshold = min(
                    config.collision_threshold_after_collision,
                    collision_threshold + config.collision_threshold_increase_after_collision
                )

                # Reset hidden state after collision
                hidden = None
                action_history.clear()
                if verbose and total_collisions % 10 == 0:
                    print(f"  Collisions: {total_collisions} (Iteration {total_iterations})")
                break

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("NAVIGATION SUMMARY")
        print("="*60)
        print(f"Goals reached: {goals_reached}")
        print(f"Total iterations: {total_iterations}")
        print(f"Average iterations per goal: {total_iterations / goals_reached:.1f}")
        print(f"Total collisions: {total_collisions}")
        print(f"Collision rate: {total_collisions / total_iterations:.2%}")
        print(f"Total stuck events: {total_stuck_events}")
        print("="*60)

    return {
        'goals_reached': goals_reached,
        'total_iterations': total_iterations,
        'avg_iterations_per_goal': total_iterations / goals_reached,
        'total_collisions': total_collisions,
        'collision_rate': total_collisions / total_iterations,
        'total_stuck_events': total_stuck_events
    }


if __name__ == '__main__':
    goals_to_reach = 2
    goal_seeking_lstm(goals_to_reach, verbose=True)
