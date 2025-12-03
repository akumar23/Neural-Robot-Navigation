import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Wander, Seek
from src.robot_navigation import simulation as sim
from src.robot_navigation.networks import Action_Conditioned_FF
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

    Creates feature vector with raw sensors + derived features + goal/temporal/spatial-goal features + action,
    then normalizes using the saved scaler.

    Args:
        sim_env: Simulation environment
        action: Steering action (-5 to +5)
        scaler: StandardScaler for normalization
        action_history: Deque of recent actions (last 5)

    Returns:
        Normalized network input tensor (20D)
    """
    sensor_readings = sim_env.raycasting()

    # Get robot state for enhanced features
    robot_pos = sim_env.robot.body.position
    robot_angle = sim_env.robot.body.angle
    goal_pos = sim_env.goal_body.position
    velocity = sim_env.robot.body.velocity

    # Use enhanced feature engineering to create 20D feature vector
    # Features: [5 sensors] + [6 spatial] + [2 goal] + [4 temporal] + [2 spatial-goal] + [action] = 20
    features = engineer_features(
        sensor_readings, action,
        robot_pos=robot_pos,
        robot_angle=robot_angle,
        goal_pos=goal_pos,
        velocity=velocity,
        action_history=action_history
    )

    # Scaler expects 20 features (no collision label during inference)
    normalized = scaler.transform(features.reshape(1, -1))
    network_param = normalized.flatten()

    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param


def predict_with_uncertainty(model, network_param, n_samples=10):
    """
    Get prediction with uncertainty using Monte Carlo Dropout.

    MC Dropout enables uncertainty estimation by running multiple forward passes
    with dropout enabled at inference time. The variance in predictions indicates
    epistemic uncertainty (model uncertainty about correct prediction).

    Args:
        model: Trained neural network with dropout layers
        network_param: Input tensor (20-dimensional enhanced features)
        n_samples: Number of forward passes with dropout (default: 10)
                   Higher values give more accurate uncertainty estimates but slower

    Returns:
        mean_pred: Average collision probability across all samples
        uncertainty: Standard deviation of predictions (higher = more uncertain)
    """
    # Store original training mode
    original_mode = model.training

    # Enable dropout by setting model to train mode
    model.train()

    predictions = []

    # Run n_samples forward passes with dropout enabled
    with torch.no_grad():  # No gradients needed during inference
        for _ in range(n_samples):
            # Forward pass with dropout active
            logit = model(network_param)
            # Apply sigmoid to convert logit to probability
            pred = torch.sigmoid(logit).item()

            # Handle NaN/inf values (treat as unsafe)
            if np.isnan(pred) or np.isinf(pred):
                pred = 1.0

            predictions.append(pred)

    # Restore original model mode
    model.train(original_mode)

    # Compute statistics
    predictions_array = np.array(predictions)
    mean_pred = np.mean(predictions_array)
    uncertainty = np.std(predictions_array)

    return mean_pred, uncertainty

def goal_seeking(goals_to_reach, config=None, use_uncertainty=True, uncertainty_penalty_factor=1.0, n_mc_samples=10):
    """
    Goal-seeking navigation with optional Monte Carlo Dropout uncertainty estimation.

    Args:
        goals_to_reach: Number of goals to reach before stopping
        config: NavigationConfig instance (uses default if None)
        use_uncertainty: Whether to use MC Dropout for uncertainty estimation
        uncertainty_penalty_factor: How much to penalize uncertain predictions (0-2)
                                    0 = ignore uncertainty, 1 = standard penalty, 2 = very conservative
        n_mc_samples: Number of MC dropout samples for uncertainty estimation
    """
    if config is None:
        config = NavigationConfig()

    sim_env = sim.SimulationEnvironment()
    action_repeat = config.simulation_action_repeat
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)

    #load model (updated to 20D input)
    models_path = Path(__file__).parent.parent / "models"
    model = Action_Conditioned_FF(input_size=20)
    model.load_state_dict(torch.load(models_path / "saved_model.pkl"))
    model.eval()

    #load normalization parameters
    scaler = pickle.load(open(models_path / "scaler.pkl", "rb"))

    # Initialize action history for temporal features
    action_history = deque(maxlen=5)

    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    robot_turned_around = False
    actions_checked = []
    goals_reached = 0
    last_position = None
    stuck_counter = 0
    collision_threshold = config.collision_threshold_initial
    consecutive_no_actions = 0  # Track consecutive iterations with no safe actions
    last_distance_to_goal = float('inf')
    no_progress_counter = 0

    # Uncertainty tracking
    uncertainty_log = []
    high_uncertainty_count = 0
    iteration = 0
    while goals_reached < goals_to_reach:
        iteration += 1

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)

        # Check if making progress toward goal
        if distance_to_goal < last_distance_to_goal - config.progress_distance_threshold:
            no_progress_counter = 0
        else:
            no_progress_counter += 1

        last_distance_to_goal = distance_to_goal

        # If not making progress for a while, increase threshold
        if no_progress_counter > config.no_progress_counter_max:
            collision_threshold = min(
                config.collision_threshold_max_no_progress,
                collision_threshold + config.collision_threshold_increase_no_progress
            )
            no_progress_counter = 0

        if distance_to_goal < config.goal_reach_distance:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            print("goal reached +1")
            goals_reached += 1
            collision_threshold = config.collision_threshold_initial
            no_progress_counter = 0
            last_distance_to_goal = float('inf')
            continue

        action_space = np.arange(config.action_space_min, config.action_space_max + 1)
        actions_available = []
        action_predictions = {}  # Store predictions for all actions
        action_uncertainties = {}  # Store uncertainties for all actions

        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler, action_history)

            if use_uncertainty:
                # Use Monte Carlo Dropout for uncertainty estimation
                mean_pred, uncertainty = predict_with_uncertainty(model, network_param, n_samples=n_mc_samples)

                # Adjust prediction for safety: more uncertain = more conservative
                # adjusted_pred = mean_pred + (uncertainty * penalty_factor)
                adjusted_pred = mean_pred + (uncertainty * uncertainty_penalty_factor)

                # Clip to valid probability range [0, 1]
                adjusted_pred = np.clip(adjusted_pred, 0.0, 1.0)

                action_predictions[action] = adjusted_pred
                action_uncertainties[action] = uncertainty
            else:
                # Standard single-pass prediction (original behavior)
                prediction = model(network_param)
                # Apply sigmoid to convert logits to probabilities (0-1 range)
                prediction_value = torch.sigmoid(prediction).item()
                # Handle NaN predictions
                if np.isnan(prediction_value):
                    prediction_value = 1.0  # Treat NaN as unsafe
                action_predictions[action] = prediction_value
                action_uncertainties[action] = 0.0

            if action_predictions[action] < collision_threshold:
                actions_available.append(action)

        # Log uncertainty statistics periodically
        if use_uncertainty and iteration % 50 == 0:
            avg_uncertainty = np.mean(list(action_uncertainties.values()))
            max_uncertainty = np.max(list(action_uncertainties.values()))
            uncertainty_log.append({
                'iteration': iteration,
                'avg_uncertainty': avg_uncertainty,
                'max_uncertainty': max_uncertainty,
                'actions_available': len(actions_available)
            })
            print(f"Iteration {iteration}: Avg uncertainty={avg_uncertainty:.4f}, "
                  f"Max uncertainty={max_uncertainty:.4f}, Safe actions={len(actions_available)}")

            # Track high uncertainty decisions
            if max_uncertainty > 0.15:  # Threshold for high uncertainty
                high_uncertainty_count += 1

        # Get desired action from steering behavior
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        
        if len(actions_available) == 0:
            # If no actions are available, gradually increase threshold to allow more actions
            consecutive_no_actions += 1
            if consecutive_no_actions > config.consecutive_no_actions_threshold:
                # Gradually increase threshold when stuck
                collision_threshold = min(
                    config.collision_threshold_max,
                    collision_threshold + config.collision_threshold_increase_step
                )
                consecutive_no_actions = 0
            else:
                # Pick actions with lowest collision predictions
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                if len(valid_predictions) == 0:
                    valid_predictions = action_predictions
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                # Take the safest actions
                safest_actions = [a[0] for a in sorted_actions[:config.safest_actions_count]]
                actions_available = safest_actions

            # Increment stuck counter when no safe actions are available
            stuck_counter += 1
            # Also limit how often we turn around to prevent infinite loops
            if stuck_counter > config.no_safe_actions_turn_threshold:
                sim_env.turn_robot_around()
                stuck_counter = 0
                last_position = None
                collision_threshold = config.collision_threshold_after_turn
                continue
        else:
            # Reset counters when we have safe actions
            consecutive_no_actions = 0
            # Gradually lower threshold back to conservative when we have safe actions
            if collision_threshold > config.collision_threshold_initial:
                collision_threshold = max(
                    config.collision_threshold_initial,
                    collision_threshold - config.collision_threshold_decrease_step
                )
        
        # Find the action closest to the desired steering direction
        if len(actions_available) == 0:
            # Should not happen, but handle it
            closest_action = 0
        else:
            min_diff, closest_action = 9999, actions_available[0]
            for a in actions_available:
                diff = abs(desired_action - a)
                if diff < min_diff:
                    min_diff = diff
                    closest_action = a
        

        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

        # Update action history with the action we're about to take
        action_history.append(closest_action)

        # Check if robot is stuck (not moving)
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < config.stuck_position_change_threshold:
                stuck_counter += 1
                if stuck_counter > config.stuck_counter_max:
                    # Force a turn to get unstuck
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    last_position = None
                    collision_threshold = config.collision_threshold_after_turn
                    continue
            else:
                stuck_counter = 0  # Reset counter if robot is moving
        last_position = current_position
        
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                steering_behavior.reset_action()
                stuck_counter = 0  # Reset stuck counter on collision
                last_position = None
                # Slightly increase threshold after collision to be more permissive
                collision_threshold = min(
                    config.collision_threshold_after_collision,
                    collision_threshold + config.collision_threshold_increase_after_collision
                )
                break

    # Print uncertainty summary if enabled
    if use_uncertainty:
        print("\n" + "="*60)
        print("MONTE CARLO DROPOUT UNCERTAINTY SUMMARY")
        print("="*60)
        print(f"Total iterations: {iteration}")
        print(f"Goals reached: {goals_reached}")
        print(f"High uncertainty decisions: {high_uncertainty_count}")
        print(f"MC samples per prediction: {n_mc_samples}")
        print(f"Uncertainty penalty factor: {uncertainty_penalty_factor}")
        if uncertainty_log:
            all_avg_unc = [log['avg_uncertainty'] for log in uncertainty_log]
            all_max_unc = [log['max_uncertainty'] for log in uncertainty_log]
            print(f"\nUncertainty Statistics:")
            print(f"  Mean avg uncertainty: {np.mean(all_avg_unc):.4f}")
            print(f"  Mean max uncertainty: {np.mean(all_max_unc):.4f}")
            print(f"  Overall max uncertainty: {np.max(all_max_unc):.4f}")
        print("="*60)


if __name__ == '__main__':
    goals_to_reach = 2
    # Run with uncertainty estimation enabled
    goal_seeking(goals_to_reach, use_uncertainty=True, uncertainty_penalty_factor=1.0, n_mc_samples=10)
