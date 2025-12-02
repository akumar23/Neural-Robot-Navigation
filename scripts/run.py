import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Wander, Seek
from src.robot_navigation import simulation as sim
from src.robot_navigation.networks import Action_Conditioned_FF
from src.robot_navigation.feature_engineering import engineer_features

import pickle
import numpy as np
import torch
import numpy.linalg as la


def get_network_param(sim_env, action, scaler):
    """
    Get network input with feature engineering.

    Creates feature vector with raw sensors + derived features + action,
    then normalizes using the saved scaler.
    """
    sensor_readings = sim_env.raycasting()

    # Use feature engineering to create enhanced feature vector
    # Features: [5 raw sensors] + [6 derived features] + [action] = 12 features
    features = engineer_features(sensor_readings, action)

    # Add dummy collision label (0) to match scaler shape, then remove after transform
    features_with_label = np.append(features, 0)
    normalized = scaler.transform(features_with_label.reshape(1, -1))
    network_param = normalized.flatten()[:-1]  # Remove the collision label

    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param

def goal_seeking(goals_to_reach):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 20
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)

    #load model
    models_path = Path(__file__).parent.parent / "models"
    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load(models_path / "saved_model.pkl"))
    model.eval()

    #load normalization parameters
    scaler = pickle.load(open(models_path / "scaler.pkl", "rb"))

    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    robot_turned_around = False
    actions_checked = []
    goals_reached = 0
    last_position = None
    stuck_counter = 0
    collision_threshold = 0.3  # Slightly higher initial threshold for more actions
    consecutive_no_actions = 0  # Track consecutive iterations with no safe actions
    last_distance_to_goal = float('inf')
    no_progress_counter = 0
    while goals_reached < goals_to_reach:

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)
        
        # Check if making progress toward goal
        if distance_to_goal < last_distance_to_goal - 5:  # Making progress
            no_progress_counter = 0
        else:
            no_progress_counter += 1
        
        last_distance_to_goal = distance_to_goal
        
        # If not making progress for a while, increase threshold
        if no_progress_counter > 30:
            collision_threshold = min(0.8, collision_threshold + 0.1)
            no_progress_counter = 0
        
        if distance_to_goal < 50:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            print("goal reached +1")
            goals_reached += 1
            collision_threshold = 0.3  # Reset threshold on goal reach
            no_progress_counter = 0
            last_distance_to_goal = float('inf')
            continue

        action_space = np.arange(-5,6)
        actions_available = []
        action_predictions = {}  # Store predictions for all actions
        
        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler)
            prediction = model(network_param)
            # Apply sigmoid to convert logits to probabilities (0-1 range)
            # Lower probability = safer action (less likely to collide)
            prediction_value = torch.sigmoid(prediction).item()
            # Handle NaN predictions
            if np.isnan(prediction_value):
                prediction_value = 1.0  # Treat NaN as unsafe
            action_predictions[action] = prediction_value
            if prediction_value < collision_threshold:
                actions_available.append(action)

        # Get desired action from steering behavior
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        
        if len(actions_available) == 0:
            # If no actions are available, gradually increase threshold to allow more actions
            consecutive_no_actions += 1
            if consecutive_no_actions > 3:  # Faster threshold increase
                # Gradually increase threshold when stuck
                collision_threshold = min(0.85, collision_threshold + 0.15)
                consecutive_no_actions = 0
            else:
                # Pick actions with lowest collision predictions
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                if len(valid_predictions) == 0:
                    valid_predictions = action_predictions
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                # Take the top 5 safest actions (more options)
                safest_actions = [a[0] for a in sorted_actions[:5]]
                actions_available = safest_actions
            
            # Increment stuck counter when no safe actions are available
            stuck_counter += 1
            # Also limit how often we turn around to prevent infinite loops
            if stuck_counter > 15:  # Turn around sooner
                sim_env.turn_robot_around()
                stuck_counter = 0
                last_position = None
                collision_threshold = 0.5  # Reset to moderate threshold after turn
                continue
        else:
            # Reset counters when we have safe actions
            consecutive_no_actions = 0
            # Gradually lower threshold back to conservative when we have safe actions
            if collision_threshold > 0.3:
                collision_threshold = max(0.3, collision_threshold - 0.03)
        
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
        
        # If the closest action is 0 (straight ahead) and we have other options,
        # prefer a slightly turning action to ensure forward movement
        if closest_action == 0 and len(actions_available) > 1:
            # Prefer actions that are close to the desired action but not exactly 0
            # This helps avoid getting stuck in place
            for a in actions_available:
                if a != 0 and abs(action - a) <= min + 1:
                    closest_action = a
                    break

        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        
        # Check if robot is stuck (not moving)
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < 3.0:  # More sensitive stuck detection
                stuck_counter += 1
                if stuck_counter > 8:  # Turn around sooner when stuck
                    # Force a turn to get unstuck
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    last_position = None
                    collision_threshold = 0.5  # Reset threshold
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
                collision_threshold = min(0.6, collision_threshold + 0.05)
                break


if __name__ == '__main__':
    goals_to_reach = 2
    goal_seeking(goals_to_reach)
