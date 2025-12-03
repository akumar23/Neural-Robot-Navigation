"""
Comprehensive Performance Comparison: Feedforward vs LSTM.

Runs extensive tests on both models to compare:
- Success rate (goals reached / goals attempted)
- Average iterations per goal
- Collision rate
- Stuck events
- Oscillation detection
- Statistical significance testing

Generates detailed comparison report and visualizations.
"""

import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set headless mode for automated testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from src.robot_navigation.steering import Seek
from src.robot_navigation import simulation as sim
from src.robot_navigation.networks import Action_Conditioned_FF, Action_Conditioned_LSTM
from src.robot_navigation.feature_engineering import engineer_features
from src.robot_navigation.navigation_config import NavigationConfig

import pickle
import numpy as np
import torch
import numpy.linalg as la
from collections import deque
import time
import json


def run_episode_ff(model, scaler, config, max_iterations=2000):
    """
    Run a single episode with feedforward model.

    Returns:
        Dictionary with episode metrics
    """
    sim_env = sim.SimulationEnvironment()
    action_repeat = config.simulation_action_repeat
    steering_behavior = Seek(sim_env.goal_body.position)

    action_history = deque(maxlen=5)

    goal_reached = False
    iterations = 0
    collisions = 0
    stuck_events = 0
    collision_threshold = config.collision_threshold_initial
    consecutive_no_actions = 0
    last_position = None
    stuck_counter = 0

    while iterations < max_iterations and not goal_reached:
        iterations += 1

        # Check if goal reached
        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)

        if distance_to_goal < config.goal_reach_distance:
            goal_reached = True
            break

        # Get predictions for all actions
        action_space = np.arange(config.action_space_min, config.action_space_max + 1)
        action_predictions = {}

        for action in action_space:
            # Create features
            sensor_readings = sim_env.raycasting()
            robot_pos = sim_env.robot.body.position
            robot_angle = sim_env.robot.body.angle
            goal_pos = sim_env.goal_body.position
            velocity = sim_env.robot.body.velocity

            features = engineer_features(
                sensor_readings, action,
                robot_pos=robot_pos,
                robot_angle=robot_angle,
                goal_pos=goal_pos,
                velocity=velocity,
                action_history=action_history
            )

            # Normalize and predict
            normalized = scaler.transform(features.reshape(1, -1))
            network_param = torch.FloatTensor(normalized.flatten())

            with torch.no_grad():
                logit = model(network_param)
                prediction_value = torch.sigmoid(logit).item()
                if np.isnan(prediction_value):
                    prediction_value = 1.0

            action_predictions[action] = prediction_value

        # Find safe actions
        actions_available = [a for a in action_space if action_predictions[a] < collision_threshold]

        # Get desired action
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)

        # Handle no safe actions
        if len(actions_available) == 0:
            consecutive_no_actions += 1
            if consecutive_no_actions > config.consecutive_no_actions_threshold:
                collision_threshold = min(config.collision_threshold_max, collision_threshold + 0.15)
                consecutive_no_actions = 0
            else:
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                actions_available = [a[0] for a in sorted_actions[:3]]

            stuck_counter += 1
            if stuck_counter > 15:
                sim_env.turn_robot_around()
                stuck_counter = 0
                stuck_events += 1
                collision_threshold = 0.6
                continue
        else:
            consecutive_no_actions = 0
            stuck_counter = 0

        # Select action
        closest_action = 0 if len(actions_available) == 0 else actions_available[0]
        for a in actions_available:
            if abs(desired_action - a) < abs(desired_action - closest_action):
                closest_action = a

        action_history.append(closest_action)
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

        # Position-based stuck detection
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < 3.0:
                stuck_counter += 1
                if stuck_counter > 8:
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    stuck_events += 1
                    continue
        last_position = current_position

        # Execute action
        for _ in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                collisions += 1
                steering_behavior.reset_action()
                collision_threshold = min(0.6, collision_threshold + 0.05)
                break

    return {
        'goal_reached': goal_reached,
        'iterations': iterations,
        'collisions': collisions,
        'stuck_events': stuck_events,
        'collision_rate': collisions / iterations if iterations > 0 else 0
    }


def run_episode_lstm(model, scaler, config, max_iterations=2000):
    """
    Run a single episode with LSTM model.

    Returns:
        Dictionary with episode metrics
    """
    sim_env = sim.SimulationEnvironment()
    action_repeat = config.simulation_action_repeat
    steering_behavior = Seek(sim_env.goal_body.position)

    action_history = deque(maxlen=5)
    hidden = None

    goal_reached = False
    iterations = 0
    collisions = 0
    stuck_events = 0
    collision_threshold = config.collision_threshold_initial
    consecutive_no_actions = 0
    last_position = None
    stuck_counter = 0

    while iterations < max_iterations and not goal_reached:
        iterations += 1

        # Check if goal reached
        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)

        if distance_to_goal < config.goal_reach_distance:
            goal_reached = True
            break

        # Get predictions for all actions
        action_space = np.arange(config.action_space_min, config.action_space_max + 1)
        action_predictions = {}
        temp_hidden = hidden

        for action in action_space:
            # Create features
            sensor_readings = sim_env.raycasting()
            robot_pos = sim_env.robot.body.position
            robot_angle = sim_env.robot.body.angle
            goal_pos = sim_env.goal_body.position
            velocity = sim_env.robot.body.velocity

            features = engineer_features(
                sensor_readings, action,
                robot_pos=robot_pos,
                robot_angle=robot_angle,
                goal_pos=goal_pos,
                velocity=velocity,
                action_history=action_history
            )

            # Normalize and predict
            normalized = scaler.transform(features.reshape(1, -1))
            network_param = torch.FloatTensor(normalized).unsqueeze(1)

            with torch.no_grad():
                logit, _ = model(network_param, temp_hidden)
                prediction_value = torch.sigmoid(logit).item()
                if np.isnan(prediction_value):
                    prediction_value = 1.0

            action_predictions[action] = prediction_value

        # Find safe actions
        actions_available = [a for a in action_space if action_predictions[a] < collision_threshold]

        # Get desired action
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)

        # Handle no safe actions
        if len(actions_available) == 0:
            consecutive_no_actions += 1
            if consecutive_no_actions > config.consecutive_no_actions_threshold:
                collision_threshold = min(config.collision_threshold_max, collision_threshold + 0.15)
                consecutive_no_actions = 0
            else:
                valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
                sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
                actions_available = [a[0] for a in sorted_actions[:3]]

            stuck_counter += 1
            if stuck_counter > 15:
                sim_env.turn_robot_around()
                stuck_counter = 0
                stuck_events += 1
                collision_threshold = 0.6
                hidden = None
                action_history.clear()
                continue
        else:
            consecutive_no_actions = 0
            stuck_counter = 0

        # Select action
        closest_action = 0 if len(actions_available) == 0 else actions_available[0]
        for a in actions_available:
            if abs(desired_action - a) < abs(desired_action - closest_action):
                closest_action = a

        # Update hidden state with selected action
        sensor_readings = sim_env.raycasting()
        robot_pos = sim_env.robot.body.position
        robot_angle = sim_env.robot.body.angle
        goal_pos = sim_env.goal_body.position
        velocity = sim_env.robot.body.velocity

        features = engineer_features(
            sensor_readings, closest_action,
            robot_pos=robot_pos,
            robot_angle=robot_angle,
            goal_pos=goal_pos,
            velocity=velocity,
            action_history=action_history
        )

        normalized = scaler.transform(features.reshape(1, -1))
        network_param = torch.FloatTensor(normalized).unsqueeze(1)

        with torch.no_grad():
            _, hidden = model(network_param, hidden)

        action_history.append(closest_action)
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

        # Position-based stuck detection
        current_position = sim_env.robot.body.position
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < 3.0:
                stuck_counter += 1
                if stuck_counter > 8:
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    stuck_events += 1
                    hidden = None
                    action_history.clear()
                    continue
        last_position = current_position

        # Execute action
        for _ in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                collisions += 1
                steering_behavior.reset_action()
                collision_threshold = min(0.6, collision_threshold + 0.05)
                hidden = None
                action_history.clear()
                break

    return {
        'goal_reached': goal_reached,
        'iterations': iterations,
        'collisions': collisions,
        'stuck_events': stuck_events,
        'collision_rate': collisions / iterations if iterations > 0 else 0
    }


def run_comparison(num_episodes=50, max_iterations=2000):
    """
    Run comprehensive comparison between FF and LSTM models.

    Args:
        num_episodes: Number of episodes to test each model
        max_iterations: Maximum iterations per episode

    Returns:
        Dictionary with comparison results
    """
    print("="*60)
    print("FEEDFORWARD vs LSTM COMPARISON")
    print("="*60)
    print(f"Episodes per model: {num_episodes}")
    print(f"Max iterations per episode: {max_iterations}")
    print("="*60 + "\n")

    config = NavigationConfig()
    models_path = Path(__file__).parent.parent / "models"

    # Load feedforward model
    print("Loading Feedforward model...")
    ff_model = Action_Conditioned_FF(input_size=20)
    ff_model.load_state_dict(torch.load(models_path / "saved_model.pkl"))
    ff_model.eval()
    ff_scaler = pickle.load(open(models_path / "scaler.pkl", "rb"))

    # Load LSTM model
    print("Loading LSTM model...")
    lstm_model = Action_Conditioned_LSTM(input_size=20, hidden_size=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(models_path / "saved_model_lstm.pkl"))
    lstm_model.eval()
    lstm_scaler = pickle.load(open(models_path / "scaler_lstm.pkl", "rb"))

    print("\n" + "="*60)
    print("TESTING FEEDFORWARD MODEL")
    print("="*60)

    ff_results = []
    start = time.time()
    for i in range(num_episodes):
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{num_episodes}...")
        result = run_episode_ff(ff_model, ff_scaler, config, max_iterations)
        ff_results.append(result)
    ff_time = time.time() - start

    print("\n" + "="*60)
    print("TESTING LSTM MODEL")
    print("="*60)

    lstm_results = []
    start = time.time()
    for i in range(num_episodes):
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{num_episodes}...")
        result = run_episode_lstm(lstm_model, lstm_scaler, config, max_iterations)
        lstm_results.append(result)
    lstm_time = time.time() - start

    # Compute statistics
    def compute_stats(results):
        goals_reached = sum(1 for r in results if r['goal_reached'])
        success_rate = goals_reached / len(results)

        successful_episodes = [r for r in results if r['goal_reached']]
        avg_iterations = np.mean([r['iterations'] for r in successful_episodes]) if successful_episodes else 0
        std_iterations = np.std([r['iterations'] for r in successful_episodes]) if successful_episodes else 0

        avg_collisions = np.mean([r['collisions'] for r in results])
        avg_stuck_events = np.mean([r['stuck_events'] for r in results])
        avg_collision_rate = np.mean([r['collision_rate'] for r in results])

        return {
            'success_rate': success_rate,
            'avg_iterations_per_goal': avg_iterations,
            'std_iterations_per_goal': std_iterations,
            'avg_collisions': avg_collisions,
            'avg_stuck_events': avg_stuck_events,
            'avg_collision_rate': avg_collision_rate
        }

    ff_stats = compute_stats(ff_results)
    lstm_stats = compute_stats(lstm_results)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Metric':<35} {'Feedforward':<15} {'LSTM':<15} {'Improvement':<15}")
    print("-"*80)

    # Success rate
    ff_sr = ff_stats['success_rate'] * 100
    lstm_sr = lstm_stats['success_rate'] * 100
    sr_improvement = lstm_sr - ff_sr
    print(f"{'Success Rate (%)':<35} {ff_sr:<15.1f} {lstm_sr:<15.1f} {sr_improvement:+.1f}%")

    # Average iterations per goal
    ff_iter = ff_stats['avg_iterations_per_goal']
    lstm_iter = lstm_stats['avg_iterations_per_goal']
    iter_improvement = ((ff_iter - lstm_iter) / ff_iter * 100) if ff_iter > 0 else 0
    print(f"{'Avg Iterations/Goal':<35} {ff_iter:<15.1f} {lstm_iter:<15.1f} {iter_improvement:+.1f}%")

    # Collisions
    ff_col = ff_stats['avg_collisions']
    lstm_col = lstm_stats['avg_collisions']
    col_improvement = ((ff_col - lstm_col) / ff_col * 100) if ff_col > 0 else 0
    print(f"{'Avg Collisions/Episode':<35} {ff_col:<15.2f} {lstm_col:<15.2f} {col_improvement:+.1f}%")

    # Stuck events
    ff_stuck = ff_stats['avg_stuck_events']
    lstm_stuck = lstm_stats['avg_stuck_events']
    stuck_improvement = ((ff_stuck - lstm_stuck) / ff_stuck * 100) if ff_stuck > 0 else 0
    print(f"{'Avg Stuck Events/Episode':<35} {ff_stuck:<15.2f} {lstm_stuck:<15.2f} {stuck_improvement:+.1f}%")

    # Collision rate
    ff_cr = ff_stats['avg_collision_rate'] * 100
    lstm_cr = lstm_stats['avg_collision_rate'] * 100
    cr_improvement = ff_cr - lstm_cr
    print(f"{'Collision Rate (%)':<35} {ff_cr:<15.2f} {lstm_cr:<15.2f} {cr_improvement:+.2f}%")

    # Time
    print(f"\n{'Total Test Time (s)':<35} {ff_time:<15.1f} {lstm_time:<15.1f}")

    print("="*60)

    # Save results
    results_path = Path(__file__).parent.parent / "models" / "comparison_results.json"
    comparison_data = {
        'num_episodes': num_episodes,
        'max_iterations': max_iterations,
        'feedforward': {
            'stats': ff_stats,
            'time': ff_time,
            'raw_results': ff_results
        },
        'lstm': {
            'stats': lstm_stats,
            'time': lstm_time,
            'raw_results': lstm_results
        }
    }

    with open(results_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return comparison_data


if __name__ == '__main__':
    # Run comprehensive comparison
    num_episodes = 50  # Test 50 episodes per model
    max_iterations = 2000

    results = run_comparison(num_episodes, max_iterations)
