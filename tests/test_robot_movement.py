"""
Test script to verify robot movement and detect issues like oscillation or getting stuck.
This script runs the simulation and tracks the robot's position over time.

Usage:
    # Run a single test
    python test_robot_movement.py
    
    # Run multiple tests
    python test_robot_movement.py --tests 5
    
    # Customize test parameters
    python test_robot_movement.py --iterations 2000 --goals 3 --tests 3
    
    # Quiet mode (less output)
    python test_robot_movement.py --quiet

The script will:
- Track robot position over time
- Detect if robot is stuck or oscillating
- Calculate movement statistics
- Save results to JSON files
"""

import os
import sys
from pathlib import Path

# Set headless mode before importing SimulationEnvironment
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Seek
from src.robot_navigation import simulation as sim
from src.robot_navigation.networks import Action_Conditioned_FF
from src.robot_navigation.feature_engineering import engineer_features
from src.robot_navigation.action_smoother import ActionSmoother
from src.robot_navigation.spatial_memory import SpatialMemory
from src.robot_navigation.wall_follower import WallFollower
from src.robot_navigation.openness_scorer import OpennessScorer
from src.robot_navigation.waypoint_planner import WaypointPlanner

import pickle
import numpy as np
import torch
import numpy.linalg as la
import json
from datetime import datetime

# Set headless mode in the simulation module
sim.HEADLESS = True


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

    # Scaler was trained only on features (not labels), so we can transform directly
    normalized = scaler.transform(features.reshape(1, -1))
    network_param = normalized.flatten()

    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param


def test_robot_movement(max_iterations=1000, goals_to_reach=2, verbose=True):
    """
    Test robot movement and track statistics.
    
    Args:
        max_iterations: Maximum number of iterations to run
        goals_to_reach: Number of goals to reach before stopping
        verbose: Whether to print progress
    
    Returns:
        Dictionary with test results and statistics
    """
    sim_env = sim.SimulationEnvironment()
    action_repeat = 20
    steering_behavior = Seek(sim_env.goal_body.position)

    # Load model
    models_path = Path(__file__).parent.parent / "models"
    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load(models_path / "saved_model.pkl"))
    model.eval()

    # Load normalization parameters
    scaler = pickle.load(open(models_path / "scaler.pkl", "rb"))

    # Initialize action smoother to reduce oscillation
    action_smoother = ActionSmoother(history_length=5, momentum_weight=0.4)

    # Initialize spatial memory to prevent oscillation loops
    spatial_memory = SpatialMemory(grid_size=30, decay_rate=0.97)

    # Initialize wall follower for systematic escape behavior
    wall_follower = WallFollower(target_distance=70, max_follow_steps=100)

    # Initialize openness scorer for evaluating open space
    openness_scorer = OpennessScorer(max_sensor_range=150)

    # Initialize waypoint planner for intermediate targets
    waypoint_planner = WaypointPlanner(waypoint_distance=120)

    # Tracking variables
    positions = []
    goal_positions = []
    actions_taken = []
    collisions = []
    distances_to_goal = []
    stuck_events = []
    turn_around_events = []
    
    goals_reached = 0
    last_position = None
    stuck_counter = 0
    iteration = 0
    total_collisions = 0
    collision_threshold = 0.3  # Slightly higher initial threshold for more actions
    consecutive_no_actions = 0  # Track consecutive iterations with no safe actions
    last_distance_to_goal = float('inf')
    no_progress_counter = 0
    
    initial_position = sim_env.robot.body.position
    initial_goal = sim_env.goal_body.position
    
    if verbose:
        print(f"Starting test...")
        print(f"Initial robot position: ({initial_position.x:.1f}, {initial_position.y:.1f})")
        print(f"Initial goal position: ({initial_goal.x:.1f}, {initial_goal.y:.1f})")
        print(f"Initial distance to goal: {la.norm(initial_goal - initial_position):.1f}")
        print("-" * 60)
    
    while goals_reached < goals_to_reach and iteration < max_iterations:
        iteration += 1
        
        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        distance_to_goal = la.norm(seek_vector)
        distances_to_goal.append(distance_to_goal)
        
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
        
        current_position = sim_env.robot.body.position
        positions.append((current_position.x, current_position.y))
        goal_positions.append((sim_env.goal_body.position.x, sim_env.goal_body.position.y))

        # Add current position to spatial memory
        spatial_memory.add_position(current_position)

        # Decay spatial memory every 10 iterations
        if iteration % 10 == 0:
            spatial_memory.decay_visits()

        if distance_to_goal < 50:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            goals_reached += 1
            collision_threshold = 0.3  # Reset threshold on goal reach
            no_progress_counter = 0
            last_distance_to_goal = float('inf')
            action_smoother.reset()  # Clear action history on goal reach
            spatial_memory.reset()  # Clear spatial memory on goal reach
            wall_follower.reset()  # Reset wall following on goal reach
            waypoint_planner.reset()  # Clear waypoints on goal reach
            if verbose:
                print(f"Iteration {iteration}: Goal reached! ({goals_reached}/{goals_to_reach})")
            continue

        # Get sensor readings for wall follower and waypoint planner
        sensor_readings = sim_env.raycasting()

        # Get navigation target (waypoint or goal) using waypoint planner
        target = waypoint_planner.get_target(
            sim_env.robot.body.position,
            sim_env.robot.body.angle,
            sim_env.goal_body.position,
            sensor_readings
        )

        # Update steering behavior to seek the current target
        steering_behavior.update_goal(target)

        # Print waypoint status if using waypoint
        if verbose and waypoint_planner.current_waypoint is not None:
            print(f"Iteration {iteration}: Using waypoint at ({target[0]:.1f}, {target[1]:.1f})")

        # Update wall follower state
        wall_follower.update(sensor_readings, stuck_counter)

        action_space = np.arange(-5, 6)
        actions_available = []
        action_predictions = {}  # Store predictions for all actions

        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler)
            prediction = model(network_param)
            # Apply sigmoid to convert logits to probabilities (0-1 range)
            prediction_value = torch.sigmoid(prediction).item()
            # Handle NaN predictions
            if np.isnan(prediction_value):
                prediction_value = 1.0  # Treat NaN as unsafe
            action_predictions[action] = prediction_value
            if prediction_value < collision_threshold:
                actions_available.append(action)

        # Get desired action from steering behavior (now pointing to target, not original goal)
        desired_action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)

        # Check if wall-following is active
        if wall_follower.active:
            # Get wall-following action
            wall_action = wall_follower.get_wall_following_action(sensor_readings)

            # Check if wall action is safe
            if wall_action in actions_available:
                closest_action = wall_action
            else:
                # Find closest safe action to wall action
                if len(actions_available) > 0:
                    closest_action = min(actions_available, key=lambda x: abs(x - wall_action))
                else:
                    # No safe actions - use wall action anyway (emergency)
                    closest_action = wall_action

            if verbose:
                print(f"Iteration {iteration}: Wall-following active ({wall_follower.preferred_side}): action={closest_action}")

            actions_taken.append(closest_action)
            action_smoother.add_action(closest_action)
            steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

            # Execute wall-following action
            for action_timestep in range(action_repeat):
                _, collision, _ = sim_env.step(steering_force)
                if collision:
                    total_collisions += 1
                    collisions.append(iteration)
                    stuck_counter = 0
                    last_position = None
                    action_smoother.reset()
                    collision_threshold = min(0.6, collision_threshold + 0.05)
                    break
            continue

        # Check for oscillation using spatial memory (only check every 5 iterations to avoid over-triggering)
        if iteration % 5 == 0 and spatial_memory.detect_oscillation(current_position, window=20, threshold=60.0):
            # Force a random strong action to break out of oscillation
            strong_actions = [a for a in actions_available if abs(a) >= 3]
            if strong_actions:
                closest_action = np.random.choice(strong_actions)
                if verbose:
                    print(f"Iteration {iteration}: Spatial oscillation detected, forcing strong action: {closest_action}")
                actions_taken.append(closest_action)
                action_smoother.add_action(closest_action)
                # Clear position history to give robot a fresh start
                spatial_memory.position_history.clear()
                steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)

                # Execute forced action
                for action_timestep in range(action_repeat):
                    _, collision, _ = sim_env.step(steering_force)
                    if collision:
                        total_collisions += 1
                        collisions.append(iteration)
                        stuck_counter = 0
                        last_position = None
                        action_smoother.reset()
                        collision_threshold = min(0.6, collision_threshold + 0.05)
                        break
                continue

        if len(actions_available) == 0:
            # If no actions are available, gradually increase threshold to allow more actions
            consecutive_no_actions += 1
            if consecutive_no_actions > 3:  # Faster threshold increase
                # Gradually increase threshold when stuck
                collision_threshold = min(0.85, collision_threshold + 0.15)
                consecutive_no_actions = 0

            # Always pick safest actions when none are available
            # Pick actions with lowest collision predictions
            valid_predictions = {k: v for k, v in action_predictions.items() if not np.isnan(v)}
            if len(valid_predictions) == 0:
                valid_predictions = action_predictions
            sorted_actions = sorted(valid_predictions.items(), key=lambda x: x[1])
            # Take the top 5 safest actions (more options)
            safest_actions = [a[0] for a in sorted_actions[:5]]
            actions_available = safest_actions
            
            turn_around_events.append(iteration)
            # Only turn around if we've been stuck for a while
            if stuck_counter > 15:  # Turn around sooner
                sim_env.turn_robot_around()
                stuck_counter = 0
                last_position = None
                action_smoother.reset()  # Clear action history when turning around
                collision_threshold = 0.5  # Reset to moderate threshold after turn
                if verbose:
                    print(f"Iteration {iteration}: No safe actions, threshold={collision_threshold:.2f}, turning around")
                continue
            elif verbose:
                print(f"Iteration {iteration}: No safe actions, threshold={collision_threshold:.2f}")
        else:
            # Reset counters when we have safe actions
            consecutive_no_actions = 0
            # Gradually lower threshold back to conservative when we have safe actions
            if collision_threshold > 0.3:
                collision_threshold = max(0.3, collision_threshold - 0.03)

        # Apply spatial memory repulsion to action selection (only if we have actions)
        # Calculate repulsion scores for each available action
        action_scores = {}
        if len(actions_available) > 0:
            # Get openness scores for all available actions
            openness_scores = openness_scorer.score_all_actions(actions_available, sensor_readings)

            # Adaptive openness weight: increase when stuck to prioritize escape
            openness_weight = 1.5 if stuck_counter > 3 else 0.8

            robot_angle = sim_env.robot.body.angle
            for action in actions_available:
                # Estimate future position if we take this action
                # action * 0.1 * pi gives the steering angle
                action_angle = robot_angle + action * 0.1 * np.pi
                # Estimate position ~30 pixels ahead in that direction
                future_x = current_position.x + 30 * np.cos(action_angle)
                future_y = current_position.y + 30 * np.sin(action_angle)
                future_position = (future_x, future_y)

                # Get repulsion score for estimated future position
                repulsion = spatial_memory.get_repulsion_score(future_position)

                # Calculate action desirability (closer to desired_action is better)
                distance_from_desired = abs(action - desired_action)

                # Get openness score for this action
                openness = openness_scores[action]

                # Score = preference for desired action - penalty for visited areas + openness bonus
                # Higher score is better
                score = -distance_from_desired - (repulsion * 2.0) + (openness * openness_weight)
                action_scores[action] = score

            # Detect thrashing and handle it
            if action_smoother.detect_thrashing(threshold=3):
                # Robot is oscillating - pick a random safe action with stronger turn
                strong_actions = [a for a in actions_available if abs(a) >= 2]
                if strong_actions:
                    closest_action = np.random.choice(strong_actions)
                    if verbose:
                        print(f"Iteration {iteration}: Thrashing detected, forcing strong turn: {closest_action}")
                else:
                    # Fall back to smoother if no strong actions available
                    closest_action = action_smoother.get_smoothed_action(actions_available, desired_action)
            else:
                # Select action with highest score (considering both desired direction and spatial repulsion)
                best_action = max(action_scores.items(), key=lambda x: x[1])[0]
                # Use action smoother to smooth the spatially-aware action
                closest_action = action_smoother.get_smoothed_action(actions_available, best_action)

        actions_taken.append(closest_action)
        action_smoother.add_action(closest_action)  # Record the action taken
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        
        # Check if robot is stuck
        if last_position is not None:
            position_change = la.norm(current_position - last_position)
            if position_change < 3.0:  # More sensitive stuck detection
                stuck_counter += 1
                if stuck_counter > 8:  # Turn around sooner when stuck
                    stuck_events.append(iteration)
                    sim_env.turn_robot_around()
                    stuck_counter = 0
                    last_position = None
                    action_smoother.reset()  # Clear action history when stuck
                    collision_threshold = 0.5  # Reset threshold
                    if verbose:
                        print(f"Iteration {iteration}: Robot stuck, forcing turn")
                    continue
            else:
                stuck_counter = 0
        last_position = current_position
        
        # Execute action
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                total_collisions += 1
                collisions.append(iteration)
                stuck_counter = 0
                last_position = None
                action_smoother.reset()  # Clear action history on collision
                # Slightly increase threshold after collision to be more permissive
                collision_threshold = min(0.6, collision_threshold + 0.05)
                break
        
        # Print progress every 50 iterations
        if verbose and iteration % 50 == 0:
            print(f"Iteration {iteration}: Position=({current_position.x:.1f}, {current_position.y:.1f}), "
                  f"Distance to goal={distance_to_goal:.1f}, "
                  f"Action={closest_action}, "
                  f"Available actions={len(actions_available)}")
    
    # Calculate statistics
    positions_array = np.array(positions)
    if len(positions_array) > 1:
        total_distance_traveled = np.sum(np.linalg.norm(np.diff(positions_array, axis=0), axis=1))
        net_displacement = la.norm(positions_array[-1] - positions_array[0])
        
        # Check for oscillation (robot returns to similar positions)
        oscillation_score = 0
        if len(positions_array) > 20:
            # Check if robot visits similar positions multiple times
            for i in range(len(positions_array) - 20):
                for j in range(i + 10, len(positions_array)):
                    if la.norm(positions_array[i] - positions_array[j]) < 20:
                        oscillation_score += 1
    else:
        total_distance_traveled = 0
        net_displacement = 0
        oscillation_score = 0
    
    # Final statistics
    final_position = positions[-1] if positions else (initial_position.x, initial_position.y)
    final_goal = goal_positions[-1] if goal_positions else (initial_goal.x, initial_goal.y)
    final_distance = la.norm(np.array(final_goal) - np.array(final_position))
    
    results = {
        'test_completed': goals_reached >= goals_to_reach,
        'iterations': iteration,
        'goals_reached': goals_reached,
        'total_collisions': total_collisions,
        'collision_events': collisions,
        'stuck_events': stuck_events,
        'turn_around_events': turn_around_events,
        'initial_position': (initial_position.x, initial_position.y),
        'final_position': final_position,
        'initial_goal': (initial_goal.x, initial_goal.y),
        'final_goal': final_goal,
        'initial_distance_to_goal': la.norm(initial_goal - initial_position),
        'final_distance_to_goal': final_distance,
        'total_distance_traveled': float(total_distance_traveled),
        'net_displacement': float(net_displacement),
        'oscillation_score': oscillation_score,
        'average_distance_to_goal': float(np.mean(distances_to_goal)) if distances_to_goal else 0,
        'min_distance_to_goal': float(np.min(distances_to_goal)) if distances_to_goal else 0,
        'max_distance_to_goal': float(np.max(distances_to_goal)) if distances_to_goal else 0,
        'action_distribution': dict(zip(*np.unique(actions_taken, return_counts=True))) if actions_taken else {},
        'positions': positions[:1000] if len(positions) > 1000 else positions,  # Limit for file size
    }
    
    return results


def print_test_results(results):
    """Print formatted test results."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test completed: {'✓' if results['test_completed'] else '✗'}")
    print(f"Iterations: {results['iterations']}")
    print(f"Goals reached: {results['goals_reached']}")
    print(f"Total collisions: {results['total_collisions']}")
    print(f"Stuck events: {len(results['stuck_events'])}")
    print(f"Turn around events: {len(results['turn_around_events'])}")
    print()
    print("Position Analysis:")
    print(f"  Initial position: ({results['initial_position'][0]:.1f}, {results['initial_position'][1]:.1f})")
    print(f"  Final position: ({results['final_position'][0]:.1f}, {results['final_position'][1]:.1f})")
    print(f"  Net displacement: {results['net_displacement']:.1f} pixels")
    print(f"  Total distance traveled: {results['total_distance_traveled']:.1f} pixels")
    print()
    print("Goal Analysis:")
    print(f"  Initial distance to goal: {results['initial_distance_to_goal']:.1f} pixels")
    print(f"  Final distance to goal: {results['final_distance_to_goal']:.1f} pixels")
    print(f"  Average distance to goal: {results['average_distance_to_goal']:.1f} pixels")
    print(f"  Min distance to goal: {results['min_distance_to_goal']:.1f} pixels")
    print(f"  Max distance to goal: {results['max_distance_to_goal']:.1f} pixels")
    print()
    print("Movement Analysis:")
    if results['net_displacement'] > 0:
        efficiency = results['net_displacement'] / results['total_distance_traveled'] if results['total_distance_traveled'] > 0 else 0
        print(f"  Movement efficiency: {efficiency:.2%}")
    print(f"  Oscillation score: {results['oscillation_score']} (lower is better)")
    if results['oscillation_score'] > 50:
        print("  ⚠ WARNING: High oscillation detected - robot may be stuck in a loop!")
    print()
    if results['action_distribution']:
        print("Action Distribution:")
        for action, count in sorted(results['action_distribution'].items()):
            print(f"  Action {action:2d}: {count:4d} times ({count/len(results['action_distribution'])*100:.1f}%)")
    print("=" * 60)


def save_test_results(results, filename=None):
    """Save test results to a JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
    
    filename = results_dir / filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    json_results = convert_to_native(results)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nTest results saved to: {filename}")
    return filename


def run_multiple_tests(num_tests=5, goals_per_test=2, max_iterations=1000):
    """Run multiple tests and aggregate results."""
    print(f"Running {num_tests} tests...")
    all_results = []
    
    for i in range(num_tests):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{num_tests}")
        print(f"{'='*60}")
        results = test_robot_movement(
            max_iterations=max_iterations,
            goals_to_reach=goals_per_test,
            verbose=True
        )
        all_results.append(results)
        print_test_results(results)
    
    # Aggregate statistics
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print(f"Tests completed: {sum(1 for r in all_results if r['test_completed'])}/{num_tests}")
    print(f"Average iterations: {np.mean([r['iterations'] for r in all_results]):.1f}")
    print(f"Average collisions: {np.mean([r['total_collisions'] for r in all_results]):.1f}")
    print(f"Average stuck events: {np.mean([len(r['stuck_events']) for r in all_results]):.1f}")
    print(f"Average net displacement: {np.mean([r['net_displacement'] for r in all_results]):.1f}")
    print(f"Average oscillation score: {np.mean([r['oscillation_score'] for r in all_results]):.1f}")
    
    # Save aggregate results
    aggregate = {
        'num_tests': num_tests,
        'tests_completed': sum(1 for r in all_results if r['test_completed']),
        'average_iterations': float(np.mean([r['iterations'] for r in all_results])),
        'average_collisions': float(np.mean([r['total_collisions'] for r in all_results])),
        'average_stuck_events': float(np.mean([len(r['stuck_events']) for r in all_results])),
        'average_net_displacement': float(np.mean([r['net_displacement'] for r in all_results])),
        'average_oscillation_score': float(np.mean([r['oscillation_score'] for r in all_results])),
        'individual_results': all_results
    }
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"aggregate_test_results_{timestamp}.json"
    save_test_results(aggregate, filename)
    
    return aggregate


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test robot movement and detect issues')
    parser.add_argument('--iterations', type=int, default=1000, help='Maximum iterations per test')
    parser.add_argument('--goals', type=int, default=2, help='Number of goals to reach')
    parser.add_argument('--tests', type=int, default=1, help='Number of tests to run')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if args.tests == 1:
        # Single test
        results = test_robot_movement(
            max_iterations=args.iterations,
            goals_to_reach=args.goals,
            verbose=not args.quiet
        )
        print_test_results(results)
        save_test_results(results)
    else:
        # Multiple tests
        aggregate = run_multiple_tests(
            num_tests=args.tests,
            goals_per_test=args.goals,
            max_iterations=args.iterations
        )

