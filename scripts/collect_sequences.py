"""
Sequence Data Collection for LSTM Training.

Collects episodes (sequences) instead of individual samples for temporal learning.
Each episode represents a complete trajectory from start to termination:
- Goal reached: Robot gets within goal_reach_distance of goal
- Collision: Robot collides with wall
- Timeout: Episode exceeds max_steps without goal/collision
- Stuck: Robot gets stuck (no movement for extended period)

Episode structure:
- List of timesteps, each containing:
  - features: 20D feature vector (sensors + spatial + goal + temporal + spatial-goal + action)
  - action: Steering action taken (-5 to +5)
  - collision: Binary collision label (0 or 1) for this timestep
  - goal_reached: Binary goal reached label (0 or 1)

Saves as pickled list of episodes for preservation of sequence structure.
"""

import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set headless mode for data collection (no visualization needed)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from src.robot_navigation.steering import Wander
from src.robot_navigation import simulation as sim
from src.robot_navigation.feature_engineering import engineer_features

import numpy as np
import pickle
from collections import deque


def collect_episode(sim_env, steering_behavior, max_steps=500, action_repeat=100):
    """
    Collect a single episode (sequence) of robot navigation.

    Args:
        sim_env: Simulation environment
        steering_behavior: Steering behavior (Wander for exploration)
        max_steps: Maximum timesteps before episode timeout
        action_repeat: Number of physics steps per action decision

    Returns:
        episode_data: List of timestep dictionaries
        episode_info: Dictionary with episode metadata
    """
    episode_data = []
    action_history = deque(maxlen=5)

    # Episode metadata
    episode_info = {
        'length': 0,
        'outcome': 'timeout',  # 'goal', 'collision', 'timeout', 'stuck'
        'total_collisions': 0,
        'distance_traveled': 0
    }

    last_position = sim_env.robot.body.position
    stuck_counter = 0
    stuck_threshold = 8  # Consecutive steps with no movement

    for step in range(max_steps):
        # Get action from steering behavior
        action, steering_force = steering_behavior.get_action(step, sim_env.robot.body.angle)

        # Get sensor readings before taking action
        sensor_readings = sim_env.raycasting()

        # Get robot state for enhanced features
        robot_pos = sim_env.robot.body.position
        robot_angle = sim_env.robot.body.angle
        goal_pos = sim_env.goal_body.position
        velocity = sim_env.robot.body.velocity

        # Create enhanced feature vector (20D)
        features = engineer_features(
            sensor_readings, action,
            robot_pos=robot_pos,
            robot_angle=robot_angle,
            goal_pos=goal_pos,
            velocity=velocity,
            action_history=action_history
        )

        # Execute action and observe outcome
        collision_occurred = False
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                collision_occurred = True
                steering_behavior.reset_action()
                break

        # Check if goal reached
        distance_to_goal = np.linalg.norm(
            np.array([goal_pos.x - robot_pos.x, goal_pos.y - robot_pos.y])
        )
        goal_reached = distance_to_goal < 20.0  # Goal reach threshold

        # Store timestep data
        timestep = {
            'features': features,
            'action': action,
            'collision': 1 if collision_occurred else 0,
            'goal_reached': 1 if goal_reached else 0
        }
        episode_data.append(timestep)

        # Update action history
        action_history.append(action)

        # Update episode metadata
        episode_info['length'] += 1
        if collision_occurred:
            episode_info['total_collisions'] += 1

        # Calculate distance traveled
        current_position = sim_env.robot.body.position
        position_change = np.linalg.norm(
            np.array([current_position.x - last_position.x,
                     current_position.y - last_position.y])
        )
        episode_info['distance_traveled'] += position_change

        # Check for stuck (no movement)
        if position_change < 3.0:
            stuck_counter += 1
        else:
            stuck_counter = 0

        last_position = current_position

        # Episode termination conditions
        if goal_reached:
            episode_info['outcome'] = 'goal'
            break
        elif collision_occurred:
            episode_info['outcome'] = 'collision'
            # Continue episode after collision (don't terminate)
            # This allows learning collision recovery behaviors
        elif stuck_counter >= stuck_threshold:
            episode_info['outcome'] = 'stuck'
            break

    return episode_data, episode_info


def collect_training_sequences(num_episodes, max_steps_per_episode=500):
    """
    Collect multiple episodes for LSTM training.

    Args:
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum timesteps per episode

    Returns:
        all_episodes: List of episodes (each episode is a list of timesteps)
        all_episode_info: List of episode metadata
    """
    all_episodes = []
    all_episode_info = []

    # Statistics
    total_timesteps = 0
    outcomes = {'goal': 0, 'collision': 0, 'timeout': 0, 'stuck': 0}

    # Set up environment
    sim_env = sim.SimulationEnvironment()
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    print(f"Collecting {num_episodes} episodes for LSTM training...")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"This may take a while. Progress will be shown every 10 episodes.\n")

    for episode_idx in range(num_episodes):
        # Reset environment for new episode
        sim_env.reset()
        steering_behavior.reset_action()

        # Collect episode
        episode_data, episode_info = collect_episode(
            sim_env, steering_behavior,
            max_steps=max_steps_per_episode,
            action_repeat=action_repeat
        )

        # Store episode if it has data
        if len(episode_data) > 0:
            all_episodes.append(episode_data)
            all_episode_info.append(episode_info)

            # Update statistics
            total_timesteps += episode_info['length']
            outcomes[episode_info['outcome']] += 1

        # Progress report
        if (episode_idx + 1) % 10 == 0 or episode_idx == 0:
            progress = 100 * (episode_idx + 1) / num_episodes
            avg_length = total_timesteps / (episode_idx + 1) if episode_idx > 0 else episode_info['length']
            print(f"Progress: {progress:.1f}% ({episode_idx + 1}/{num_episodes} episodes)")
            print(f"  Avg episode length: {avg_length:.1f} timesteps")
            print(f"  Total timesteps: {total_timesteps}")
            print(f"  Outcomes - Goal: {outcomes['goal']}, Collision: {outcomes['collision']}, "
                  f"Timeout: {outcomes['timeout']}, Stuck: {outcomes['stuck']}\n")

    print("\n" + "="*60)
    print("SEQUENCE COLLECTION SUMMARY")
    print("="*60)
    print(f"Total episodes collected: {len(all_episodes)}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Average episode length: {total_timesteps / len(all_episodes):.1f}")
    print(f"\nEpisode outcomes:")
    for outcome, count in outcomes.items():
        pct = 100 * count / len(all_episodes) if len(all_episodes) > 0 else 0
        print(f"  {outcome.capitalize()}: {count} ({pct:.1f}%)")

    # Calculate collision rate across all timesteps
    total_collisions = sum(
        sum(1 for timestep in episode if timestep['collision'] == 1)
        for episode in all_episodes
    )
    collision_rate = total_collisions / total_timesteps if total_timesteps > 0 else 0
    print(f"\nOverall collision rate: {collision_rate:.2%}")
    print("="*60)

    return all_episodes, all_episode_info


def main():
    # Collect episodes for LSTM training
    # Target: 1500 episodes to generate ~150k-200k total timesteps
    num_episodes = 1500
    max_steps_per_episode = 500  # Timeout after 500 steps

    print("="*60)
    print("LSTM SEQUENCE DATA COLLECTION")
    print("="*60)
    print(f"Collecting {num_episodes} episodes...")
    print(f"Each episode will be a sequence from start to termination.")
    print(f"Expected total timesteps: ~150k-200k")
    print("="*60 + "\n")

    # Collect sequences
    all_episodes, all_episode_info = collect_training_sequences(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode
    )

    # Save to data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    sequences_path = data_dir / "training_sequences.pkl"
    metadata_path = data_dir / "sequences_metadata.pkl"

    # Save episodes as pickle files
    with open(sequences_path, 'wb') as f:
        pickle.dump(all_episodes, f)
    print(f"\nSaved sequences to: {sequences_path}")

    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_episode_info, f)
    print(f"Saved metadata to: {metadata_path}")

    print("\nSequence collection complete!")


if __name__ == '__main__':
    main()
