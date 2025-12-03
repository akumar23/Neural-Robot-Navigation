import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.steering import Wander
from src.robot_navigation import simulation as sim
from src.robot_navigation.feature_engineering import engineer_features

import numpy as np
from collections import deque


def collect_training_data(total_actions):
    # set-up environment
    sim_env = sim.SimulationEnvironment()

    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    network_params = []

    # Track action history for temporal features (last 5 actions)
    action_history = deque(maxlen=5)

    for action_i in range(total_actions):
        progress = 100 * float(action_i) / total_actions
        if action_i % 1000 == 0:
            print(f'Collecting Training Data {progress:.1f}%', flush=True)

        # steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()

                if action_timestep < action_repeat * .3:  # in case prior action caused collision
                    network_params[-1][-1] = collision  # share collision result with prior action
                break

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

        # Add action to history after using it (for next iteration)
        action_history.append(action)

        # Add collision label to features (20 features + 1 label = 21 columns)
        features = np.append(features, collision)

        network_params.append(features)

    # Save to data directory
    data_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    np.savetxt(data_path, network_params, delimiter=",")

    # Verify dimensions
    print(f"\nData collection complete!")
    print(f"Total samples: {len(network_params)}")
    print(f"Features per sample: {len(network_params[0])} (20 features + 1 collision label)")


if __name__ == '__main__':
    # Collect training data for model accuracy
    # Using 100,000 samples for full dataset (as per requirements)
    total_actions = 100000
    print(f"Collecting {total_actions} training samples with enhanced features (20D)...")
    print("This may take a while. Progress will be shown.")
    collect_training_data(total_actions)
    print(f"\nTraining data collection complete! Collected {total_actions} samples.")
