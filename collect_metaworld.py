# File: collect_metaworld.py

import metaworld
from metaworld.policies import *
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import random
import math

def collect_metaworld_data(env_name, num_trajectories, max_path_length, save_path,
                          scaling_prob=0.8, scaling_ranges=None, noise_mean=0.0, noise_std=0.10, noise_prob=0.3, wall_prob=0.0):
    """
    Collects trajectories from a specified Metaworld environment with variable speeds,
    ensuring a more uniform speed distribution by varying scaling factor ranges across trajectory groups.

    Parameters:
    - env_name (str): Name of the Metaworld environment.
    - num_trajectories (int): Number of trajectories to collect.
    - max_path_length (int): Maximum length of each trajectory.
    - save_path (str): File path to save the collected data.
    - scaling_prob (float): Probability of applying scaling (both per-trajectory and per-action).
    - scaling_ranges (list of tuples): List of (min_scale, max_scale) tuples for each group.
    """

    # Define default scaling ranges if not provided
    if scaling_ranges is None:
        scaling_ranges = [
            (0.7, 0.9),  # Group 1: Slower trajectories
            (0.9, 1.1),  # Group 2: Normal trajectories
            (1.1, 1.3)   # Group 3: Faster trajectories
        ]

    group_size = num_trajectories // len(scaling_ranges)

    # Initialize Metaworld environment
    mt = metaworld.MT1(env_name)
    env = mt.train_classes[env_name]()
    tasks = mt.train_tasks  # All available tasks
    mt2 = metaworld.MT1('button-press-wall-v2')
    env2 = mt2.train_classes['button-press-wall-v2']()
    tasks2 = mt2.train_tasks  # All available tasks

    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'timeouts': [],
        'success': [],
    }

    # Initialize the appropriate policy based on the environment
    # Add more environment-policy mappings as needed
    policy_map = {
        'drawer-close-v2': SawyerDrawerCloseV2Policy(),
        'door-open-v2': SawyerDoorOpenV2Policy(),
        'reach-v2': SawyerReachV2Policy(),
        'pick-place-v2': SawyerPickPlaceV2Policy(),
        'button-press-v2': SawyerButtonPressV2Policy(),
        'button-press-wall-v2': SawyerButtonPressWallV2Policy(),
        'push-v2': SawyerPushV2Policy(),
        'hand-insert-v2': SawyerHandInsertV2Policy(),
        # Add other mappings here
    }

    if env_name not in policy_map:
        raise ValueError(f"No policy defined for environment '{env_name}'.")

    policy = policy_map[env_name]
    policy2=policy_map['button-press-wall-v2']

    # Calculate the number of groups
    num_groups = math.ceil(num_trajectories / group_size)
    # If there are more groups than scaling_ranges, cycle through scaling_ranges
    extended_scaling_ranges = scaling_ranges * (num_groups // len(scaling_ranges)) + scaling_ranges[:num_groups % len(scaling_ranges)]

    for traj_idx in tqdm(range(num_trajectories), desc=f"Collecting trajectories for '{env_name}'"):
        # Assign a unique seed for each trajectory to ensure diversity
        wall=random.random() < wall_prob
        current_env = env2 if wall else env
        tasks_to_use = tasks2 if wall else tasks
        current_policy= policy2 if wall else policy
        
        
        # Set the environment's task and reset
        current_env.seed(traj_idx)
        task = random.choice(tasks_to_use)
        current_env.set_task(task)


        obs= current_env.reset()
        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        success = False

        # Determine the group index for the current trajectory
        group_idx = traj_idx // group_size
        scaling_range = extended_scaling_ranges[group_idx]  # Tuple (min_scale, max_scale)

        # Decide whether to apply scaling to this trajectory
        apply_scaling = random.random() < scaling_prob

        # Determine scaling strategy if scaling is applied
        if apply_scaling:
            scaling_strategy = random.choice(['per_trajectory'])
            if scaling_strategy == 'per_trajectory':
                # Generate a single scaling factor for the entire trajectory
                scaling_factor = np.random.uniform(scaling_range[0], scaling_range[1])
        else:
            scaling_strategy = None 

        for t in range(max_path_length):
            action = current_policy.get_action(obs) 
            if apply_scaling:
                if scaling_strategy == 'per_trajectory':
                    # Scale all but the last action component (gripper) uniformly
                    scaled_action = action.copy()
                    scaled_action[:-1] *= scaling_factor
                elif scaling_strategy == 'per_action':
                    # Generate a unique scaling factor for this action
                    scaling_factor = np.random.uniform(scaling_range[0], scaling_range[1])
                    # Scale all but the last action component (gripper)
                    scaled_action = action.copy()
                    scaled_action[:-1] *= scaling_factor
                else:
                    # Fallback to original action if strategy is unknown
                    scaled_action = action.copy()
            else:
                # No scaling applied
                scaled_action = action.copy()

            # Apply Gaussian noise with the specified probability
            if random.random() < noise_prob:
                noise = np.random.normal(noise_mean, noise_std, size=scaled_action[:-1].shape)
                scaled_action[:-1] += noise  # Only apply noise to the movement components, not the gripper


            # Clip the scaled action to ensure it remains within valid bounds
            scaled_action[:-1] = np.clip(scaled_action[:-1], current_env.action_space.low[:-1], current_env.action_space.high[:-1])

            # Execute the (possibly scaled) action
            next_obs, reward, _, info = current_env.step(scaled_action)
            done = int(info.get('success', False)) == 1

            # Record data
            observations.append(obs)
            actions.append(scaled_action)
            rewards.append(reward)
            terminals.append(done)
            timeouts.append(t == (max_path_length - 1))
            obs = next_obs

            if done:
                if info.get('success', False):
                    success = True
                break

        # Append trajectory data
        data['observations'].append(np.array(observations))
        data['actions'].append(np.array(actions))
        data['rewards'].append(np.array(rewards))
        data['terminals'].append(np.array(terminals))
        data['timeouts'].append(np.array(timeouts))
        data['success'].append(success)

    # Save the collected data
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to '{save_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Metaworld Trajectories with Variable Speeds")
    parser.add_argument('--env_name', type=str, default='drawer-close-v2',
                        help='Name of the Metaworld environment (e.g., drawer-close-v2).')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of trajectories to collect.')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Maximum number of steps per trajectory.')
    parser.add_argument('--save_path', type=str, default='metaworld_drawer_close_data_uniform.pkl',
                        help='File path to save the collected data.')
    parser.add_argument('--scaling_prob', type=float, default=0.9,
                        help='Probability of applying speed scaling to a trajectory.')


    args = parser.parse_args()

    default_scaling_ranges = [
        #(0.01, 0.1),
        #(0.1, 0.2),
        (0.2, 0.25),
        (0.25, 0.3),
        (0.3, 0.5),
        (0.5,0.6),   
        (0.6, 0.8),
        (0.8, 0.9),
    ]

    collect_metaworld_data(
        env_name=args.env_name,
        num_trajectories=args.num_trajectories,
        max_path_length=args.max_path_length,
        save_path=args.save_path,
        scaling_prob=args.scaling_prob,
        scaling_ranges=default_scaling_ranges
    )
