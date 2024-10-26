import metaworld
from metaworld.policies import *
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import random

def collect_metaworld_data(env_name, num_trajectories, max_path_length, save_path):
    # Initialize Metaworld environment
    mt = metaworld.ML1(env_name)
    env = mt.train_classes[env_name]()
    task = random.choice(mt.train_tasks)
    env.set_task(task)

    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'timeouts': [],
        'success': [],
    }
    policy = SawyerDrawerCloseV2Policy()

    for traj_idx in tqdm(range(num_trajectories), desc=f"Collecting trajectories for {env_name}"):
        env.seed(traj_idx)
        obs= env.reset()
        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        success = False
        for t in range(max_path_length):
            action = policy.get_action(obs)
            next_obs, reward, _, info = env.step(action)
            done = int(info['success']) == 1
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            timeouts.append(t == (max_path_length - 1))
            obs = next_obs
            if done:
                if info.get('success', False):  # Check if the task was successful
                    success = True
                break
        data['observations'].append(np.array(observations))
        data['actions'].append(np.array(actions))
        data['rewards'].append(np.array(rewards))
        data['terminals'].append(np.array(terminals))
        data['timeouts'].append(np.array(timeouts))
        data['success'].append(success)

    # Save the data
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='drawer-close-v2')
    parser.add_argument('--num_trajectories', type=int, default=3000)
    parser.add_argument('--max_path_length', type=int, default=500)
    parser.add_argument('--save_path', type=str, default='metaworld_drawer_close_data2.pkl')
    args = parser.parse_args()
    collect_metaworld_data(args.env_name, args.num_trajectories, args.max_path_length, args.save_path)
