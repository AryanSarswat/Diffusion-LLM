import metaworld
from metaworld.policies import *
import numpy as np
import pickle
import imageio
from tqdm import tqdm
import argparse
from gymnasium.spaces import Box
import random
import os

os.environ['MUJOCO_GL'] = 'egl'

def collect_metaworld_data(env_name, num_trajectories, max_path_length, save_path, render_videos=False, video_path=None, video_fps=30, video_dim=(256, 256)):
    # Initialize Metaworld environment
    mt = metaworld.MT1(env_name)
    env = mt.train_classes[env_name](render_mode='rgb_array')
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

    policy_map = {
        'drawer-close-v2': SawyerDrawerCloseV2Policy(),
        'door-open-v2': SawyerDoorOpenV2Policy(),
        'reach-v2': SawyerReachV2Policy(),
        'pick-place-v2': SawyerPickPlaceV2Policy(),
        'button-press-v2': SawyerButtonPressV2Policy(),
        'button-press-wall-v2': SawyerButtonPressWallV2Policy(),
        'push-v2': SawyerPushV2Policy(),
        'hand-insert-v2': SawyerHandInsertV2Policy(),
        'pick-place-v2' : SawyerPickPlaceV2Policy(),
        'pick-place-wall-v2' : SawyerPickPlaceWallV2Policy(),
        # Add other mappings here
    }

    if env_name not in policy_map:
        raise ValueError(f"No policy defined for environment '{env_name}'.")

    policy = policy_map[env_name]

    # Ensure video path exists
    if render_videos and video_path is not None:
        os.makedirs(video_path, exist_ok=True)

    for traj_idx in tqdm(range(num_trajectories), desc=f"Collecting trajectories for {env_name}"):
        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        success = False
        timeout = False
        done = False

        # Set up video frames list if rendering is enabled
        frames = []

        env = mt.train_classes[env_name](render_mode='rgb_array')
        task = random.choice(mt.train_tasks)
        env.set_task(task)
        obs, _ = env.reset()
        
        for t in range(max_path_length):
            action = policy.get_action(obs)
            next_obs, reward, _, _, info = env.step(action)
            done = int(info['success']) == 1
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            obs = next_obs

            # Render and store frames if enabled
            if render_videos and (traj_idx % 100 == 0):
                img = env.render()
                frames.append(img)

            if done:
                if info.get('success', False):  # Check if the task was successful
                    success = True
                else:
                    timeout = True
                break
                
            timeouts.append(timeout)
        
        # Save trajectory data
        data['observations'].append(np.array(observations))
        data['actions'].append(np.array(actions))
        data['rewards'].append(np.array(rewards))
        data['terminals'].append(np.array(terminals))
        data['timeouts'].append(np.array([timeout]))
        data['success'].append(success)

        # Save video if enabled
        if render_videos and (video_path is not None) and (traj_idx % 100 == 0):
            video_file = os.path.join(video_path, f'TEST_trajectory_{traj_idx}.mp4')
            imageio.mimwrite(video_file, frames, fps=video_fps)
            print(f"Saved video to {video_file}")

    # Save the dataset
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='button-press-wall-v2')
    parser.add_argument('--num_trajectories', type=int, default=1000)
    parser.add_argument('--max_path_length', type=int, default=250)
    parser.add_argument('--save_path', type=str, default='metaworld_drawer_close_data2.pkl')
    parser.add_argument('--render_videos', action='store_true', help="Enable video rendering")
    parser.add_argument('--video_path', type=str, default='videos', help="Directory to save rendered videos")
    parser.add_argument('--video_fps', type=int, default=30, help="Frames per second for the video")
    parser.add_argument('--video_dim', type=int, nargs=2, default=[256, 256], help="Dimensions for the rendered video (width height)")
    args = parser.parse_args()

    collect_metaworld_data(
        args.env_name,
        args.num_trajectories,
        args.max_path_length,
        args.save_path,
        render_videos=args.render_videos,
        video_path=args.video_path,
        video_fps=args.video_fps,
        video_dim=args.video_dim
    )

