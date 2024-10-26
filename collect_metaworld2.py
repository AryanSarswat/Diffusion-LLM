import metaworld
from metaworld.policies import *
import numpy as np
import pickle
import imageio
from tqdm import tqdm
import argparse
import random
import os


def collect_metaworld_data(env_name, num_trajectories, max_path_length, save_path, render_videos=False, video_path=None, video_fps=10, video_dim=(256, 256)):
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

    # Ensure video path exists
    if render_videos and video_path is not None:
        os.makedirs(video_path, exist_ok=True)

    for traj_idx in tqdm(range(num_trajectories), desc=f"Collecting trajectories for {env_name}"):
        obs = env.reset()
        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        success = False
        done = False

        # Set up video frames list if rendering is enabled
        frames = []

        for t in range(max_path_length):
            action = policy.get_action(obs)
            next_obs, reward, _, info = env.step(action)
            done = int(info['success']) == 1
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            obs = next_obs

            # Render and store frames if enabled
            if render_videos:
                img = env.render(offscreen=True)
                frames.append(img)

            if done:
                if info.get('success', False):  # Check if the task was successful
                    success = True
                break

        # Save trajectory data
        data['observations'].append(np.array(observations))
        data['actions'].append(np.array(actions))
        data['rewards'].append(np.array(rewards))
        data['terminals'].append(np.array(terminals))
        data['timeouts'].append(np.array(timeouts))
        data['success'].append(success)

        # Save video if enabled
        if render_videos and video_path is not None:
            video_file = os.path.join(video_path, f'trajectory_{traj_idx}.mp4')
            imageio.mimwrite(video_file, frames, fps=video_fps)
            print(f"Saved video to {video_file}")

    # Save the dataset
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='drawer-close-v2')
    parser.add_argument('--num_trajectories', type=int, default=1000)
    parser.add_argument('--max_path_length', type=int, default=500)
    parser.add_argument('--save_path', type=str, default='metaworld_drawer_close_data2.pkl')
    parser.add_argument('--render_videos', action='store_true', help="Enable video rendering")
    parser.add_argument('--video_path', type=str, default='videos', help="Directory to save rendered videos")
    parser.add_argument('--video_fps', type=int, default=10, help="Frames per second for the video")
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

