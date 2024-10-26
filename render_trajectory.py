# render_trajectory.py

import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio

import metaworld
from mujoco_py import MjRenderContextOffscreen, MjSim, load_model_from_path

def parse_args():
    parser = argparse.ArgumentParser(description="Render a trajectory from the Metaworld dataset.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset pickle file.')
    parser.add_argument('--env', type=str, required=True, help='Metaworld task name (e.g., "drawer-close-v2").')
    parser.add_argument('--episode', type=int, default=None, help='Episode index to render. If not set, selects randomly.')
    parser.add_argument('--savepath', type=str, default='trajectory_video.mp4', help='Path to save the rendered video.')
    parser.add_argument('--dim', type=int, nargs=2, default=[256, 256], help='Dimensions for rendering images (height width).')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video.')
    parser.add_argument('--show_first', action='store_true', help='Whether to display the first frame.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    return parser.parse_args()

def load_data(data_path):
    """
    Loads the dataset from the pickle file.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def initialize_env(env_name, seed=None):
    """
    Initializes the Metaworld environment.
    """
    mt = metaworld.ML1(env_name)
    env = mt.train_classes[env_name]()  # Use 'rgb_array' for offscreen rendering
    task = random.choice(mt.train_tasks)
    env.set_task(task)
    if seed is not None:
        env.seed(seed)
    return env

def select_episode(data, episode_index=None):
    """
    Selects a specific episode from the dataset.
    If episode_index is None, selects a random episode.
    """
    n_episodes = len(data['observations'])
    if episode_index is None:
        episode_index = random.randint(0, n_episodes - 1)
    elif episode_index < 0 or episode_index >= n_episodes:
        raise ValueError(f'Episode index out of range. Must be between 0 and {n_episodes - 1}')
    print(f'Selected Episode: {episode_index}')
    return episode_index

def set_env_state(env, observation):
    """
    Sets the environment's state based on the observation.
    This function needs to be tailored based on how observations map to the environment's state.
    
    Typically, observations include positions and velocities which correspond to qpos and qvel in MuJoCo.
    """
    sim = env.sim
    # Example mapping: First N elements are qpos, next M are qvel
    nq = sim.model.nq
    nv = sim.model.nv
    qpos = observation[:nq]
    qvel = observation[nq:nq + nv]
    
    # Ensure correct shapes
    qpos = np.asarray(qpos).flatten()
    qvel = np.asarray(qvel).flatten()
    
    # Set the simulation state
    sim.reset()
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.forward()

def render_episode(env, data, episode_index, savepath, dim=(256, 256), fps=10, show_first=False):
    """
    Renders a specific episode and saves it as a video.
    """
    observations = data['observations'][episode_index]  # [episode_length x observation_dim]
    actions = data['actions'][episode_index]  # [episode_length x action_dim]
    
    # Initialize the renderer
    viewer = MjRenderContextOffscreen(env.sim)
    env.sim.add_render_context(viewer)
    
    frames = []
    
    for step in range(len(observations)):
        obs = observations[step]
        set_env_state(env, obs)
        
        # Render the scene
        viewer.render(*dim)
        img = viewer.read_pixels(*dim, depth=False)
        img = img[::-1, :, :]  # Flip vertically
        frames.append(img)
    
    # Save as video
    imageio.mimwrite(savepath, frames, fps=fps)
    print(f'Saved video to {savepath}')
    
    # Optionally display the first frame
    if show_first:
        plt.imshow(frames[0])
        plt.title('First Frame of the Episode')
        plt.axis('off')
        plt.show()

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Load data
    print('Loading data...')
    data = load_data(args.data_path)
    print("DATA",data)
    pr
    
    # Initialize environment
    print('Initializing environment...')
    env = initialize_env(args.env, seed=args.seed)
    
    # Select episode
    episode_index = select_episode(data, args.episode)
    
    # Render and save video
    print('Rendering episode...')
    render_episode(
        env=env,
        data=data,
        episode_index=episode_index,
        savepath=args.savepath,
        dim=tuple(args.dim),
        fps=args.fps,
        show_first=args.show_first
    )
    
    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
