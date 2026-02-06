import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MetaworldRenderer:
    '''
        Renderer tailored for Metaworld environments.
        Provides the same interface as MuJoCoRenderer.
    '''

    def __init__(self, env):
        '''
            Initialize the renderer with a Metaworld environment.

            Parameters:
            - env (metaworld.envs.mujoco.env_name): An instance of a Metaworld environment.
        '''
        self.env = env

        # Access simulation attributes from the environment
        self.sim = getattr(self.env, 'sim', None) or getattr(self.env, '_sim', None)
        self.model = getattr(self.env, 'model', None)
        self.data = getattr(self.env, 'data', None)

        if self.sim is None:
            # Attempt to access sim via a method if not directly available
            try:
                self.sim = self.env._get_sim()
            except AttributeError:
                raise AttributeError('Cannot access simulation in the environment.')

        # Determine observation and action dimensions
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)

        # Initialize the offscreen renderer using the existing sim
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.sim)
            self.sim.add_render_context(self.viewer)
        except Exception as e:
            print('[MetaworldRenderer] Warning: could not initialize offscreen renderer:', e)
            self.viewer = None


    def pad_observation(self, observation):
        '''
            Pad the observation if necessary. This method can be customized based on observation structure.

            Parameters:
            - observation (np.array): The observation to pad.

            Returns:
            - state (np.array): The padded state.
        '''
        # Example: Adding a dummy position element if required
        # Adjust based on your observation structure
        # Here, we assume no padding is necessary
        return observation

    def pad_observations(self, observations):
        '''
            Pad multiple observations.

            Parameters:
            - observations (np.array): Array of observations [batch_size x obs_dim].

            Returns:
            - states (np.array): Array of padded states.
        '''
        return observations  # Modify if padding is needed

    def set_state_from_observation(self, observation):
        '''
            Set the simulation state based on the observation.

            Parameters:
            - observation (np.array): The observation to map to the simulation state.
        '''
        # Mapping depends on the observation structure.
        # Here's a generic approach; customize based on your specific environment.

        # Example for Sawyer environments:
        # Assuming observation structure: [ee_pos(3), ee_vel(3), obj_pos(3), ...]
        # And simulation state consists of [qpos (e.g., 7 for Sawyer arm), qvel (7), ...]
        
        # Number of qpos and qvel in the simulation
        nq = self.sim.model.nq
        nv = self.sim.model.nv

        # Initialize qpos and qvel with zeros
        qpos = self.sim.data.qpos.ravel().copy()
        qvel = self.sim.data.qvel.ravel().copy()

        # Example mapping: Assign the first nq elements from observation to qpos
        # Adjust indices based on actual observation structure
        qpos[:len(qpos)] = observation[:len(qpos)]
        # Similarly for qvel if available
        if len(observation) >= nq + nv:
            qvel[:len(qvel)] = observation[len(qpos):len(qpos) + len(qvel)]

        # Set the state in the simulation
        self.sim.reset()
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def set_state(self, qpos, qvel):
        '''
            Directly set the simulation state.

            Parameters:
            - qpos (np.array): Joint positions.
            - qvel (np.array): Joint velocities.
        '''
        self.sim.reset()
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def render(self, observation, dim=256, render_kwargs=None, partial=False, qvel=True, conditions=None):
        '''
            Render a single observation.

            Parameters:
            - observation (np.array): The observation to render.
            - dim (int or tuple): Dimensions of the rendered image.
            - render_kwargs (dict): Additional rendering parameters.
            - partial (bool): Whether to partially render (if applicable).
            - qvel (bool): Whether to include velocities in the state.
            - conditions (dict): Additional conditions for rendering.

            Returns:
            - img (np.array): The rendered image.
        '''
        if isinstance(dim, int):
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), dtype=np.uint8)

        # Default rendering parameters
        if render_kwargs is None:
            render_kwargs = {
                'distance': 1.5,
                'lookat': [0, 0, 1],
                'elevation': -20,
                'azimuth': 90,
            }

        # Set camera attributes correctly
        if 'lookat' in render_kwargs:
            self.viewer.cam.lookat[:] = render_kwargs['lookat']
        if 'distance' in render_kwargs:
            self.viewer.cam.distance = render_kwargs['distance']
        if 'elevation' in render_kwargs:
            self.viewer.cam.elevation = render_kwargs['elevation']
        if 'azimuth' in render_kwargs:
            self.viewer.cam.azimuth = render_kwargs['azimuth']

        # Optionally modify observation based on 'partial'
        if partial:
            observation = self.pad_observation(observation)
        
        # Optionally include velocities
        if not qvel:
            # Zero out velocities if not required
            # Adjust based on your observation structure
            # Here, we assume velocities are in the latter part of the observation
            qpos_dim = self.sim.model.nq
            observation = observation.copy()
            observation[qpos_dim:qpos_dim + self.sim.model.nv] = 0

        # Set the simulation state
        self.set_state_from_observation(observation)

        # Render the image
        self.viewer.render(*dim)
        img = self.viewer.read_pixels(*dim, depth=False)
        img = img[::-1, :, :]  # Flip vertically

        return img

    def _renders(self, observations, **kwargs):
        '''
            Render multiple observations.

            Parameters:
            - observations (np.array): Array of observations [batch_size x obs_dim].

            Returns:
            - images (np.array): Array of rendered images [batch_size x H x W x C].
        '''
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, **kwargs):
        '''
            Alias for _renders.

            Parameters:
            - samples (np.array): Array of observations.

            Returns:
            - images (np.array): Array of rendered images.
        '''
        return self._renders(samples, **kwargs)

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        '''
            Composite multiple rendered images into a single image.

            Parameters:
            - savepath (str): Path to save the composite image.
            - paths (list or np.array): List of trajectories [batch_size x horizon x obs_dim].
            - dim (tuple): Dimensions for rendering each image.
            - **kwargs: Additional keyword arguments for rendering.

            Returns:
            - composite (np.array): The composite image.
        '''
        render_kwargs = {
        'distance': 1.5,
        'lookat': [0, 0, 1],
        'elevation': -20,
        'azimuth': 90,
        }
        
        images = []
        for path in paths:
            # Ensure the path is a 2D array
            path = atmost_2d(path)
            
            # Render the path as images
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs)
            images.append(img)
        
        # Stack images along the batch axis (assumes each img is H x W x C)
        images = np.concatenate(images, axis=0)

        # If images have more than 3 dimensions, reduce to 3D (H x W x C)
        if images.ndim > 3:
            # Example: Take the first batch dimension
            images = images[0]  # Or process it in another way if this isn't sufficient

        # Ensure images are 3D before saving
        assert images.ndim == 3, f"Expected 3D image array (H x W x C), got {images.ndim}D."

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images
        
    def render_rollout(self, savepath, states, **video_kwargs):
        '''
            Render a rollout and save as a video.

            Parameters:
            - savepath (str): Path to save the video.
            - states (list or np.array): List of states [H x obs_dim].
            - **video_kwargs: Additional keyword arguments for video saving.
        '''
        if isinstance(states, list):
            states = np.array(states)
        images = self.renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        '''
            Render a planned trajectory alongside the real trajectory.

            Parameters:
            - savepath (str): Path to save the video.
            - actions (list or np.array): List of actions [batch_size x horizon x action_dim].
            - observations_pred (list or np.array): Predicted observations [batch_size x horizon x obs_dim].
            - state (np.array): Initial state [obs_dim].
            - fps (int): Frames per second for the video.
        '''
        # Generate real observations based on actions
        observations_real = self.rollouts_from_state(state, actions)

        # Ensure observations_real and observations_pred have the same length
        observations_real = observations_real[:, :-1, :]  # Trim the last state

        # Render predicted and real observations
        images_pred = np.stack([
            self.renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])
        images_real = np.stack([
            self.renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        # Concatenate images side by side
        images = np.concatenate([images_pred, images_real], axis=-2)

        # Save the video
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            Render the diffusion process over multiple timesteps.

            Parameters:
            - savepath (str): Path to save the video.
            - diffusion_path (np.array): Array representing the diffusion path [n_steps x batch_size x 1 x horizon x joined_dim].
            - **video_kwargs: Additional keyword arguments for video saving.
        '''
        render_kwargs = {
            'distance': 1.5,
            'lookat': [0, 0, 1],
            'elevation': -20,
            'azimuth': 90,
        }

        diffusion_path = to_np(diffusion_path)
        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[MetaworldRenderer] Diffusion: {t} / {n_diffusion_steps}')

            # [batch_size x horizon x observation_dim]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(512, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def rollouts_from_state(self, state, actions_l):
        '''
            Generate rollouts from a given state and a list of actions.

            Parameters:
            - state (np.array): Initial state [obs_dim].
            - actions_l (list or np.array): List of actions [batch_size x horizon x action_dim].

            Returns:
            - rollouts (np.array): Generated rollouts [batch_size x (horizon+1) x obs_dim].
        '''
        rollouts = np.stack([
            self.rollout_from_state(state, actions)
            for actions in actions_l
        ])
        return rollouts

    def rollout_from_state(self, state, actions):
        '''
            Generate a single rollout from a given state and a list of actions.

            Parameters:
            - state (np.array): Initial state [obs_dim].
            - actions (list or np.array): List of actions [horizon x action_dim].

            Returns:
            - observations (np.array): Generated observations [horizon+1 x obs_dim].
        '''
        # Set the initial state
        self.set_state_from_observation(state)

        observations = [self.env._get_obs()]
        for act in actions:
            obs, rew, term, _ = self.env.step(act)
            observations.append(obs)
            if term:
                break
        # Pad observations if terminated early
        for _ in range(len(observations), len(actions) + 1):
            observations.append(np.zeros_like(observations[0]))
        return np.stack(observations)

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
