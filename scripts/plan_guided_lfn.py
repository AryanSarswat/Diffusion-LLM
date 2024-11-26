# File: /scripts/plan_guided.py

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.guides import CustomGuide
import imageio
import os
import torch
import numpy as np
from tqdm import tqdm

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'your-dataset-name'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## Load diffusion model from disk
diffusion_experiment = utils.load_diffusion(
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)


diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset 
renderer = diffusion_experiment.renderer



def predefined_loss_fn_(x, obs_dim,action_dim,normalizer):
    # Extract actions: shape [batch_size, horizon, action_dim]
    actions = x[:, :, :action_dim]
    actions = actions * torch.tensor(normalizer.normalizers['actions'].stds, device=x.device) + torch.tensor(normalizer.normalizers['actions'].means, device=x.device)
    # Compute the norm (speed) of each action vector: shape [batch_size, horizon]
    speeds = torch.linalg.norm(actions, dim=-1)

    # Compute the mean speed per trajectory: shape [batch_size]
    mean_speeds = torch.mean(speeds, dim=1)

    # Define loss as negative mean speed: shape [batch_size]
    loss_per_trajectory = mean_speeds

    return loss_per_trajectory  # Shape: [batch_size]
import torch.nn.functional as F
def predefined_loss_fn(x, obs_dim,action_dim,normalizer, wall_pos=None, min_safe_dist=0.3):
    # Extract hand positions from observations: shape [batch_size, horizon, 3]
    scale=10
    if wall_pos is None:
        wall_pos = torch.tensor([0.1, 0.6, 0.0], device=x.device, dtype=torch.double)
    else:
        wall_pos = wall_pos.to(x.device)
        
    # Unnormalize observations
    obs = x[:, :, action_dim:].to(dtype=torch.double)  # [batch_size, horizon, obs_dim]
    obs_stds = torch.from_numpy(normalizer.normalizers['observations'].stds).to(x.device).to(x.dtype)
    obs_means = torch.from_numpy(normalizer.normalizers['observations'].means).to(x.device).to(x.dtype)
    
    # Reshape for broadcasting
    obs_stds = obs_stds.view(1, 1, -1)
    obs_means = obs_means.view(1, 1, -1)
    
    obs = obs * obs_stds + obs_means   # [batch_size, horizon, obs_dim]
    
    # Extract hand positions
    print("OBS_LOSS",obs)
    hand_positions = obs[:, :, :3] # [batch_size, horizon, 3]

    # Compute distances
    distances_to_wall = torch.linalg.norm(hand_positions - wall_pos, dim=-1)*100  # [batch_size, horizon]

    # Penalties using F.relu to maintain differentiability
    penalties = F.relu(min_safe_dist - distances_to_wall + 1e-4) ** 2  # [batch_size, horizon]

    # Mean penalty per trajectory
    loss_per_trajectory = penalties.mean(dim=1)  # [batch_size]
    print("LOSS: ",loss_per_trajectory)

    return loss_per_trajectory  # Shape: [batch_size]

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def predefined_loss_fn1(x, obs_dim, action_dim, normalizer, 
                       wall_pos=[0.1, 0.6, 0.075], 
                       wall_half_size=[0.1, 0.01, 0.12], 
                       min_safe_dist=0.1, 
                       delta_t=1.0, 
                       scaling_factor=10.0, 
                       return_penetration=False):
    """
    Loss function to penalize trajectories that bring the hand too close to a specified wall.
    Penalizes only actions moving towards the wall within the unsafe zone.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, horizon, transition_dim].
        obs_dim (int): Observation dimension offset in x.
        action_dim (int): Action dimension in x.
        normalizer (object): Normalizer with 'actions' and 'observations' containing 'stds' and 'means'.
        wall_pos (list or tuple): Center position of the wall [x, y, z].
        wall_half_size (list or tuple): Half-dimensions of the wall along each axis [dx, dy, dz].
        min_safe_dist (float): Minimum safe distance from the wall.
        delta_t (float): Time step size for action integration.
        scaling_factor (float): Scaling factor for penalties.
        return_penetration (bool): Whether to return detailed penetration information.

    Returns:
        torch.Tensor: Loss per trajectory of shape [batch_size].
        (Optional) dict: Penetration details if `return_penetration` is True.
    """
    # Convert wall parameters to tensors and reshape for broadcasting
    wall_pos = torch.tensor(wall_pos, device=x.device, dtype=x.dtype).view(1, 1, 3)  # [1, 1, 3]
    wall_half_size = torch.tensor(wall_half_size, device=x.device, dtype=x.dtype).view(1, 1, 3)  # [1, 1, 3]

    # Split into actions and observations
    actions = x[:, :, :action_dim]  # [batch_size, horizon, action_dim]
    obs = x[:, :, action_dim:]      # [batch_size, horizon, obs_dim]

    # Unnormalize actions and observations
    action_stds = torch.from_numpy(normalizer.normalizers['actions'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    action_means = torch.from_numpy(normalizer.normalizers['actions'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    actions = actions * action_stds + action_means  # [batch_size, horizon, action_dim]

    obs_stds = torch.from_numpy(normalizer.normalizers['observations'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    obs_means = torch.from_numpy(normalizer.normalizers['observations'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    obs = obs * obs_stds + obs_means  # [batch_size, horizon, obs_dim]

    # Get current hand positions (initial positions)
    current_positions = obs[:, 0, :3]  # [batch_size, 3]

    # Initialize a list to store predicted positions
    predicted_positions = [current_positions]  # List of [batch_size, 3]

    # Iterate over the horizon to compute predicted positions sequentially
    for t in range(actions.shape[1]):
        action_t = actions[:, t, :3]  # [batch_size, 3]
        next_pos = predicted_positions[-1] + action_t * delta_t  # [batch_size, 3]
        predicted_positions.append(next_pos)

    # Stack predicted positions: [batch_size, horizon + 1, 3]
    predicted_positions = torch.stack(predicted_positions, dim=1)

    # Exclude the initial position to match the horizon length
    predicted_positions = predicted_positions[:, 1:, :]  # [batch_size, horizon, 3]

    # Compute vectors from predicted positions to wall center
    vectors_to_wall = wall_pos - predicted_positions  # [batch_size, horizon, 3]

    # Compute distances to wall boundaries
    distances = torch.abs(predicted_positions - wall_pos) - wall_half_size  # [batch_size, horizon, 3]

    # Apply safe distance
    safe_distances = distances - min_safe_dist  # [batch_size, horizon, 3]

    # Calculate penalties where the hand is too close to the wall
    proximity_penalty = torch.relu(-safe_distances) ** 2  # [batch_size, horizon, 3]
    scaled_penalty = proximity_penalty * scaling_factor  # [batch_size, horizon, 3]

    # Sum penalties across dimensions to get total penalty per timestep
    total_penetration = scaled_penalty.sum(dim=-1)  # [batch_size, horizon]

    # Create a mask for actions moving towards the wall and within the unsafe zone
    # Compute dot product between action and direction to wall
    # Positive dot product indicates movement towards the wall
    distances_to_wall = torch.linalg.norm(vectors_to_wall, dim=-1, keepdim=True) + 1e-8  # [batch_size, horizon, 1]
    directions_to_wall = vectors_to_wall / distances_to_wall  # [batch_size, horizon, 3]
    dot_product = (actions[..., :3] * directions_to_wall).sum(dim=-1)  # [batch_size, horizon]

    movement_towards_wall = dot_product > 0  # [batch_size, horizon]
    within_unsafe_zone = total_penetration > 0  # [batch_size, horizon]
    mask = movement_towards_wall & within_unsafe_zone  # [batch_size, horizon]

    # Apply the mask to the total penetration
    directional_penetration = torch.where(mask, total_penetration, torch.zeros_like(total_penetration))  # [batch_size, horizon]

    # Compute mean penalty per trajectory
    loss = directional_penetration.mean(dim=1)  # [batch_size]

    if return_penetration:
        penetration_info = {
            'total_penetration': torch.sqrt(total_penetration),  # [batch_size, horizon]
            'num_violations': mask.sum().item(),
            'max_penetration': torch.sqrt(total_penetration).max().item(),
            'mean_penetration': torch.sqrt(total_penetration[mask]).mean().item() if mask.any() else 0.0
        }
        print("Penetration Info:", penetration_info)
        return loss, penetration_info

    return loss


def predefined_loss_fn2(x, obs_dim, action_dim, normalizer, 
                                  wall_pos=[0.1, 0.6, 0.075], 
                                  wall_half_size=[0.1, 0.01, 0.13], 
                                  min_safe_dist=0.07, 
                                  delta_t=1.0, 
                                  return_penetration=False):
    """
    Loss function to penalize trajectories that bring the hand too close to a specified wall.
    Actions are applied sequentially to simulate trajectory over time.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, horizon, transition_dim].
        obs_dim (int): Observation dimension offset in x.
        action_dim (int): Action dimension in x.
        normalizer (object): Normalizer with 'actions' and 'observations' containing 'stds' and 'means'.
        wall_pos (list or tuple): Center position of the wall [x, y, z].
        wall_half_size (list or tuple): Half-dimensions of the wall along each axis [dx, dy, dz].
        min_safe_dist (float): Minimum safe distance from the wall.
        delta_t (float): Time step size for action integration.
        return_penetration (bool): Whether to return detailed penetration information.
    
    Returns:
        torch.Tensor: Loss per trajectory of shape [batch_size].
        (Optional) dict: Penetration details if `return_penetration` is True.
    """
    # Convert wall parameters to tensors and reshape for broadcasting
    wall_pos = torch.tensor(wall_pos, device=x.device, dtype=x.dtype).view(1, 1, 3)  # [1, 1, 3]
    wall_half_size = torch.tensor(wall_half_size, device=x.device, dtype=x.dtype).view(1, 1, 3)  # [1, 1, 3]

    # Split into actions and observations
    actions = x[:, :, :action_dim]  # [batch_size, horizon, action_dim]
    obs = x[:, :, action_dim:]      # [batch_size, horizon, obs_dim]

    # Unnormalize actions and observations
    action_stds = torch.from_numpy(normalizer.normalizers['actions'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    action_means = torch.from_numpy(normalizer.normalizers['actions'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    actions = actions * action_stds + action_means  # [batch_size, horizon, action_dim]

    obs_stds = torch.from_numpy(normalizer.normalizers['observations'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    obs_means = torch.from_numpy(normalizer.normalizers['observations'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    obs = obs * obs_stds + obs_means  # [batch_size, horizon, obs_dim]

    # Get current hand positions (initial positions)
    current_positions = obs[:, 0, :3]  # [batch_size, 3]

    # Initialize a list to store predicted positions
    predicted_positions = [current_positions]  # List of [batch_size, 3]

    # Iterate over the horizon to compute predicted positions sequentially
    for t in range(actions.shape[1]):
        action_t = actions[:, t, :3]  # [batch_size, 3]
        next_pos = predicted_positions[-1] + action_t * delta_t  # [batch_size, 3]
        predicted_positions.append(next_pos)

    # Stack predicted positions: [batch_size, horizon + 1, 3]
    predicted_positions = torch.stack(predicted_positions, dim=1)

    # Exclude the initial position to match the horizon length
    predicted_positions = predicted_positions[:, 1:, :]  # [batch_size, horizon, 3]

    # Compute distances to wall boundaries
    distances = torch.abs(predicted_positions - wall_pos) - wall_half_size  # [batch_size, horizon, 3]

    # Apply safe distance
    safe_distances = distances - min_safe_dist  # [batch_size, horizon, 3]

    # Calculate penalties where the hand is too close to the wall
    penalties = torch.relu(-safe_distances) ** 2  # [batch_size, horizon, 3]
    scaled_p=penalties

    # Sum penalties across dimensions to get total penalty per timestep
    total_penetration = scaled_p.sum(dim=-1)  # [batch_size, horizon]
    step_weights = [1.0, 1.0, 1.0] + [1.0]*5

    if step_weights is not None:
        step_weights = torch.tensor(step_weights, device=x.device, dtype=x.dtype).view(1, -1)  # [1, horizon]
        if step_weights.shape[1] != total_penetration.shape[1]:
            raise ValueError("Length of step_weights must match the horizon length.")
        total_penetration = total_penetration * step_weights  # [batch_size, horizon]
    
    # Compute mean penalty per trajectory
    loss = total_penetration.mean(dim=1)  # [batch_size]


    if return_penetration:
        penetration_info = {
            'total_penetration': torch.sqrt(total_penetration),  # [batch_size, horizon]
            'num_violations': (total_penetration > 0).sum().item(),
            'max_penetration': torch.sqrt(total_penetration).max().item(),
            'mean_penetration': torch.sqrt(total_penetration).mean().item() if (total_penetration > 0).any() else 0.0
        }
        print("Penetration Info:", penetration_info)
        return loss, penetration_info

    return loss



def _loss_fn (x, obs_dim,action_dim,normalizer ):
    actions = x[:, :, :action_dim]
    actions = actions * torch.tensor(normalizer.normalizers['actions'].stds, device=x.device) + torch.tensor(normalizer.normalizers['actions'].means, device=x.device)
    actions=actions[:,:,:action_dim-1]

    # Compute the speed (norm of each action vector): shape [batch_size, horizon]
    speeds = torch.linalg.norm(actions, dim=-1)
    max_speed = 0.8
    min_speed = 0.5

    # Penalize speeds outside the [min_speed, max_speed] range
    speed_loss = torch.maximum(speeds - max_speed, torch.tensor(0.0)) + \
                 torch.maximum(min_speed - speeds, torch.tensor(0.0))

    # Compute the mean loss per trajectory (average across the horizon): shape [batch_size]
    loss_per_trajectory = speed_loss.mean(dim=1)

    return loss_per_trajectory  # Shape: [batch_size]

def pick_place_wall_loss_fn(x, obs_dim, action_dim, normalizer,
                          wall_pos=[0.1, 0.75, .06],  # Wall midpoint from environment
                          wall_dims=[0.12, 0.01, 0.06],  # Approximate wall dimensions
                          min_safe_dist=0.3,
                          delta_t=1.0,
                          return_diagnostics=False):
    """
    Loss function for pick-place task with wall obstacle avoidance.
    Penalizes:
    1. Trajectories that come too close to the wall
    2. Trajectories that don't successfully grasp the object
    3. Trajectories that don't move the object to the target
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, horizon, transition_dim]
        obs_dim (int): Observation dimension offset in x
        action_dim (int): Action dimension in x
        normalizer (object): Normalizer with means and stds for actions/observations
        wall_pos (list): Center position of the wall [x, y, z]
        wall_dims (list): Full dimensions of the wall [width, thickness, height]
        min_safe_dist (float): Minimum safe distance from wall
        delta_t (float): Time step for action integration
        return_diagnostics (bool): Whether to return detailed diagnostic information
    
    Returns:
        torch.Tensor: Combined loss of shape [batch_size]
        (Optional) dict: Diagnostic information if return_diagnostics is True
    """
    import torch
    
    # Convert parameters to tensors
    wall_pos = torch.tensor(wall_pos, device=x.device, dtype=x.dtype).view(1, 1, 3)
    wall_half_size = torch.tensor(wall_dims, device=x.device, dtype=x.dtype).view(1, 1, 3) / 2

    # Split and unnormalize actions and observations
    actions = x[:, :, :action_dim]
    obs = x[:, :, action_dim:]

    # Unnormalize
    action_stds = torch.from_numpy(normalizer.normalizers['actions'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    action_means = torch.from_numpy(normalizer.normalizers['actions'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    actions = actions * action_stds + action_means

    obs_stds = torch.from_numpy(normalizer.normalizers['observations'].stds).to(x.device).to(x.dtype).view(1, 1, -1)
    obs_means = torch.from_numpy(normalizer.normalizers['observations'].means).to(x.device).to(x.dtype).view(1, 1, -1)
    obs = obs * obs_stds + obs_means

    # Extract relevant positions from observations
    hand_pos = obs[:, :, :3]  # [batch_size, horizon, 3]
    gripper_state = obs[:, :, 3:4]  # [batch_size, horizon, 1]
    obj_pos = obs[:, :, 4:7]  # [batch_size, horizon, 3]
    target_pos = obs[:, :, -3:]  # [batch_size, horizon, 3]

    # 1. Wall avoidance loss
    distances_to_wall = torch.abs(hand_pos - wall_pos) - wall_half_size
    safe_distances = distances_to_wall - min_safe_dist
    wall_penalties = torch.relu(-safe_distances) ** 2
    wall_loss = wall_penalties.sum(dim=-1).mean(dim=1)  # [batch_size]

    # 2. Grasping loss
    hand_to_obj = torch.norm(hand_pos - obj_pos, dim=-1)  # [batch_size, horizon]
    grasp_dist_loss = torch.exp(-5.0 * hand_to_obj) * gripper_state.squeeze(-1)
    grasp_loss = grasp_dist_loss.mean(dim=1)  # [batch_size]

    # 3. Target reaching loss
    obj_to_target = torch.norm(obj_pos - target_pos, dim=-1)  # [batch_size, horizon]
    target_loss = obj_to_target.mean(dim=1)  # [batch_size]

    # 4. Smooth motion loss
    action_diff = actions[:, 1:] - actions[:, :-1]
    smoothness_loss = torch.norm(action_diff, dim=-1).mean(dim=1)  # [batch_size]

    # Combine losses with weights
    total_loss = (
        15.0 * wall_loss +      # Strongly avoid wall collisions
        1.0 * grasp_loss +     # Encourage grasping
        1.5 * target_loss +    # Prioritize reaching target
        1 * smoothness_loss  # Slight preference for smooth motion
    )

    if return_diagnostics:
        diagnostics = {
            'wall_loss': wall_loss.mean().item(),
            'grasp_loss': grasp_loss.mean().item(),
            'target_loss': target_loss.mean().item(),
            'smoothness_loss': smoothness_loss.mean().item(),
            'total_loss': total_loss.mean().item(),
            'num_wall_violations': (wall_penalties.sum(-1) > 0).float().sum().item(),
            'max_wall_penetration': wall_penalties.max().item(),
            'min_obj_target_dist': obj_to_target.min().item(),
            'min_hand_obj_dist': hand_to_obj.min().item()
        }
        return total_loss, diagnostics

    return total_loss


## Initialize custom guide with your loss function
guide = CustomGuide(loss_fn=pick_place_wall_loss_fn, model=diffusion,normalizer=dataset.normalizer)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## Initialize policy with the custom guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## Sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
    descending=args.descending.lower()=="true",
)

print("Order of loss?", "Descending" if args.descending.lower()=="true" else "Ascending")
logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

import metaworld

for i in range(15):
    mt = metaworld.MT1(args.dataset)
    env = mt.train_classes[args.dataset]()
    # breakpoint()
    tasks = mt.train_tasks
    env.set_task(tasks[0])

    observation = env.reset()

    ## Observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    frames = []
    speed_list = []
    pos_list = []
    d_list={}
    done=False
    for t in tqdm(range(args.max_episode_length)):

        if t % 10 == 0: print(args.savepath, flush=True)

        ## Save state for rendering only
        #state = env.state_vector().copy()

        ## Format current observation and goal for conditioning
        conditions = {0: observation}

        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## Execute action in environment
        action[:-1] = np.clip(action[:-1], env.action_space.low[:-1], env.action_space.high[:-1])
        next_observation, reward, terminal, info = env.step(action)

        ###########
        wall_body_pos = torch.tensor([0.1, 0.75, .06], device='cuda')  # [0.1, 0.6, 0]
        wall_size = torch.tensor([0.12, 0.01, 0.06], device='cuda')  # from wall.xml

        min_safe_dist=0.3
        current_positions = torch.tensor(observation[:3],device='cuda')
        actions=torch.tensor(action[:-1],device='cuda')
        predicted_positions = current_positions + actions
        
        # Calculate distances to wall SURFACES (not center)
        dx = torch.abs(predicted_positions[..., 0] - wall_body_pos[0]) - wall_size[0]
        dy = torch.abs(predicted_positions[..., 1] - wall_body_pos[1]) - wall_size[1]
        dz = torch.abs(predicted_positions[..., 2] - wall_body_pos[2]) - wall_size[2]
        
        # Check violations (negative distance means we're inside the wall)
        x_violation = dx < min_safe_dist
        y_violation = dy < min_safe_dist
        z_violation = dz < min_safe_dist
        
        # Position violates if within bounds in all dimensions
        in_violation = x_violation & y_violation & z_violation
        
        # Calculate penetration (how much we're violating the safe distance)
        x_pen = torch.relu(min_safe_dist - dx)
        y_pen = torch.relu(min_safe_dist - dy)
        z_pen = torch.relu(min_safe_dist - dz)
        
        penetration = torch.where(
            in_violation,
            x_pen**2 + y_pen**2 + z_pen**2,
            torch.zeros_like(dx)
        )
        print("DISTANCE TO WALL: ",torch.sqrt(penetration), "  Hand POS: ",current_positions)
        ###########
        done = int(info.get('success', False)) == 1

        action = torch.tensor(action) if isinstance(action, np.ndarray) else action
        speed = torch.linalg.norm(action).item()

        speed_list.append(speed)
        pos_list.append(observation[:3])
        d_list[t]={'dx':x_violation,'dy':y_violation,'dz':z_violation}
        ## Print reward and score
        total_reward += reward
        #score = env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} | R: {total_reward:.2f} | '
            f'values: {samples.values} | scale: {args.scale}',
            flush=True,
        )

        if args.render_videos:
            img = env.render(offscreen=True)
            frames.append(img)

        ## Update rollout observations
        rollout.append(next_observation.copy())

        ## Render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)

        if done:
            break

        observation = next_observation

    if args.render_videos:
        video_file = os.path.join("videos/pick-place-v2/", f'trajectory_{i}_guided_spatial_{args.dataset}_{args.horizon}.mp4')
        imageio.mimwrite(video_file, frames, fps=30)
        print(f"Saved video to {video_file}")
