# File: /scripts/plan_guided.py

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.guides import CustomGuide
import imageio
import os
import torch
import numpy as np

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
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer



def predefined_loss_fn(x, obs_dim):
    # Extract actions: shape [batch_size, horizon, action_dim]
    actions = x[:, :, obs_dim:-1]

    # Compute the norm (speed) of each action vector: shape [batch_size, horizon]
    speeds = torch.linalg.norm(actions, dim=-1)

    # Compute the mean speed per trajectory: shape [batch_size]
    mean_speeds = torch.mean(speeds, dim=1)

    # Define loss as negative mean speed: shape [batch_size]
    loss_per_trajectory = - mean_speeds

    return loss_per_trajectory  # Shape: [batch_size]

def _loss_fn (x, obs_dim ):
    actions = x[:, :, obs_dim:-1]

    # Compute the speed (norm of each action vector): shape [batch_size, horizon]
    speeds = torch.linalg.norm(actions, dim=-1)
    max_speed = 0.4
    min_speed = 0.38

    # Penalize speeds outside the [min_speed, max_speed] range
    speed_loss = torch.maximum(speeds - max_speed, torch.tensor(0.0)) + \
                 torch.maximum(min_speed - speeds, torch.tensor(0.0))

    # Compute the mean loss per trajectory (average across the horizon): shape [batch_size]
    loss_per_trajectory = speed_loss.mean(dim=1)

    return loss_per_trajectory  # Shape: [batch_size]

## Initialize custom guide with your loss function
guide = CustomGuide(loss_fn=_loss_fn, model=diffusion)

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

env = dataset.env
observation = env.reset()


## Observations for rendering
rollout = [observation.copy()]

total_reward = 0
frames = []
speed_list = []
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## Save state for rendering only
    #state = env.state_vector().copy()

    ## Format current observation and goal for conditioning
    conditions = {0: observation}

    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    
    ## Execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    action = torch.tensor(action) if isinstance(action, np.ndarray) else action
    speed = torch.linalg.norm(action).item()

    speed_list.append(speed)
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

    if terminal:
        break

    observation = next_observation

if args.render_videos:
    video_file = os.path.join("videos", f'trajectory_guided_{args.horizon}.mp4')
    imageio.mimwrite(video_file, frames, fps=30)
    print(f"Saved video to {video_file}")
    text_file = os.path.join("videos", f'speed_guided_{args.horizon}.txt')
    try:
        with open(text_file, 'w') as f:
            for step, speed in enumerate(speed_list):
                f.write(f"Step {step}: Speed {speed:.2f}\n")
        print(f"Saved speed data to {text_file}")
    except IOError as e:
        print(f"Failed to save speed data: {e}")
## Write results to json file at `args.savepath`
#logger.finish(t, score, total_reward, terminal, diffusion_experiment, None)  # No value_experiment
