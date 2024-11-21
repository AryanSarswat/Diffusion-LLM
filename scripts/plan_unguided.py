import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import imageio
import os
import torch
import numpy as np
import json

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load only the diffusion model from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset 
renderer = diffusion_experiment.renderer

## Initialize logger
logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

logger = logger_config()

## Initialize unguided policy
policy_config = utils.Config(
    'sampling.UnguidedPolicy',  # Updated to use UnguidedPolicy
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs (if any)
    #sample_kwargs={},  # Add any specific sampling kwargs if needed
)

policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
import metaworld
mt = metaworld.MT1(args.dataset)
env = mt.train_classes[args.dataset]()
tasks = mt.train_tasks
success_r=[]
for i,task in enumerate(tasks):
    env.set_task(task)
    observation = env.reset()

    ## observations for rendering
    #rollout = [observation.copy()]

    total_reward = 0
    frames = []
    speed_list = []
    pos_list = []
    success = 0
    for t in range(args.max_episode_length):

        if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        #state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)

        done = int(info.get('success', False)) == 1

        action = torch.tensor(action) if isinstance(action, np.ndarray) else action
        speed = torch.linalg.norm(action).item()

        speed_list.append(speed)
        pos_list.append(observation[:3])

        ## print reward and score
        total_reward += reward
        #score = env.get_normalized_score(total_reward)
        """ print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | '
            f'values: {samples.values} | scale: {args.scale}',
            flush=True,
        ) """

        if args.render_videos:
            img = env.render(offscreen=True)
            frames.append(img)

        ## update rollout observations
        #rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)

        if done:
            success=1
            break

        observation = next_observation
    
    success_r.append(success)
    if args.render_videos:
        video_file = os.path.join("videos/test3_button_unguided_wall", f'trajectory_{i}_{args.dataset}_{args.horizon}.mp4')
        imageio.mimwrite(video_file, frames, fps=30)
        print(f"Saved video to {video_file}")
        trajectory_data = {
            "trajectory_id": i,
            "success": success,
            "steps": [
                {"step": step, "speed": speed, "hand_position": pos_list[step].tolist()}
                for step, speed in enumerate(speed_list)
            ]
        }

        # Append to a main JSON file for all trajectories
        json_file = "videos/test3_button_unguided_wall/all_trajectory_data.json"
        try:
            # Load existing data if file exists, otherwise initialize empty list
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    all_trajectory_data = json.load(f)
            else:
                all_trajectory_data = []

            # Add the new trajectory data
            all_trajectory_data.append(trajectory_data)

            # Save updated data back to JSON
            with open(json_file, 'w') as f:
                json.dump(all_trajectory_data, f, indent=4)
            print(f"Appended trajectory data to {json_file}")
        except IOError as e:
            print(f"Failed to save trajectory data: {e}")

overall_success_rate = (sum(success_r) / len(tasks)) * 100
with open("videos/test3_button_unguided_wall/res.txt", 'w') as f:
    f.write(f"Overall Success Rate: {overall_success_rate}%\n")
print(f"Overall Success Rate: {overall_success_rate}%")
## write results to json file at `args.savepath`
#logger.finish(t, score, total_reward, terminal, diffusion_experiment, None)  # No value_experiment

