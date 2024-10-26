import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import imageio
import os

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

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
frames = []
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    #state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    #score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    if args.render_videos:
        img = env.render(offscreen=True)
        frames.append(img)

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    #logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation
    
if args.render_videos:
    video_file = os.path.join("videos", f'trajectory_{args.horizon}_2.mp4')
    imageio.mimwrite(video_file, frames, fps=30)
    print(f"Saved video to {video_file}")
## write results to json file at `args.savepath`
#logger.finish(t, score, total_reward, terminal, diffusion_experiment, None)  # No value_experiment

