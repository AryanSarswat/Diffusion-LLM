# Zero-Shot Adaptation for Guided Robot Policies

**Aryan Sarswat, Dhruv Patel, Kushal Shah, Woo Chul Shin** | Georgia Institute of Technology

This project combines diffusion-based trajectory planners with large language models (LLMs) to enable real-time, zero-shot robot policy adaptation. Given a natural language description of a new task constraint (e.g., "avoid the wall" or "move faster"), an LLM generates a differentiable loss function that guides the diffusion model's sampling process—adapting robot behavior at test time without any retraining.

Built on top of [Diffuser](https://github.com/jannerm/diffuser) (Janner et al., ICML 2022), extended with [MetaWorld](https://meta-world.github.io/) manipulation task support and LLM-guided adaptation.

---

## Approach

```
                          Natural Language Instruction
                                    |
                                    v
                      +----------------------------+
                      |   LLM (GPT-4o / Claude)    |
                      |   Generates differentiable  |
                      |   loss function in PyTorch   |
                      +----------------------------+
                                    |
                          Loss function L(trajectory)
                                    |
                                    v
+----------------+    +----------------------------+    +----------------+
| Demonstration  |--->| Diffusion Trajectory       |--->| Adapted        |
| Data (MetaWorld)|   | Planner (Temporal U-Net)   |    | Trajectory     |
+----------------+    | + Loss-Guided Sampling     |    +----------------+
                      +----------------------------+
                                    |
                                    v
                          Robot executes adapted
                          trajectory in MetaWorld
```

**Pipeline:**
1. **Train** a diffusion model on expert demonstrations for a base MetaWorld task (e.g., pick-and-place, door-close)
2. **Describe** a new constraint or adaptation in natural language (e.g., "avoid the wall at position [0.1, 0.75, 0.06]")
3. **Generate** a differentiable PyTorch loss function from the description using an LLM
4. **Guide** the diffusion model's reverse sampling process using gradients of the generated loss
5. **Execute** the adapted trajectory zero-shot in MetaWorld

---

## Key Results

### Spatial Constraints

| Task | Guidance | Success Rate ± Std Dev |
|------|----------|------------------------|
| Button Press | No guidance (baseline) | 1.00 ± 0.00 |
| Button Press w/ Wall | No guidance | 0.53 ± 0.11 |
| Button Press w/ Wall | Wall Penalization + Goal Incentive | **0.63 ± 0.10** |
| Button Press w/ Wall | Wall Penalization | 0.26 ± 0.08 |
| Pick and Place | No guidance (baseline) | 0.80 ± 0.07 |
| Pick and Place w/ Wall | No guidance | 0.40 ± 0.03 |
| Pick and Place w/ Wall | **Wall Penalization** | **0.80 ± 0.02** |

**Key findings:**
- **Pick-and-Place with Wall:** Loss-guided diffusion achieves **+40% absolute improvement** in success rate by generating spatial constraint loss functions that avoid obstacles
- **Button Press with Wall:** Combining wall avoidance with goal incentive losses improves success rate by **+10%** over unguided baseline with wall

### Speed Constraints

| Speed | Success Rate | Overall Average Speed |
|-------|--------------|----------------------|
| Unguided (baseline) | 1.0 | 1.635 |
| Slower | 0.9 | 1.352 |
| Faster | 1.0 | 1.643 |

The system successfully modifies trajectory speed on the door-close task while maintaining high task success, demonstrating behavioral adaptation without retraining.

---

## Project Structure

```
.
├── config/
│   └── locomotion.py          # Hyperparameters for all tasks
├── diffuser/
│   ├── datasets/
│   │   ├── metaworld_sequence.py  # MetaWorld dataset loader
│   │   ├── sequence.py        # D4RL dataset loader
│   │   └── ...
│   ├── models/
│   │   ├── diffusion.py       # Gaussian diffusion model
│   │   ├── temporal.py        # Temporal U-Net architecture
│   │   └── ...
│   ├── sampling/
│   │   ├── functions.py       # Loss-guided sampling step
│   │   ├── guides.py          # CustomGuide (loss-fn guidance)
│   │   ├── policies.py        # GuidedPolicy, UnguidedPolicy
│   │   └── ...
│   └── utils/
│       └── ...
├── scripts/
│   ├── train_metaworld.py     # Train diffusion model
│   ├── plan_guided_lfn.py     # Run with loss-function guidance
│   ├── plan_unguided.py       # Run without guidance (baseline)
│   └── ...
├── collect_metaworld.py       # Collect expert demonstrations
├── environment.yml            # Conda environment specification
└── setup.py
```

---

## Installation

### Prerequisites
- CUDA-capable GPU (NVIDIA)
- Conda package manager
- MuJoCo 2.0 with valid license

### Setup

```bash
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

---

## Usage

### 1. Collect Training Data

Collect expert demonstrations from MetaWorld tasks using scripted policies:

```bash
python collect_metaworld.py \
    --env_name pick-place-v2 \
    --num_trajectories 150 \
    --max_path_length 250 \
    --save_path data/metaworld_pick_place_data.pkl
```

### 2. Train a Diffusion Model

Train the diffusion model on your collected demonstrations:

```bash
python scripts/train_metaworld.py \
    --dataset pick-place-v2 \
    --data_path data/metaworld_pick_place_data.pkl \
    --val_data_path data/metaworld_pick_place_val.pkl
```

Default hyperparameters are in [`config/locomotion.py`](config/locomotion.py). Override with command-line flags:

```bash
python scripts/train_metaworld.py \
    --dataset pick-place-v2 \
    --data_path data/metaworld_pick_place_data.pkl \
    --val_data_path data/metaworld_pick_place_val.pkl \
    --n_diffusion_steps 250 \
    --batch_size 512 \
    --horizon 8 \
    --attention True
```

Models are saved to `logs/<dataset>/diffusion/<config>/`.

### 3. Run with Loss-Guided Diffusion

Use LLM-generated loss functions to adapt behavior at test time:

```bash
python scripts/plan_guided_lfn.py \
    --dataset pick-place-wall-v2 \
    --diffusion_loadpath logs/pick-place-v2/diffusion/metaworld_H8_T250 \
    --horizon 8 \
    --descending False
```

The `--descending` flag controls trajectory ranking: `False` selects the trajectory with the lowest loss (default for constraint satisfaction), `True` selects the highest loss.

### 4. Run Unguided Baseline

Run the baseline diffusion model without guidance:

```bash
python scripts/plan_unguided.py \
    --dataset pick-place-v2 \
    --diffusion_loadpath logs/pick-place-v2/diffusion/metaworld_H8_T250 \
    --horizon 8
```

---

## How Loss-Guided Adaptation Works

The core mechanism is implemented in [`diffuser/sampling/guides.py`](diffuser/sampling/guides.py) (`CustomGuide` class):

1. A loss function `L(trajectory)` is provided—either hand-crafted or generated by an LLM
2. During each reverse diffusion step, gradients of `L` with respect to the trajectory are computed
3. These gradients nudge the sampled trajectory toward lower loss (e.g., away from walls, toward faster speeds)
4. The `CustomGuide` class accepts loss functions as Python callables or as code strings, enabling dynamic LLM-generated functions via `exec()`

The guided sampling step is implemented in [`diffuser/sampling/functions.py`](diffuser/sampling/functions.py) (`n_step_guided_p_sample` function), which applies the loss gradients during the reverse diffusion process.

**Example loss function for wall avoidance:**

```python
def guidance_loss_fn(task_id, x0_pred, robot_state):
    # x0_pred shape: (batch, horizon, action_dim)
    # Extract end-effector positions (first 3 dims of observation)
    positions = x0_pred[:, :, :3]

    # Wall at position [0.1, 0.75, 0.06] with dimensions
    wall_center = torch.tensor([0.1, 0.75, 0.06])
    wall_half_extents = torch.tensor([0.05, 0.05, 0.15])

    # Compute distance to wall and penalize penetration
    distance = torch.abs(positions - wall_center) - wall_half_extents
    penetration = torch.clamp(-distance, min=0.0)
    loss = penetration.sum()

    return loss
```

This loss function is provided to the `CustomGuide`, which computes gradients and applies them during sampling.

---

## Supported MetaWorld Tasks

Task configurations are in [`config/locomotion.py`](config/locomotion.py):

- **pick-place-v2** / **pick-place-wall-v2** — Pick up an object and place it at a target location (with/without wall obstacle)
- **button-press-v2** / **button-press-wall-v2** — Press a button (with/without wall obstacle)
- **drawer-close-v2** — Close a drawer
- **door-open-v2** — Open a door
- **reach-v2** — Reach to a target position
- **push-v2** — Push an object to a goal location

---

## Acknowledgements

This project builds on [Diffuser](https://github.com/jannerm/diffuser) by Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine (ICML 2022). The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). The repository organization is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.

---

## Citation

If you use this code or build upon this work, please cite the original Diffuser paper:

```bibtex
@inproceedings{janner2022diffuser,
  title     = {Planning with Diffusion for Flexible Behavior Synthesis},
  author    = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year      = {2022},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
