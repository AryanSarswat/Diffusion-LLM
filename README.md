# Planning with Diffusion &nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing)


Training and visualizing of diffusion models from [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).

The [main branch](https://github.com/jannerm/diffuser/tree/main) contains code for training diffusion models and planning via value-function guided sampling on the D4RL locomotion environments.
The [kuka branch](https://github.com/jannerm/diffuser/tree/kuka) contains block-stacking experiments.
The [maze2d branch](https://github.com/jannerm/diffuser/tree/maze2d) contains goal-reaching via inpainting in the Maze2D environments.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

**Updates**
- 12/09/2022: Diffuser (the RL model) has been integrated into 🤗 Diffusers (the Hugging Face diffusion model library)! See [these docs](https://huggingface.co/docs/diffusers/using-diffusers/rl) for how to run Diffuser using their pipeline.
- 10/17/2022: A bug in the value function scaling has been fixed in [this commit](https://github.com/jannerm/diffuser/commit/3d7361c2d028473b601cc04f5eecd019e14eb4eb). Thanks to [Philemon Brakel](https://scholar.google.com/citations?user=Q6UMpRYAAAAJ&hl=en) for catching it!

## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing).


## Installation

```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Using pretrained models

### Downloading weights



<details>
<summary>To create the table of offline RL results from the paper, run <code>python plotting/table.py</code>. This will print a table that can be copied into a Latex document. (Expand to view table source.)</summary>

```
\definecolor{tblue}{HTML}{1F77B4}
\definecolor{tred}{HTML}{FF6961}
\definecolor{tgreen}{HTML}{429E9D}
\definecolor{thighlight}{HTML}{000000}
\newcolumntype{P}{>{\raggedleft\arraybackslash}X}
\begin{table*}[hb!]
\centering
\small
\begin{tabularx}{\textwidth}{llPPPPPPPPr}
\toprule
\multicolumn{1}{r}{\bf \color{black} Dataset} & \multicolumn{1}{r}{\bf \color{black} Environment} & \multicolumn{1}{r}{\bf \color{black} BC} & \multicolumn{1}{r}{\bf \color{black} CQL} & \multicolumn{1}{r}{\bf \color{black} IQL} & \multicolumn{1}{r}{\bf \color{black} DT} & \multicolumn{1}{r}{\bf \color{black} TT} & \multicolumn{1}{r}{\bf \color{black} MOPO} & \multicolumn{1}{r}{\bf \color{black} MOReL} & \multicolumn{1}{r}{\bf \color{black} MBOP} & \multicolumn{1}{r}{\bf \color{black} Diffuser} \\ 
\midrule
Medium-Expert & HalfCheetah & $55.2$ & $91.6$ & $86.7$ & $86.8$ & $95.0$ & $63.3$ & $53.3$ & $\textbf{\color{thighlight}105.9}$ & $88.9$ \scriptsize{\raisebox{1pt}{$\pm 0.3$}} \\ 
Medium-Expert & Hopper & $52.5$ & $\textbf{\color{thighlight}105.4}$ & $91.5$ & $\textbf{\color{thighlight}107.6}$ & $\textbf{\color{thighlight}110.0}$ & $23.7$ & $\textbf{\color{thighlight}108.7}$ & $55.1$ & $103.3$ \scriptsize{\raisebox{1pt}{$\pm 1.3$}} \\ 
Medium-Expert & Walker2d & $\textbf{\color{thighlight}107.5}$ & $\textbf{\color{thighlight}108.8}$ & $\textbf{\color{thighlight}109.6}$ & $\textbf{\color{thighlight}108.1}$ & $101.9$ & $44.6$ & $95.6$ & $70.2$ & $\textbf{\color{thighlight}106.9}$ \scriptsize{\raisebox{1pt}{$\pm 0.2$}} \\ 
\midrule
Medium & HalfCheetah & $42.6$ & $44.0$ & $\textbf{\color{thighlight}47.4}$ & $42.6$ & $\textbf{\color{thighlight}46.9}$ & $42.3$ & $42.1$ & $44.6$ & $42.8$ \scriptsize{\raisebox{1pt}{$\pm 0.3$}} \\ 
Medium & Hopper & $52.9$ & $58.5$ & $66.3$ & $67.6$ & $61.1$ & $28.0$ & $\textbf{\color{thighlight}95.4}$ & $48.8$ & $74.3$ \scriptsize{\raisebox{1pt}{$\pm 1.4$}} \\ 
Medium & Walker2d & $75.3$ & $72.5$ & $\textbf{\color{thighlight}78.3}$ & $74.0$ & $\textbf{\color{thighlight}79.0}$ & $17.8$ & $\textbf{\color{thighlight}77.8}$ & $41.0$ & $\textbf{\color{thighlight}79.6}$ \scriptsize{\raisebox{1pt}{$\pm 0.55$}} \\ 
\midrule
Medium-Replay & HalfCheetah & $36.6$ & $45.5$ & $44.2$ & $36.6$ & $41.9$ & $\textbf{\color{thighlight}53.1}$ & $40.2$ & $42.3$ & $37.7$ \scriptsize{\raisebox{1pt}{$\pm 0.5$}} \\ 
Medium-Replay & Hopper & $18.1$ & $\textbf{\color{thighlight}95.0}$ & $\textbf{\color{thighlight}94.7}$ & $82.7$ & $\textbf{\color{thighlight}91.5}$ & $67.5$ & $\textbf{\color{thighlight}93.6}$ & $12.4$ & $\textbf{\color{thighlight}93.6}$ \scriptsize{\raisebox{1pt}{$\pm 0.4$}} \\ 
Medium-Replay & Walker2d & $26.0$ & $77.2$ & $73.9$ & $66.6$ & $\textbf{\color{thighlight}82.6}$ & $39.0$ & $49.8$ & $9.7$ & $70.6$ \scriptsize{\raisebox{1pt}{$\pm 1.6$}} \\ 
\midrule
\multicolumn{2}{c}{\bf Average} & 51.9 & \textbf{\color{thighlight}77.6} & \textbf{\color{thighlight}77.0} & 74.7 & \textbf{\color{thighlight}78.9} & 42.1 & 72.9 & 47.8 & \textbf{\color{thighlight}77.5} \hspace{.6cm} \\ 
\bottomrule
\end{tabularx}
\vspace{-.0cm}
\caption{
}
\label{table:locomotion}
\end{table*}

```

![](https://github.com/diffusion-planning/diffusion-planning.github.io/blob/master/images/table.png)
</details>

### Planning

To plan with guided sampling, run:
```
python scripts/plan_guided_lfn.py     --dataset button-press-wall-v2     --diffusion_loadpath /home/kurt/HRI/diffuser/logs/button-press-v2/diffusion/metaworld_H8_T20_noisy --horizon 8 --descending False '
```
Where "descending" controls how the loss values are ordered to pick the first trajectory.

To run unguided:

```
python scripts/plan_unguided.py     --dataset button-press-v2     --diffusion_loadpath /home/kurt/HRI/diffuser/logs/button-press-v2/diffusion/metaworld_H8_T20_noisy --horizon 8
```
## Training from scratch

1. Train a diffusion model with:
```
python scripts/train_metaworld.py --dataset button-press-v2 --data_path '/home/kurt/HRI/diffuser/metaworld_button_press_data_150_uniform_noisy.pkl' --val_data_path '/home/kurt/HRI/diffuser/metaworld_button_press_validation.pkl'
```
You can pass your dataset for training in "data_path" and you validation dataset on "val_data_path"

The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.
Be sure to add parameters you want to change for each task if required

2. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:

#Using Loss function for guidance
```
python scripts/plan_guided_lfn.py  --dataset drawer-close-v2 --diffusion_loadpath /home/kurt/HRI/diffuser/logs/drawer-close-v2/diffusion/metaworld_H8_T20 --horizon 8 --descending False
```
Where "descending" controls how the loss values are ordered to pick the first trajectory.
#Without any guidance
```
python scripts/plan_unguided.py --dataset drawer-close-v2  --diffusion_loadpath /home/kurt/HRI/diffuser/logs/drawer-close-v2/diffusion/metaworld_H8_T20 --horizon 8
```
See [locomotion:plans](config/locomotion.py#L110-L149) for the corresponding default hyperparameters.

**Deferred f-strings.** Note that some planning script arguments, such as `--n_diffusion_steps` or `--discount`,
do not actually change any logic during planning, but simply load a different model using a deferred f-string.
For example, the following flags:
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
will resolve to a value checkpoint path of `values/defaults_H32_T20_d0.997`. It is possible to
change the horizon of the diffusion model after training (see [here](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing) for an example),
but not for the value function.




## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
