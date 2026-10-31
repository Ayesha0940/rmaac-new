# How the H=25 denoiser builds its input, and how BC is queried

Source: `RMA-AC/experiments/train.py` (`collect_diffusion_data`, `train_diffusion`,
`testRobustnessAP`, `diffusion_denoise_action`). Plotted sweeps load
`diffusion_models/<scenario>_m3ddpg.pt` (the H=25 model), not the `*_H1` files.

Episodes are 25 steps (`--max-episode-len` default 25) and H = 25, so each stored
trajectory is one whole episode: `states[N,25,Ds]`, `actions[N,25,Da]`, N = 2000.

## Training (H=25 denoiser)
- Target: the full 25-step clean joint-action trajectory, z-scored per action dim
  (mean/std over all N x 25 entries).
- Condition: `states[:, 0, :]` only, i.e. the episode's initial state. The states at
  steps 1..24 are stored but never used.
- Loss: DDPM epsilon-prediction MSE, t ~ U[0, 100).

## Test, at every env step
- Condition: the current step's joint observation `concat(obs_n)` (any of steps 0..24).
- Trajectory input: zeros of shape (1, 25, Da) in normalized space. Slot 0 is the
  z-scored perturbed joint action of the current step. Slots 1..24 stay zero (the
  dataset mean action). The perturbed action is used directly as x at t_start; there
  is no forward noising.
- The reverse chain runs from t_start (20 or 40; default 40) down to 0.
- Executed action: slot 0 only, un-normalized, split per agent, clipped to
  [act_low, act_high] = [-1, 1], then `env.step`. Slots 1..24 are discarded.

## Consequence
Slot 0 is trained as the step-0 action given the step-0 state. At test steps 1..24 it
is given the current state as condition, which it never saw as a condition in
training.

## BC (state-only baseline)
- Queried with the current joint state at every step; executes one action per step
  through the same concat/split/clip path as the denoiser output. No perturbation.
- Trained on all 25 (state_t, action_t) pairs of each episode (50k pairs) from the
  same dataset D, so it is trained on the states it is queried with.
- The comparison is therefore matched in what is queried and executed per step, but
  not in training coverage: BC covers all 25 steps, the denoiser's condition covers
  only the initial state.
