# Action-Attack Mechanisms: `run_delay_sweep.sh` vs `run_all_scenarios_stuck_at.sh`

Both scripts drive the same test harness (`train.py --mode test`) but exercise two
different entries in `ActionAttack` (`action_attacks.py`): **`delay`** (comms/actuation
lag) and **`stuck_at`** (frozen actuator channels). Neither script implements the attack
itself — they just sweep a parameter and let `train.py` invoke `ActionAttack.perturb()`.

## Shared pipeline

Both scripts:

1. Set `SCENARIO` / `NUM_ADVERSARIES`, resolve a per-scenario diffusion checkpoint at
   `diffusion_models/${SCENARIO}_m3ddpg.pt`.
2. Loop over 4 policy variants by remapping a friendly suffix to a `--variant` flag:
   - `maddpg` → `maddpg-none`
   - `earnie` → `maddpg-earnie`
   - `rmaac` → `maddpg-act_adv`
   - `m3ddpg` → `m3ddpg`
3. For each variant, call `train.py --mode test` with `--attack {delay|stuck_at}`,
   an attack-specific sweep list, `--act-low -1.0 --act-high 1.0` (the clip bound applied
   *after* the attack), and diffusion-denoiser settings (`--diffusion-model-path`,
   `--diffusion-steps 100`, `--t-start-list 20 40`).
4. `train.py` writes one CSV per variant: `${exp_name}_delay_sweep.csv` or
   `${exp_name}_stuck_at_sweep.csv`.

`run_all_scenarios_stuck_at.sh` is a thin outer loop: it just calls
`run_stuck_at_sweep.sh` once per scenario in `(simple_push, simple_speaker_listener,
simple_spread, simple_tag)` with the right `--num-adversaries` for each (0, 0, 0, 3).
`run_delay_sweep.sh` is called directly on a single scenario (no outer-loop counterpart
in this pair, though the same pattern applies).

## Where the attack actually happens (`train.py`, test loop)

Inside the per-episode/per-step test loop (`train.py:1150-1210`):

```python
attack = ActionAttack(arglist)
for ep in range(n_episodes):
    ...
    attack.reset(n_agents=env.n)          # clears per-agent history buffers
    for step in range(max_episode_len):
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]  # clean policy actions
        action_n_noisy = [attack.perturb(a, i) for i, a in enumerate(action_n)]  # <-- attack applied here
        ...
        # optional diffusion denoiser tries to recover the clean action from action_n_noisy
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_clean)
        attack.step()                      # advances internal clock (self.t += 1)
```

So the attack sits strictly between the trained policy's clean action and the
environment step, and (optionally) a diffusion-based denoiser tries to reconstruct the
clean action from the corrupted one before it hits `env.step`.

## Mechanism 1 — `delay` (`action_attacks.py:156-159`)

```python
elif kind == "delay":
    st["hist"].append(x.copy())
    k = self.a.delay_k
    out = st["hist"][-1 - k] if len(st["hist"]) > k else st["hist"][0]
```

- Each agent keeps its own action history buffer `st["hist"]` (reset every episode).
- Every step, the current clean action `x` is appended to history.
- The action actually sent to the env is the action from **`k` steps ago**
  (`st["hist"][-1-k]`), simulating a fixed-length actuation/communication delay of `k`
  control steps.
- Before the buffer has `k+1` entries (i.e., at the start of an episode), it falls back
  to the very first action in history — so the effective delay is "clamped" during the
  first `k` steps rather than indexing out of bounds.
- Swept parameter: `--delay-k-list` = `1 2 3 5 8` (`run_delay_sweep.sh:26`). `train.py`
  loops over this list (`train.py:1497`), sets `arglist.delay_k = k` for each value, runs
  `NUM_TEST_EPISODES=800` test episodes without the denoiser, then (if the denoiser is
  enabled) reruns with the denoiser at each `t_start` in `--t-start-list` (`20 40`) to see
  how much of the delay-induced reward loss diffusion denoising recovers. Results land in
  one row per `k` in `${exp_name}_delay_sweep.csv`.

## Mechanism 2 — `stuck_at` (`action_attacks.py:121-127`)

```python
elif kind == "stuck_at":
    # each dim independently freezes (prob sp_prob) at its first-seen value
    if st["last"] is None:
        st["last"] = x.copy()
        st["frozen_mask"] = np.random.rand(*x.shape) < self.a.sp_prob
        st["frozen_val"] = x.copy()
    out = np.where(st["frozen_mask"], st["frozen_val"], x)
```

- On the **first step of each episode**, per action dimension, a Bernoulli mask
  `frozen_mask` is drawn with probability `sp_prob` per element, and the first clean
  action `x` is captured as `frozen_val`.
- For every subsequent step in that episode, dimensions where `frozen_mask` is `True`
  are hard-pinned to their frozen first-step value (`frozen_val`), regardless of what
  the policy outputs; unmasked dimensions pass the live action through unchanged. This
  models an actuator/sensor channel that gets stuck at its initial reading for the whole
  episode (a persistent partial actuator fault), as opposed to a per-step random dropout.
- Swept parameter: `--sp-prob-list` = `0.0 0.1 0.25 0.5 0.75 1.0` (`run_stuck_at_sweep.sh:26`).
  `train.py` loops over this list (`train.py:1544`), sets `arglist.sp_prob = p` each time,
  runs the no-denoiser pass, then the denoiser pass at each `t_start`, and writes one row
  per `p` to `${exp_name}_stuck_at_sweep.csv`.

## Common post-attack step: clipping and denoising

After `perturb()` computes `out`, `ActionAttack._clip()` clamps it to
`[act_low, act_high] = [-1.0, 1.0]` (both scripts pass these explicitly). If a diffusion
denoiser is active (`use_denoiser` / `diffusion_policy` in `train.py`), the corrupted
action vector plus current state is fed to `diffusion_denoise_action(...)` with a given
`t_start` (how many diffusion steps to run from the noisy action) to produce a "cleaned"
action that is what's actually sent to `env.step`. The sweep compares reward with no
denoiser vs. reward at each `t_start`, reporting the best `t_start` and the percentage
improvement over the no-denoiser case in the CSV.

## Summary of what differs between the two scripts

| | `run_delay_sweep.sh` | `run_all_scenarios_stuck_at.sh` (→ `run_stuck_at_sweep.sh`) |
|---|---|---|
| Attack | `delay` — replay action from `k` steps ago | `stuck_at` — freeze a random subset of action dims at their episode-start value |
| State kept per agent | Growing action history list | One-time frozen mask + frozen value, set once per episode |
| Swept CLI arg | `--delay-k-list` (1,2,3,5,8) | `--sp-prob-list` (0.0–1.0) |
| Scope | Single scenario per invocation | Loops over 4 scenarios (`simple_push`, `simple_speaker_listener`, `simple_spread`, `simple_tag`), each with scenario-specific `--num-adversaries` |
| Output | `${exp_name}_delay_sweep.csv` | `${exp_name}_stuck_at_sweep.csv` (per scenario × variant) |
