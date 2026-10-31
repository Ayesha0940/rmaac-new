# Action-recovery experiments: denoiser vs base policy vs behavior cloning (BC)

Scope: the RMA-AC multi-agent particle-environment experiments in this repo
(simple_adversary, simple_push, simple_speaker_listener, simple_spread, simple_tag
= predator-prey). All numbers are read from the CSVs already in `noise_sweeps/`
(and `RMA-AC/experiments/results/`, plus the new BC runs in `noise_sweeps/bc/`).
Reward is higher-is-better. "±" is the spread across seeds.

Status: BC is finished for 4 scenarios. **BC for simple_tag (predator-prey) is
still training, so it is missing from every table below.** Re-run
`noise_sweeps/bc/make_bc_clean_table.py` when it lands.

---

## 1. Short version

1. **The denoiser beats the un-recovered, perturbed policy in most cases**: in 85% of
   the 500 (scenario × policy × fault × level) points with a fault present, and it
   wins by large margins when the fault is severe (stuck actuators, long delay,
   biased or large noise). Example: simple_adversary, stuck-at probability 1.0:
   -69.7 without recovery vs -3.3 with the denoiser.
2. **A classical Kalman filter does almost nothing** in these tests. It stays within a
   few percent of "no recovery" at every level.
3. **The denoiser is not free when there is little or no fault.** With no
   perturbation it costs reward in simple_speaker_listener (-32 → -56),
   simple_tag (60.7 → 57.5 with zero noise, 57.5 → 32.0 with zero stuck-at) and
   simple_push (-8.5 → -10.7).
4. **BC, which ignores the policy and the fault entirely, beats the denoiser in
   almost every comparison we have** (4 scenarios). BC's clean reward is close to
   its teacher policy, so a state-only imitator is already a strong baseline. The
   denoiser only beats BC where fault levels are very small, and there the gap is
   tiny.
5. So the honest claim the data supports is: *the denoiser recovers a lot of the
   reward lost to faults compared with doing nothing or using a Kalman filter. It
   does not beat a state-only imitator.* For simple_speaker_listener the denoiser's
   result is almost identical to a state-only generative control, which suggests it
   mostly ignores the noisy action there.

---

## 2. What each method sees (inputs and conditioning)

| Method | Input it uses | Conditioned on | Output executed | Notes |
|---|---|---|---|---|
| **Base policy** (MADDPG, ERNIE, RMAAC, M3DDPG) | Each agent's own observation | own observation | own action | The clean reference. Four trained variants. |
| **Perturbed policy ("no recovery")** | Base-policy action plus a fault | own observation (through the policy) | the perturbed action | Faults: Gaussian noise (zero-mean or biased ±1), stuck-at actuator, delay. |
| **Kalman filter** | The perturbed action only (per dimension, over time) | nothing (no state) | the filtered action | Classical baseline. A per-point best-of-grid version exists for predator-prey. |
| **Denoiser, H=25 (main, in all plots)** | The perturbed joint action of all agents, placed in slot 0 of a 25-slot trajectory (other slots zero) | the joint observation of all agents, concatenated, **at the current step** | slot 0, un-normalised, clipped to [-1, 1] | Diffusion model (100 steps, reverse chain starts at step 20 or 40). Centralised: one network for all agents. |
| **Denoiser, H=1** | Same, but a 1-slot trajectory | same | same | Trained on 2000 step-0 pairs only. Single-run results in `results/diffusion_h1`. |
| **Diffusion-policy control** | Nothing from the perturbed action | joint observation | slot 0 | Same weights as the denoiser, but the chain starts from pure noise. A state-only *generative* policy. Run for 3 scenarios. |
| **BC (new)** | Nothing from the policy or the fault | joint observation at the current step | the joint action, un-normalised, clipped to [-1, 1] | Deterministic MLP (state → 256 → 256 → action), MSE loss, no perturbation. Replaces policy and denoiser. |

How the H=25 denoiser is built (from the code):

- It is trained on whole 25-step episodes of clean M3DDPG actions, but conditioned
  **only on the first state of the episode**. States from steps 1 to 24 are never used
  as a condition during training.
- At test time, at every step, it is conditioned on the **current** state and its
  first slot gets the perturbed action. Slots 1 to 24 stay at zero and are discarded.
- BC is trained on all 25 steps of the same dataset, so BC has seen the states it is
  queried with. The denoiser's condition has not. The comparison is matched in what is
  queried and executed each step, but not in training coverage.
- One denoiser is trained per scenario, from M3DDPG data only, and is then applied to
  all four policy variants.

Further variants that exist in the code (not tabulated here because they were not
re-analysed for this report): predator-only and prey-only denoising on
simple_tag, a "good-only" simple_tag denoiser, and plots in `noise_sweeps/plots/`
(`m3ddpg_predatoronly`, `m3ddpg_preyonly`, `m3ddpg_ablation`).

---

## 3. Experiments run

| Experiment | Levels | Seeds | Episodes | Where |
|---|---|---|---|---|
| Zero-mean Gaussian noise | std 0, 0.4 … 3 | several (per sweep) | 800 | `guassian_noise/` |
| Biased Gaussian noise (mean +1 and -1) | std 0 … 3 | several | 800 | `guassian_noise/` |
| Delay (action arrives k steps late) | k = 1, 2, 3, 5, 8 | 3 | 800 | `delay_sweeps/` |
| Stuck-at actuator (probability a dimension is frozen) | p = 0 … 1 | 3 | 800 | `actuation_fault/` |
| Kalman filter on the same faults | same levels | 3 | 800 | `filters/`, `filters_oracle/` |
| Predator-prey (simple_tag) with capture and survival metrics | noise, delay, stuck-at | 2 to 3 | 800 | `*/predprey/`, `predprey_kalman_oracle/` |
| Diffusion-policy control | std 0 … 3 | 1 | 800 | `RMA-AC/experiments/*diffusion_policy_ablation*.csv` |
| H=1 denoiser | std 0 … 3 | 1 | 800 | `RMA-AC/experiments/results/diffusion_h1` |
| **BC baseline (new)** | clean only; flat across levels | 47, 48, 49 | 800 | `noise_sweeps/bc/` |

Where "denoiser" appears below it means `best_reward_with_diffusion`: the better of
the two starting steps (20 or 40) at each point. See caveats (section 7).

---

## 4. Results

### 4.1 Clean reward: BC vs the base policies (mean over seeds 47 to 49)

| Scenario | BC | MADDPG | ERNIE | RMAAC | M3DDPG (BC's teacher) |
|---|---|---|---|---|---|
| simple_adversary | 3.69 | 4.72 | 3.93 | 3.29 | 4.02 |
| simple_push | -8.43 | -9.03 | -8.88 | -9.06 | -8.63 |
| simple_speaker_listener | -31.92 | -29.16 | -58.74 | -58.30 | -33.06 |
| simple_spread | -468.69 | -410.13 | -403.69 | -424.93 | -435.29 |
| simple_tag | pending | pending | pending | pending | pending |

BC is close to its M3DDPG teacher: slightly below on adversary, slightly above on push
and speaker_listener, and about 8% below on spread, where it is below all four policies. In speaker_listener it is far better than ERNIE
and RMAAC. Full per-policy numbers with standard deviations:
`noise_sweeps/bc/bc_clean_reward_table.csv`.

### 4.2 Zero-mean Gaussian noise (M3DDPG policy; denoiser = best of t_start 20/40)

Cells show reward with no recovery / with denoiser. BC is flat across all levels.

| Scenario | std 0 | std 1 | std 2 | std 3 | BC (any level) |
|---|---|---|---|---|---|
| simple_adversary | 4.12 / 3.75 | -3.41 / -0.61 | -15.14 / -5.06 | -22.38 / -6.70 | **3.69** |
| simple_push | -8.50 / -10.67 | -17.35 / -12.22 | -24.61 / -13.13 | -27.73 / -13.52 | **-8.43** |
| simple_speaker_listener | -31.85 / -56.04 | -83.50 / -59.07 | -142.12 / -58.84 | -178.84 / -60.77 | **-31.92** |
| simple_spread | -448.66 / -440.43 | -546.10 / -527.24 | -635.80 / -590.48 | -684.41 / -617.45 | **-468.69** |
| simple_tag (total reward) | 60.71 / 57.52 | 16.67 / 12.48 | 5.11 / 8.77 | -1.33 / 4.50 | pending |

Kalman at std 2 and 3 is within about 1 reward unit of "no recovery" in
adversary, push and speaker_listener (e.g. adversary std 3: -22.13 vs -22.38). In
simple_spread it is also essentially level with no recovery at std 2 and 3
(-637.55 vs -635.80, -684.50 vs -684.41).

### 4.3 Delay (M3DDPG)

| Scenario | k = 3 none / denoiser | k = 8 none / denoiser | BC |
|---|---|---|---|
| simple_adversary | -1.89 / 0.49 | -20.01 / -1.58 | 3.69 |
| simple_push | -13.46 / -11.42 | -24.03 / -12.73 | -8.43 |
| simple_speaker_listener | -40.70 / -56.09 | -98.41 / -60.70 | -31.92 |
| simple_spread | -499.70 / -483.07 | -685.55 / -589.33 | -468.69 |
| simple_tag (total) | 21.41 / 18.66 | -2.26 / 1.51 | pending |

At mild delay (k = 1 to 3) the denoiser is often *worse* than doing nothing (it loses
in speaker_listener at k = 1 to 3, and in spread/push/tag at k = 1). It only
clearly wins at longer delays.

### 4.4 Stuck-at actuator (M3DDPG)

| Scenario | p = 0.5 none / denoiser | p = 1.0 none / denoiser | BC |
|---|---|---|---|
| simple_adversary | -12.25 / -0.30 | -69.69 / -3.29 | 3.69 |
| simple_push | -21.66 / -12.52 | -40.94 / -14.06 | -8.43 |
| simple_speaker_listener | -132.15 / -58.10 | -381.75 / -61.35 | -31.92 |
| simple_spread | -584.95 / -555.68 | -1080.55 / -767.10 | -468.69 |
| simple_tag (total) | 2.00 / 1.43 | -54.77 / -38.58 | pending |

These are the largest recoveries in the whole study (speaker_listener p = 1.0 gains
about 320 reward units), because without recovery performance collapses.

### 4.5 Predator-prey (simple_tag, M3DDPG), per-metric

Predator reward (higher is better for the predators):

| Fault | none | denoiser | Kalman |
|---|---|---|---|
| noise std 0 | 96.88 | 79.09 | 103.69 |
| noise std 2 | 21.32 | 21.70 | – |
| noise std 3 | 16.84 | 18.05 | 20.18 |
| delay k = 3 | 33.39 | 27.45 | 36.41 |
| delay k = 8 | 20.16 | 17.42 | 22.05 |
| stuck-at p = 0.5 | 35.40 | 25.34 | 38.89 |
| stuck-at p = 1.0 | 14.45 | 14.54 | 16.16 |

Captures follow the same pattern (e.g. noise std 3: 0.56 none, 0.60 denoiser,
0.67 Kalman). Survival time of the prey is *longer* with the denoiser (e.g. 17.8 vs
14.6 with no fault), which for predators is a loss, not a gain. **In predator-prey
the denoiser does not help the predators.** At best it matches no-recovery under
heavy noise, and it is below the Kalman filter at most points.

### 4.6 Denoiser variants

**Starting step (t_start 20 vs 40), win rate against "no recovery" over noise, delay
and stuck-at points with a fault present, all four policies:**

| Scenario | t = 20 | t = 40 | best of both |
|---|---|---|---|
| simple_adversary | 82% | 98% | 98% |
| simple_push | 88% | 76% | 88% |
| simple_speaker_listener | 80% | 70% | 80% |
| simple_spread | 75% | 7% | 76% |
| simple_tag | 49% | 26% | 65% |

On simple_spread, t = 40 loses to no recovery almost always (average -47 reward),
while t = 20 gains +47. The choice of starting step matters a lot.

**H=1 vs H=25 denoiser (M3DDPG, zero-mean noise; H=1 is a single run, H=25 is the
multi-seed best-of-t_start number):**

| Scenario | H=1 std 2 | H=25 std 2 | H=1 std 0 | H=25 std 0 |
|---|---|---|---|---|
| simple_adversary | 2.5 | -5.06 | 3.9 | 3.75 |
| simple_push | -17.1 | -13.13 | -13.1 | -10.67 |
| simple_speaker_listener | -66.2 | -58.84 | -61.5 | -56.04 |
| simple_spread | -553.5 | -590.48 | -491.5 | -440.43 |
| simple_tag | 17.6 | 5.11 | 18.9 | 57.52 |

H=1 is better on adversary, spread and tag at std 2, and worse on push and
speaker_listener. At zero noise H=1 costs more than H=25 in four of the five
scenarios. It is not a clear winner either way. (The H=1 numbers are one run, so
differences of a few reward units are not conclusive.)

**Diffusion-policy control (start from pure noise, ignore the perturbed action),
M3DDPG, mean over noise std 1 to 3:**

| Scenario | Control | Denoiser | No recovery |
|---|---|---|---|
| simple_speaker_listener | about -60 | -59 to -60 | -81 to -186 |
| simple_adversary | about -9 to -10 | -1.2 to -5.6 | -3 to -22 |
| simple_spread | -628 to -878 (two runs) | -520 to -620 | -540 to -690 |

---

## 5. Where the denoiser wins

**Against no recovery (the fault reaches the environment unchanged):**

- 85% of 500 fault points overall. Per scenario: adversary 99%, push 91%,
  speaker_listener 85%, spread 83%, tag 67%.
- The win grows with fault severity. In simple_adversary the denoiser's reward stays
  between about +3 and -7 across *all* stuck-at, delay and noise levels, while no
  recovery falls to -70, -20 and -22 at the worst levels.
- It wins most on persistent, structured faults: stuck actuators and biased noise
  (average gain about 10 to 116 reward units depending on scenario, excluding simple_tag) compared with delay.
- Averaged over all faulted points, the gain is positive in every scenario except
  simple_tag actuator faults and delay (about -2.7 and -0.4).

**Against the Kalman filter:** the denoiser is generally ahead at medium and high fault levels in
the four non-predator-prey scenarios, because Kalman stays level with "no recovery"
(not at mild delay in speaker_listener, where it loses).
Predator-prey is the exception (section 4.5).

**Against BC:** only in simple_spread at very low noise (std ≤ 0.4: -464 vs BC
-469), which is a tie within the spread of the seeds. Everywhere else BC is higher.
Concretely, for M3DDPG with zero-mean noise BC is above the denoiser at every
level in adversary (except a 0.06 tie at std 0), push, and speaker_listener, and at
std ≥ 0.8 in spread. BC is also above the denoiser at delay k = 8 and stuck-at p = 0.5
and 1.0 in those four scenarios. **simple_tag has no BC result yet.**

### Why the denoiser wins against no recovery (and what the evidence says)

- **The clean action is mostly determined by the state.** BC, which only sees the
  state, reaches reward close to the teacher (section 4.1). So the state carries
  enough information to recover the action. A model that sees both the state and the
  corrupted action can pull the action back toward what that state calls for.
- **The more corrupt the action, the more the model leans on the state.** That is
  consistent with the gain growing with severity, and with the flat denoiser curves
  (the denoiser's reward barely changes from std 1 to std 3, while no recovery keeps
  falling).
- **A temporal filter cannot do this.** Kalman uses no state and assumes the fault
  looks like noise around the true action. Under a stuck or biased fault the
  filtered action stays wrong, so it scores the same as no recovery.
- **Evidence the denoiser really uses the noisy action** (not only the state): in
  simple_spread the pure-noise control is much worse than the denoiser
  (-628 to -878 vs -520 to -620). So in that scenario the perturbed action carries
  useful information.
- **Evidence that it often does not:** in simple_speaker_listener the control from
  pure noise (≈ -60) matches the denoiser (-59 to -60) at every level. The denoiser's
  curve there is flat at about -58, which is what a state-only policy would do. In
  that scenario the "win" over no recovery is largely a state-only policy replacing
  a badly corrupted one. BC, a *deterministic* state-only policy, is much better
  than that generative one (-32 vs -60).

---

## 6. Where the denoiser does not win

- **No or small fault**: it adds error. speaker_listener -32 → -56, tag stuck-at p = 0
  57.5 → 32.0, push -8.5 → -10.7. Averaged over the zero-fault points across the four
  policies: speaker_listener -12.8, simple_tag -20.9, push -1.0 (spread +4.5,
  adversary +0.6).
- **Mild delay** (k = 1 to 3): often below doing nothing.
- **Versus BC**: BC is ahead nearly everywhere (section 5).
- **Predator-prey**: no gain for predators (section 4.5).
- **Starting step**: the wrong choice (t = 40 on spread) turns a gain into a loss.

---

## 7. Caveats

1. **The "denoiser" number is the best of two starting steps (20, 40) per point,
   chosen with the test results.** It is therefore optimistic. With a fixed
   starting step the win rates are lower (section 4.6), especially t = 40.
2. **Seeds differ between experiments.** Plotted denoiser sweeps use older seeds
   (e.g. 0, 1, 5, 10, 15, 42, 48, 49), while BC uses seeds 47 to 49. BC rows are therefore
   compared with numbers from different episodes. The BC across-seed spread is small
   (std 0.1 to 0.2 on adversary, push and speaker_listener) but I did not run both on
   identical seeds.
3. **BC does not see any fault by construction**, so its flat line is the value
   *with no perturbation*. It is a reference, not a method that is "robust" in a
   tested sense. BC's per-level rows are one rollout per seed, repeated at each
   level.
4. The denoiser is trained on M3DDPG data and applied to four policies. BC is also
   trained from M3DDPG data.
5. The H=1, diffusion-policy control and predator-only/prey-only results are
   single-run and not seed-averaged.
6. BC in the denoiser's own comparison is trained on all 25 steps while the
   denoiser's condition covers only the first step (section 2).

## 8. Not included

- **simple_tag BC** (training in progress).
- **The VMAS reconstructor plan** you pasted (C1/C2 supervised reconstructors,
  8-sector denoiser): no VMAS code or results exist in this repo, and I have not run
  or implemented any of it. Nothing in this report comes from VMAS.
- The `marl_vla` project (robomimic tasks) is a different codebase and is not covered.
- simple_crypto has data but was not in the sweeps.
