# State-only BC vs the diffusion denoiser

Question this answers: *is the denoiser's gain due to diffusion, or does a plain
network that only sees the state do as well?* BC is that plain network. This report
compares only BC and the denoiser. Reward is higher-is-better.

Scope and status:
- Testbeds: simple_adversary, simple_push, simple_speaker_listener, simple_spread.
- **simple_tag (predator-prey) is not included: its BC model is still training.** No
  claim is made about it here.
- Numbers come from `noise_sweeps/plots/aggregated_diffusion_and_filters_combined.csv`
  (denoiser, multi-seed), `noise_sweeps/bc/` (BC, seeds 47 to 49, 800 episodes), and
  the per-seed denoiser CSVs for the fixed-starting-step check.

---

## 1. Result in one paragraph

BC beats the denoiser at almost every fault level in these four testbeds. Of 464
points (scenario × policy × fault type × level, including zero-fault points), the
denoiser is ahead of BC in 38 (8%). Every one of those is in simple_spread or
simple_adversary at zero or very small faults. In simple_push and
simple_speaker_listener the denoiser is never ahead of BC. So the denoiser's gain over
the un-recovered policy is not evidence that diffusion is needed: a state-only
imitator that ignores both the policy and the fault does better.

---

## 2. What each one sees

| | Denoiser (H=25, main) | BC |
|---|---|---|
| Uses the perturbed action | **Yes**: placed in slot 0 of a 25-slot trajectory (other slots zero) | **No** |
| Uses the base policy | Indirectly: its input is the policy's action after the fault | **No** |
| State it is conditioned on | Joint observation of all agents, **current step** | Joint observation of all agents, **current step** |
| Model | Diffusion (DDPM, 100 steps; reverse chain starts at step 20 or 40), epsilon-prediction MLP, hidden 256 | Deterministic MLP, state → 256 → 256 → action, hidden 256 |
| Loss | Noise-prediction MSE | MSE on normalised actions |
| Training data | Same dataset D: 2000 clean M3DDPG episodes of 25 steps | Same dataset D |
| What in D it trains on | Whole 25-step action trajectories, conditioned on the **first state of the episode only** | All 25 (state, action) pairs of each episode (50,000 pairs) |
| Executed action | Slot 0, un-normalised, split per agent, clipped to [-1, 1] | Un-normalised, split per agent, clipped to [-1, 1] |
| Fault applied to its output | No | No |
| Result across fault levels | Changes with the fault | Identical at every level (one rollout per seed, repeated per level) |

Both are centralised and see the same state at test time. The difference is that the
denoiser also receives the corrupted action. BC's output does not depend on the
fault at all, so its curve is flat by construction.

---

## 3. No fault

Reward with no fault, M3DDPG. The denoiser column is zero-noise (std 0) and BC is
from clean rollouts.

| Scenario | Clean M3DDPG policy | Denoiser (std 0) | BC |
|---|---|---|---|
| simple_adversary | 4.02 | 3.75 | 3.69 |
| simple_push | -8.63 | -10.67 | -8.43 |
| simple_speaker_listener | -33.06 | -56.04 | -31.92 |
| simple_spread | -435.29 | -440.43 | -468.69 |

- BC stays close to its teacher in push, speaker_listener and adversary.
- The denoiser loses reward even with no fault in push (-2.0) and especially
  speaker_listener (-23.0).
- simple_spread is the one place BC is clearly worse than the policy (-468.7 vs
  -435.3, about 8% lower), so BC is a weaker imitator there.

---

## 4. Under faults (M3DDPG policy, denoiser = best of starting steps 20 and 40)

Denoiser reward at each fault level. The BC column is the same single number at every
level. **Bold** marks the better of the two.

| Scenario | BC | noise std 1 | noise std 2 | noise std 3 | delay k=3 | delay k=8 | stuck p=0.5 | stuck p=1.0 | bias +1, std 2 | bias -1, std 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| simple_adversary | **3.7** | -0.6 | -5.1 | -6.7 | 0.5 | -1.6 | -0.3 | -3.3 | -5.0 | -4.3 |
| simple_push | **-8.4** | -12.2 | -13.1 | -13.5 | -11.4 | -12.7 | -12.5 | -14.1 | -13.3 | -13.0 |
| simple_speaker_listener | **-31.9** | -59.1 | -58.8 | -60.8 | -56.1 | -60.7 | -58.1 | -61.4 | -59.9 | -59.6 |
| simple_spread | **-468.7** | -527.2 | -590.5 | -617.5 | -483.1 | -589.3 | -555.7 | -767.1 | -588.0 | -589.9 |

BC is the better one in every cell above. How far ahead, averaged over all faulted
points and all four policies (BC minus denoiser):

| Scenario | Average BC advantage (reward units) | In the M3DDPG table above |
|---|---|---|
| simple_adversary | +5.5 | denoiser is 3 to 10 units below BC |
| simple_push | +4.1 | denoiser loses 35% to 67% more reward than BC |
| simple_speaker_listener | +27.8 | denoiser loses about 1.8x to 1.9x more reward than BC |
| simple_spread | +86.4 | gap ranges from 3% (delay k=3) to 64% (stuck p=1.0) more loss; the largest single gap over all policies is 298 |

The denoiser's curve is nearly flat across fault levels (e.g. speaker_listener stays at
about -59 to -61 from std 1 to std 3). It trades a worse level at low faults for
protection at high faults. BC is already better than that level at every fault
level.

---

## 5. Where the denoiser is ahead of BC

38 of 464 points, nowhere else:

| Scenario | Points ahead | Where |
|---|---|---|
| simple_spread | 28 of 116 | zero fault; noise std 0 and 0.4; stuck p up to 0.1; delay k = 1 and 2 |
| simple_adversary | 10 of 116 | 8 at zero fault; plus ERNIE at noise std 0.4 and at stuck p=0.1 |
| simple_push | 0 | never |
| simple_speaker_listener | 0 | never |

- 22 of the 38 are zero-fault points. The other 16 are at the lowest fault levels: 14 in simple_spread and 2 in simple_adversary (ERNIE only). In spread, the denoiser's lead there is mostly because BC
  is a weaker imitator (section 3): at zero fault the policy itself is better than BC.
- With one fixed starting step instead of picking the best of two, the denoiser is
  ahead of BC even less often (M3DDPG, 208 faulted points across noise/delay/stuck): t = 20:
  15% of spread points and 0% elsewhere; t = 40: 12% of spread, 2% of adversary, 0% elsewhere.

---

## 6. Why BC is ahead (what the data support)

- **The state carries most of the information.** BC reaches reward close to the
  clean policy in three of four testbeds, so the clean action is largely a function
  of the state. A model that ignores the fault entirely can therefore do well.
- **The denoiser has its own error even with no fault.** At std 0 it loses reward in
  push and speaker_listener (section 3), so it starts below BC before any fault
  appears.
- **In speaker_listener the denoiser looks like a state-only policy.** An earlier
  control, the same network started from pure noise (so it ignores the perturbed
  action), scored about -60 at every noise level. The denoiser scored -59 to -60. So
  the denoiser adds essentially nothing over a state-only *generative* policy there,
  and BC (a deterministic state-only policy) is far better than both (-32).
- **In simple_spread the denoiser does use the perturbed action.** The same control
  scored -628 to -878 there, much worse than the denoiser (-520 to -620). So the
  perturbed action helps the denoiser, just not enough to pass BC once faults exceed
  small levels.

Not tested (hypotheses only): whether the denoiser's gap to BC comes from conditioning
on the first state only during training (BC sees all 25 states). The H=1 denoiser has
the same first-state-only conditioning, and was not compared with BC.

---

## 7. Caveats

1. **The denoiser numbers are best-of-two starting steps chosen from test results**,
   which favours the denoiser. Even so, it loses to BC nearly everywhere.
2. **Different seeds.** BC used seeds 47 to 49. The denoiser sweeps used other seeds
   (0, 1, 5, 10, 15, 42, 48, 49 depending on the experiment). BC's across-seed
   spread is small (std 0.1 to 0.2 on adversary, push, speaker_listener; 2.2 on
   spread) and BC is a single flat number per scenario.
3. **BC is a reference, not a robust method in a tested sense.** Its output is never
   perturbed, so its curve equals its clean score. The fair statement is: with the
   fault model used here, a model that never sees the policy action does at least as
   well as the denoiser.
4. The comparison is not matched in training coverage (section 2): BC trains on 25
   states per episode, the denoiser's condition on 1.
5. BC is trained from M3DDPG data. The denoiser is also trained on M3DDPG data
   and applied to four policies. This report uses M3DDPG for the level tables, and
   all four policies for the 464-point counts.
6. **simple_tag is missing** (BC not finished). In the earlier combined results the
   denoiser did not help the predators there, so it should not be assumed to win.
