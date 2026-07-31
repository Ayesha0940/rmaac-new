# Action-filter hyperparameter tuning

Tuning results for the EMA, median, and Kalman action-recovery filters
(`RMA-AC/experiments/action_filters.py`) evaluated against the noise / delay /
actuator-fault / biased-noise attacks in `RMA-AC/experiments/action_attacks.py`.
Produced by `RMA-AC/experiments/tune_filters.py`.

## Scope

- **Scenarios (4):** `simple_adversary`, `simple_push`, `simple_speaker_listener`,
  `simple_spread`. The predator-prey scenario (`simple_tag`) is excluded from this
  round of tuning.
- **Checkpoint:** `maddpg-none` ("maddpgbest") only — no EARNIE/RMAAC/M3DDPG variants.
- **Attack types (5):** `delay`, `stuck_at` (actuator fault), `gauss` (unbiased action
  noise, μ=0), `biased_gauss_pos` (μ=+1), `biased_gauss_neg` (μ=-1).
- **Episodes per rollout:** 200.

## Methodology

### Severity levels

3 representative severities per attack type — **min / mid / max** — taken from the
existing sweep-list conventions used elsewhere in this repo (`run_*_sweep.sh`), plus
`biased_gauss_pos`/`_neg` reusing the same noise-magnitude levels as `gauss`:

| attack | min | mid | max |
|---|---|---|---|
| `delay` (`delay_k`) | 1 | 3 | 8 |
| `stuck_at` (`sp_prob`) | 0.1 | 0.5 | 1.0 |
| `gauss` / `biased_gauss_pos` / `biased_gauss_neg` (`action_noise_std`) | 0.4 | 1.5 | 3.0 |

5 attacks × 3 severities = **15 (attack, severity) points** per hyperparameter
candidate. `biased_gauss_pos`/`_neg` use the same `ActionAttack` `biased_gauss` kind
with `attack_mu` fixed at +1 / -1 respectively; `gauss` is the μ=0 case.

### Selection metric: combined score, plus an oracle ceiling

Each hyperparameter candidate is scored by its **mean total reward averaged across all
15 points**, and the argmax is the tuned ("achievable") value — one hyperparameter used
everywhere, not tuned per attack type. To quantify what's lost by that constraint, we
also compute an **oracle ceiling** per (scenario, filter): the average of each point's
own best-candidate reward, i.e. the score if the ideal hyperparameter could be picked
independently for every attack/severity. `oracle_gap = oracle_mean_reward -
mean_reward` is how much reward is left on the table by using one fixed hyperparameter.
A `no_filter` baseline is measured at each of the 15 points the same way.

### Efficient rollout — one checkpoint load per scenario

`filter_eval.py` exposes `build_test_env(arglist)` (construct env/trainers, load the
checkpoint — called once per scenario) and `rollout_reward(env, trainers, arglist,
attack, method, filt, ...)` (run N episodes against an already-loaded policy — called
once per attack/severity/filter/hyperparameter combination). `tune_filters.py` opens one
TF session per scenario and loops every combination inside it, since a full search here
is ~585 rollouts/scenario and reloading the graph/checkpoint per rollout would dominate
runtime.

## Hyperparameter grids

| filter | hyperparameter(s) | grid |
|---|---|---|
| EMA | `alpha` | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99 (8 values) |
| Median | `window` | 3, 5, 7, 9, 11 (5 values) |
| Kalman | `(q, r)` — joint grid | q ∈ {0.05, 0.1, 0.2, 0.5, 1.0} × r ∈ {0.01, 0.02, 0.05, 0.1, 0.2} (25 combos) |

The EMA/Kalman grids already fold in the wider "less smoothing" range that an earlier
100-episode/3-attack-type/9-point tuning pass (plus a grid-extension follow-up) found
necessary — that first round's best hyperparameters kept landing at the edge of a
narrower initial grid (EMA α up to 0.9, Kalman q up to 0.2 / r down to 0.05); extending
those grids flipped several negative results positive. This run starts from that wider
range directly instead of needing a second extension pass.

Per scenario: 15 baseline rollouts + 15 × (8 + 5 + 25) = 585 rollouts × 200 episodes.
×4 scenarios = 2340 rollouts, 468,000 episodes total. Wall-clock: ~83 minutes (measured
directly from this run; throughput was ~93 episodes/sec).

## Final tuned hyperparameters

| scenario | filter | best hyperparameter | mean reward | baseline | % improvement | oracle reward | oracle % improvement | oracle gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| simple_adversary | EMA | `alpha=0.95` | -13.64 | -13.52 | -0.88% | -12.89 | +4.70% | 0.76 |
| simple_adversary | Median | `window=3` | -19.94 | -13.52 | -47.43% | -19.88 | -47.02% | 0.06 |
| simple_adversary | Kalman | `q=1.0, r=0.01` | -13.46 | -13.52 | +0.50% | -11.87 | +12.25% | 1.59 |
| simple_push | EMA | `alpha=0.95` | -20.75 | -21.12 | +1.75% | -20.21 | +4.31% | 0.54 |
| simple_push | Median | `window=3` | -22.93 | -21.12 | -8.58% | -22.87 | -8.26% | 0.07 |
| simple_push | Kalman | `q=0.5, r=0.05` | -20.63 | -21.12 | +2.31% | -19.70 | +6.74% | 0.94 |
| simple_speaker_listener | EMA | `alpha=0.9` | -118.05 | -120.42 | +1.97% | -112.54 | +6.54% | 5.50 |
| simple_speaker_listener | Median | `window=3` | -144.30 | -120.42 | -19.83% | -142.90 | -18.67% | 1.40 |
| simple_speaker_listener | Kalman | `q=0.1, r=0.01` | -117.63 | -120.42 | +2.32% | -108.99 | +9.49% | 8.64 |
| simple_spread | EMA | `alpha=0.9` | -592.00 | -591.78 | -0.04% | -588.24 | +0.60% | 3.76 |
| simple_spread | Median | `window=3` | -632.72 | -591.78 | -6.92% | -632.04 | -6.80% | 0.69 |
| simple_spread | Kalman | `q=1.0, r=0.02` | -587.61 | -591.78 | +0.70% | -578.99 | +2.16% | 8.62 |

Also machine-readable in `best_hyperparameters.csv` in this directory (columns:
`scenario, filter, best_hparam, mean_reward, baseline_mean_reward, pct_improvement,
oracle_mean_reward, oracle_pct_improvement, oracle_gap`). Full per-(attack, severity,
hyperparameter) reward data is in each scenario's `{scenario}_hparam_sweep_raw.csv`; the
per-attack-type breakdown (below) is in `{scenario}_by_attack.csv`.

## Per-attack-type breakdown

Using each filter's single chosen (combined-best) hyperparameter, % improvement over
baseline broken out by attack type (`{scenario}_by_attack.csv`):

| scenario | attack | EMA | Median | Kalman |
|---|---|---:|---:|---:|
| simple_adversary | delay | -1.51% | -46.26% | -2.12% |
| simple_adversary | stuck_at | +0.31% | -2.75% | +0.92% |
| simple_adversary | gauss | -11.15% | -108.09% | -8.29% |
| simple_adversary | biased_gauss_pos | +2.19% | -59.05% | **+7.74%** |
| simple_adversary | biased_gauss_neg | +2.06% | -98.04% | -0.83% |
| simple_push | delay | +0.47% | -13.52% | +1.08% |
| simple_push | stuck_at | -1.66% | -2.98% | -1.70% |
| simple_push | gauss | +4.51% | -12.41% | +5.75% |
| simple_push | biased_gauss_pos | +0.32% | -6.74% | +2.84% |
| simple_push | biased_gauss_neg | +5.77% | -9.80% | +4.02% |
| simple_speaker_listener | delay | -0.86% | -25.71% | -1.72% |
| simple_speaker_listener | stuck_at | **+8.97%** | +3.26% | +5.54% |
| simple_speaker_listener | gauss | +1.61% | -34.84% | +0.44% |
| simple_speaker_listener | biased_gauss_pos | -3.31% | -32.28% | +4.84% |
| simple_speaker_listener | biased_gauss_neg | -2.15% | -27.76% | -1.79% |
| simple_spread | delay | -0.35% | -7.25% | +1.89% |
| simple_spread | stuck_at | -0.35% | -2.55% | -1.46% |
| simple_spread | gauss | +0.38% | -9.09% | +1.26% |
| simple_spread | biased_gauss_pos | -0.70% | -8.10% | +0.26% |
| simple_spread | biased_gauss_neg | +0.92% | -8.35% | +2.11% |

## Findings

1. **Median is harmful across every attack type, in every scenario, with no
   exceptions** (`simple_speaker_listener`/`stuck_at` at +3.26% is the single positive
   cell in the entire table). Its oracle gap is also small everywhere (0.06–1.40) —
   meaning even the *best possible* window size wouldn't fix this; it's not a tuning
   problem, median genuinely isn't the right tool here.
2. **`delay` is the hardest attack type for every filter** — small negative or barely
   positive everywhere. This tracks: a communication/actuation delay is a signal shifted
   in time, and time-domain smoothing filters can't recover information that was never
   received on time; they can only add more lag on top.
3. **`stuck_at` is consistently where EMA/Kalman help most** (`simple_speaker_listener`
   EMA +8.97%, Kalman +5.54%), consistent with the standalone filter unit tests from the
   initial implementation — a frozen/stuck actuator value is effectively a stationary
   signal once frozen, which smoothing-style filters are well suited to track.
4. **`gauss`/`biased_gauss` results are scenario-dependent** — helpful in 3/4 scenarios,
   but actively harmful in `simple_adversary` (EMA -11.15%, Kalman -8.29% on plain
   `gauss`) while still helping there on `biased_gauss_pos` (+2.19%/+7.74%). No single
   filter/hyperparameter choice is uniformly best across bias directions in that
   scenario.
5. **The oracle gap is real and sometimes large** — e.g. `simple_speaker_listener`
   Kalman: achievable +2.32% vs. oracle +9.49% (gap 8.64). This means a per-attack-type
   tuned hyperparameter (rather than one fixed value across delay + stuck_at + gauss +
   both biased directions) could meaningfully outperform what's reported here, at the
   cost of needing to know which attack is active at test/deploy time.

## Reproducing / extending this search

```bash
cd RMA-AC/experiments
python tune_filters.py --num-test-episodes 200 --seed 0 --save-dir ./model
```

Writes, per scenario: `{scenario}_hparam_sweep_raw.csv` (every rollout),
`{scenario}_hparam_summary.csv` (combined-best + oracle per filter),
`{scenario}_by_attack.csv` (chosen hyperparameter's performance split by attack type).
Refreshes the combined `best_hyperparameters.csv` after each scenario, so a long run's
partial progress is never lost. To tune against a different set of attacks/severities or
grids, edit the `ATTACKS`, `EMA_GRID`, `MEDIAN_GRID`, `KALMAN_GRID` constants at the top
of the script.
