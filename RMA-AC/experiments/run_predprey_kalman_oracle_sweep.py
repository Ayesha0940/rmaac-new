"""
run_predprey_kalman_oracle_sweep.py
=====================================
Per-point Kalman hyperparameter oracle sweep for the predator-prey scenario
(simple_tag), with full benchmark metrics (predator reward, prey reward,
captures, survival time) -- see run_hparam_oracle_sweep.py for the same
"sweep a grid, keep the best-achievable value at each exact point" oracle
methodology applied to the other 4 (non-benchmark) scenarios.

Differences from that script: Kalman-only (EMA/median dropped, matching the
recent narrowing of the combined-plot scope to Kalman), benchmark=True,
num_adversaries=3, attack_target="all". Each of the 4 benchmark metrics is
maximized INDEPENDENTLY across the swept (q,r) grid at each point -- matching
the existing convention already used for the DDPM predator-prey sweep (see
noise_sweeps/plot_predprey.py), where best_reward_pred_with_diffusion /
best_reward_prey_with_diffusion / best_captures_with_diffusion /
best_survival_with_diffusion are each their own independent max, not all tied
to one "winning" t_start. The best (q,r) achieving each metric is recorded
and may differ metric-to-metric.

Scope: all 4 policy variants (maddpg/earnie/rmaac/m3ddpg). 5 perturbation types x 3 severities
each (min/mid/max, 0 = no perturbation) x a 9-combo Kalman (q,r) grid, 800
episodes/rollout.

Builds one TF session and reuses it across all attack/severity/hyperparameter
combinations (matches the efficient pattern in run_hparam_oracle_sweep.py).

Output: noise_sweeps/predprey_kalman_oracle/{delay,actuation_fault,noise,
biased_noise_pos,biased_noise_neg}/seed{N}/ -- a new, separate tree; nothing
in noise_sweeps/filters_oracle/ (the other 4 scenarios' non-benchmark oracle
data) is read or written here.

Usage:
    python run_predprey_kalman_oracle_sweep.py --num-test-episodes 800 --seed 0 --save-dir ./model
"""
import csv
import os
import random

import numpy as np
import tensorflow as tf

from train import parse_args, apply_variant_defaults, r2
import maddpg.common.tf_util as U
from filter_eval import build_test_env, rollout_reward
from action_attacks import ActionAttack
from action_filters import ActionFilter

SCENARIO = "simple_tag"
NUM_ADVERSARIES = 3

# (variant flag, exp-name suffix)
ALGORITHMS = [
    ("maddpg-none", "maddpg"),
    ("maddpg-earnie", "earnie"),
    ("maddpg-act_adv", "rmaac"),
    ("m3ddpg", "m3ddpg"),
]

_SIGMA_LEVELS = [0.0, 1.5, 3.0]
_STUCK_LEVELS = [0.0, 0.5, 1.0]


def _set_delay(a, v):
    a.delay_k = v


def _set_stuck(a, v):
    a.sp_prob = v


def _set_gauss(a, v):
    a.attack_sigma = v
    a.attack_mu = 0.0


def _set_biased_pos(a, v):
    a.attack_sigma = v
    a.attack_mu = 1.0


def _set_biased_neg(a, v):
    a.attack_sigma = v
    a.attack_mu = -1.0


# (label, ActionAttack kind, severities [min, mid, max], setter, output subdir, csv severity-column name)
ATTACKS = [
    ("delay",             "delay",        [0, 3, 8],     _set_delay,       "delay",            "delay_k"),
    ("stuck_at",          "stuck_at",     _STUCK_LEVELS,  _set_stuck,       "actuation_fault",  "sp_prob"),
    ("gauss",             "gauss",        _SIGMA_LEVELS,  _set_gauss,       "noise",            "action_noise_std"),
    ("biased_gauss_pos",  "biased_gauss", _SIGMA_LEVELS,  _set_biased_pos,  "biased_noise_pos", "action_noise_std"),
    ("biased_gauss_neg",  "biased_gauss", _SIGMA_LEVELS,  _set_biased_neg,  "biased_noise_neg", "action_noise_std"),
]

KALMAN_GRID = [(q, r) for q in [0.05, 0.2, 1.0] for r in [0.01, 0.05, 0.2]]

METRIC_NAMES = ["pred_reward", "prey_reward", "captures", "survival"]


def _sweep_point(env, trainers, arglist, attack, num_adversaries, n_episodes):
    """Sweep the Kalman grid at the current attack/severity setting (already
    applied to arglist). Returns (baseline_tuple, per_candidate) where
    per_candidate maps (q, r) -> (pred_reward, prey_reward, captures, survival).
    """
    baseline = rollout_reward(env, trainers, arglist, attack, "no_filter", None,
                               "all", num_adversaries, n_episodes)

    per_candidate = {}
    for q, r_ in KALMAN_GRID:
        arglist.kalman_q, arglist.kalman_r = q, r_
        filt = ActionFilter(arglist, kind="kalman")
        result = rollout_reward(env, trainers, arglist, attack, "kalman", filt,
                                 "all", num_adversaries, n_episodes)
        per_candidate[(q, r_)] = result
    return baseline, per_candidate


def _best_per_metric(per_candidate):
    """Independently maximize each of the 4 benchmark metrics across the
    grid. Returns {metric_name: (best_value, best_hparam_repr)}."""
    best = {}
    for i, name in enumerate(METRIC_NAMES):
        best_val, best_hp = None, None
        for (q, r_), result in per_candidate.items():
            v = result[i]
            if best_val is None or v > best_val:
                best_val, best_hp = v, "q={},r={}".format(q, r_)
        best[name] = (best_val, best_hp)
    return best


def run_predprey(base_arglist, variant, suffix, n_episodes, seed, out_root):
    arglist = base_arglist
    arglist.scenario = SCENARIO
    arglist.variant = variant
    arglist.exp_name = "{}__{}best".format(SCENARIO, suffix)
    arglist.num_adversaries = NUM_ADVERSARIES
    arglist.benchmark = True
    apply_variant_defaults(arglist)

    print("\n{}".format("=" * 70))
    print("Predator-prey Kalman oracle sweep: {}".format(arglist.exp_name))
    print("{}".format("=" * 70))

    rows_by_label = {label: [] for label, *_ in ATTACKS}

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)
        attack = ActionAttack(arglist)

        for label, kind, severities, setter, _subdir, _sev_name in ATTACKS:
            arglist.attack = kind
            for sev in severities:
                setter(arglist, sev)

                baseline, per_candidate = _sweep_point(
                    env, trainers, arglist, attack, num_adversaries, n_episodes)
                best = _best_per_metric(per_candidate)

                pred_b, prey_b, cap_b, surv_b = baseline
                row = [
                    sev, r2(pred_b), r2(prey_b), r2(cap_b), r2(surv_b),
                    r2(best["pred_reward"][0]), best["pred_reward"][1],
                    r2(best["prey_reward"][0]), best["prey_reward"][1],
                    r2(best["captures"][0]), best["captures"][1],
                    r2(best["survival"][0]), best["survival"][1],
                ]
                rows_by_label[label].append(row)
                print("  [{}] severity={} pred_base={} prey_base={} "
                      "best_pred={} best_prey={} best_cap={} best_surv={}".format(
                          label, sev, r2(pred_b), r2(prey_b),
                          r2(best["pred_reward"][0]), r2(best["prey_reward"][0]),
                          r2(best["captures"][0]), r2(best["survival"][0])))

    for label, kind, severities, setter, subdir, sev_name in ATTACKS:
        path = os.path.join(out_root, subdir, "seed{}".format(seed),
                             "{}_{}_oracle_sweep.csv".format(arglist.exp_name, label))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sev_name, "pred_reward_baseline", "prey_reward_baseline",
                              "captures_baseline", "survival_baseline",
                              "best_pred_reward", "best_pred_reward_hparam",
                              "best_prey_reward", "best_prey_reward_hparam",
                              "best_captures", "best_captures_hparam",
                              "best_survival", "best_survival_hparam"])
            writer.writerows(rows_by_label[label])
        print("Saved {}".format(path))


if __name__ == '__main__':
    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)

    if arglist.seed is not None:
        np.random.seed(arglist.seed)
        random.seed(arglist.seed)
        tf.set_random_seed(arglist.seed)

    n_episodes = arglist.num_test_episodes
    seed = arglist.seed if arglist.seed is not None else 0
    out_root = os.path.join("..", "..", "noise_sweeps", "predprey_kalman_oracle")

    for variant, suffix in ALGORITHMS:
        run_predprey(arglist, variant, suffix, n_episodes, seed, out_root)

    print("\nPredator-prey Kalman oracle sweep complete.")
