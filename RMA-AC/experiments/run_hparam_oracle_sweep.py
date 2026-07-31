"""
run_hparam_oracle_sweep.py
===========================
Per-point hyperparameter-oracle sweep for the EMA / median / Kalman action
filters: rather than one fixed hyperparameter applied across every severity
(as in tune_filters.py's combined-score selection, or the fixed
hyperparameters applied in run_full_filter_sweeps.py), this searches a
coarse hyperparameter grid *independently at every (scenario, attack,
severity) point* and keeps only the best-achievable ("oracle") reward at
that exact point. This answers "how much better could the policy do if a
filter's hyperparameter were perfectly matched to the exact perturbation,"
independent of whether any single fixed value can capture that.

Scope (first pass): maddpg-none only, 4 scenarios, 5 perturbation types x 3
severities each (min/mid/max, with min=0 = no perturbation for every type),
800 episodes/rollout. The ALGORITHMS list below is deliberately a simple
constant so extending to earnie/rmaac/m3ddpg later is a one-line change.

Builds one TF session per (scenario, algorithm) and reuses it across all 15
(attack, severity) points x the ~16-point hyperparameter grid, rather than
reloading the checkpoint per rollout (matches the efficient pattern in
tune_filters.py).

Output: noise_sweeps/filters_oracle/{delay,actuation_fault,noise,
biased_noise_pos,biased_noise_neg}/seed{N}/ -- a new, separate directory
tree from noise_sweeps/filters/ (the fixed-hyperparameter full-sweep
results) and noise_sweeps/filters/tuning/ (the tuning-phase results).
Nothing in those existing trees is read or written by this script.

Usage:
    python run_hparam_oracle_sweep.py --num-test-episodes 800 --seed 0 --save-dir ./model
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

SCENARIOS = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]

# (variant flag, exp-name suffix) -- maddpg-only for this first pass.
ALGORITHMS = [
    ("maddpg-none", "maddpg"),
]

_SIGMA_LEVELS = [0.0, 1.5, 3.0]


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


_STUCK_LEVELS = [0.0, 0.5, 1.0]

# (label, ActionAttack kind, severities [min, mid, max], setter, output subdir, csv suffix)
ATTACKS = [
    ("delay",             "delay",        [0, 3, 8],     _set_delay,       "delay",            "delay_k"),
    ("stuck_at",          "stuck_at",     _STUCK_LEVELS,  _set_stuck,       "actuation_fault",  "sp_prob"),
    ("gauss",             "gauss",        _SIGMA_LEVELS,  _set_gauss,       "noise",            "action_noise_std"),
    ("biased_gauss_pos",  "biased_gauss", _SIGMA_LEVELS,  _set_biased_pos,  "biased_noise_pos", "action_noise_std"),
    ("biased_gauss_neg",  "biased_gauss", _SIGMA_LEVELS,  _set_biased_neg,  "biased_noise_neg", "action_noise_std"),
]

EMA_GRID = [0.1, 0.3, 0.6, 0.95]
MEDIAN_GRID = [3, 7, 11]
KALMAN_GRID = [(q, r) for q in [0.05, 0.2, 1.0] for r in [0.01, 0.05, 0.2]]

FILTER_GRIDS = {
    "ema": (EMA_GRID, lambda a, c: setattr(a, "ema_alpha", c), lambda c: "alpha={}".format(c)),
    "median": (MEDIAN_GRID, lambda a, c: setattr(a, "median_window", c), lambda c: "window={}".format(c)),
    "kalman": (KALMAN_GRID, lambda a, c: (setattr(a, "kalman_q", c[0]), setattr(a, "kalman_r", c[1])),
               lambda c: "q={},r={}".format(c[0], c[1])),
}
FILTER_LIST = ["ema", "median", "kalman"]


def _best_at_point(env, trainers, arglist, attack, num_adversaries, n_episodes, filt_name):
    """Sweep filt_name's hyperparameter grid at the current attack/severity
    setting (already applied to arglist) and return (best_hparam_repr, best_reward)."""
    grid, set_hparams, hparam_repr = FILTER_GRIDS[filt_name]
    best_reward, best_repr = None, None
    for candidate in grid:
        set_hparams(arglist, candidate)
        filt = ActionFilter(arglist, kind=filt_name)
        r = rollout_reward(env, trainers, arglist, attack, filt_name, filt,
                            "all", num_adversaries, n_episodes)
        if best_reward is None or r > best_reward:
            best_reward, best_repr = r, hparam_repr(candidate)
    return best_repr, best_reward


def run_scenario_algo(base_arglist, scenario, variant, suffix, n_episodes, seed, out_root):
    arglist = base_arglist
    arglist.scenario = scenario
    arglist.variant = variant
    arglist.exp_name = "{}__{}best".format(scenario, suffix)
    arglist.num_adversaries = 0
    arglist.benchmark = False
    apply_variant_defaults(arglist)

    print("\n{}".format("=" * 70))
    print("Oracle sweep: {} / {}".format(scenario, arglist.exp_name))
    print("{}".format("=" * 70))

    # rows_by_attack_label[label] = list of [severity, reward_no_filter,
    #   reward_ema, reward_median, reward_kalman, ema_best_hparam,
    #   median_best_hparam, kalman_best_hparam]
    rows_by_label = {label: [] for label, *_ in ATTACKS}

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)
        attack = ActionAttack(arglist)

        for label, kind, severities, setter, _subdir, _sev_name in ATTACKS:
            arglist.attack = kind
            for sev in severities:
                setter(arglist, sev)

                baseline = rollout_reward(env, trainers, arglist, attack, "no_filter", None,
                                           "all", num_adversaries, n_episodes)

                row = [sev, r2(baseline)]
                hparam_cols = []
                for filt_name in FILTER_LIST:
                    best_repr, best_reward = _best_at_point(
                        env, trainers, arglist, attack, num_adversaries, n_episodes, filt_name)
                    row.append(r2(best_reward))
                    hparam_cols.append(best_repr)
                row += hparam_cols
                rows_by_label[label].append(row)

    for label, kind, severities, setter, subdir, sev_name in ATTACKS:
        path = os.path.join(out_root, subdir, "seed{}".format(seed),
                             "{}_{}_oracle_sweep.csv".format(arglist.exp_name, label))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sev_name, "reward_no_filter", "reward_ema", "reward_median", "reward_kalman",
                              "ema_best_hparam", "median_best_hparam", "kalman_best_hparam"])
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
    out_root = os.path.join("..", "..", "noise_sweeps", "filters_oracle")

    for scenario in SCENARIOS:
        for variant, suffix in ALGORITHMS:
            run_scenario_algo(arglist, scenario, variant, suffix, n_episodes, seed, out_root)

    print("\nHyperparameter-oracle sweep complete.")
