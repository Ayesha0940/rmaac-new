"""
run_full_filter_sweeps.py
==========================
Full-scale (default 800-episode) noise / delay / actuator-fault sweep
evaluation of the EMA / median / Kalman action filters, using the
per-scenario tuned hyperparameters found by tune_filters.py (see
noise_sweeps/filters/tuning/best_hyperparameters.csv), applied across all 4
policy variants (maddpg, earnie, rmaac, m3ddpg) for each of the 4 scenarios.

NOTE: the tuned hyperparameters were found/validated ONLY on maddpg-none
checkpoints (see HYPERPARAMETER_TUNING.md). Applying the same per-scenario
values to earnie/rmaac/m3ddpg here is exploratory -- those policies'
sensitivity to filtering was not separately tuned.

Writes CSVs with the same schema/filenames filter_eval.py's CLI sweep
produces (`_delay_filter_sweep.csv`, `_stuck_at_filter_sweep.csv`,
`_mu{mu}_actstd_filter_sweep.csv`), into the existing
noise_sweeps/filters/{delay,actuation_fault,noise}/seed{N}/ directories, so
plot_filter_combined.py works unchanged.

Builds one TF session per (scenario, variant) and reuses it across all three
sweep types (delay + stuck_at + noise = 20 severities x 4 methods), rather
than reloading the checkpoint per severity value.

Usage:
    python run_full_filter_sweeps.py --num-test-episodes 800 --seed 0 --save-dir ./model
"""
import csv
import os
import random

import numpy as np
import tensorflow as tf

from train import parse_args, apply_variant_defaults, r2
import maddpg.common.tf_util as U
from filter_eval import build_test_env, rollout_reward, _row_for, _header_for
from action_attacks import ActionAttack
from action_filters import ActionFilter

SCENARIOS = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]

# per-scenario tuned hyperparameters, from noise_sweeps/filters/tuning/best_hyperparameters.csv
# (tuned/validated on maddpg-none only -- see HYPERPARAMETER_TUNING.md)
TUNED_HYPERPARAMS = {
    "simple_adversary":        {"ema_alpha": 0.95, "median_window": 3, "kalman_q": 1.0, "kalman_r": 0.01},
    "simple_push":              {"ema_alpha": 0.95, "median_window": 3, "kalman_q": 0.5, "kalman_r": 0.05},
    "simple_speaker_listener":  {"ema_alpha": 0.9,  "median_window": 3, "kalman_q": 0.1, "kalman_r": 0.01},
    "simple_spread":            {"ema_alpha": 0.9,  "median_window": 3, "kalman_q": 1.0, "kalman_r": 0.02},
}

# (variant flag, exp-name suffix)
VARIANTS = [
    ("maddpg-none", "maddpg"),
    ("maddpg-earnie", "earnie"),
    ("maddpg-act_adv", "rmaac"),
    ("m3ddpg", "m3ddpg"),
]

DELAY_K_LIST = [1, 2, 3, 5, 8]
SP_PROB_LIST = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
ACT_STD_LIST = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0]

FILTER_LIST = ["ema", "median", "kalman"]


def _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                severity_list, apply_severity):
    rows = []
    for sev in severity_list:
        apply_severity(arglist, sev)
        results = {"no_filter": rollout_reward(env, trainers, arglist, attack, "no_filter", None,
                                                 "all", num_adversaries, n_episodes)}
        for filt_name in FILTER_LIST:
            results[filt_name] = rollout_reward(env, trainers, arglist, attack, filt_name, filters[filt_name],
                                                  "all", num_adversaries, n_episodes)
        rows.append(_row_for(sev, results, FILTER_LIST, False))
    return rows


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved {}".format(path))


def run_scenario_variant(base_arglist, scenario, variant, suffix, n_episodes, seed, out_root):
    arglist = base_arglist
    arglist.scenario = scenario
    arglist.variant = variant
    arglist.exp_name = "{}__{}best".format(scenario, suffix)
    arglist.num_adversaries = 0
    arglist.benchmark = False
    apply_variant_defaults(arglist)

    hp = TUNED_HYPERPARAMS[scenario]
    arglist.ema_alpha = hp["ema_alpha"]
    arglist.median_window = hp["median_window"]
    arglist.kalman_q = hp["kalman_q"]
    arglist.kalman_r = hp["kalman_r"]

    print("\n{}".format("=" * 70))
    print("Full sweep: {} / {} (ema_alpha={}, median_window={}, kalman_q={}, kalman_r={})".format(
        scenario, arglist.exp_name, hp["ema_alpha"], hp["median_window"], hp["kalman_q"], hp["kalman_r"]))
    print("{}".format("=" * 70))

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)
        attack = ActionAttack(arglist)
        filters = {kind: ActionFilter(arglist, kind=kind) for kind in FILTER_LIST}

        # --- delay sweep ---
        arglist.attack = "delay"
        delay_rows = _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                                 DELAY_K_LIST, lambda a, v: setattr(a, "delay_k", v))
        delay_path = os.path.join(out_root, "delay", "seed{}".format(seed),
                                   "{}_delay_filter_sweep.csv".format(arglist.exp_name))
        _write_csv(delay_path, _header_for("delay_k", FILTER_LIST, False), delay_rows)

        # --- stuck_at (actuator fault) sweep ---
        arglist.attack = "stuck_at"
        stuck_rows = _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                                 SP_PROB_LIST, lambda a, v: setattr(a, "sp_prob", v))
        stuck_path = os.path.join(out_root, "actuation_fault", "seed{}".format(seed),
                                   "{}_stuck_at_filter_sweep.csv".format(arglist.exp_name))
        _write_csv(stuck_path, _header_for("sp_prob", FILTER_LIST, False), stuck_rows)

        # --- gauss action-noise sweep (mu=0) ---
        arglist.attack = "gauss"
        arglist.noise_mu = 0

        def _set_gauss(a, v):
            a.attack_sigma = v
            a.attack_mu = 0.0

        noise_rows = _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                                 ACT_STD_LIST, _set_gauss)
        noise_path = os.path.join(out_root, "noise", "seed{}".format(seed),
                                   "{}_mu0.0_actstd_filter_sweep.csv".format(arglist.exp_name))
        _write_csv(noise_path, _header_for("action_noise_std", FILTER_LIST, False), noise_rows)


if __name__ == '__main__':
    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)

    if arglist.seed is not None:
        np.random.seed(arglist.seed)
        random.seed(arglist.seed)
        tf.set_random_seed(arglist.seed)

    n_episodes = arglist.num_test_episodes
    seed = arglist.seed if arglist.seed is not None else 0
    out_root = os.path.join("..", "..", "noise_sweeps", "filters")

    for scenario in SCENARIOS:
        for variant, suffix in VARIANTS:
            run_scenario_variant(arglist, scenario, variant, suffix, n_episodes, seed, out_root)

    print("\nFull filter sweep complete.")
