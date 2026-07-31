"""
run_full_filter_sweeps_biased.py
=================================
Companion to run_full_filter_sweeps.py: full-scale (default 800-episode)
biased action-noise sweep (mu=+1 and mu=-1, full sigma severity list),
using the same per-scenario tuned hyperparameters, across all 4 scenarios x
4 policy variants.

Split out from run_full_filter_sweeps.py (which only covered delay, stuck_at,
and unbiased gauss/mu=0) because it was scoped too narrowly the first time --
biased noise was added to the *tuning* objective specifically because it
revealed scenario-specific effects (e.g. simple_adversary: Kalman -8.29% on
unbiased gauss but +7.74% on biased_gauss_pos), so the full evaluation needs
it too. Writes to separate `noise_sweeps/filters/biased_noise_{pos,neg}/`
directories rather than mixing into the existing `noise_sweeps/filters/noise/`
tree, since plot_filter_combined.py's loader matches any mu value in that
directory and would otherwise silently average unbiased and biased runs
together as if they were repeated samples of the same condition.

Usage:
    python run_full_filter_sweeps_biased.py --num-test-episodes 800 --seed 0 --save-dir ./model
"""
import os
import random

import numpy as np
import tensorflow as tf

from train import parse_args, apply_variant_defaults
import maddpg.common.tf_util as U
from filter_eval import build_test_env, _header_for
from action_attacks import ActionAttack
from action_filters import ActionFilter
from run_full_filter_sweeps import (SCENARIOS, VARIANTS, TUNED_HYPERPARAMS, ACT_STD_LIST,
                                     FILTER_LIST, _run_sweep, _write_csv)


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
    print("Biased-noise sweep: {} / {} (ema_alpha={}, median_window={}, kalman_q={}, kalman_r={})".format(
        scenario, arglist.exp_name, hp["ema_alpha"], hp["median_window"], hp["kalman_q"], hp["kalman_r"]))
    print("{}".format("=" * 70))

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)
        attack = ActionAttack(arglist)
        filters = {kind: ActionFilter(arglist, kind=kind) for kind in FILTER_LIST}

        arglist.attack = "biased_gauss"

        # --- mu = +1 ---
        def _set_biased_pos(a, v):
            a.attack_sigma = v
            a.attack_mu = 1.0

        pos_rows = _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                               ACT_STD_LIST, _set_biased_pos)
        pos_path = os.path.join(out_root, "biased_noise_pos", "seed{}".format(seed),
                                 "{}_mu1.0_actstd_filter_sweep.csv".format(arglist.exp_name))
        _write_csv(pos_path, _header_for("action_noise_std", FILTER_LIST, False), pos_rows)

        # --- mu = -1 ---
        def _set_biased_neg(a, v):
            a.attack_sigma = v
            a.attack_mu = -1.0

        neg_rows = _run_sweep(env, trainers, arglist, attack, filters, num_adversaries, n_episodes,
                               ACT_STD_LIST, _set_biased_neg)
        neg_path = os.path.join(out_root, "biased_noise_neg", "seed{}".format(seed),
                                 "{}_mu-1.0_actstd_filter_sweep.csv".format(arglist.exp_name))
        _write_csv(neg_path, _header_for("action_noise_std", FILTER_LIST, False), neg_rows)


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

    print("\nBiased-noise full filter sweep complete.")
