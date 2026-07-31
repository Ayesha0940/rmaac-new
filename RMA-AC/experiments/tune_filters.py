"""
tune_filters.py
================
Hyperparameter search for the EMA / median / Kalman action filters
(action_filters.py / filter_eval.py), tuned per scenario using only the
maddpg-none ("maddpgbest") checkpoint. Skips the predator-prey scenario
(simple_tag) per the current scope.

For each scenario, each filter's hyperparameter(s) are swept against a fixed
grid of 5 attack types x 3 severities each (min/mid/max) = 15 (attack,
severity) points:
    delay              -- delay_k in [1, 3, 8]
    stuck_at           -- sp_prob in [0.1, 0.5, 1.0]
    gauss              -- unbiased action noise, attack_sigma in [0.4, 1.5, 3.0]
    biased_gauss_pos   -- biased action noise, mu=+1, same sigma levels
    biased_gauss_neg   -- biased action noise, mu=-1, same sigma levels

Each hyperparameter candidate is scored by its reward averaged across all 15
points; the argmax per (scenario, filter) is the tuned value ("achievable"
score, since it's one fixed hyperparameter used everywhere). We also report
an "oracle" ceiling per (scenario, filter): the reward if the ideal
hyperparameter could be picked independently for each of the 15 points,
i.e. the average of each point's own best-candidate reward. The gap between
oracle and achievable quantifies how much is left on the table by using one
fixed hyperparameter across every attack type and severity.

The EMA/Kalman grids here already include the wider "less smoothing" range
found by a first-round tuning pass + grid-extension follow-up (see
noise_sweeps/filters/tuning/HYPERPARAMETER_TUNING.md) -- this version folds
that into a single grid instead of needing a second extension pass.

The checkpoint/session/env are built once per scenario (via
filter_eval.build_test_env) and reused across every (attack, severity,
filter, hyperparameter) combination for that scenario -- with ~500+
rollouts/scenario, rebuilding the TF graph per rollout would dominate
runtime.

Usage:
    python tune_filters.py --num-test-episodes 200 --seed 0 --save-dir ./model
"""
import csv
import os

import numpy as np
import tensorflow as tf

from train import parse_args, apply_variant_defaults, r2
import maddpg.common.tf_util as U
from filter_eval import build_test_env, rollout_reward
from action_attacks import ActionAttack
from action_filters import ActionFilter

SCENARIOS = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]

# (label, ActionAttack kind, severity values, setter(arglist, severity_value))
_SIGMA_LEVELS = [0.4, 1.5, 3.0]


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


ATTACKS = [
    ("delay", "delay", [1, 3, 8], _set_delay),
    ("stuck_at", "stuck_at", [0.1, 0.5, 1.0], _set_stuck),
    ("gauss", "gauss", _SIGMA_LEVELS, _set_gauss),
    ("biased_gauss_pos", "biased_gauss", _SIGMA_LEVELS, _set_biased_pos),
    ("biased_gauss_neg", "biased_gauss", _SIGMA_LEVELS, _set_biased_neg),
]
ATTACK_LABELS = [a[0] for a in ATTACKS]
ALL_POINTS = [(label, sev) for label, _, severities, _ in ATTACKS for sev in severities]

EMA_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
MEDIAN_GRID = [3, 5, 7, 9, 11]
KALMAN_GRID = [(q, r) for q in [0.05, 0.1, 0.2, 0.5, 1.0] for r in [0.01, 0.02, 0.05, 0.1, 0.2]]


def _apply_point(arglist, label, sev):
    for lbl, kind, _, setter in ATTACKS:
        if lbl == label:
            arglist.attack = kind
            setter(arglist, sev)
            return
    raise ValueError("unknown attack label: {}".format(label))


def _pct_inc(best, baseline):
    if baseline == 0:
        return 0.0
    return (best - baseline) / abs(baseline) * 100.0


def _rollout_all_points(env, trainers, arglist, attack, method, filt, num_adversaries, n_episodes, raw_rows,
                         hparam_repr):
    """Roll out `method` (with filter `filt`, or None for baseline) at every
    (attack, severity) point. Returns {(label, sev): reward}."""
    point_rewards = {}
    for label, sev in ALL_POINTS:
        _apply_point(arglist, label, sev)
        r = rollout_reward(env, trainers, arglist, attack, method, filt,
                            "all", num_adversaries, n_episodes)
        point_rewards[(label, sev)] = r
        raw_rows.append([label, sev, method, hparam_repr, r2(r)])
    return point_rewards


def _score_grid(env, trainers, arglist, attack, num_adversaries, n_episodes,
                 filter_kind, grid, set_hparams, hparam_repr, raw_rows):
    """Returns (best_hparam, best_mean_reward, best_point_rewards, oracle_mean_reward)."""
    candidate_points = {}
    for candidate in grid:
        set_hparams(arglist, candidate)
        filt = ActionFilter(arglist, kind=filter_kind)
        candidate_points[candidate] = _rollout_all_points(
            env, trainers, arglist, attack, filter_kind, filt, num_adversaries, n_episodes,
            raw_rows, hparam_repr(candidate))

    scores = {c: float(np.mean(list(pts.values()))) for c, pts in candidate_points.items()}
    best = max(scores, key=scores.get)
    best_reward = scores[best]
    best_point_rewards = candidate_points[best]

    oracle_point_rewards = {pt: max(candidate_points[c][pt] for c in grid) for pt in ALL_POINTS}
    oracle_mean = float(np.mean(list(oracle_point_rewards.values())))

    return hparam_repr(best), best_reward, best_point_rewards, oracle_mean


def _label_avg(point_rewards, label):
    vals = [v for (lbl, sev), v in point_rewards.items() if lbl == label]
    return float(np.mean(vals))


def tune_scenario(base_arglist, scenario, n_episodes, out_dir):
    arglist = base_arglist
    arglist.scenario = scenario
    arglist.variant = "maddpg-none"
    arglist.exp_name = "{}__maddpgbest".format(scenario)
    arglist.num_adversaries = 0
    arglist.benchmark = False
    apply_variant_defaults(arglist)

    print("\n{}".format("=" * 70))
    print("Tuning scenario: {}".format(scenario))
    print("{}".format("=" * 70))

    raw_rows = []  # [attack, severity, method, hparam_repr, reward]

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)
        attack = ActionAttack(arglist)

        baseline_points = _rollout_all_points(
            env, trainers, arglist, attack, "no_filter", None, num_adversaries, n_episodes,
            raw_rows, "")
        baseline_mean = float(np.mean(list(baseline_points.values())))

        filter_grids = {
            "ema": (EMA_GRID, lambda a, c: setattr(a, "ema_alpha", c), lambda c: "alpha={}".format(c)),
            "median": (MEDIAN_GRID, lambda a, c: setattr(a, "median_window", c), lambda c: "window={}".format(c)),
            "kalman": (KALMAN_GRID, lambda a, c: (setattr(a, "kalman_q", c[0]), setattr(a, "kalman_r", c[1])),
                       lambda c: "q={},r={}".format(c[0], c[1])),
        }

        filter_results = {}
        for filt_name, (grid, set_hparams, hparam_repr) in filter_grids.items():
            filter_results[filt_name] = _score_grid(
                env, trainers, arglist, attack, num_adversaries, n_episodes,
                filt_name, grid, set_hparams, hparam_repr, raw_rows)

    os.makedirs(out_dir, exist_ok=True)

    # --- raw CSV ---
    raw_path = os.path.join(out_dir, "{}_hparam_sweep_raw.csv".format(scenario))
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "severity", "filter", "hparam", "reward"])
        writer.writerows(raw_rows)
    print("Saved {}".format(raw_path))

    # --- per-scenario summary (combined score + oracle ceiling) ---
    summary_rows = []
    for filt_name, (best_hparam, best_reward, best_points, oracle_mean) in filter_results.items():
        summary_rows.append([
            filt_name, best_hparam, r2(best_reward), r2(baseline_mean),
            r2(_pct_inc(best_reward, baseline_mean)),
            r2(oracle_mean), r2(_pct_inc(oracle_mean, baseline_mean)),
            r2(oracle_mean - best_reward),
        ])
    summary_path = os.path.join(out_dir, "{}_hparam_summary.csv".format(scenario))
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filter", "best_hparam", "mean_reward", "baseline_mean_reward", "pct_improvement",
                          "oracle_mean_reward", "oracle_pct_improvement", "oracle_gap"])
        writer.writerows(summary_rows)
    print("Saved {}".format(summary_path))

    # --- per-attack breakdown, using each filter's chosen (combined-best) hyperparameter ---
    by_attack_rows = []
    for label in ATTACK_LABELS:
        by_attack_rows.append(["no_filter", label, r2(_label_avg(baseline_points, label)), "", ""])
    for filt_name, (best_hparam, best_reward, best_points, oracle_mean) in filter_results.items():
        for label in ATTACK_LABELS:
            frew = _label_avg(best_points, label)
            brew = _label_avg(baseline_points, label)
            by_attack_rows.append([filt_name, label, r2(frew), r2(brew), r2(_pct_inc(frew, brew))])
    by_attack_path = os.path.join(out_dir, "{}_by_attack.csv".format(scenario))
    with open(by_attack_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filter", "attack", "mean_reward", "baseline_mean_reward", "pct_improvement"])
        writer.writerows(by_attack_rows)
    print("Saved {}".format(by_attack_path))

    return [[scenario] + row for row in summary_rows]


if __name__ == '__main__':
    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)

    scenarios = arglist.scenario_list if getattr(arglist, "scenario_list", None) else SCENARIOS
    n_episodes = arglist.num_test_episodes
    out_dir = os.path.join("..", "..", "noise_sweeps", "filters", "tuning")

    all_summary_rows = []
    for scenario in scenarios:
        all_summary_rows += tune_scenario(arglist, scenario, n_episodes, out_dir)

        # write/refresh the combined summary after every scenario so partial
        # progress is never lost on a long run
        combined_path = os.path.join(out_dir, "best_hyperparameters.csv")
        with open(combined_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "filter", "best_hparam", "mean_reward",
                              "baseline_mean_reward", "pct_improvement",
                              "oracle_mean_reward", "oracle_pct_improvement", "oracle_gap"])
            writer.writerows(all_summary_rows)
        print("Saved {}".format(combined_path))

    print("\nHyperparameter tuning complete.")
