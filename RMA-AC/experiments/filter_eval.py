"""
filter_eval.py
==============
Standalone evaluation of classical action-recovery filters (EMA, median,
Kalman — see `action_filters.py`) against the same noise / delay /
actuator-fault attacks already implemented in `action_attacks.py`.

Deliberately kept separate from train.py's DDPM-diffusion sweep machinery
(`testRobustnessAP` + the `__main__` dispatch in train.py). Reuses train.py's
general-purpose helpers (arg parsing, env/trainer construction, checkpoint
loading) rather than duplicating them, since MADDPG/M3DDPG trainer
construction needs the full hyperparameter surface train.py's parser already
defines.

Usage (mirrors train.py's test-mode sweep scripts):
    python filter_eval.py --mode test --scenario simple_tag --num-adversaries 3 \\
        --variant maddpg-act_adv --exp-name simple_tag__rmaacbest \\
        --save-dir ./model --attack delay --delay-k-list 1 2 3 5 8 \\
        --num-test-episodes 800 --seed 5 [--benchmark]
"""

import csv
import os
import random
import time

import numpy as np
import tensorflow as tf

# train.py appends the maddpg package's parent dirs to sys.path at import
# time, so it must be imported before `maddpg.common.tf_util`.
from train import (parse_args, apply_variant_defaults, make_env, get_trainers,
                    resolve_checkpoint, r2, r2ms)
import maddpg.common.tf_util as U
from action_attacks import ActionAttack
from action_filters import ActionFilter

FILTER_KINDS = ["ema", "median", "kalman"]


def build_test_env(arglist):
    """Construct env + trainers once and load the checkpoint. Must be called
    inside an open `U.single_threaded_session()` (after `tf.reset_default_graph()`).
    Returns (env, trainers, num_adversaries).
    """
    env = make_env(arglist.scenario, arglist, arglist.benchmark)

    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

    print('Testing using good policy {} and adv policy {}'.format(
        arglist.good_policy, arglist.adv_policy))

    U.initialize()

    load_dir, exp_name = resolve_checkpoint(arglist)
    print('Loading trained model from {}'.format(load_dir))
    U.load_state(load_dir, exp_name=exp_name)

    return env, trainers, num_adversaries


def rollout_reward(env, trainers, arglist, attack, method, filt, attack_target,
                    num_adversaries, n_episodes):
    """Run `n_episodes` of the already-loaded policy under `attack`, applying
    `filt` (an ActionFilter, or None for the "no_filter" baseline) to the
    attacked agents' actions. Returns `mean_total_reward` (float), or with
    `arglist.benchmark` a `(pred_reward, prey_reward, mean_captures,
    mean_survival)` tuple. Does not touch the TF graph/session/checkpoint —
    callers that need to sweep many configs against the same policy should
    build the env/trainers once (see `build_test_env`) and call this
    repeatedly.
    """
    max_episode_len = arglist.max_episode_len
    all_rewards = []
    all_captures = []
    all_survivals = []

    for ep in range(n_episodes):
        obs_n = env.reset()
        episode_reward = np.zeros(env.n)
        attack.reset(n_agents=env.n)
        if filt is not None:
            filt.reset(n_agents=env.n)
        capture_count = 0
        survival_step = None

        for step in range(max_episode_len):
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            n_agents = len(action_n)

            if attack_target == "predator":
                attacked_idx = range(num_adversaries)
            elif attack_target == "prey":
                attacked_idx = range(num_adversaries, n_agents)
            else:
                attacked_idx = range(n_agents)

            action_n_noisy = [attack.perturb(a, i) if i in attacked_idx else a
                              for i, a in enumerate(action_n)]

            if filt is None:
                action_n_final = action_n_noisy
            else:
                action_n_final = [
                    np.clip(filt.apply(a, i), arglist.act_low, arglist.act_high)
                    if i in attacked_idx else a
                    for i, a in enumerate(action_n_noisy)
                ]

            new_obs_n, rew_n, done_n, info_n = env.step(action_n_final)
            attack.step()

            if arglist.benchmark:
                step_collisions = sum(info_n['n'][:num_adversaries])
                capture_count += step_collisions
                if survival_step is None and step_collisions > 0:
                    survival_step = step

            episode_reward += rew_n
            obs_n = new_obs_n

            if arglist.display:
                env.render()
                time.sleep(0.05)

            if all(done_n):
                break

        all_rewards.append(episode_reward)
        if arglist.benchmark:
            if survival_step is None:
                survival_step = step
            all_captures.append(capture_count)
            all_survivals.append(survival_step)

    mean_total_reward = np.mean(np.sum(all_rewards, axis=1))
    print("  [{}] mean total reward over {} episodes: {:.3f}".format(
        method, n_episodes, mean_total_reward))

    if arglist.benchmark:
        pred_reward = np.mean([np.sum(r[:num_adversaries]) for r in all_rewards])
        prey_reward = np.mean([np.sum(r[num_adversaries:]) for r in all_rewards])
        mean_captures = np.mean(all_captures)
        mean_survival = np.mean(all_survivals)
        return (pred_reward, prey_reward, mean_captures, mean_survival)
    return mean_total_reward


def test_with_filters(arglist, filter_list, attack_target="all"):
    """Roll out the trained policy under the configured attack, evaluating the
    no-recovery baseline plus one pass per filter in `filter_list`.

    Returns a dict: {"no_filter": stats, "ema": stats, "median": stats, "kalman": stats}
    (only keys in filter_list plus "no_filter" are present), where `stats` is
    either `mean_total_reward` (float) or, with `arglist.benchmark`, a
    `(pred_reward, prey_reward, mean_captures, mean_survival)` tuple.
    """
    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, num_adversaries = build_test_env(arglist)

        n_episodes = arglist.num_test_episodes
        attack = ActionAttack(arglist)
        filters = {kind: ActionFilter(arglist, kind=kind) for kind in filter_list}

        results = {}
        for method in ["no_filter"] + list(filter_list):
            filt = None if method == "no_filter" else filters[method]
            results[method] = rollout_reward(env, trainers, arglist, attack, method, filt,
                                              attack_target, num_adversaries, n_episodes)
        return results


def _row_for(severity, results, filter_list, benchmark):
    if benchmark:
        row = []
        for method in ["no_filter"] + list(filter_list):
            pred, prey, cap, surv = results[method]
            row += [r2(pred), r2(prey), r2(cap), r2(surv)]
        return [severity] + row
    else:
        row = [r2(results[m]) for m in ["no_filter"] + list(filter_list)]
        return [severity] + row


def _header_for(severity_name, filter_list, benchmark):
    if benchmark:
        header = [severity_name]
        for method in ["no_filter"] + list(filter_list):
            header += ["reward_pred_{}".format(method), "reward_prey_{}".format(method),
                       "captures_{}".format(method), "survival_{}".format(method)]
        return header
    else:
        return [severity_name] + ["reward_{}".format(m) for m in ["no_filter"] + list(filter_list)]


if __name__ == '__main__':
    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)

    if arglist.seed is not None:
        np.random.seed(arglist.seed)
        random.seed(arglist.seed)
        tf.set_random_seed(arglist.seed)

    filter_list = arglist.filter_list
    attack_target_suffix = ("" if arglist.attack_target == "all"
                             else "_{}only".format(arglist.attack_target))

    attack_kind = getattr(arglist, "attack", "gauss")

    if attack_kind == "delay":
        severity_list = arglist.delay_k_list
        severity_name = "delay_k"
        csv_filename = "{}_delay_filter_sweep{}.csv".format(arglist.exp_name, attack_target_suffix)

        results_rows = []
        for k in severity_list:
            arglist.delay_k = k
            print("\n=== [filters] Delay k = {} ===".format(k))
            results = test_with_filters(arglist, filter_list, attack_target=arglist.attack_target)
            results_rows.append(_row_for(k, results, filter_list, arglist.benchmark))

    elif attack_kind == "stuck_at":
        severity_list = arglist.sp_prob_list
        severity_name = "sp_prob"
        csv_filename = "{}_stuck_at_filter_sweep{}.csv".format(arglist.exp_name, attack_target_suffix)

        results_rows = []
        for p in severity_list:
            arglist.sp_prob = p
            print("\n=== [filters] sp_prob = {} ===".format(p))
            results = test_with_filters(arglist, filter_list, attack_target=arglist.attack_target)
            results_rows.append(_row_for(p, results, filter_list, arglist.benchmark))

    else:
        # gauss / biased_gauss action-noise-std sweep (default)
        severity_list = arglist.act_std_list
        severity_name = "action_noise_std"
        csv_filename = "{}_mu{}_actstd_filter_sweep{}.csv".format(
            arglist.exp_name, arglist.noise_mu, attack_target_suffix)

        results_rows = []
        for act_std in severity_list:
            arglist.attack_sigma = act_std
            arglist.attack_mu = arglist.noise_mu
            print("\n=== [filters] action noise std = {} ===".format(act_std))
            results = test_with_filters(arglist, filter_list, attack_target=arglist.attack_target)
            results_rows.append(_row_for(act_std, results, filter_list, arglist.benchmark))

    header = _header_for(severity_name, filter_list, arglist.benchmark)

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results_rows)

    print("\nSaved filter sweep results to {}".format(csv_filename))
