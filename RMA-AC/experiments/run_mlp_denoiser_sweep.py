"""
run_mlp_denoiser_sweep.py
=========================
Evaluates the MLP joint-action denoiser (mlp_denoiser.py) on maddpg-none
checkpoints against a no-recovery baseline, on the low and high severity of
each attack. Kalman also runs live at sigma=1, the one grid point the existing
filter sweeps (noise_sweeps/filters/) don't have.

Writes one CSV per scenario; refuses to overwrite an existing one. DDPM/Kalman
results are joined in afterwards by noise_sweeps/plot_mlp_denoiser.py.

Usage:
    python run_mlp_denoiser_sweep.py --num-test-episodes 800 --seed 0 --save-dir ./model
"""
import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf

from train import parse_args, apply_variant_defaults, concat_actions, split_actions, r2
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack
from action_filters import ActionFilter
from mlp_denoiser import LoadedMLPDenoiser
from run_full_filter_sweeps import SCENARIOS, TUNED_HYPERPARAMS, _write_csv

HEADER = ["attack", "noise_mu", "severity", "level",
          "reward_no_recovery", "reward_mlp", "reward_kalman"]

# (attack, noise_mu, low, high)
ATTACK_GRID = [("delay", "", 1, 8),
               ("stuck_at", "", 0.1, 1.0),
               ("gauss", 0.0, 1.0, 3.0),
               ("biased_gauss", 1.0, 1.0, 3.0),
               ("biased_gauss", -1.0, 1.0, 3.0)]


def _set_attack(arglist, attack_name, mu, severity):
    arglist.attack = attack_name
    if attack_name == "delay":
        arglist.delay_k = severity
    elif attack_name == "stuck_at":
        arglist.sp_prob = severity
    else:
        arglist.attack_sigma = severity
        arglist.attack_mu = mu


def rollout_joint(env, trainers, arglist, attack, method, n_episodes, denoiser=None, filt=None):
    """Same episode loop as filter_eval.rollout_reward, but the recovery step
    sees the whole joint action and observation rather than one agent's action."""
    all_rewards = []
    for ep in range(n_episodes):
        obs_n = env.reset()
        episode_reward = np.zeros(env.n)
        attack.reset(n_agents=env.n)
        if filt is not None:
            filt.reset(n_agents=env.n)

        for step in range(arglist.max_episode_len):
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            n_agents = len(action_n)
            action_n_noisy = [attack.perturb(a, i) for i, a in enumerate(action_n)]

            if denoiser is not None:
                action_vec = denoiser.denoise(concat_actions(action_n_noisy), np.concatenate(obs_n, axis=0))
                action_n_final = split_actions(action_vec, n_agents, [len(a) for a in action_n])
                action_n_final = [np.clip(a, arglist.act_low, arglist.act_high) for a in action_n_final]
            elif filt is not None:
                action_n_final = [np.clip(filt.apply(a, i), arglist.act_low, arglist.act_high)
                                  for i, a in enumerate(action_n_noisy)]
            else:
                action_n_final = action_n_noisy

            new_obs_n, rew_n, done_n, _ = env.step(action_n_final)
            attack.step()
            episode_reward += rew_n
            obs_n = new_obs_n
            if all(done_n):
                break

        all_rewards.append(episode_reward)

    mean_total_reward = np.mean(np.sum(all_rewards, axis=1))
    print("  [{}] mean total reward over {} episodes: {:.3f}".format(method, n_episodes, mean_total_reward))
    return mean_total_reward


def run_scenario(arglist, scenario, n_episodes, model_dir, out_dir):
    arglist.scenario = scenario
    arglist.variant = "maddpg-none"
    arglist.exp_name = "{}__maddpgbest".format(scenario)
    arglist.num_adversaries = 0
    arglist.benchmark = False
    apply_variant_defaults(arglist)

    hp = TUNED_HYPERPARAMS[scenario]
    arglist.kalman_q = hp["kalman_q"]
    arglist.kalman_r = hp["kalman_r"]

    out_path = os.path.join(out_dir, "{}_mlp_sweep.csv".format(arglist.exp_name))
    if os.path.exists(out_path):
        print("{} already exists -- skipping {}.".format(out_path, scenario))
        return

    denoiser = LoadedMLPDenoiser(os.path.join(model_dir, "{}_m3ddpg.pt".format(scenario)))

    tf.reset_default_graph()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        attack = ActionAttack(arglist)
        kalman = ActionFilter(arglist, kind="kalman")

        rows = []
        for attack_name, mu, low, high in ATTACK_GRID:
            for level, severity in (("low", low), ("high", high)):
                print("\n=== {}: {} mu={} severity={} ({}) ===".format(scenario, attack_name, mu, severity, level))
                _set_attack(arglist, attack_name, mu, severity)
                base = rollout_joint(env, trainers, arglist, attack, "no_recovery", n_episodes)
                mlp = rollout_joint(env, trainers, arglist, attack, "mlp", n_episodes, denoiser=denoiser)
                kal = ""
                if mu != "" and severity == 1.0:
                    kal = r2(rollout_joint(env, trainers, arglist, attack, "kalman", n_episodes, filt=kalman))
                rows.append([attack_name, mu, severity, level, r2(base), r2(mlp), kal])

    _write_csv(out_path, HEADER, rows)


if __name__ == '__main__':
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mlp-model-dir", type=str, default="./mlp_denoiser_models")
    pre.add_argument("--mlp-out-root", type=str, default=os.path.join("..", "..", "noise_sweeps", "mlp_denoiser"))
    pre.add_argument("--mlp-scenarios", type=str, nargs="+", default=SCENARIOS)
    pre_args, rest = pre.parse_known_args()
    sys.argv = sys.argv[:1] + rest  # train.parse_args rejects unknown flags

    arglist = parse_args()
    arglist = apply_variant_defaults(arglist)

    seed = arglist.seed if arglist.seed is not None else 0
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    out_dir = os.path.join(pre_args.mlp_out_root, "seed{}".format(seed))
    for scenario in pre_args.mlp_scenarios:
        run_scenario(arglist, scenario, arglist.num_test_episodes, pre_args.mlp_model_dir, out_dir)

    print("\nMLP denoiser sweep complete.")
