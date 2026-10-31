"""
run_bc_sweep.py
===============
Test the state-only BC baseline (see train_bc.py) and the clean base policies.

BC replaces the whole policy+denoiser pipeline. At every env step:
    state_vec = concat(obs_n)            (current joint observation)
    a_vec     = BC(state_vec)            (un-normalized with act_mean/act_std)
    action_n  = split_actions(a_vec, per-agent dims taken from the policy's own action lengths)
    action_n  = [np.clip(a, act_low, act_high)]   <- identical to the denoiser path
    env.step(action_n)
concat_actions / split_actions are imported from train.py. NO attack is applied
(faults sit upstream of the denoiser, hence upstream of BC), so BC's result is
independent of the perturbation level. It is therefore rolled out ONCE per seed
and the same value is written at every level of every sweep family, with the
same level grids as the existing sweeps, so it plots as a flat line.

Also measures each base policy's clean (no-attack) reward for the 4 variants
with the same seed, for the BC-vs-policy table.

Output (noise_sweeps/bc/):
    clean/seed{N}/<scenario>_clean.csv                  (BC + 4 policies, clean)
    {delay,actuation_fault,noise_mu-1.0,noise_mu0.0,noise_mu1.0}/seed{N}/<scenario>_bc_sweep.csv

Usage:
    python run_bc_sweep.py --scenario simple_spread --num-test-episodes 800 --seed 47
"""
import csv
import os
import random

import numpy as np
import tensorflow as tf
import torch

from train import (parse_args, apply_variant_defaults, r2,
                   concat_actions, split_actions)
import maddpg.common.tf_util as U
from filter_eval import build_test_env, rollout_reward
from action_attacks import ActionAttack
from train_bc import load_bc_model

HERE = os.path.dirname(os.path.abspath(__file__))

NUM_ADV = {"simple_adversary": 0, "simple_push": 0, "simple_speaker_listener": 0,
           "simple_spread": 0, "simple_tag": 3}

ALGORITHMS = [("maddpg-none", "maddpg"), ("maddpg-earnie", "earnie"),
              ("maddpg-act_adv", "rmaac"), ("m3ddpg", "m3ddpg")]

# Level grids: union of those used by the existing sweeps (incl. predator-prey oracle).
SWEEPS = [
    ("delay",           "delay_k",          [0, 1, 2, 3, 5, 8]),
    ("actuation_fault", "sp_prob",          [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]),
    ("noise_mu-1.0",    "action_noise_std", [0, 1, 1.5, 2, 3]),
    ("noise_mu0.0",     "action_noise_std", [0, 1, 1.5, 2, 3]),
    ("noise_mu1.0",     "action_noise_std", [0, 1, 1.5, 2, 3]),
]


def bc_rollout(env, trainers, arglist, bc_model, act_mean, act_std, num_adversaries, n_episodes):
    """Same loop as filter_eval.rollout_reward, with the policy+attack+denoiser
    replaced by BC(state). Returns mean total reward, or with arglist.benchmark
    (pred, prey, captures, survival)."""
    all_rewards, all_captures, all_survivals = [], [], []
    for ep in range(n_episodes):
        obs_n = env.reset()
        episode_reward = np.zeros(env.n)
        capture_count, survival_step = 0, None
        for step in range(arglist.max_episode_len):
            # per-agent action dims come from the policy's own action lengths,
            # exactly as in testRobustnessAP
            action_dim_per_agent = [len(agent.action(obs)) for agent, obs in zip(trainers, obs_n)]
            n_agents = len(action_dim_per_agent)
            state_vec = np.concatenate(obs_n, axis=0)
            with torch.no_grad():
                a = bc_model(torch.from_numpy(state_vec).float().unsqueeze(0))
                a_vec = (a * act_std + act_mean)[0].numpy()
            assert a_vec.shape[0] == sum(action_dim_per_agent)
            action_n = split_actions(a_vec, n_agents, action_dim_per_agent)
            action_n = [np.clip(x, arglist.act_low, arglist.act_high) for x in action_n]

            obs_n, rew_n, done_n, info_n = env.step(action_n)
            if arglist.benchmark:
                sc = sum(info_n['n'][:num_adversaries])
                capture_count += sc
                if survival_step is None and sc > 0:
                    survival_step = step
            episode_reward += rew_n
            if all(done_n):
                break
        all_rewards.append(episode_reward)
        if arglist.benchmark:
            all_captures.append(capture_count)
            all_survivals.append(step if survival_step is None else survival_step)

    total = np.mean(np.sum(all_rewards, axis=1))
    print("  [bc] mean total reward over {} episodes: {:.3f}".format(n_episodes, total))
    if arglist.benchmark:
        return (np.mean([np.sum(r[:num_adversaries]) for r in all_rewards]),
                np.mean([np.sum(r[num_adversaries:]) for r in all_rewards]),
                np.mean(all_captures), np.mean(all_survivals))
    return total


def _seed(seed):
    np.random.seed(seed); random.seed(seed); tf.set_random_seed(seed)


def _as_tuple(res, benchmark):
    return tuple(res) if benchmark else (res,)


def _cols(prefix, benchmark):
    if benchmark:
        return ["{}_{}".format(m, prefix) for m in ("reward_pred", "reward_prey", "captures", "survival")]
    return ["reward_{}".format(prefix)]


if __name__ == "__main__":
    arglist = parse_args()
    scenario = arglist.scenario
    seed = arglist.seed if arglist.seed is not None else 0
    n_episodes = arglist.num_test_episodes
    arglist.num_adversaries = NUM_ADV[scenario]
    arglist.benchmark = (scenario == "simple_tag")
    bench = arglist.benchmark
    out_root = os.environ.get("BC_OUT_ROOT") or os.path.join(HERE, "..", "..", "noise_sweeps", "bc")

    bc_path = os.path.join(os.environ.get("BC_MODEL_DIR") or os.path.join(HERE, "bc_models"),
                           "{}_m3ddpg_bc.pt".format(scenario))
    bc_model, act_mean, act_std = load_bc_model(bc_path)

    results = {}   # label -> tuple of metrics
    for variant, suffix in ALGORITHMS:
        arglist.variant = variant
        arglist.exp_name = "{}__{}best".format(scenario, suffix)
        apply_variant_defaults(arglist)
        arglist.attack = "none"
        print("\n=== {} | seed {} ===".format(arglist.exp_name, seed))
        tf.reset_default_graph()
        with U.single_threaded_session():
            env, trainers, num_adv = build_test_env(arglist)
            _seed(seed)
            attack = ActionAttack(arglist)
            results["policy_" + suffix] = _as_tuple(
                rollout_reward(env, trainers, arglist, attack, "clean_" + suffix, None,
                               "all", num_adv, n_episodes), bench)
            if suffix == "m3ddpg":
                # BC is policy-independent; trainers are used only for per-agent action dims.
                _seed(seed)
                results["bc"] = _as_tuple(
                    bc_rollout(env, trainers, arglist, bc_model, act_mean, act_std,
                               num_adv, n_episodes), bench)

    # --- clean table ---
    path = os.path.join(out_root, "clean", "seed{}".format(seed), "{}_clean.csv".format(scenario))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = ["bc"] + ["policy_" + s for _, s in ALGORITHMS]
    header = []
    for lab in labels:
        header += _cols(lab, bench)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([r2(v) for lab in labels for v in results[lab]])
    print("Saved {}".format(path))

    # --- flat per-level CSVs: BC is rolled out once, value replicated at each level ---
    bc_vals = [r2(v) for v in results["bc"]]
    for subdir, sev_name, levels in SWEEPS:
        p = os.path.join(out_root, subdir, "seed{}".format(seed), "{}_bc_sweep.csv".format(scenario))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([sev_name] + _cols("bc", bench))
            for lv in levels:
                w.writerow([lv] + bc_vals)
        print("Saved {}".format(p))
