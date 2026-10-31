"""
eval_decentral_c1.py -- decentralized protocol (all agents faulted and corrected), supervised reconstructor C1.
C1 = MatchedNet with the action input, trained on the S1 forward-diffusion corruption, predicts the normalized clean action.
Inference "unscaled": x = (clip(fault(a)) - act_mean) / act_std fed directly, one pass, un-normalised, clipped.
One job = (scenario, fault cell); variants:
  noisy   : no correction (regression check against noise_sweeps/decentralized)
  c1_full : every agent corrected by its full-local-observation C1
  c1_red  : the reduced agent(s) (REDUCED map) corrected by the reduced-observation C1, every other agent by its full C1
Reward summed over agents. Models from --model-dir (default models/c1_decentral).
"""
import os
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)
import argparse
import csv
import random
import sys
import time
import zlib

import numpy as np

import eval_recon as E
import eval_il as EIL
import eval_decentral_reduced as ER
import common as C
import info_levels as IL
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack

LEVEL = {"simple_speaker_listener": ("speaker", "removed"), "simple_adversary": ("good", "removed"),
         "simple_push": ("good", "goal-hidden"), "simple_spread": ("shared", "others-removed")}
COLUMNS = ["scenario", "role", "info_level", "fault", "level", "mu", "seed", "method", "reward_mean", "reward_std",
           "action_mse_per_agent", "msg_acc", "start_hash", "fault_state_hash", "n_episodes", "seconds", "model_sha"]


def run_variant(arglist, scen, kind, seed, episodes, model_dir):
    role, lvl = LEVEL[scen]
    red = ER.REDUCED[(scen, role, lvl)]
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n = env.n
        corr = {}
        if kind != "noisy":
            for i in range(n):
                key = "c1_s1" + (IL.SUFFIX[lvl] if (kind == "c1_red" and i in red) else "")
                corr[i] = EIL.ILCorr(key, scen, i, seed, model_dir)
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        lo, hi = arglist.act_low, arglist.act_high
        ep_tot, mse, acc, steps, start_hash = [], np.zeros(n), 0.0, 0, 0
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n)
            ep_r = np.zeros(n)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean_n]
                ahat = [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]
                out = list(ahat)
                if kind != "noisy":
                    for i in range(n):
                        cr = corr[i]
                        xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)               # unscaled input
                        out[i] = cr.net.direct(cr.cond(obs_n[i]), xn) * cr.as_ + cr.am
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n, dims)]
                for i in range(n):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                if scen == "simple_speaker_listener":
                    acc += float(np.argmax(action_n[0]) == np.argmax(clean_n[0]))
                obs_n, rew_n, done_n, _ = env.step(action_n)
                attack.step()
                ep_r += rew_n
                steps += 1
                if all(done_n):
                    break
            ep_tot.append(float(np.sum(ep_r)))
    return {"reward_mean": float(np.mean(ep_tot)), "reward_std": float(np.std(ep_tot)),
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps) for x in mse),
            "msg_acc": "{:.4f}".format(acc / steps) if scen == "simple_speaker_listener" else "",
            "start_hash": start_hash, "fault_state_hash": fs.state_hash(), "n_episodes": episodes,
            "seconds": "{:.1f}".format(time.time() - t0), "model_sha": "|".join(sorted(set(c.sha for c in corr.values())))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=sorted(LEVEL))
    p.add_argument("--fault", required=True, choices=["gauss", "biased", "delay", "stuck"])
    p.add_argument("--flevel", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--only", nargs="+", default=["noisy", "c1_full", "c1_red"])
    p.add_argument("--model-dir", default=os.path.join(C.MODEL_DIR, "c1_decentral"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(E.HERE, a.model_dir)
    a.out = a.out if os.path.isabs(a.out) else os.path.join(E.HERE, a.out)
    sys.argv = [sys.argv[0], "--scenario", a.scenario, "--variant", "m3ddpg",
                "--exp-name", "{}__m3ddpgbest".format(a.scenario), "--save-dir", "./model", "--num-adversaries", "0"]
    arglist = apply_variant_defaults(parse_args())
    arglist.benchmark = False
    assert arglist.act_low == C.ACT_LOW and arglist.act_high == C.ACT_HIGH
    if a.fault == "gauss":
        arglist.attack, arglist.attack_mu, arglist.attack_sigma = "gauss", 0.0, a.flevel
    elif a.fault == "biased":
        arglist.attack, arglist.attack_mu, arglist.attack_sigma = "biased_gauss", a.mu, a.flevel
    elif a.fault == "delay":
        arglist.attack, arglist.delay_k = "delay", int(a.flevel)
    else:
        arglist.attack, arglist.sp_prob = "stuck_at", a.flevel
    role, lvl = LEVEL[a.scenario]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for kind in a.only:
            r = run_variant(arglist, a.scenario, kind, a.seed, a.episodes, a.model_dir)
            row = {"scenario": a.scenario, "role": role, "info_level": lvl, "fault": a.fault, "level": a.flevel, "mu": a.mu,
                   "seed": a.seed, "method": kind}
            row.update(r)
            w.writerow(row); f.flush()
            print("[{} {} {} mu={} {}] reward={:.2f} mse={} ({}s)".format(a.scenario, a.fault, a.flevel, a.mu, kind,
                  r["reward_mean"], r["action_mse_per_agent"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
