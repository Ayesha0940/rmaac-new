"""
eval_decentral.py -- decentralized (per-agent) diffusion denoiser vs noisy policy, M3DDPG, all agents faulted and corrected.
Each agent is corrected by its own S1 per-agent model (own full local observation [+ one-hot ID]); Config B:
x_t = sqrt(abar_t) * normalized(clip(fault(a))), standard DDPM posterior chain, t in {10, 20, 40} x {one-step, chain}.
One job = (scenario, fault, level[, mu]).  --fault clean runs only the clean policy (for reward_no_noise).
Faults: gauss (mu = 0), biased (biased_gauss with --mu), delay, stuck. Reward = summed over agents (original plot convention).
"""
# bit-for-bit reproducible BLAS: must be set before numpy is imported anywhere
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
import eval_recon_s1 as T
import common as C
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack

T_GRID = [10, 20, 40]
COLUMNS = ["scenario", "fault", "level", "mu", "seed", "method", "mode", "t", "reward_mean", "reward_std",
           "action_mse_per_agent", "start_hash", "fault_state_hash", "n_episodes", "seconds", "model_sha"]


def variants(fault):
    if fault == "clean":
        return [dict(kind="clean")]
    V = [dict(kind="noisy")]
    for t in T_GRID:
        V.append(dict(kind="den", t=t, one=True, mode="1step"))
        V.append(dict(kind="den", t=t, one=False, mode="chain"))
    return V


def run_variant(arglist, scenario, v, seed, episodes, model_dir, S1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n = env.n
        corr = [T.Corr("den_s1", scenario, "full", i, seed, model_dir) for i in range(n)] if v["kind"] == "den" else None
        sha = ""
        if corr:
            import hashlib
            sha = "|".join(sorted(set(hashlib.sha256(open(C.model_path("den_s1", scenario, *reversed(C.agent_tag_and_info(scenario, i, "full")), seed), "rb").read()).hexdigest()[:16] for i in range(n))))
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        rng = np.random.RandomState(seed + 999)
        betas, alphas, abar = S1
        lo, hi = arglist.act_low, arglist.act_high
        ep_tot, mse, steps, start_hash = [], np.zeros(n), 0, 0
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n)
            ep_r = np.zeros(n)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean_n]
                if v["kind"] == "clean":
                    out = list(clean_n)
                else:
                    ahat = [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]
                    out = list(ahat)
                    if v["kind"] == "den":
                        for i, cr in enumerate(corr):
                            xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                            xt = (np.sqrt(abar[v["t"]]) * xn).astype(np.float32)
                            an = C.ddpm_chain(cr.net, xt, cr.cond(obs_n[i], None), v["t"], v["one"], rng, betas, alphas, abar)
                            out[i] = an * cr.as_ + cr.am
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n, dims)]
                for i in range(n):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                obs_n, rew_n, done_n, _ = env.step(action_n)
                if v["kind"] != "clean":
                    attack.step()
                ep_r += rew_n
                steps += 1
                if all(done_n):
                    break
            ep_tot.append(float(np.sum(ep_r)))
    return {"reward_mean": float(np.mean(ep_tot)), "reward_std": float(np.std(ep_tot)),
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps) for x in mse), "start_hash": start_hash,
            "fault_state_hash": fs.state_hash() if v["kind"] != "clean" else "", "n_episodes": episodes,
            "seconds": "{:.1f}".format(time.time() - t0), "model_sha": sha}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"])
    p.add_argument("--fault", required=True, choices=["clean", "gauss", "biased", "delay", "stuck"])
    p.add_argument("--flevel", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--model-dir", default=C.MODEL_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(E.HERE, a.model_dir)
    a.out = a.out if os.path.isabs(a.out) else os.path.join(E.HERE, a.out)
    sys.argv = [sys.argv[0], "--scenario", a.scenario, "--variant", "m3ddpg",
                "--exp-name", "{}__m3ddpgbest".format(a.scenario), "--save-dir", "./model", "--num-adversaries", "0"]
    arglist = apply_variant_defaults(parse_args())
    arglist.benchmark = False
    assert arglist.act_low == C.ACT_LOW and arglist.act_high == C.ACT_HIGH
    if a.fault in ("gauss", "clean"):
        arglist.attack, arglist.attack_mu, arglist.attack_sigma = "gauss", 0.0, a.flevel
    elif a.fault == "biased":
        arglist.attack, arglist.attack_mu, arglist.attack_sigma = "biased_gauss", a.mu, a.flevel
    elif a.fault == "delay":
        arglist.attack, arglist.delay_k = "delay", int(a.flevel)
    else:
        arglist.attack, arglist.sp_prob = "stuck_at", a.flevel
    b, al, ab = C.schedule("cosine")
    S1 = (b.numpy().astype(np.float64), al.numpy().astype(np.float64), ab.numpy().astype(np.float64))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for v in variants(a.fault):
            r = run_variant(arglist, a.scenario, v, a.seed, a.episodes, a.model_dir, S1)
            row = {"scenario": a.scenario, "fault": a.fault, "level": a.flevel, "mu": a.mu, "seed": a.seed,
                   "method": v["kind"], "mode": v.get("mode", ""), "t": v.get("t", "")}
            row.update(r)
            w.writerow(row); f.flush()
            print("[{} {} {} mu={} {} {}{}] reward={:.2f} ({}s)".format(a.scenario, a.fault, a.flevel, a.mu, v["kind"],
                  v.get("mode", ""), "@{}".format(v["t"]) if "t" in v else "", r["reward_mean"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
