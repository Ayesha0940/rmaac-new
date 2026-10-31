"""
eval_decentral_reduced.py -- decentralized protocol (all agents faulted), with the REDUCED-observation agent(s) using a
reduced-input model. One job = (scenario, reduced level, fault cell). Variants (t in {10,20,40} x {one-step, chain}):
  den_red : reduced agent(s) corrected by the reduced-input S1 denoiser; every other agent by its full S1 denoiser
  bc_red  : reduced agent(s) act with BC on the same reduced input (their fault is drawn but not applied);
            every other agent faulted and corrected by its full S1 denoiser with the same (t, mode)
  noisy   : optional (--with-noisy), for the regression check against the decentralized run
Reward summed over agents; good-agent reward also logged (agents >= 1 for simple_adversary / simple_push).
"""
import os
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)
import argparse
import csv
import hashlib
import random
import sys
import time
import zlib

import numpy as np

import eval_recon as E
import eval_recon_s1 as T
import eval_il as EIL
import common as C
import info_levels as IL
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack

T_GRID = [10, 20, 40]
REDUCED = {("simple_speaker_listener", "speaker", "removed"): [0], ("simple_speaker_listener", "listener", "removed"): [1],
           ("simple_adversary", "good", "removed"): [1, 2], ("simple_push", "good", "goal-hidden"): [1],
           ("simple_spread", "shared", "others-removed"): [0, 1, 2]}
COLUMNS = ["scenario", "role", "info_level", "fault", "level", "mu", "seed", "method", "mode", "t", "reward_mean", "reward_std",
           "reward_good_mean", "reward_good_std", "action_mse_per_agent", "start_hash", "fault_state_hash", "n_episodes",
           "seconds", "model_sha"]


def sha16(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def run_variant(arglist, scen, role, lvl, red, v, seed, episodes, model_dir, S1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n = env.n
        corr, shas = {}, set()
        if v["kind"] in ("den_red", "bc_red"):
            for i in range(n):
                if i in red:
                    key = ("den_s1" if v["kind"] == "den_red" else "bc") + IL.SUFFIX[lvl]
                    corr[i] = EIL.ILCorr(key, scen, i, seed, model_dir)
                    tag, info_f = C.agent_tag_and_info(scen, i, "full")
                    shas.add(sha16(C.model_path(key, scen, info_f, tag, seed)))
                else:
                    corr[i] = T.Corr("den_s1", scen, "full", i, seed, model_dir)
                    tag, info_f = C.agent_tag_and_info(scen, i, "full")
                    shas.add(sha16(C.model_path("den_s1", scen, info_f, tag, seed)))
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        rng = np.random.RandomState(seed + 999)
        betas, alphas, abar = S1
        lo, hi = arglist.act_low, arglist.act_high
        ep_tot, ep_good, mse, steps, start_hash = [], [], np.zeros(n), 0, 0
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n)
            ep_r = np.zeros(n)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean_n]
                ahat = [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]   # drawn for every agent (stream parity)
                out = list(ahat)
                if v["kind"] in ("den_red", "bc_red"):
                    for i in range(n):
                        cr = corr[i]
                        if v["kind"] == "bc_red" and i in red:
                            out[i] = cr.net.direct(cr.cond(obs_n[i])) * cr.as_ + cr.am       # BC: no fault applied
                            continue
                        cond = cr.cond(obs_n[i]) if i in red else cr.cond(obs_n[i], None)
                        xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                        xt = (np.sqrt(abar[v["t"]]) * xn).astype(np.float32)
                        an = C.ddpm_chain(cr.net, xt, cond, v["t"], v["one"], rng, betas, alphas, abar)
                        out[i] = an * cr.as_ + cr.am
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n, dims)]
                for i in range(n):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                obs_n, rew_n, done_n, _ = env.step(action_n)
                attack.step()
                ep_r += rew_n
                steps += 1
                if all(done_n):
                    break
            ep_tot.append(float(np.sum(ep_r)))
            ep_good.append(float(np.mean(ep_r[1:])))
    good = scen in ("simple_adversary", "simple_push")
    return {"reward_mean": float(np.mean(ep_tot)), "reward_std": float(np.std(ep_tot)),
            "reward_good_mean": float(np.mean(ep_good)) if good else "", "reward_good_std": float(np.std(ep_good)) if good else "",
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps) for x in mse), "start_hash": start_hash,
            "fault_state_hash": fs.state_hash(), "n_episodes": episodes, "seconds": "{:.1f}".format(time.time() - t0),
            "model_sha": "|".join(sorted(shas))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--info-level", required=True)
    p.add_argument("--fault", required=True, choices=["gauss", "biased", "delay", "stuck"])
    p.add_argument("--flevel", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--with-noisy", action="store_true")
    p.add_argument("--only", nargs="+", default=None, help="subset of den_red bc_red noisy")
    p.add_argument("--model-dir", default=C.MODEL_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(E.HERE, a.model_dir)
    a.out = a.out if os.path.isabs(a.out) else os.path.join(E.HERE, a.out)
    red = REDUCED[(a.scenario, a.role, a.info_level)]
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
    b, al, ab = C.schedule("cosine")
    S1 = (b.numpy().astype(np.float64), al.numpy().astype(np.float64), ab.numpy().astype(np.float64))
    V = ([dict(kind="noisy")] if a.with_noisy else [])
    for kind in ("den_red", "bc_red"):
        for t in T_GRID:
            V.append(dict(kind=kind, t=t, one=True, mode="1step"))
            V.append(dict(kind=kind, t=t, one=False, mode="chain"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for v in V:
            if a.only and v["kind"] not in a.only:
                continue
            r = run_variant(arglist, a.scenario, a.role, a.info_level, red, v, a.seed, a.episodes, a.model_dir, S1)
            row = {"scenario": a.scenario, "role": a.role, "info_level": a.info_level, "fault": a.fault, "level": a.flevel,
                   "mu": a.mu, "seed": a.seed, "method": v["kind"], "mode": v.get("mode", ""), "t": v.get("t", "")}
            row.update(r)
            w.writerow(row); f.flush()
            print("[{} {} {} {} {} mu={} {} {}{}] reward={:.2f} ({}s)".format(a.scenario, a.role, a.info_level, a.fault, a.flevel, a.mu,
                  v["kind"], v.get("mode", ""), "@{}".format(v["t"]) if "t" in v else "", r["reward_mean"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
