"""
eval_state_only.py -- STATE-ONLY diffusion: the existing per-agent denoiser weights (conditional diffusion model of the clean
action given the agent's own observation, trained on clean actions only) sampled from PURE NOISE: x_99 ~ N(0, I), standard DDPM
posterior chain t = 99 ... 0 (S1 cosine), conditioned on the agent's own (full or reduced) observation + ID. The (faulted) action
is never used and no fault is applied, so the result is fault-independent.
  --mode full    : every agent uses its full-observation model
  --mode reduced : the reduced agent(s) use the reduced-observation model, every other agent its full-observation model
Also runs the clean policy as a reference. --probe-fault applies a fault stream anyway (smoke test: results must not change).
"""
import os
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)
import argparse, csv, hashlib, random, sys, time, zlib
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


def sha16(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def run(scen, kind, mode, seed, episodes, model_dir, arglist, S1, probe_fault):
    role, lvl = LEVEL[scen]
    red = ER.REDUCED[(scen, role, lvl)]
    tf.reset_default_graph(); tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n = env.n
        corr, shas = {}, set()
        if kind == "state_only":
            for i in range(n):
                tag, info_f = C.agent_tag_and_info(scen, i, "full")
                if mode == "reduced" and i in red:
                    key = "den_s1" + IL.SUFFIX[lvl]
                    corr[i] = EIL.ILCorr(key, scen, i, seed, model_dir)
                else:
                    key = "den_s1"
                    corr[i] = EIL.ILCorr(key, scen, i, seed, model_dir)
                shas.add(sha16(C.model_path(key, scen, info_f, tag, seed)))
        np.random.seed(seed); random.seed(seed)
        fs = E.FaultStream(ActionAttack(arglist), seed) if probe_fault else None
        attack = fs.attack if fs else None
        rng = np.random.RandomState(seed + 999)
        betas, alphas, abar = S1
        lo, hi = arglist.act_low, arglist.act_high
        tot, mse, cosv, acc, steps, sh = [], np.zeros(n), np.zeros(n), 0.0, 0, 0
        for ep in range(episodes):
            obs_n = env.reset(); sh = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), sh)
            if attack:
                attack.reset(n_agents=n)
            r = np.zeros(n)
            for step in range(arglist.max_episode_len):
                clean = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean]
                if fs:                                     # probe only: draw the fault, never use it
                    _ = [fs.perturb(a, i) for i, a in enumerate(clean)]
                if kind == "clean":
                    out = list(clean)
                else:
                    out = []
                    for i in range(n):
                        cr = corr[i]
                        x99 = rng.standard_normal(dims[i]).astype(np.float32)                 # pure noise x_99 ~ N(0, I)
                        an = C.ddpm_chain(cr.net, x99, cr.cond(obs_n[i]), 99, False, rng, betas, alphas, abar)
                        out.append(an * cr.as_ + cr.am)
                act = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n, dims)]
                for i in range(n):
                    mse[i] += np.mean((act[i] - clean[i]) ** 2)
                    cosv[i] += float(np.dot(act[i], clean[i]) / (np.linalg.norm(act[i]) * np.linalg.norm(clean[i]) + 1e-12))
                if scen == "simple_speaker_listener":
                    acc += float(np.argmax(act[0]) == np.argmax(clean[0]))
                obs_n, rew, done, _ = env.step(act)
                if attack:
                    attack.step()
                r += rew
                steps += 1
                if all(done):
                    break
            tot.append(float(np.sum(r)))
    return dict(reward_mean=float(np.mean(tot)), reward_std=float(np.std(tot)),
                action_mse_per_agent="|".join("{:.4f}".format(x / steps) for x in mse),
                action_cos_per_agent="|".join("{:.4f}".format(x / steps) for x in cosv),
                msg_acc="{:.4f}".format(acc / steps) if scen == "simple_speaker_listener" else "", start_hash=sh,
                seconds="{:.1f}".format(time.time() - t0), model_sha="|".join(sorted(shas)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--mode", required=True, choices=["full", "reduced"])
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-fault", default=None, help="smoke test: gauss3 | stuck1 (drawn but never used)")
    p.add_argument("--model-dir", default=C.MODEL_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(E.HERE, a.model_dir)
    sys.argv = [sys.argv[0], "--scenario", a.scenario, "--variant", "m3ddpg", "--exp-name", "{}__m3ddpgbest".format(a.scenario),
                "--save-dir", "./model", "--num-adversaries", "0"]
    arglist = apply_variant_defaults(parse_args()); arglist.benchmark = False
    if a.probe_fault == "gauss3":
        arglist.attack, arglist.attack_mu, arglist.attack_sigma = "gauss", 0.0, 3.0
    elif a.probe_fault == "stuck1":
        arglist.attack, arglist.sp_prob = "stuck_at", 1.0
    b, al, ab = C.schedule("cosine")
    S1 = (b.numpy().astype(np.float64), al.numpy().astype(np.float64), ab.numpy().astype(np.float64))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = ["scenario", "method", "mode", "seed", "reward_mean", "reward_std", "action_mse_per_agent", "action_cos_per_agent",
            "msg_acc", "n_episodes", "start_hash", "seconds", "model_sha"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for kind in ("clean", "state_only"):
            r = run(a.scenario, kind, a.mode, a.seed, a.episodes, a.model_dir, arglist, S1, a.probe_fault)
            row = dict(scenario=a.scenario, method=kind, mode=a.mode, seed=a.seed, n_episodes=a.episodes); row.update(r)
            w.writerow(row); f.flush()
            print("[{} {} {}] reward = {:.2f} (std {:.2f}) mse {} acc {}".format(a.scenario, kind, a.mode, r["reward_mean"], r["reward_std"],
                  r["action_mse_per_agent"], r["msg_acc"]), flush=True)


if __name__ == "__main__":
    main()
