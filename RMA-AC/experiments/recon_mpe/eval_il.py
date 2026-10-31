"""
eval_il.py -- information-level sweep, one job = (task, varied role, information level, fault cell).

Protocol (CORRECTED agents = info_levels.CORRECTED[(task, varied)]):
  speaker varied  : only agent 0 is faulted and corrected, the listener acts on the clean policy
  listener varied : only agent 1 is faulted and corrected, the speaker acts on the clean policy (message is clean)
  good (adversary): agents 1, 2 faulted and corrected, the adversary acts clean; metric = good-agent reward
  shared (spread) : all three agents faulted and corrected
Runs: clean, noisy, [bc (only with --with-bc)], and denoiser variants t in {5,10,20,30,40,60,80,99} x {1step, chain}
with scaled input x_t = sqrt(abar_t) * normalized(clip(fault(a))) on the S1 cosine schedule (standard DDPM posterior).
Same seeding / fault stream / projection as eval_pa.py.
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
import hashlib

import numpy as np

import eval_recon as E
import common as C
import info_levels as IL
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack

T_GRID = [5, 10, 20, 30, 40, 60, 80, 99]
COLUMNS = ["phase", "scenario", "varied", "info_level", "method", "mode", "t", "fault", "level", "seed",
           "reward_mean", "reward_std", "reward_adv_mean", "reward_adv_std", "reward_good_mean", "reward_good_std",
           "action_mse_per_agent", "action_cos_per_agent", "msg_acc", "start_hash", "fault_state_hash",
           "n_episodes", "seconds", "model_sha"]
FAULT_ATTACK = {"gauss": "gauss", "biased": "biased_gauss", "delay": "delay", "stuck": "stuck_at"}
FAULT_NAME = {"gauss": "gauss", "biased": "biased_gauss_mu+1", "delay": "delay", "stuck": "stuck"}


def variants(with_bc, step2=False, confirm=False, shift=False):
    V = [dict(name="clean", kind="clean"), dict(name="noisy", kind="noisy")]
    if with_bc:
        V.append(dict(name="bc", kind="bc"))
    for t in T_GRID:
        V.append(dict(name="den", kind="den", t=t, one=True, mode="1step"))
        V.append(dict(name="den", kind="den", t=t, one=False, mode="chain"))
    if step2:
        # B_mean: same denoiser models / variants, noisy action replaced by the dataset mean (normalized 0 -> x_t = 0)
        for t in T_GRID:
            V.append(dict(name="bmean", kind="bmean", t=t, one=True, mode="1step"))
            V.append(dict(name="bmean", kind="bmean", t=t, one=False, mode="chain"))
        # C1: unscaled normalized noisy action, and scaled by sqrt(abar_t) for each t of the grid
        V.append(dict(name="c1", kind="c1", mode="unscaled"))
        for t in T_GRID:
            V.append(dict(name="c1", kind="c1", t=t, mode="scaled"))
    if confirm:
        # C1-linear: C1 trained on the linear S0 forward process; unscaled, and scaled by sqrt(abar_t) of S0
        V.append(dict(name="c1lin", kind="c1lin", mode="unscaled"))
        for t in T_GRID:
            V.append(dict(name="c1lin", kind="c1lin", t=t, mode="scaled"))
    if shift:
        # C1-gauss: reconstructor trained ONLY on the Gaussian fault model clip(a + sigma*eps), sigma ~ U(0, 3) (method c2)
        V.append(dict(name="c1g", kind="c1g", mode="unscaled"))
        for t in T_GRID:
            V.append(dict(name="c1g", kind="c1g", t=t, mode="scaled"))
    # random policy for the corrected agents: uniform over the projection bounds [-1, 1] (opt-in via --only random)
    V.append(dict(name="random", kind="random"))
    return V


class ILCorr:
    def __init__(self, key, scenario, agent, seed, model_dir):
        tag, info_f = C.agent_tag_and_info(scenario, agent, "full")
        C.MODEL_DIR = model_dir
        path = C.model_path(key, scenario, info_f, tag, seed)
        model, ck = C.load_model(path)
        with open(path, "rb") as fh:
            self.sha = hashlib.sha256(fh.read()).hexdigest()[:16]
        self.net, self.ck, self.agent = C.NumpyNet(model), ck, agent
        n = ck["norm"][agent]
        self.om, self.os, self.am, self.as_ = n["obs_mean"], n["obs_std"], n["act_mean"][0], n["act_std"][0]
        self.layout = ck.get("level_layout") if ck.get("level", "full") != "full" else None
        self.spec = ck["cond_spec"]

    def cond(self, obs):
        if self.layout is not None:
            return IL.build_cond_level(self.layout, obs[None], self.om, self.os, self.agent)[0]
        return C.build_cond(self.spec, ((obs[None] - self.om) / self.os), self.agent)[0]


def model_key(method, level):
    return (method if method == "bc" else method + "_s1") + IL.SUFFIX[level]


def run_variant(arglist, task, varied, level, v, seed, episodes, model_dir, S1, S0=None):
    CS = IL.CORRECTED[(task, varied)]
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n_agents = env.n
        corr = {}
        if v["kind"] in ("den", "bc", "bmean", "c1", "c1lin", "c1g"):
            key = model_key({"bc": "bc", "c1": "c1"}.get(v["kind"], "den"), level)
            if v["kind"] == "c1lin":
                key = "c1" + IL.SUFFIX[level]                          # S0-trained C1 (no _s1)
            if v["kind"] == "c1g":
                key = "c2" + IL.SUFFIX[level]                          # Gaussian-fault-trained reconstructor
            corr = dict((i, ILCorr(key, task, i, seed, model_dir)) for i in CS)
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        rng = np.random.RandomState(seed + 999)
        rand_rs = np.random.RandomState(seed + 555)
        steps_total, start_hash = 0, 0
        ep_tot, ep_adv, ep_good = [], [], []
        mse = np.zeros(n_agents); cos = np.zeros(n_agents); acc = 0.0
        lo, hi = arglist.act_low, arglist.act_high
        betas, alphas, abar = S1
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n_agents)
            ep_r = np.zeros(n_agents)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean_n]
                faulty = v["kind"] in ("noisy", "den", "bmean", "c1", "c1lin", "c1g")      # bmean draws the fault too (identical streams)
                ahat = [np.clip(fs.perturb(a, i), lo, hi) if (faulty and i in CS) else a for i, a in enumerate(clean_n)]
                out = list(ahat)                                    # agents outside CS execute the clean action
                for i in CS:
                    if v["kind"] == "random":
                        out[i] = rand_rs.uniform(lo, hi, size=dims[i])
                    elif v["kind"] == "bc":
                        cr = corr[i]
                        out[i] = cr.net.direct(cr.cond(obs_n[i])) * cr.as_ + cr.am
                    elif v["kind"] in ("c1", "c1lin", "c1g"):
                        cr = corr[i]
                        xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                        if "t" in v:
                            sab = S0[2] if v["kind"] == "c1lin" else abar     # c1lin uses its own (S0) schedule
                            xn = (np.sqrt(sab[v["t"]]) * xn).astype(np.float32)
                        out[i] = cr.net.direct(cr.cond(obs_n[i]), xn) * cr.as_ + cr.am
                    elif v["kind"] in ("den", "bmean"):
                        cr = corr[i]
                        if v["kind"] == "bmean":
                            xn = np.zeros(len(cr.am), dtype=np.float32)       # dataset mean action, normalized
                        else:
                            xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                        t = v["t"]
                        xt = (np.sqrt(abar[t]) * xn).astype(np.float32)
                        an = C.ddpm_chain(cr.net, xt, cr.cond(obs_n[i]), t, v["one"], rng, betas, alphas, abar)
                        out[i] = an * cr.as_ + cr.am
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n_agents, dims)]
                for i in range(n_agents):
                    e, c = action_n[i], clean_n[i]
                    mse[i] += np.mean((e - c) ** 2)
                    cos[i] += float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-12))
                if task == "simple_speaker_listener":
                    acc += float(np.argmax(action_n[0]) == np.argmax(clean_n[0]))
                obs_n, rew_n, done_n, _ = env.step(action_n)
                if v["kind"] != "clean":
                    attack.step()
                ep_r += rew_n
                steps_total += 1
                if all(done_n):
                    break
            ep_tot.append(float(np.sum(ep_r)))
            ep_adv.append(float(ep_r[0]))
            ep_good.append(float(np.mean(ep_r[1:])))
    adv = task == "simple_adversary"
    return {"reward_mean": float(np.mean(ep_tot)), "reward_std": float(np.std(ep_tot)),
            "reward_adv_mean": float(np.mean(ep_adv)) if adv else "", "reward_adv_std": float(np.std(ep_adv)) if adv else "",
            "reward_good_mean": float(np.mean(ep_good)) if adv else "", "reward_good_std": float(np.std(ep_good)) if adv else "",
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in mse),
            "action_cos_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in cos),
            "msg_acc": "{:.4f}".format(acc / steps_total) if task == "simple_speaker_listener" else "",
            "start_hash": start_hash, "fault_state_hash": fs.state_hash() if v["kind"] in ("noisy", "den", "bmean", "c1", "c1lin", "c1g") else "",
            "model_sha": "|".join(sorted(set(cr.sha for cr in corr.values()))) if corr else "",
            "n_episodes": episodes, "seconds": "{:.1f}".format(time.time() - t0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="il")
    p.add_argument("--scenario", required=True, choices=["simple_speaker_listener", "simple_adversary", "simple_spread"])
    p.add_argument("--varied", required=True, choices=["speaker", "listener", "good", "shared"])
    p.add_argument("--info-level", required=True)
    p.add_argument("--fault", required=True, choices=list(FAULT_ATTACK))
    p.add_argument("--flevel", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--with-bc", action="store_true")
    p.add_argument("--step2", action="store_true", help="add B_mean and C1 variants")
    p.add_argument("--confirm", action="store_true", help="add C1-linear (S0-trained C1) variants")
    p.add_argument("--shift", action="store_true", help="add C1-gauss (Gaussian-fault-trained) variants")
    p.add_argument("--only", nargs="+", default=None, help="subset of: clean noisy bc den")
    p.add_argument("--ts", type=int, nargs="+", default=None, help="subset of the t grid (smoke tests)")
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
    arglist.attack = FAULT_ATTACK[a.fault]
    arglist.attack_mu = 1.0 if a.fault == "biased" else 0.0
    if a.fault in ("gauss", "biased"):
        arglist.attack_sigma = a.flevel
    elif a.fault == "delay":
        arglist.delay_k = int(a.flevel)
    else:
        arglist.sp_prob = a.flevel
    b, al, ab = C.schedule("cosine")
    S1 = (b.numpy().astype(np.float64), al.numpy().astype(np.float64), ab.numpy().astype(np.float64))
    b0, al0, ab0 = C.schedule("linear")
    S0 = (b0.numpy().astype(np.float64), al0.numpy().astype(np.float64), ab0.numpy().astype(np.float64))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for v in variants(a.with_bc, a.step2, a.confirm, a.shift):
            if v["kind"] == "random" and not (a.only and "random" in a.only):
                continue
            if a.only and v["kind"] not in a.only and v["name"] not in a.only:
                continue
            if a.ts and "t" in v and v["t"] not in a.ts:
                continue
            r = run_variant(arglist, a.scenario, a.varied, a.info_level, v, a.seed, a.episodes, a.model_dir, S1, S0)
            row = {"phase": a.phase, "scenario": a.scenario, "varied": a.varied, "info_level": a.info_level,
                   "method": v["name"], "mode": v.get("mode", ""), "t": v.get("t", ""), "fault": FAULT_NAME[a.fault],
                   "level": a.flevel, "seed": a.seed}
            row.update(r)
            w.writerow(row); f.flush()
            print("[s{} {} {} {} {} {}{}] reward={:.2f} mse={} acc={} ({}s)".format(
                a.seed, a.scenario, a.varied, a.info_level, FAULT_NAME[a.fault], v["name"],
                "" if "mode" not in v else " {}@{}".format(v["mode"], v.get("t", "-")), r["reward_mean"],
                r["action_mse_per_agent"], r["msg_acc"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
