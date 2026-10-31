"""
eval_pa.py -- per-agent (decentralized) denoiser on MPE, S0 vs S1, several inference variants.
Corrector input per agent = own noisy action + own full observation + one-hot agent id (models trained by
train_recon.py, method 'den' / 'den_s1'). Same seeding / fault stream / projection as eval_recon.py.

Variants (name): clean, noisy,
  <S>_legacy_chain@10        existing sampler, full chain from t=10 (unscaled input)          [deployable]
  <S>_1step@{10,20,40}       legacy one-step x0_hat at fixed t (unscaled input)                [deployable]
  <S>_ORACLE_1step_match_{scaled,unscaled}   one-step at t_match                          [oracle: upper bound, not deployable]
  <S>_ORACLE_chain_match_{scaled,unscaled}   standard DDPM posterior chain t_match -> 0   [oracle: upper bound, not deployable]
  scaled: x_t = sqrt(abar_t) * a~_norm ; unscaled: x_t = a~_norm.   <S> in {S0 linear, S1 cosine}.
t_match = step whose noise level sqrt((1-abar)/abar) is closest to the injected error in normalized units
(mean over action dims of rms(a~ - a)/act_std, simulated on the dataset for the fault being evaluated).
"""
# bit-for-bit reproducible BLAS: must be set before numpy is imported anywhere
import os
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)
import argparse
import csv
import os
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

COLUMNS = E.COLUMNS + ["schedule", "oracle", "reward_adv_mean", "reward_adv_std", "reward_good_mean", "reward_good_std"]
FAULT_ATTACK = {"gauss": "gauss", "biased": "biased_gauss", "delay": "delay", "stuck": "stuck_at"}


def variants():
    V = [dict(name="clean", kind="clean"), dict(name="noisy", kind="noisy")]
    for sch, key in (("S0", "den"), ("S1", "den_s1")):
        V.append(dict(name=sch + "_legacy_chain@10", kind="old", key=key, sch=sch, t=10, one=False, oracle=0))
        for t in (10, 20, 40):
            V.append(dict(name="{}_1step@{}".format(sch, t), kind="old", key=key, sch=sch, t=t, one=True, oracle=0))
        for scale in (True, False):
            tg = "scaled" if scale else "unscaled"
            V.append(dict(name="{}_ORACLE_1step_match_{}".format(sch, tg), kind="ddpm", key=key, sch=sch, one=True, scale=scale, oracle=1))
            V.append(dict(name="{}_ORACLE_chain_match_{}".format(sch, tg), kind="ddpm", key=key, sch=sch, one=False, scale=scale, oracle=1))
    return V


def schedules():
    out = {}
    for sch, nm in (("S0", "linear"), ("S1", "cosine")):
        b, al, ab = C.schedule(nm)
        out[sch] = {"f32": (b.numpy(), al.numpy(), ab.numpy()),
                    "f64": (b.numpy().astype(np.float64), al.numpy().astype(np.float64), ab.numpy().astype(np.float64)),
                    "lv": C.noise_levels(ab.numpy())}
    return out


def tmatch_for(scenario, fault, level, SCH):
    """{(agent, sch): t_match} for this fault cell."""
    d = np.load(os.path.join(C.DATA_DIR, "{}_m3ddpg.npz".format(scenario)))
    A = d["actions"]
    ao = np.cumsum([0] + C.SCENARIOS[scenario]["act_dims"])
    tm = {}
    for i in range(len(ao) - 1):
        a = A[:, :, ao[i]:ao[i + 1]]
        _, astd = C.norm_stats(a.reshape(-1, a.shape[2]))
        e = C.fault_error_norm(a, fault, level, astd[0])
        for sch in SCH:
            tm[(i, sch)] = C.t_match(e, SCH[sch]["lv"])
    return tm


def run_variant(arglist, scenario, v, seed, fault, level, episodes, model_dir, SCH, tm):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n_agents = env.n
        p_vals = [tr.p_debug["p_values"] for tr in trainers]
        corr = None
        if v["kind"] in ("old", "ddpm"):
            corr = [T.Corr(v["key"], scenario, "full", i, seed, model_dir) for i in range(n_agents)]
        tused = None
        if v["kind"] == "ddpm":
            tused = [tm[(i, v["sch"])] for i in range(n_agents)]
        elif v["kind"] == "old":
            tused = [v["t"]] * n_agents
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        rng = np.random.RandomState(seed + 999)
        steps_total, start_hash = 0, 0
        ep_tot, ep_adv, ep_good = [], [], []
        mse = np.zeros(n_agents); mse_det = np.zeros(n_agents); acc = acc_det = 0.0
        lo, hi = arglist.act_low, arglist.act_high
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n_agents)
            ep_r = np.zeros(n_agents)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                det_n = [E.softmax(pv(o[None])[0]) for pv, o in zip(p_vals, obs_n)]
                dims = [len(a) for a in clean_n]
                ahat = clean_n if v["kind"] == "clean" else [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]
                if v["kind"] in ("clean", "noisy"):
                    out = ahat
                else:
                    out = []
                    for i, cr in enumerate(corr):
                        cond = cr.cond(obs_n[i], None)
                        xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                        sc = SCH[v["sch"]]
                        if v["kind"] == "old":
                            an = C.sample_chain(cr.net, xn, cond, tused[i], v["one"], rng, sc["f32"][1], sc["f32"][2])
                        else:
                            t = tused[i]
                            xt = (np.sqrt(sc["f64"][2][t]) * xn).astype(np.float32) if v["scale"] else xn
                            an = C.ddpm_chain(cr.net, xt, cond, t, v["one"], rng, *sc["f64"])
                        out.append(an * cr.as_ + cr.am)
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n_agents, dims)]
                for i in range(n_agents):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                    mse_det[i] += np.mean((action_n[i] - det_n[i]) ** 2)
                if scenario == "simple_speaker_listener":
                    acc += float(np.argmax(action_n[0]) == np.argmax(clean_n[0]))
                    acc_det += float(np.argmax(action_n[0]) == np.argmax(det_n[0]))
                obs_n, rew_n, done_n, _ = env.step(action_n)
                if v["kind"] != "clean":
                    attack.step()
                ep_r += rew_n
                steps_total += 1
                if all(done_n):
                    break
            ep_tot.append(float(np.sum(ep_r)))
            ep_adv.append(float(ep_r[0]))                    # simple_adversary: agent 0 = adversary
            ep_good.append(float(np.mean(ep_r[1:])))         # good agents share one reward value; mean == each one's return
    adv = scenario == "simple_adversary"
    sl = scenario == "simple_speaker_listener"
    return {"reward_mean": float(np.mean(ep_tot)), "reward_std": float(np.std(ep_tot)),
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in mse),
            "action_mse_det_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in mse_det),
            "msg_acc": "{:.4f}".format(acc / steps_total) if sl else "",
            "msg_acc_det": "{:.4f}".format(acc_det / steps_total) if sl else "",
            "start_hash": start_hash, "fault_state_hash": fs.state_hash() if v["kind"] != "clean" else "",
            "n_episodes": episodes, "seconds": "{:.1f}".format(time.time() - t0),
            "t_start": "" if tused is None else "|".join("A{}:{}".format(i, t) for i, t in enumerate(tused)),
            "schedule": v.get("sch", ""), "oracle": v.get("oracle", ""),
            "reward_adv_mean": float(np.mean(ep_adv)) if adv else "", "reward_adv_std": float(np.std(ep_adv)) if adv else "",
            "reward_good_mean": float(np.mean(ep_good)) if adv else "", "reward_good_std": float(np.std(ep_good)) if adv else ""}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="pa")
    p.add_argument("--scenario", required=True, choices=list(C.SCENARIOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fault", required=True, choices=list(FAULT_ATTACK))
    p.add_argument("--level", type=float, required=True)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--variants", nargs="+", default=None)
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
        arglist.attack_sigma = a.level
    elif a.fault == "delay":
        arglist.delay_k = int(a.level)
    else:
        arglist.sp_prob = a.level

    SCH = schedules()
    tm = tmatch_for(a.scenario, a.fault, a.level, SCH)
    print("t_match (agent, schedule):", dict(sorted(tm.items())), flush=True)
    fname = {"gauss": "gauss", "biased": "biased_gauss_mu+1", "delay": "delay", "stuck": "stuck"}[a.fault]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for v in variants():
            if a.variants and v["name"] not in a.variants:
                continue
            r = run_variant(arglist, a.scenario, v, a.seed, a.fault, a.level, a.episodes, a.model_dir, SCH, tm)
            row = {"phase": a.phase, "scenario": a.scenario, "info_level": "full", "method": v["name"], "policy": "m3ddpg",
                   "fault": fname, "level": a.level, "seed": a.seed}
            row.update(r)
            w.writerow(row); f.flush()
            print("[{} {} {} {}] reward={:.2f} mse={} acc={} t={} ({}s)".format(
                a.scenario, fname, a.level, v["name"], r["reward_mean"], r["action_mse_per_agent"], r["msg_acc"],
                r["t_start"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
