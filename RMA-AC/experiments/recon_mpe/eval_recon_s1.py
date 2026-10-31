"""
eval_recon_s1.py -- second screen: S1 (cosine) denoiser variants, oracle-matched / fixed start step,
speaker-history variants. Same seeding, projection, fault stream and logging as eval_recon.py
(imported, not copied, for FaultStream / softmax / COLUMNS).

Variants (method column):
  noisy, bc, denoiser@10 (S0 linear, legacy sampler), c1 (S0), c2, c1_s1 (S1 matched corruption)
  den_s1_1step_match / den_s1_chain_match : S1 denoiser, one step / standard DDPM chain from t_match
        (per agent; step whose noise level sqrt((1-abar)/abar) is closest to the effective injected std in
        normalized units after the clip projection; oracle-matched). Input x_t = sqrt(abar_t) * a~_norm.
  den_s1_1step_fixed / den_s1_chain_fixed : same, at ONE fixed t (= t_match for std 1) at every level
  den_s1_1step_match_noscale : control, one-step at t_match with x_t = a~_norm (no sqrt(abar) scaling)
  speaker history (simple_speaker_listener hidden only): bc_hist, c1_hist (S1 corruption), c2_hist,
        den_hist_1step_match, den_hist_chain_match. Speaker input = ID + previous 5 noisy (post-fault,
        post-clip, pre-correction) messages, zero-padded at episode start; listener models unchanged.
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

import eval_recon as E          # sets BLAS/TF env, chdir(EXP), sys.path; provides helpers
import common as C
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env
from action_attacks import ActionAttack

COLUMNS = E.COLUMNS + ["schedule"]
LISTENER_KEY = {"bc_hist": "bc", "c1_hist": "c1_s1", "c2_hist": "c2", "den_hist": "den_s1"}
STDS = [0.0, 0.4, 1.0, 2.0, 3.0]


def variant_list(scenario, info):
    V = [
        dict(name="noisy", kind="noisy"),
        dict(name="bc", kind="direct", key="bc"),
        dict(name="denoiser", kind="old", key="den", t=10, sched="S0"),
        dict(name="c1", kind="direct", key="c1", sched="S0"),
        dict(name="c2", kind="direct", key="c2"),
        dict(name="c1_s1", kind="direct", key="c1_s1", sched="S1"),
    ]
    for nm, one, tsel, scale in [("den_s1_1step_match", True, "match", True), ("den_s1_chain_match", False, "match", True),
                                 ("den_s1_1step_fixed", True, "fixed", True), ("den_s1_chain_fixed", False, "fixed", True),
                                 ("den_s1_1step_match_noscale", True, "match", False)]:
        V.append(dict(name=nm, kind="ddpm", key="den_s1", one=one, tsel=tsel, scale=scale, sched="S1"))
    if scenario == "simple_speaker_listener" and info == "hidden":
        V += [dict(name="bc_hist", kind="direct", key="bc_hist"),
              dict(name="c1_hist", kind="direct", key="c1_hist", sched="S1"),
              dict(name="c2_hist", kind="direct", key="c2_hist"),
              dict(name="den_hist_1step_match", kind="ddpm", key="den_hist", one=True, tsel="match", scale=True, sched="S1"),
              dict(name="den_hist_chain_match", kind="ddpm", key="den_hist", one=False, tsel="match", scale=True, sched="S1")]
    return V


def tmatch_table(scenario, lv1):
    """{(agent, sigma): t_match} from the effective injected std in normalized units (after the clip projection)."""
    tab = {}
    for i, (o, a) in enumerate(C.load_agent_data(scenario)):
        _, astd = C.norm_stats(a)
        for s in STDS:
            tab[(i, s)] = C.t_match(C.effective_noise_norm(a, astd[0], s), lv1)
    return tab


class Corr:
    def __init__(self, key, scenario, info, agent, seed, model_dir):
        tag, info_f = C.agent_tag_and_info(scenario, agent, info)
        if "hist" in key and agent == 0:
            tag, info_f = "speaker", info
        C.MODEL_DIR = model_dir
        model, ck = C.load_model(C.model_path(key, scenario, info_f, tag, seed))
        self.net, self.ck, self.spec, self.agent = C.NumpyNet(model), ck, ck["cond_spec"], agent
        n = ck["norm"][agent]
        self.om, self.os, self.am, self.as_ = n["obs_mean"][0], n["obs_std"][0], n["act_mean"][0], n["act_std"][0]
        self.hist = self.spec["kind"] == "id_hist"
        self.n_hist = self.spec.get("n_hist", 0)

    def cond(self, obs, hist_bufs):
        if self.hist:
            return np.concatenate([np.array([1.0, 0.0], dtype=np.float32)] + hist_bufs).astype(np.float32)
        return C.build_cond(self.spec, ((obs - self.om) / self.os)[None], self.agent)[0]


def run_variant(arglist, scenario, info, v, seed, level, episodes, model_dir, sched, tm):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t0 = time.time()
    S0, S1, = sched
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n_agents = env.n
        p_vals = [tr.p_debug["p_values"] for tr in trainers]
        corr = None
        if v["kind"] in ("direct", "old", "ddpm"):
            corr = []
            for i in range(n_agents):
                key = v["key"]
                if "hist" in key and i != 0:
                    key = LISTENER_KEY[key]
                corr.append(Corr(key, scenario, info, i, seed, model_dir))
        # step used per agent for ddpm variants
        tused = None
        if v["kind"] == "ddpm":
            tused = [tm[(i, float(level))] if v["tsel"] == "match" else tm[(i, 1.0)] for i in range(n_agents)]
        elif v["kind"] == "old":
            tused = [v["t"]] * n_agents
        np.random.seed(seed); random.seed(seed)
        attack = ActionAttack(arglist)
        fs = E.FaultStream(attack, seed)
        samp_rng = np.random.RandomState(seed + 999)
        steps_total, ep_rewards, start_hash = 0, [], 0
        mse = np.zeros(n_agents); mse_det = np.zeros(n_agents); acc = acc_det = 0.0
        lo, hi = arglist.act_low, arglist.act_high
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n_agents)
            hist_n = corr[0].n_hist if corr and corr[0].hist else 0
            hbuf = [np.zeros(corr[0].ck["action_dim"], dtype=np.float32) for _ in range(hist_n)]
            ep_r = np.zeros(n_agents)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                det_n = [E.softmax(pv(o[None])[0]) for pv, o in zip(p_vals, obs_n)]
                dims = [len(a) for a in clean_n]
                ahat = clean_n if v["name"] == "bc" else [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]
                if v["kind"] == "noisy":
                    out = ahat
                else:
                    out = []
                    for i, cr in enumerate(corr):
                        cond = cr.cond(obs_n[i], hbuf if cr.hist else None)
                        xn = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                        if v["name"] == "bc":
                            an = cr.net.direct(cond)
                        elif v["name"] == "bc_hist" and i == 0:
                            an = cr.net.direct(cond)                       # history only, ignores the current action
                        elif v["kind"] == "direct" or (v["name"] == "bc_hist"):
                            an = cr.net.direct(cond, xn) if cr.net.use_action else cr.net.direct(cond)
                        elif v["kind"] == "old":
                            an = C.sample_chain(cr.net, xn, cond, tused[i], False, samp_rng, S0[1], S0[2])
                        else:
                            t = tused[i]
                            xt = (np.sqrt(S1[2][t]) * xn).astype(np.float32) if v["scale"] else xn
                            an = C.ddpm_chain(cr.net, xt, cond, t, v["one"], samp_rng, S1[0], S1[1], S1[2])
                        out.append(an * cr.as_ + cr.am)
                    if corr[0].hist:       # push the current noisy speaker message (normalized) into the history
                        hbuf = [((ahat[0] - corr[0].am) / corr[0].as_).astype(np.float32)] + hbuf[:-1]
                action_n = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n_agents, dims)]
                for i in range(n_agents):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                    mse_det[i] += np.mean((action_n[i] - det_n[i]) ** 2)
                if scenario == "simple_speaker_listener":
                    acc += float(np.argmax(action_n[0]) == np.argmax(clean_n[0]))
                    acc_det += float(np.argmax(action_n[0]) == np.argmax(det_n[0]))
                obs_n, rew_n, done_n, _ = env.step(action_n)
                if v["name"] != "bc":
                    attack.step()
                ep_r += rew_n
                steps_total += 1
                if all(done_n):
                    break
            ep_rewards.append(float(np.sum(ep_r)))
    sl = scenario == "simple_speaker_listener"
    tstr = "" if tused is None else "|".join("A{}:{}".format(i, t) for i, t in enumerate(tused))
    return {"reward_mean": float(np.mean(ep_rewards)), "reward_std": float(np.std(ep_rewards)),
            "action_mse_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in mse),
            "action_mse_det_per_agent": "|".join("{:.6f}".format(x / steps_total) for x in mse_det),
            "msg_acc": "{:.4f}".format(acc / steps_total) if sl else "",
            "msg_acc_det": "{:.4f}".format(acc_det / steps_total) if sl else "",
            "start_hash": start_hash, "fault_state_hash": fs.state_hash() if v["name"] != "bc" else "",
            "n_episodes": episodes, "seconds": "{:.1f}".format(time.time() - t0), "t_start": tstr,
            "schedule": v.get("sched", "")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="screen2")
    p.add_argument("--scenario", required=True, choices=list(C.SCENARIOS))
    p.add_argument("--info", default="full", choices=["hidden", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", type=float, nargs="+", required=True)
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
    arglist.attack, arglist.attack_mu = "gauss", 0.0

    b0, a0, ab0 = C.schedule("linear")
    S0 = (b0.numpy(), a0.numpy(), ab0.numpy())      # float32, exactly as eval_recon.py (reference rows reproduce)
    b1, a1, ab1 = C.schedule("cosine")
    S1 = (b1.numpy().astype(np.float64), a1.numpy().astype(np.float64), ab1.numpy().astype(np.float64))
    lv1 = C.noise_levels(S1[2])
    tm = tmatch_table(a.scenario, lv1)
    print("t_match per (agent, std):", {k: v for k, v in sorted(tm.items())}, flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for level in a.levels:
            arglist.attack_sigma = float(level)
            for v in variant_list(a.scenario, a.info):
                if a.variants and v["name"] not in a.variants:
                    continue
                r = run_variant(arglist, a.scenario, a.info, v, a.seed, level, a.episodes, a.model_dir, (S0, S1), tm)
                row = {"phase": a.phase, "scenario": a.scenario, "info_level": a.info, "method": v["name"],
                       "policy": "m3ddpg", "fault": "noise", "level": level, "seed": a.seed}
                row.update(r)
                w.writerow(row); f.flush()
                print("[{} {} lvl={} {} {}] reward={:.2f} mse={} acc={} ({}s)".format(
                    a.scenario, a.info, level, v["name"], r["t_start"], r["reward_mean"], r["action_mse_per_agent"],
                    r["msg_acc"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
