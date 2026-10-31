"""
eval_recon.py -- evaluate noisy policy / BC / denoiser / C1 / C2 on MPE under action faults.

Per env step, per agent:
    clean = M3DDPG policy sample (Gumbel-softmax; always drawn, also for BC, so every method
            consumes the same number of policy random draws)
    a~    = clip(fault(clean), -1, 1)               (fault stream: dedicated RNG state)
    exec  = corrector(a~, own obs)                  -> concat_actions -> split_actions -> clip -> env.step
BC ignores the policy action and is executed WITHOUT fault.

Reproducibility: a fresh TF graph/session is built for every method run (tf.set_random_seed(seed)
first) so the policy's Gumbel noise is the same stream for every method; np/random/env seeded
per run; fault noise / stuck masks / delay buffers come from a dedicated RandomState swapped in
around ActionAttack.perturb (action_attacks.py is not modified); sampler noise from its own
numpy Generator.

Methods: noisy, bc, denoiser@{10,20,40}, denoiser_1step@{10,20,40}, c1, c2.
"""
import argparse
import csv
import os
import random
import sys
import time
import zlib

# bit-for-bit reproducible BLAS (must be set before numpy is imported)
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
               ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")
os.chdir(EXP)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

import tensorflow as tf  # noqa: E402

import common as C  # noqa: E402
from train import (parse_args, apply_variant_defaults, concat_actions, split_actions)  # noqa: E402
import maddpg.common.tf_util as U  # noqa: E402
from filter_eval import build_test_env  # noqa: E402
from action_attacks import ActionAttack  # noqa: E402

START_STEPS = [10, 20, 40]
COLUMNS = ["phase", "scenario", "info_level", "method", "policy", "fault", "level", "t_start", "seed",
           "reward_mean", "reward_std", "action_mse_per_agent", "action_mse_det_per_agent",
           "msg_acc", "msg_acc_det", "start_hash", "fault_state_hash", "n_episodes", "seconds"]


def all_methods(only=None):
    ms = [("noisy", None), ("bc", None)]
    ms += [("denoiser", t) for t in START_STEPS]
    ms += [("denoiser_1step", t) for t in START_STEPS]
    ms += [("c1", None), ("c2", None)]
    if only:
        ms = [m for m in ms if m[0] in only]
    return ms


class FaultStream:
    """Runs ActionAttack.perturb with a dedicated numpy RNG state (global np.random untouched)."""

    def __init__(self, attack, seed):
        self.attack = attack
        self.state = np.random.RandomState(seed + 777).get_state()

    def perturb(self, a, i):
        saved = np.random.get_state()
        np.random.set_state(self.state)
        out = self.attack.perturb(a, i)
        self.state = np.random.get_state()
        np.random.set_state(saved)
        return out

    def state_hash(self):
        return zlib.crc32(self.state[1].tobytes()) ^ int(self.state[2])


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class AgentCorrector:
    def __init__(self, method_key, scenario, info, agent, seed, model_dir):
        tag, info_f = C.agent_tag_and_info(scenario, agent, info)
        C.MODEL_DIR = model_dir
        model, ck = C.load_model(C.model_path(method_key, scenario, info_f, tag, seed))
        self.net = C.NumpyNet(model)
        self.spec, self.agent = ck["cond_spec"], agent
        n = ck["norm"][agent]
        self.om, self.os = n["obs_mean"][0], n["obs_std"][0]
        self.am, self.as_ = n["act_mean"][0], n["act_std"][0]
        self.ck = ck

    def cond(self, obs):
        return C.build_cond(self.spec, ((obs - self.om) / self.os)[None], self.agent)[0]


def run_method(arglist, scenario, info, method, t_start, seed, fault, level, episodes, model_dir, rng_ab):
    alphas, alphas_bar = rng_ab
    cfg_key = {"denoiser": "den", "denoiser_1step": "den", "c1": "c1", "c2": "c2", "bc": "bc"}.get(method)

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    t_wall = time.time()
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n_agents = env.n
        p_vals = [tr.p_debug["p_values"] for tr in trainers]
        correctors = None
        if cfg_key is not None:
            correctors = [AgentCorrector(cfg_key, scenario, info, i, seed, model_dir) for i in range(n_agents)]

        np.random.seed(seed)
        random.seed(seed)
        attack = ActionAttack(arglist)
        fs = FaultStream(attack, seed)
        samp_rng = np.random.RandomState(seed + 999)
        dims = None
        steps_total = 0
        ep_rewards = []
        mse = np.zeros(n_agents); mse_det = np.zeros(n_agents)
        acc = acc_det = 0.0
        start_hash = 0
        lo, hi = arglist.act_low, arglist.act_high
        for ep in range(episodes):
            obs_n = env.reset()
            start_hash = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), start_hash)
            attack.reset(n_agents=n_agents)
            ep_r = np.zeros(n_agents)
            for step in range(arglist.max_episode_len):
                clean_n = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                det_n = [softmax(pv(o[None])[0]) for pv, o in zip(p_vals, obs_n)]
                dims = [len(a) for a in clean_n]
                if method == "bc":
                    ahat = clean_n                                          # unused; shape only
                else:
                    ahat = [np.clip(fs.perturb(a, i), lo, hi) for i, a in enumerate(clean_n)]
                if method == "noisy":
                    out = ahat
                else:
                    out = []
                    for i, cr in enumerate(correctors):
                        cond = cr.cond(obs_n[i])
                        if method == "bc":
                            an = cr.net.direct(cond)
                        elif method in ("c1", "c2"):
                            an = cr.net.direct(cond, ((ahat[i] - cr.am) / cr.as_).astype(np.float32))
                        else:
                            x = ((ahat[i] - cr.am) / cr.as_).astype(np.float32)
                            an = C.sample_chain(cr.net, x, cond, t_start, method == "denoiser_1step",
                                                samp_rng, alphas, alphas_bar)
                        out.append(an * cr.as_ + cr.am)
                action_vec = concat_actions(out)
                action_n = [np.clip(a, lo, hi) for a in split_actions(action_vec, n_agents, dims)]
                for i in range(n_agents):
                    mse[i] += np.mean((action_n[i] - clean_n[i]) ** 2)
                    mse_det[i] += np.mean((action_n[i] - det_n[i]) ** 2)
                if scenario == "simple_speaker_listener":
                    acc += float(np.argmax(action_n[0]) == np.argmax(clean_n[0]))
                    acc_det += float(np.argmax(action_n[0]) == np.argmax(det_n[0]))
                obs_n, rew_n, done_n, _ = env.step(action_n)
                if method != "bc":
                    attack.step()
                ep_r += rew_n
                steps_total += 1
                if all(done_n):
                    break
            ep_rewards.append(float(np.sum(ep_r)))

    is_sl = scenario == "simple_speaker_listener"
    return {
        "reward_mean": float(np.mean(ep_rewards)), "reward_std": float(np.std(ep_rewards)),
        "action_mse_per_agent": "|".join("{:.6f}".format(v / steps_total) for v in mse),
        "action_mse_det_per_agent": "|".join("{:.6f}".format(v / steps_total) for v in mse_det),
        "msg_acc": "{:.4f}".format(acc / steps_total) if is_sl else "",
        "msg_acc_det": "{:.4f}".format(acc_det / steps_total) if is_sl else "",
        "start_hash": start_hash, "fault_state_hash": fs.state_hash() if method != "bc" else "",
        "n_episodes": episodes, "seconds": "{:.1f}".format(time.time() - t_wall),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--scenario", required=True, choices=list(C.SCENARIOS))
    p.add_argument("--info", default="full", choices=["hidden", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fault", default="noise", choices=["noise", "delay", "stuck"])
    p.add_argument("--levels", type=float, nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--methods", nargs="+", default=None, help="subset of method names")
    p.add_argument("--model-dir", default=C.MODEL_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    # eval chdirs to experiments/; resolve relative paths against recon_mpe/
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(HERE, a.model_dir)
    a.out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)

    sys.argv = [sys.argv[0], "--scenario", a.scenario, "--variant", "m3ddpg",
                "--exp-name", "{}__m3ddpgbest".format(a.scenario), "--save-dir", "./model",
                "--num-adversaries", "0"]
    arglist = apply_variant_defaults(parse_args())
    arglist.benchmark = False
    assert arglist.act_low == C.ACT_LOW and arglist.act_high == C.ACT_HIGH
    arglist.attack = {"noise": "gauss", "delay": "delay", "stuck": "stuck_at"}[a.fault]
    arglist.attack_mu = 0.0

    _, al, ab = C.beta_schedule()
    rng_ab = (al.numpy(), ab.numpy())

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for level in a.levels:
            if a.fault == "noise":
                arglist.attack_sigma = float(level)
            elif a.fault == "delay":
                arglist.delay_k = int(level)
            else:
                arglist.sp_prob = float(level)
            for method, t_start in all_methods(a.methods):
                r = run_method(arglist, a.scenario, a.info, method, t_start, a.seed, a.fault, level,
                               a.episodes, a.model_dir, rng_ab)
                row = {"phase": a.phase, "scenario": a.scenario, "info_level": a.info, "method": method,
                       "policy": "m3ddpg", "fault": a.fault, "level": level,
                       "t_start": "" if t_start is None else t_start, "seed": a.seed}
                row.update(r)
                w.writerow(row); f.flush()
                print("[{} {} {} lvl={} {}{}] reward={:.2f} mse={} acc={} ({}s)".format(
                    a.scenario, a.info, a.fault, level, method, "" if t_start is None else "@{}".format(t_start),
                    r["reward_mean"], r["action_mse_per_agent"], r["msg_acc"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
