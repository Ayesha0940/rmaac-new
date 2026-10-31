"""BC on the FULL local observation (per-agent, decentralized): every agent acts with its own per-agent BC model
(own full observation [+ ID]); no fault (BC never sees the policy action, so it is fault-independent). One run per scenario.
Also runs the clean policy in the same job as a reference (reward summed over agents, 800 episodes, seed 0)."""
import os
for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_CBWR", "COMPATIBLE")]:
    os.environ.setdefault(_k, _v)
import argparse, csv, random, sys, zlib
import numpy as np
import eval_recon as E
import eval_il as EIL
import common as C
import tensorflow as tf
from train import parse_args, apply_variant_defaults, concat_actions, split_actions
import maddpg.common.tf_util as U
from filter_eval import build_test_env


def run(scen, kind, seed, episodes, model_dir, arglist):
    tf.reset_default_graph(); tf.set_random_seed(seed)
    with U.single_threaded_session():
        env, trainers, _ = build_test_env(arglist)
        n = env.n
        corr = [EIL.ILCorr("bc", scen, i, seed, model_dir) for i in range(n)] if kind == "bc" else None
        np.random.seed(seed); random.seed(seed)
        lo, hi = arglist.act_low, arglist.act_high
        tot, sh = [], 0
        for ep in range(episodes):
            obs_n = env.reset(); sh = zlib.crc32(np.concatenate(obs_n).astype(np.float32).tobytes(), sh)
            r = np.zeros(n)
            for step in range(arglist.max_episode_len):
                clean = [tr.action(o) for tr, o in zip(trainers, obs_n)]
                dims = [len(a) for a in clean]
                out = list(clean) if kind == "clean" else [cr.net.direct(cr.cond(obs_n[i])) * cr.as_ + cr.am for i, cr in enumerate(corr)]
                act = [np.clip(a, lo, hi) for a in split_actions(concat_actions(out), n, dims)]
                obs_n, rew, done, _ = env.step(act)
                r += rew
                if all(done):
                    break
            tot.append(float(np.sum(r)))
    return float(np.mean(tot)), float(np.std(tot)), sh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-dir", default=C.MODEL_DIR)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.model_dir = a.model_dir if os.path.isabs(a.model_dir) else os.path.join(E.HERE, a.model_dir)
    sys.argv = [sys.argv[0], "--scenario", a.scenario, "--variant", "m3ddpg", "--exp-name", "{}__m3ddpgbest".format(a.scenario),
                "--save-dir", "./model", "--num-adversaries", "0"]
    arglist = apply_variant_defaults(parse_args()); arglist.benchmark = False
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario", "method", "seed", "reward_mean", "reward_std", "n_episodes", "start_hash"])
        for kind in ("clean", "bc"):
            m, s, sh = run(a.scenario, kind, a.seed, a.episodes, a.model_dir, arglist)
            w.writerow([a.scenario, kind, a.seed, "{:.2f}".format(m), "{:.2f}".format(s), a.episodes, sh]); f.flush()
            print("[{}] {} reward = {:.2f} (std {:.2f})".format(a.scenario, kind, m, s), flush=True)


if __name__ == "__main__":
    main()
