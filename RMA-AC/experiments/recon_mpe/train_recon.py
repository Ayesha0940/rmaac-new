"""
train_recon.py -- one training loop for BC, C1, C2 and the per-agent denoiser.

  bc  : deterministic, input = cond only; MSE on normalized actions.
  c1  : matched corruption: t ~ U{0..99}, x = sqrt(abar_t) a_norm + sqrt(1-abar_t) eps; predicts a_norm.
  c2  : noise-aware (knows the test noise model): sigma ~ U(0, max_std) per sample,
        noisy = clip(a_raw + sigma*eps, -1, 1) in raw units, then normalized; predicts a_norm.
  den : conditional DDPM (H=1), same corruption as c1, predicts epsilon.

Same data, normalization, backbone width, optimizer, batch, epochs and loop for all four.
Fresh noise every batch. Adam lr 1e-3, batch 1024, 200 epochs, drop_last (shuffled
index batches, equivalent to DataLoader(shuffle=True, drop_last=True)).

Usage:
  python train_recon.py --scenario simple_speaker_listener --info hidden --method den --seed 0
  python train_recon.py --scenario simple_spread --info full --method c2 --seed 0
(speaker_listener trains the speaker for the given --info and the listener once, shared by both info levels)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import info_levels as IL


def _train_one(scenario, info, agent_idxs, method, data, seed, args, out_path):
    """agent_idxs: agents served by this model (speaker alone, listener alone, or all spread agents)."""
    spec_all = [C.cond_spec(scenario, info, i) for i in agent_idxs]
    act_dim = C.SCENARIOS[scenario]["act_dims"][agent_idxs[0]]
    cdim = C.cond_dim(scenario, info, agent_idxs[0])

    lay, enc_in, enc_d = None, 0, 0
    if args.level != "full":
        role = IL.role_for(scenario, agent_idxs[0])
        lay = IL.cond_layout(scenario, role, args.level)
        cdim, enc_in, enc_d = lay["model_cond"], lay["enc_in"], lay["enc_d"]
        print("[level] {} {} {}: cond_in {} -> model cond {} (encoder {}->{})".format(
            scenario, role, args.level, lay["cond_in"], cdim, enc_in, enc_d), flush=True)
    # per-agent normalization stats (spread: one set per agent, shared network)
    stats, conds, acts_raw = {}, [], []
    for i, spec in zip(agent_idxs, spec_all):
        o, a = data[i]
        om, os_ = C.norm_stats(o)
        am, as_ = C.norm_stats(a)
        stats[i] = {"obs_mean": om, "obs_std": os_, "act_mean": am, "act_std": as_}
        if args.level == "full":
            conds.append(C.build_cond(spec, (o - om) / os_, i))
        else:
            conds.append(IL.build_cond_level(lay, o, om, os_, i))
        acts_raw.append(a)
    cond = torch.from_numpy(np.concatenate(conds)).float()
    if args.hist:
        assert scenario == "simple_speaker_listener" and list(agent_idxs) == [0]
        n_hist = args.n_hist
        step_in_ep = torch.from_numpy(np.arange(cond.shape[0]) % 25)       # data order is episode-major, 25 steps
        onehot = torch.from_numpy(C.build_cond({"kind": "id_only", "n_ids": 2}, np.zeros((1, 1), np.float32), 0)[0])
        cdim = 2 + n_hist * act_dim
    a_raw = torch.from_numpy(np.concatenate(acts_raw)).float()
    amean = torch.from_numpy(np.concatenate([np.repeat(stats[i]["act_mean"], len(acts_raw[k]), 0)
                                              for k, i in enumerate(agent_idxs)])).float()
    astd = torch.from_numpy(np.concatenate([np.repeat(stats[i]["act_std"], len(acts_raw[k]), 0)
                                             for k, i in enumerate(agent_idxs)])).float()
    a_norm = (a_raw - amean) / astd
    N = cond.shape[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = C.make_model(method, cdim, act_dim, enc_in, enc_d)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    _, alphas, abar = C.schedule(args.schedule, args.steps)

    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, N // args.batch_size          # drop_last
        for b in range(nb):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            c, an = cond[idx], a_norm[idx]
            B = c.shape[0]
            if args.hist:
                # history = previous n_hist NOISY messages of the same episode (test noise model, sigma shared per
                # sample, ~U(0, max_std)), z-scored, zero-padded before the episode start.
                hsig = torch.rand(B, 1) * args.max_std
                feats = []
                for k in range(1, n_hist + 1):
                    valid = (step_in_ep[idx] >= k).float().unsqueeze(-1)
                    src = torch.clamp(idx - k, min=0)
                    nk = torch.clamp(a_raw[src] + hsig * torch.randn(B, act_dim), C.ACT_LOW, C.ACT_HIGH)
                    feats.append(((nk - amean[idx]) / astd[idx]) * valid)
                c = torch.cat([onehot.unsqueeze(0).expand(B, -1)] + feats, dim=1)
            if method == "bc":
                loss = F.mse_loss(model(c), an)
            elif method == "c2":
                sigma = hsig if args.hist else torch.rand(B, 1) * args.max_std
                noisy = torch.clamp(a_raw[idx] + sigma * torch.randn(B, act_dim), C.ACT_LOW, C.ACT_HIGH)
                noisy = (noisy - amean[idx]) / astd[idx]
                loss = F.mse_loss(model(c, x=noisy), an)
            else:
                t = torch.randint(0, args.steps, (B,))
                eps = torch.randn(B, act_dim)
                ab = abar[t].unsqueeze(-1)
                x = torch.sqrt(ab) * an + torch.sqrt(1.0 - ab) * eps
                if method == "c1":
                    loss = F.mse_loss(model(c, x=x), an)
                else:
                    loss = F.mse_loss(model(c, x=x, t=t), eps)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print("[{}] {} {} agents={} epoch {}/{} loss={:.5f} ({:.0f}s)".format(
                method, scenario, info, agent_idxs, epoch + 1, args.epochs, tot / nb, time.time() - t0), flush=True)

    torch.save({"model_state_dict": model.state_dict(), "method": method, "scenario": scenario,
                "info": info, "agents": list(agent_idxs), "cond_dim": cdim, "action_dim": act_dim,
                "cond_spec": ({"kind": "id_hist", "n_ids": 2, "n_hist": args.n_hist} if args.hist else spec_all[0]),
                "schedule": args.schedule, "hist": bool(args.hist), "key": args.key,
                "level": args.level, "level_layout": lay, "enc_in": enc_in, "enc_d": enc_d,
                "norm": {int(i): stats[i] for i in agent_idxs},
                "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs, "seed": seed,
                "hidden": C.HIDDEN, "diffusion_steps": args.steps, "beta_start": C.BETA_START,
                "beta_end": C.BETA_END, "max_std": args.max_std if method == "c2" else None,
                "corruption": {"bc": "none", "c1": "forward-diffusion (matched)",
                               "c2": "clip(a+sigma*eps,-1,1), sigma~U(0,max_std) [knows noise model]",
                               "den": "forward-diffusion + eps-prediction"}[method],
                "n_samples": int(N)}, out_path)
    print("saved", out_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(C.SCENARIOS))
    p.add_argument("--info", default="full", choices=["hidden", "full"])
    p.add_argument("--method", required=True, choices=C.METHODS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-std", type=float, default=3.0)
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--level", default="full", help="information level (see info_levels.py); full = unchanged path")
    p.add_argument("--role", default="all", choices=["all", "speaker", "listener", "good", "shared"])
    p.add_argument("--hist", action="store_true", help="speaker input = ID + last --n-hist noisy messages")
    p.add_argument("--n-hist", type=int, default=5)
    p.add_argument("--model-dir", default=None, help="override output dir (smoke tests)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.model_dir:
        C.MODEL_DIR = args.model_dir
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    # checkpoint key: <method>[_s1 if cosine & no history][_hist]; defaults reproduce the earlier file names
    key = args.method
    if args.schedule == "cosine" and not args.hist:
        key += "_s1"
    if args.hist:
        key += "_hist"
    key += IL.SUFFIX[args.level]
    args.key = key
    data = C.load_agent_data(args.scenario)
    if args.role != "all":
        jobs = {"speaker": [([0], args.info)], "listener": [([1], "any")],
                "good": [([1], "full")] if args.scenario == "simple_push" else [([1, 2], "full")],
                "shared": [([0, 1, 2], "full")]}[args.role]
    elif args.hist:
        jobs = [([0], args.info)]
    elif args.scenario == "simple_speaker_listener":
        jobs = [([0], args.info), ([1], "any")]
    elif args.scenario == "simple_push":
        jobs = [([0], "full"), ([1], "full")]          # adversary and good agent: separate networks
    elif args.scenario == "simple_adversary":
        jobs = [([0], "full"), ([1, 2], "full")]      # adversary alone; the two good agents share one network
    else:
        jobs = [([0, 1, 2], "full")]
    for agent_idxs, info in jobs:
        tag, info_f = C.agent_tag_and_info(args.scenario, agent_idxs[0], args.info)
        path = C.model_path(key, args.scenario, info_f, tag, args.seed)
        if os.path.exists(path) and not args.force:
            print("exists, skipping", path, flush=True)
            continue
        _train_one(args.scenario, args.info, agent_idxs, args.method, data, args.seed, args, path)


if __name__ == "__main__":
    main()
