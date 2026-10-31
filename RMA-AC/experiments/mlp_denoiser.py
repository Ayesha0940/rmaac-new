"""
mlp_denoiser.py
===============
One-pass supervised reconstruction denoiser for joint actions: an MLP mapping
(noisy joint action, joint observation) -> clean joint action. Trained on the
same clean rollouts as the DDPM denoiser in train.py, with the corruption
generated on the fly: zero-mean Gaussian, sigma ~ U[0, sigma_max] per sample,
then clipped to [act_low, act_high] as `ActionAttack._clip` does at test time.

--cond-mode selects the state paired with each training action: "current"
(that step's state) or "initial" (the episode's first state for every step, the
same pairing train_diffusion uses). Test time always passes the current state.

Only depends on numpy/torch, so training doesn't need a TF session.

Usage:
    python mlp_denoiser.py --data-path diffusion_data/simple_spread_m3ddpg.npz \\
        --model-path mlp_denoiser_models/simple_spread_m3ddpg.pt
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPORT_SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0]


class MLPDenoiser(nn.Module):
    """x_noisy:[B,Da], cond:[B,Ds] -> x_clean:[B,Da], all z-scored."""
    def __init__(self, action_dim, cond_dim, hidden_dim=256, n_hidden=3):
        super().__init__()
        layers, in_dim = [], action_dim + cond_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_noisy, cond):
        return self.net(torch.cat([x_noisy, cond], dim=-1))


def corrupt(actions, sigma, act_low, act_high):
    """Add zero-mean Gaussian noise; sigma is a scalar or a [B,1] tensor."""
    return torch.clamp(actions + sigma * torch.randn_like(actions), act_low, act_high)


def _load_episodes(data_path, val_frac, seed, cond_mode):
    """Split by episode, then flatten, so one episode never spans both sets.
    Also returns the current states of both sets: test time passes current
    states, so state normalisation must cover them. In "initial" mode, first-step
    stats alone would have std ~0 on dims such as velocity, which start at 0."""
    data = np.load(data_path)
    states, actions = data["states"], data["actions"]
    cond = states if cond_mode == "current" else np.repeat(states[:, :1, :], states.shape[1], axis=1)
    N = states.shape[0]
    perm = np.random.RandomState(seed).permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def flat(x, idx):
        return torch.from_numpy(x[idx].reshape(-1, x.shape[-1])).float()

    return (flat(cond, train_idx), flat(actions, train_idx),
            flat(cond, val_idx), flat(actions, val_idx),
            flat(states, train_idx), flat(states, val_idx))


def _predict(model, noisy, states, stats):
    x = (noisy - stats["act_mean"]) / stats["act_std"]
    s = (states - stats["state_mean"]) / stats["state_std"]
    return model(x, s) * stats["act_std"] + stats["act_mean"]


@torch.no_grad()
def validation_report(model, states, actions, stats, act_low, act_high):
    """Per-sigma MSE of the raw noisy input and of the MLP, plus sensitivity:
    the RMS output change between two noise draws over the RMS input change.
    A ratio near 0 means the model ignores the action, acting as a policy clone."""
    report = []
    for sigma in REPORT_SIGMAS:
        noisy = corrupt(actions, sigma, act_low, act_high)
        noisy2 = corrupt(actions, sigma, act_low, act_high)
        pred, pred2 = _predict(model, noisy, states, stats), _predict(model, noisy2, states, stats)
        in_change = F.mse_loss(noisy, noisy2).sqrt().item()
        report.append({"sigma": sigma,
                       "noisy": F.mse_loss(noisy, actions).item(),
                       "mlp": F.mse_loss(pred, actions).item(),
                       "sensitivity": F.mse_loss(pred, pred2).sqrt().item() / in_change if in_change > 0 else 0.0})
    return report


def train_mlp_denoiser(args):
    torch.manual_seed(args.seed)
    states, actions, val_states, val_actions, train_current, val_current = _load_episodes(
        args.data_path, args.val_frac, args.seed, args.cond_mode)
    print("[MLP] {} train / {} val steps, state_dim={}, action_dim={}".format(
        states.shape[0], val_states.shape[0], states.shape[1], actions.shape[1]))

    stats = {"act_mean": actions.mean(dim=0, keepdim=True),
             "act_std": actions.std(dim=0, keepdim=True) + 1e-6,
             "state_mean": train_current.mean(dim=0, keepdim=True),
             "state_std": train_current.std(dim=0, keepdim=True) + 1e-6}

    model = MLPDenoiser(actions.shape[1], states.shape[1], args.hidden_dim, args.n_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # fixed corruption for model selection, so val losses are comparable across epochs
    val_sigma = args.sigma_max * torch.rand(val_actions.shape[0], 1)
    val_noisy = corrupt(val_actions, val_sigma, args.act_low, args.act_high)

    N = states.shape[0]
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for b in range(max(1, N // args.batch_size)):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            sigma = args.sigma_max * torch.rand(idx.shape[0], 1)
            noisy = corrupt(actions[idx], sigma, args.act_low, args.act_high)
            pred = _predict(model, noisy, states[idx], stats)
            # loss in z-scored units, so no action dim dominates
            loss = F.mse_loss(pred / stats["act_std"], actions[idx] / stats["act_std"])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * idx.shape[0]

        with torch.no_grad():
            val_pred = _predict(model, val_noisy, val_states, stats)
            val_loss = F.mse_loss(val_pred / stats["act_std"], val_actions / stats["act_std"]).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("[MLP] Epoch {}/{} train={:.5f} val={:.5f} best_val={:.5f}".format(
                epoch + 1, args.epochs, epoch_loss / N, val_loss, best_val))

    model.load_state_dict(best_state)
    report = {}
    for name, cond in (("training pairing", val_states), ("current state", val_current)):
        report[name] = validation_report(model, cond, val_actions, stats, args.act_low, args.act_high)
        print("[MLP] Validation with {} (MSE in action units):".format(name))
        print("  sigma    noisy      mlp  sensitivity")
        for r in report[name]:
            print("  {:5.1f} {:8.4f} {:8.4f} {:12.3f}".format(r["sigma"], r["noisy"], r["mlp"], r["sensitivity"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "action_dim": actions.shape[1], "cond_dim": states.shape[1],
                "hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden,
                "stats": stats, "best_val": best_val, "val_report": report,
                "train_args": vars(args)},
               args.model_path)
    print("[MLP] Saved model to {}".format(args.model_path))


class LoadedMLPDenoiser:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = MLPDenoiser(ckpt["action_dim"], ckpt["cond_dim"], ckpt["hidden_dim"], ckpt["n_hidden"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.stats = ckpt["stats"]

    @torch.no_grad()
    def denoise(self, noisy_vec, state_vec):
        noisy = torch.from_numpy(np.asarray(noisy_vec, dtype=np.float32)).unsqueeze(0)
        state = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0)
        return _predict(self.model, noisy, state, self.stats)[0].numpy()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="clean (states, actions) .npz")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--cond-mode", type=str, default="current", choices=["current", "initial"])
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--sigma-max", type=float, default=3.0)
    p.add_argument("--act-low", type=float, default=-1.0)
    p.add_argument("--act-high", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if os.path.exists(args.model_path):
        print("[MLP] {} already exists -- skipping.".format(args.model_path))
        raise SystemExit(0)
    train_mlp_denoiser(args)
