"""
train_bc.py
===========
State-only behavior cloning (BC) baseline for the diffusion-denoiser comparison.

Deterministic, centralized MLP with the same backbone size as the denoiser's
TrajectoryDiffusion (hidden 256): joint state s -> joint action a (H = 1).
Trained on the SAME dataset D the H=25 denoiser was trained on
(diffusion_data/<scenario>_m3ddpg.npz), but every one of the 25 steps of each
episode is used as an independent (state_t, action_t) pair, since BC is queried
with the current state at every env step. No corruption; loss is MSE on
z-scored actions (per-dim mean/std over all pairs, + 1e-6, same as train_diffusion).

Torch-only (no TF / env needed).

Usage:
    python train_bc.py --scenario simple_tag
    python train_bc.py --scenario simple_spread --epochs 500
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


class BCPolicy(nn.Module):
    """state -> 256 -> 256 -> joint action (normalized space)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

    def forward(self, s):
        return self.net(s)


def load_bc_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = BCPolicy(ckpt["cond_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["act_mean"], ckpt["act_std"]


def train_bc(data_path, model_path, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    data = np.load(data_path)
    states, actions = data["states"], data["actions"]
    N, H, Ds = states.shape
    Da = actions.shape[2]
    s = torch.from_numpy(states.reshape(N * H, Ds)).float()
    a = torch.from_numpy(actions.reshape(N * H, Da)).float()
    print("[BC] {} episodes x {} steps = {} pairs; state_dim={} action_dim={}".format(
        N, H, N * H, Ds, Da))

    act_mean = a.mean(dim=0, keepdim=True)
    act_std = a.std(dim=0, keepdim=True) + 1e-6
    a = (a - act_mean) / act_std

    model = BCPolicy(Ds, Da)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    M = s.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(M)
        s, a = s[perm], a[perm]
        epoch_loss = 0.0
        for b in range(max(1, M // batch_size)):
            sl = slice(b * batch_size, min(M, (b + 1) * batch_size))
            loss = F.mse_loss(model(s[sl]), a[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * (sl.stop - sl.start)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("[BC] Epoch {}/{} loss={:.6f}".format(epoch + 1, epochs, epoch_loss / M))

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "action_dim": Da, "cond_dim": Ds,
                "act_mean": act_mean, "act_std": act_std}, model_path)
    print("[BC] Saved model to {}".format(model_path))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--data-path", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing model file")
    a = p.parse_args()

    data_path = a.data_path or os.path.join(HERE, "diffusion_data", "{}_m3ddpg.npz".format(a.scenario))
    model_path = a.model_path or os.path.join(HERE, "bc_models", "{}_m3ddpg_bc.pt".format(a.scenario))
    if os.path.exists(model_path) and not a.force:
        raise SystemExit("{} already exists; pass --force to overwrite".format(model_path))
    train_bc(data_path, model_path, a.epochs, a.batch_size, a.lr, a.seed)
