#!/usr/bin/env python3
"""Derive H=1 diffusion training data by slicing the existing H=25
<scenario>_m3ddpg.npz rollouts down to the first timestep. Read-only on
the source file; always writes to a new *_H1.npz path."""
import argparse
import os
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="existing H=25 .npz (read-only)")
    p.add_argument("--dst", required=True, help="new H=1 .npz to create")
    args = p.parse_args()

    if os.path.exists(args.dst):
        print("[make_h1] {} already exists — skipping.".format(args.dst))
        raise SystemExit(0)

    data = np.load(args.src)
    states, actions = data["states"], data["actions"]
    assert states.shape[1] >= 1 and actions.shape[1] >= 1, "source data has no timesteps"

    states_h1 = states[:, :1, :].copy()
    actions_h1 = actions[:, :1, :].copy()

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    np.savez(args.dst, states=states_h1, actions=actions_h1)
    print("[make_h1] Wrote {} trajectories (H=1) from {} -> {}".format(
        states_h1.shape[0], args.src, args.dst))
