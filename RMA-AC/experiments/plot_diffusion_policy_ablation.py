#!/usr/bin/env python3
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = "*_diffusion_policy_ablation_dp*.csv"
FILE_RE = re.compile(r"^(?P<name>.+)_diffusion_policy_ablation_(?P<dp_label>dp\w+)\.csv$")
DP_LABEL_TEXT = {
    "dpfull": "full chain, t_start=T-1",
}
REQUIRED_COLS = {
    "action_noise_std",
    "reward_noise_no_diffusion",
    "best_reward_with_diffusion",
    "reward_diffusion_policy",
}


def load_curve(csv_path):
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError("Missing columns {} in {}".format(sorted(missing), csv_path))

    df = df.sort_values("action_noise_std")
    x = df["action_noise_std"].astype(float).values
    no_diff = df["reward_noise_no_diffusion"].astype(float).values
    denoiser = df["best_reward_with_diffusion"].astype(float).values
    diff_policy = df["reward_diffusion_policy"].astype(float).values
    return x, no_diff, denoiser, diff_policy


def make_plot(csv_path, out_dir):
    fname = os.path.basename(csv_path)
    m = FILE_RE.match(fname)
    if not m:
        raise ValueError("Unrecognized filename format: {}".format(fname))
    name, dp_label = m.group("name"), m.group("dp_label")
    dp_text = DP_LABEL_TEXT.get(dp_label, dp_label.replace("dpt", "t_start="))

    x, no_diff, denoiser, diff_policy = load_curve(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(x, no_diff, marker="o", linewidth=2, label="No diffusion")
    plt.plot(x, denoiser, marker="s", linewidth=2, label="Denoiser (best t_start)")
    plt.plot(x, diff_policy, marker="^", linewidth=2, label="Diffusion policy ({})".format(dp_text))

    plt.title("{}: action-processing ablation ({})".format(name, dp_text))
    plt.xlabel("Action noise std (mu=0)")
    plt.ylabel("Average reward")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}_diffusion_policy_ablation_{}.png".format(name, dp_label))
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "plots", "diffusion_policy_ablation")

    csv_paths = sorted(glob.glob(os.path.join(here, PATTERN)))
    if not csv_paths:
        print("No matching CSV files found in {}".format(here))
        return

    for csv_path in csv_paths:
        out_path = make_plot(csv_path, out_dir)
        print("Generated {}".format(out_path))


if __name__ == "__main__":
    main()
