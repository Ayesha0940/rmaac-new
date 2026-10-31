"""Assemble eval_decentral_c1 parts into the noise_sweeps CSV convention under <out-root> (default noise_sweeps/decentralized/c1).
Columns: reward_noisy / reward_noise_no_diffusion, reward_c1_full_obs, reward_c1_reduced_obs (+ per-agent action MSE); 2 decimals.
Fails if a cell is missing, a reward is NaN, or the noisy reward / hashes differ from the decentralized run (--ref-parts).
Usage: python assemble_decentral_c1.py --parts <dir> [--ref-parts results/decentral/parts] [--out-root ...] [--seed 0]"""
import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]
KEY = ["scenario", "fault", "level", "mu"]


def r2(x):
    return "{:.2f}".format(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--ref-parts", default=os.path.join(HERE, "results", "decentral", "parts"))
    ap.add_argument("--out-root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..", "noise_sweeps", "decentralized", "c1")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ref-check", action="store_true", help="dry runs with fewer episodes only")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.parts, "*.csv"))])
    if d.reward_mean.isnull().any():
        raise SystemExit("NaN reward")
    ref = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.ref_parts, "*.csv"))])
    ref = ref[ref.method == "noisy"]
    for df in (d, ref):
        df["level"] = df.level.astype(float); df["mu"] = df.mu.astype(float)
    nz = d[d.method == "noisy"].merge(ref, on=KEY, suffixes=("", "_ref"))
    if len(nz) != len(d[d.method == "noisy"]):
        raise SystemExit("noisy cell without a decentralized reference")
    bad = nz[(abs(nz.reward_mean - nz.reward_mean_ref) > 1e-9) | (nz.start_hash != nz.start_hash_ref)
             | (nz.fault_state_hash.astype("int64") != nz.fault_state_hash_ref.astype("int64"))]
    if len(bad) and not a.skip_ref_check:
        raise SystemExit("noisy mismatch vs decentralized run:\n" + bad[KEY].to_string())
    print("noisy reward + start/fault hashes match the decentralized run in all {} cells".format(len(nz)))

    def cell_vals(cell):
        g = lambda m, c="reward_mean": cell[cell.method == m][c].iloc[0]
        if set(cell.method) != {"noisy", "c1_full", "c1_red"} or len(cell) != 3:
            raise SystemExit("incomplete cell " + str(cell[KEY].iloc[0].tolist()))
        return dict(noisy=g("noisy"), full=g("c1_full"), red=g("c1_red"),
                    mse_full=g("c1_full", "action_mse_per_agent"), mse_red=g("c1_red", "action_mse_per_agent"))

    tail = ["reward_c1_full_obs", "reward_c1_reduced_obs", "action_mse_c1_full_obs", "action_mse_c1_reduced_obs"]
    for sc in SCEN:
        x = d[d.scenario == sc]
        for fault, xcol, sub, fname in (("delay", "delay_k", "delay_sweeps", "{}__m3ddpgbest_delay_sweep.csv"),
                                        ("stuck", "sp_prob", "actuation_fault", "{}__m3ddpgbest_stuck_at_sweep.csv")):
            rows = []
            for lv in sorted(x[x.fault == fault].level.unique()):
                v = cell_vals(x[(x.fault == fault) & (x.level == lv)])
                rows.append({xcol: int(lv) if fault == "delay" else r2(lv), "reward_noisy": r2(v["noisy"]),
                             "reward_c1_full_obs": r2(v["full"]), "reward_c1_reduced_obs": r2(v["red"]),
                             "action_mse_c1_full_obs": v["mse_full"], "action_mse_c1_reduced_obs": v["mse_red"]})
            path = os.path.join(a.out_root, sub, "seed{}".format(a.seed), fname.format(sc))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame(rows, columns=[xcol, "reward_noisy"] + tail).to_csv(path, index=False)
            print("wrote", path, len(rows), "rows")
        for mu in (-1.0, 0.0, 1.0):
            fault = "gauss" if mu == 0.0 else "biased"
            g = x[(x.fault == fault) & (x.mu == mu)]
            rows = []
            for sd in sorted(g.level.unique()):
                v = cell_vals(g[g.level == sd])
                rows.append({"noise_mu": r2(mu), "action_noise_std": r2(sd), "reward_noise_no_diffusion": r2(v["noisy"]),
                             "reward_c1_full_obs": r2(v["full"]), "reward_c1_reduced_obs": r2(v["red"]),
                             "action_mse_c1_full_obs": v["mse_full"], "action_mse_c1_reduced_obs": v["mse_red"]})
            path = os.path.join(a.out_root, "guassian_noise", "seed{}".format(a.seed),
                                "{}__m3ddpgbest_mu{}_actstd_c1.csv".format(sc, mu))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame(rows, columns=["noise_mu", "action_noise_std", "reward_noise_no_diffusion"] + tail).to_csv(path, index=False)
            print("wrote", path, len(rows), "rows")


if __name__ == "__main__":
    main()
