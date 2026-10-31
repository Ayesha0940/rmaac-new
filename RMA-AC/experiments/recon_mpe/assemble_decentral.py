"""Assemble eval_decentral parts into the original noise_sweeps CSV convention under <out-root> (default
noise_sweeps/decentralized). reward_with_diff_t{t} = best of one-step / chain at t; best_reward_with_diffusion = best of the
6 variants (selected after the fact); pct_inc_vs_no_diffusion = (best - noisy) / |noisy| * 100; values rounded to 2 decimals.
Usage: python assemble_decentral.py --parts <dir> [--out-root ../../../noise_sweeps/decentralized] [--seed 0]"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ["simple_adversary", "simple_push", "simple_speaker_listener", "simple_spread"]
TS = [10, 20, 40]


def r2(x):
    return "{:.2f}".format(float(x))


def den_cols(cell):
    out, best, bvar = {}, None, ""
    for t in TS:
        vals = {}
        for mode in ("1step", "chain"):
            z = cell[(cell.method == "den") & (cell["mode"] == mode) & (cell.t.astype(float) == t)]
            if len(z) != 1:
                raise SystemExit("missing denoiser row t={} {}".format(t, mode))
            vals[mode] = float(z.reward_mean.iloc[0])
            out["reward_with_diff_t{}_{}".format(t, mode)] = vals[mode]
            if best is None or vals[mode] > best:
                best, bvar = vals[mode], "{}@{}".format(mode, t)
        out["reward_with_diff_t{}".format(t)] = max(vals.values())
    return out, best, bvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out-root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..", "noise_sweeps", "decentralized")))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.parts, "*.csv"))])
    d["mode"] = d["mode"].fillna("")
    if d.reward_mean.isnull().any():
        raise SystemExit("NaN reward")
    per_mode = ["reward_with_diff_t{}_{}".format(t, m) for t in TS for m in ("1step", "chain")]
    written = []
    for sc in SCEN:
        x = d[d.scenario == sc]
        clean = x[(x.fault == "clean")]
        if len(clean) != 1:
            raise SystemExit("missing clean run for " + sc)
        jc = float(clean.reward_mean.iloc[0])
        for fault, xcol, sub, fname in (("delay", "delay_k", "delay_sweeps", "{}__m3ddpgbest_delay_sweep.csv"),
                                        ("stuck", "sp_prob", "actuation_fault", "{}__m3ddpgbest_stuck_at_sweep.csv")):
            rows = []
            for lv in sorted(x[x.fault == fault].level.unique()):
                cell = x[(x.fault == fault) & (x.level == lv)]
                cols, best, bvar = den_cols(cell)
                noisy = float(cell[cell.method == "noisy"].reward_mean.iloc[0])
                row = {xcol: (int(lv) if fault == "delay" else r2(lv)), "reward_noisy": r2(noisy)}
                row.update({"reward_with_diff_t{}".format(t): r2(cols["reward_with_diff_t{}".format(t)]) for t in TS})
                row.update({"best_reward_with_diffusion": r2(best),
                            "pct_inc_vs_no_diffusion": r2((best - noisy) / abs(noisy) * 100.0 if noisy else 0.0),
                            "best_variant": bvar})
                row.update({k: r2(cols[k]) for k in per_mode})
                rows.append(row)
            hdr = [xcol, "reward_noisy"] + ["reward_with_diff_t{}".format(t) for t in TS] + \
                  ["best_reward_with_diffusion", "pct_inc_vs_no_diffusion", "best_variant"] + per_mode
            path = os.path.join(a.out_root, sub, "seed{}".format(a.seed), fname.format(sc))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame(rows, columns=hdr).to_csv(path, index=False)
            written.append(path)
        for mu in (-1.0, 0.0, 1.0):
            fault = "gauss" if mu == 0.0 else "biased"
            g = x[(x.fault == fault) & (x.mu.astype(float) == mu)]
            rows = []
            for sd in sorted(g.level.unique()):
                cell = g[g.level == sd]
                cols, best, bvar = den_cols(cell)
                noisy = float(cell[cell.method == "noisy"].reward_mean.iloc[0])
                row = {"noise_mu": r2(mu), "action_noise_std": r2(sd), "reward_no_noise": r2(jc), "reward_noise_no_diffusion": r2(noisy)}
                row.update({"reward_with_diff_t{}".format(t): r2(cols["reward_with_diff_t{}".format(t)]) for t in TS})
                row.update({"best_reward_with_diffusion": r2(best),
                            "pct_inc_vs_no_diffusion": r2((best - noisy) / abs(noisy) * 100.0 if noisy else 0.0),
                            "best_variant": bvar})
                row.update({k: r2(cols[k]) for k in per_mode})
                rows.append(row)
            hdr = ["noise_mu", "action_noise_std", "reward_no_noise", "reward_noise_no_diffusion"] + \
                  ["reward_with_diff_t{}".format(t) for t in TS] + ["best_reward_with_diffusion", "pct_inc_vs_no_diffusion", "best_variant"] + per_mode
            path = os.path.join(a.out_root, "guassian_noise", "seed{}".format(a.seed),
                                "{}__m3ddpgbest_mu{:.1f}_actstd_tstart_sweep.csv".format(sc, mu))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame(rows, columns=hdr).to_csv(path, index=False)
            written.append(path)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
