"""Prints S0 (linear) vs S1 (cosine) maximum noise levels, the injected noise in normalized units
(after the clip projection), t_match per agent / noise level, and writes results/s1_tmatch.csv."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STDS = [0.0, 0.4, 1.0, 2.0, 3.0]


def main():
    _, _, ab0 = C.schedule("linear")
    _, _, ab1 = C.schedule("cosine")
    lv0, lv1 = C.noise_levels(ab0.numpy()), C.noise_levels(ab1.numpy())
    print("S0 linear : abar_99 = {:.4g}  max noise level sqrt((1-abar)/abar) = {:.3f}".format(float(ab0[-1]), lv0[-1]))
    print("S1 cosine : abar_99 = {:.4g}  max noise level = {:.3f}".format(float(ab1[-1]), lv1[-1]))
    print("S1 levels at t = 0, 10, 25, 50, 75, 90, 99:", ["{:.3g}".format(lv1[t]) for t in (0, 10, 25, 50, 75, 90, 99)])
    rows = []
    for scen in C.SCENARIOS:
        data = C.load_agent_data(scen)
        worst = 0.0
        print("\n== {} (normalized units, after clip projection)".format(scen))
        for i, (o, a) in enumerate(data):
            _, astd = C.norm_stats(a)
            for s in STDS:
                eff = C.effective_noise_norm(a, astd[0], s)
                worst = max(worst, eff)
                t1 = C.t_match(eff, lv1)
                rows.append([scen, i, s, round(eff, 4), t1, round(float(lv1[t1]), 4)])
                print("  agent {} sigma {:.1f}: effective std {:.3f} (unclipped sigma/act_std = {:.2f}) -> t_match {:3d} (level {:.3f})".format(
                    i, s, eff, s / astd[0].mean(), t1, lv1[t1]))
        need = 3.0 * worst
        print("  largest effective injected std = {:.3f}; S1 needs max level >= {:.3f}: S0 {} (max {:.2f}), S1 {} (max {:.1f})".format(
            worst, need, "OK" if lv0[-1] >= need else "FAILS", lv0[-1], "OK" if lv1[-1] >= need else "FAILS", lv1[-1]))
        assert lv1[-1] >= need
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    with open(os.path.join(C.RESULT_DIR, "s1_tmatch.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "agent", "sigma", "effective_std_norm", "t_match_S1", "level_at_t_match"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
