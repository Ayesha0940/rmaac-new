"""Per-task: S0 / S1 maximum noise level vs the largest injected noise (normalized units, after the clip projection),
and t_match per agent / fault level / schedule. Writes results/pa_tmatch.csv."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TASKS = ["simple_speaker_listener", "simple_adversary", "simple_spread"]
FAULTS = [("gauss", 0.0), ("gauss", 1.0), ("gauss", 3.0), ("biased", 1.0), ("delay", 3), ("delay", 8),
          ("stuck", 0.5), ("stuck", 1.0)]


def table(task):
    d = np.load(os.path.join(C.DATA_DIR, "{}_m3ddpg.npz".format(task)))
    A = d["actions"]; cfg = C.SCENARIOS[task]; ao = np.cumsum([0] + cfg["act_dims"])
    out = {}
    for i in range(len(cfg["act_dims"])):
        a = A[:, :, ao[i]:ao[i + 1]]
        _, astd = C.norm_stats(a.reshape(-1, a.shape[2]))
        for f, l in FAULTS:
            out[(i, f, l)] = C.fault_error_norm(a, f, l, astd[0])
    return out


def main():
    ab0, ab1 = C.schedule("linear")[2].numpy(), C.schedule("cosine")[2].numpy()
    lv = {"S0": C.noise_levels(ab0), "S1": C.noise_levels(ab1)}
    rows = []
    for task in TASKS:
        eff = table(task)
        worst = max(eff.values())
        need = 3 * worst
        print("\n== {}: largest injected error {:.3f} (normalized, after clip) -> need max level >= {:.2f}".format(task, worst, need))
        print("   S0 linear max level {:.3f} -> {}   |   S1 cosine max level {:.1f} -> {}".format(
            lv["S0"][-1], "OK" if lv["S0"][-1] >= need else "FAILS", lv["S1"][-1], "OK" if lv["S1"][-1] >= need else "FAILS"))
        assert lv["S1"][-1] >= need
        n_ag = len(C.SCENARIOS[task]["act_dims"])
        print("   {:<14}".format("fault/level") + "".join("  agent{} err  t_S0  t_S1".format(i) for i in range(n_ag)))
        for f, l in FAULTS:
            line = "   {:<14}".format("{} {}".format(f, l))
            for i in range(n_ag):
                e = eff[(i, f, l)]
                t0, t1 = C.t_match(e, lv["S0"]), C.t_match(e, lv["S1"])
                line += "  {:9.3f}  {:4d}  {:4d}".format(e, t0, t1)
                rows.append([task, i, f, l, round(e, 4), t0, round(float(lv["S0"][t0]), 4), t1, round(float(lv["S1"][t1]), 4)])
            print(line)
    with open(os.path.join(C.RESULT_DIR, "pa_tmatch.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "agent", "fault", "level", "injected_error_norm", "t_match_S0", "level_S0", "t_match_S1", "level_S1"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
