"""Assemble the reduced-observation screen (decentralized protocol, M3DDPG, seed 0) and summarize it vs BC.
Inputs: <parts>/*.csv (den_red / bc_red from eval_decentral_reduced.py) and results/decentral/parts (noisy, clean and the
full-observation denoiser from the decentralized run, same cells / seed / fault stream).
Outputs: <out-root>/<scenario>__<role>__<level>/{delay_sweeps,actuation_fault,guassian_noise}/seed0/*.csv (decentralized
convention + extra columns) and <out-root>/summary.txt.
Labels per faulted cell (SE = episode std / sqrt(n), combined in quadrature; reward summed over agents):
  RESTORES = reduced denoiser beats noisy by > 2 SE;  BEATS-BC = reduced denoiser beats BC (same reduced input) by > 2 SE.
Usage: python report_reduced.py --parts <dir> [--decentral-parts results/decentral/parts] [--out-root ...]"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [("simple_speaker_listener", "speaker", "removed", "Cooperative Communication: speaker goal colour removed"),
          ("simple_speaker_listener", "listener", "removed", "Cooperative Communication: listener message removed"),
          ("simple_adversary", "good", "removed", "Physical Deception: good agents' goal position removed"),
          ("simple_push", "good", "goal-hidden", "Keep-away: good agent's goal position + own colour removed"),
          ("simple_spread", "shared", "others-removed", "Cooperative Navigation: other agents' positions (+ empty comm) removed")]
CELLS = [("delay", k, 0.0) for k in (0.0, 1.0, 2.0, 4.0)] + [("stuck", p, 0.0) for p in (0.0, 0.1, 0.2, 0.4)] + \
        [("biased" if mu else "gauss", sd, mu) for mu in (-1.0, 0.0, 1.0) for sd in (0.0, 1.0, 2.0, 3.0)]
NOFAULT = {("delay", 0.0, 0.0), ("stuck", 0.0, 0.0), ("gauss", 0.0, 0.0)}


def best(df):
    i = df.reward_mean.values.argmax()
    r = df.iloc[i]
    return r, "{}@{}".format(r["mode"], int(r.t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--decentral-parts", default=os.path.join(HERE, "results", "decentral", "parts"))
    ap.add_argument("--out-root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..", "noise_sweeps", "decentralized_reduced")))
    a = ap.parse_args()
    red = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.parts, "*.csv"))])
    dec = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.decentral_parts, "*.csv"))])
    for d in (red, dec):
        d["mode"] = d["mode"].fillna("")
        if d.reward_mean.isnull().any():
            raise SystemExit("NaN reward")
    out = []
    P = out.append
    P("Reduced-observation screen vs BC -- decentralized protocol (all agents faulted and corrected), M3DDPG, seed 0, 800 episodes.")
    P("Reward summed over agents. Denoiser / BC = best of t in {10,20,40} x {one-step, chain} per cell (selected after the fact);")
    P("in the BC run the reduced agent(s) act with BC (no fault) and every other agent is corrected by its full denoiser.")
    P("RESTORES: reduced denoiser > noisy by 2 SE. BEATS-BC: reduced denoiser > BC by 2 SE. Counts over the 17 faulted cells.\n")
    summary = []
    for scen, role, lvl, name in LEVELS:
        x = red[(red.scenario == scen) & (red.role == role) & (red.info_level == lvl)]
        if len(x) == 0:
            continue
        clean = float(dec[(dec.scenario == scen) & (dec.fault == "clean")].reward_mean.iloc[0])
        rows, nrest, nbeat, nfault, hashbad = [], 0, 0, 0, 0
        for f, lv, mu in CELLS:
            c = x[(x.fault == f) & (np.isclose(x.level, lv)) & (np.isclose(x.mu, mu))]
            dd = dec[(dec.scenario == scen) & (dec.fault == f) & (np.isclose(dec.level, lv)) & (np.isclose(dec.mu, mu))]
            if len(c) != 12 or len(dd) != 7:
                raise SystemExit("missing rows: {} {} {} {} {} ({} / {})".format(scen, lvl, f, lv, mu, len(c), len(dd)))
            noisy = dd[dd.method == "noisy"].iloc[0]
            if int(noisy.n_episodes) == int(c.n_episodes.iloc[0]):
                hashbad += int(set(c.start_hash) != {noisy.start_hash} or set(int(z) for z in c.fault_state_hash) != {int(noisy.fault_state_hash)})
            dr, dvar = best(c[c.method == "den_red"])
            br, bvar = best(c[c.method == "bc_red"])
            df_, fvar = best(dd[dd.method == "den"])
            n = float(noisy.n_episodes)
            se = lambda r1, r2: np.hypot(r1.reward_std, r2.reward_std) / np.sqrt(n)
            rest = dr.reward_mean - noisy.reward_mean > 2 * se(dr, noisy)
            beat = dr.reward_mean - br.reward_mean > 2 * se(dr, br)
            faulted = (f, lv, mu) not in NOFAULT
            if faulted:
                nfault += 1
                nrest += int(rest)
                nbeat += int(beat)
            rows.append(dict(fault=f, level=lv, mu=mu, noisy=noisy.reward_mean, bc=br.reward_mean, den_red=dr.reward_mean,
                             den_full=df_.reward_mean, dvar=dvar, bvar=bvar, fvar=fvar, rest=rest, beat=beat, faulted=faulted,
                             good_red=dr.reward_good_mean, good_bc=br.reward_good_mean))
        R = pd.DataFrame(rows)
        # ---- CSVs in the decentralized convention (+ extra columns)
        base = os.path.join(a.out_root, "{}__{}__{}".format(scen, role, lvl))
        def frame(sub, xcol, xs):
            o = pd.DataFrame({xcol: xs.level.values if xcol != "action_noise_std" else xs.level.values})
            if xcol == "action_noise_std":
                o.insert(0, "noise_mu", xs.mu.values)
                o["reward_no_noise"] = clean
                o["reward_noise_no_diffusion"] = xs.noisy.values
            else:
                o["reward_noisy"] = xs.noisy.values
            o["reward_bc_reduced"] = xs.bc.values
            o["best_reward_with_diffusion"] = xs.den_red.values
            o["best_reward_with_diffusion_full_obs"] = xs.den_full.values
            o["best_variant_reduced"] = xs.dvar.values
            o["best_variant_bc"] = xs.bvar.values
            o["restores"] = xs.rest.astype(int).values
            o["beats_bc"] = xs.beat.astype(int).values
            for col in o.columns:
                if o[col].dtype == float:
                    o[col] = o[col].map(lambda z: "{:.2f}".format(z))
            return o
        for sub, f, xcol, fname in (("delay_sweeps", "delay", "delay_k", "{}__m3ddpgbest_delay_sweep.csv"),
                                    ("actuation_fault", "stuck", "sp_prob", "{}__m3ddpgbest_stuck_at_sweep.csv")):
            pth = os.path.join(base, sub, "seed0", fname.format(scen))
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            frame(sub, xcol, R[R.fault == f]).to_csv(pth, index=False)
        for mu in (-1.0, 0.0, 1.0):
            pth = os.path.join(base, "guassian_noise", "seed0", "{}__m3ddpgbest_mu{:.1f}_actstd_tstart_sweep.csv".format(scen, mu))
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            frame("g", "action_noise_std", R[np.isclose(R.mu, mu) & R.fault.isin(["gauss", "biased"])]).to_csv(pth, index=False)
        # ---- summary block
        P("=" * 118)
        P("{}   ({} / {} / {}; clean = {:.2f})".format(name, scen, role, lvl, clean))
        P("=" * 118)
        P("{:<20}{:>11}{:>11}{:>13}{:>13}  {:<11}{:<11}{:>9}{:>9}".format("cell", "noisy", "BC", "den reduced", "den full-obs", "red. var", "BC var", "RESTORES", "BEATS-BC"))
        for _, r in R.iterrows():
            lab = "{} {}{}".format(r.fault, r.level, "" if r.fault not in ("gauss", "biased") else " mu{:+.0f}".format(r.mu))
            P("{:<20}{:>11.2f}{:>11.2f}{:>13.2f}{:>13.2f}  {:<11}{:<11}{:>9}{:>9}".format(
                lab + ("" if r.faulted else " *"), r.noisy, r.bc, r.den_red, r.den_full, r.dvar, r.bvar,
                "yes" if r.rest else "no", "yes" if r.beat else "no"))
        P("(* no fault, not counted)   RESTORES {} / {}   BEATS-BC {} / {}   hash mismatches vs decentralized run: {}".format(
            nrest, nfault, nbeat, nfault, hashbad))
        P("")
        summary.append((name, nrest, nbeat, nfault))
    P("=" * 118)
    P("SUMMARY (17 faulted cells per level)")
    for name, nr, nb, nf in summary:
        P("  {:<72} RESTORES {:>2}/{}   BEATS-BC {:>2}/{}".format(name, nr, nf, nb, nf))
    os.makedirs(a.out_root, exist_ok=True)
    open(os.path.join(a.out_root, "summary.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
