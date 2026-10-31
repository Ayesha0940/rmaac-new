"""Aggregate noise_sweeps/bc/clean/seed*/<scenario>_clean.csv (mean +- std over
seeds) into bc_clean_reward_table.csv: BC clean reward next to each base policy's."""
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for scen_path in sorted(glob.glob(os.path.join(HERE, "clean", "seed*", "*_clean.csv"))):
    scen = re.sub(r"_clean\.csv$", "", os.path.basename(scen_path))
    df = pd.read_csv(scen_path)
    df.insert(0, "scenario", scen)
    df.insert(1, "seed", os.path.basename(os.path.dirname(scen_path)))
    rows.append(df)
all_df = pd.concat(rows)
g = all_df.drop(columns="seed").groupby("scenario")
out = g.mean().round(2).add_suffix("_mean").join(g.std().round(2).add_suffix("_std"))
out["n_seeds"] = g.size()
out.to_csv(os.path.join(HERE, "bc_clean_reward_table.csv"))
print(out.T.to_string())
