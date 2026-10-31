"""Merge results/il/parts/*.csv into results/il.csv (usage: python merge_il.py <results_dir>)."""
import glob, sys
import pandas as pd
rd = sys.argv[1]
fs = sorted(glob.glob(rd + "/il/parts/*.csv"))
pd.concat([pd.read_csv(f) for f in fs]).to_csv(rd + "/il.csv", index=False)
print("il", len(fs), "parts merged")
