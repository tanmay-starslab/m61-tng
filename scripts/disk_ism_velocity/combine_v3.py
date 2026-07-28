#!/usr/bin/env python3
"""Concatenate per-SID v3 tables + merge v2 model for comparison -> vism_master_v3.csv."""
import glob
from pathlib import Path
import pandas as pd
D = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3")
V2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(D/"vism_v3_sid*.csv")))], ignore_index=True)
v2 = pd.read_csv(V2)[["sid","mode","alpha_deg","v_ism_model"]].rename(columns={"v_ism_model":"v_ism_v2"})
df = df.merge(v2, on=["sid","mode","alpha_deg"], how="left")
p = D/"vism_master_v3.csv"; df.to_csv(p, index=False)
print(f"{len(df)} rows, {df.sid.nunique()} SIDs -> {p}")
