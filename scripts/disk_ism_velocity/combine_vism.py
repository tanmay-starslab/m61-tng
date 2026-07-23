#!/usr/bin/env python3
"""Concatenate the per-SID Stage-B v_ISM tables into one master (14,400 rows)."""
import glob
from pathlib import Path
import pandas as pd

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables")


def main():
    files = sorted(glob.glob(str(OUT / "vism_sid*.csv")))
    if not files:
        print("no per-SID tables found"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    p = OUT / "vism_master_all_sightlines.csv"
    df.to_csv(p, index=False)
    nsid = df["sid"].nunique() if "sid" in df else 0
    nprim = int(df["v_ism_primary"].notna().sum()) if "v_ism_primary" in df else 0
    ndirect = int((df.get("v_mode") == "direct").sum()) if "v_mode" in df else 0
    print(f"master: {len(df)} rows from {nsid} SIDs -> {p}")
    print(f"  {nprim} with v_ISM ; {ndirect} direct / {nprim - ndirect} R95-edge")


if __name__ == "__main__":
    main()
