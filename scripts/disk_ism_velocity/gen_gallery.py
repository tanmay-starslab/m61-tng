#!/usr/bin/env python3
"""Validation gallery: ray-ISM diagnostics for representative sightlines of every SID.

Per SID: the 2 direct-cool sightlines with the most cool disk cells (clear ISM cases)
+ 1 median R95-edge sightline (beyond-disk / no-cool-ISM case). Reads the corrected
master table to choose cases, then calls ray_ism_diagnostic.diagnose for each.

Usage: python gen_gallery.py
"""
import os, sys
from pathlib import Path
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import ray_ism_diagnostic as R  # noqa: E402

MASTER = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")


def main():
    d = pd.read_csv(MASTER)
    cases = []
    for sid in sorted(d.sid.unique()):
        s = d[d.sid == sid]
        dc = s[s.v_mode == "direct_cool"].sort_values("n_cool_disk_cells", ascending=False)
        for _, r in dc.head(2).iterrows():
            cases.append((int(sid), r["mode"], int(r["alpha_deg"])))
        edge = s[s.v_mode == "R95-edge"]
        if len(edge):
            r = edge.iloc[len(edge) // 2]
            cases.append((int(sid), r["mode"], int(r["alpha_deg"])))
    print(f"{len(cases)} gallery cases across {d.sid.nunique()} SIDs")
    rows = []
    for sid, mode, alpha in cases:
        try:
            rows.append(R.diagnose(sid, mode, alpha))
        except Exception as e:
            print(f"FAIL {sid} {mode} {alpha}: {type(e).__name__}: {e}")
    if rows:
        pd.DataFrame(rows).to_csv(R.OUT / "gallery_summary.csv", index=False)
        print(f"\ngallery_summary.csv: {len(rows)} rows -> {R.OUT}")


if __name__ == "__main__":
    main()
