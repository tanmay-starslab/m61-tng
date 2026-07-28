#!/usr/bin/env python3
"""Accumulate the SPECTRUM-based detection rate vs velocity for one SID.

For every sightline and every COS-G130M line, we take the LSF mock flux on the ISM-subtracted
velocity axis  Δv = [-(c(λ/λ0-1)) - v_sys] - v_ISM_model  and mark each Δv bin "detected" if
any pixel in it shows >DET_DEPTH absorption (flux < 1-DET_DEPTH). Detection rate = fraction of
sightlines detected per bin. Uses the real spectra (not pyGad Voigt fits, not gas columns), so
the multiple Si II lines (1190/1193/1260) appear as separate curves.

Usage: python fig10_accumulate.py <sid>
Output: diagnostics_v2/detection_rate/acc_sid<sid>.npz  (counts[line,bin], n_los[line], edges)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
from pm_general import C_KMS  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402

# args: <sid> [v_ism_column] [master_csv] [out_subdir]
VCOL = sys.argv[2] if len(sys.argv) > 2 else "v_ism_model"
MASTER = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
    "/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/" +
           (sys.argv[4] if len(sys.argv) > 4 else "diagnostics_v2/detection_rate"))
# (spectrum key, rest wavelength) for all nine lines in the band
LINES = [("H_I_1216", 1215.6701), ("O_I_1302", 1302.1685), ("C_II_1335", 1334.5323),
         ("Si_II_1190", 1190.4158), ("Si_II_1193", 1193.2897), ("Si_II_1260", 1260.4221),
         ("Si_III_1206", 1206.500), ("Si_IV_1403", 1402.770), ("N_V_1239", 1238.821)]
DET_DEPTH = 0.10                      # >10% absorption counts as a detection
EDGES = np.arange(-500, 501, 10.0)    # Δv bins [km/s]
NB = len(EDGES) - 1


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MASTER); m = m[m.sid == sid].set_index(["mode", "alpha_deg"])
    counts = np.zeros((len(LINES), NB), float)
    n_los = np.zeros(len(LINES), float)
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for ag in sorted(base[f"mode={mode}"].keys(), key=lambda k: int(k.split("=")[1])):
                alpha = int(ag.split("=")[1])
                if (mode, alpha) not in m.index:
                    continue
                row = m.loc[(mode, alpha)]
                v_ism = float(row[VCOL]); v_sys = float(row.v_sys)
                if not np.isfinite(v_ism):
                    continue
                ray = base[f"mode={mode}"][ag][list(base[f"mode={mode}"][ag].keys())[0]]
                for j, (key, rest) in enumerate(LINES):
                    g = ray.get(f"spectrum_by_line/{key}/lsf")
                    if g is None:
                        continue
                    n_los[j] += 1
                    lam = g["lambda_A"][()]; fl = g["flux"][()]
                    dv = (-(C_KMS * (lam / rest - 1.0)) - v_sys) - v_ism
                    absorbed = fl < (1.0 - DET_DEPTH)
                    if absorbed.any():
                        b = np.digitize(dv[absorbed], EDGES) - 1
                        b = b[(b >= 0) & (b < NB)]
                        counts[j, np.unique(b)] += 1
    np.savez(OUT / f"acc_sid{sid}.npz", counts=counts, n_los=n_los, edges=EDGES,
             lines=np.array([k for k, _ in LINES]))
    print(f"[SID {sid}] accumulated {int(n_los.max())} sightlines/line -> {OUT}/acc_sid{sid}.npz")


if __name__ == "__main__":
    main()
