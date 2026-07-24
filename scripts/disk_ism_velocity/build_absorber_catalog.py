#!/usr/bin/env python3
"""Multi-ion absorbing-gas cell catalog for all 720 sightlines of one SID.

For every ray cell that carries a detectable column in ANY of the six gas ions available
in the combined ray (H I, C II, N V, Si II, Si III, Si IV -- the neutral -> low -> mid ->
high ionization sequence): the corrected LOS velocity v_rest and its offset from the
sightline disk-ISM velocity (dv = v_rest - v_ISM); the column carried by EACH ion; the
full galaxy-frame 3D kinematics
  v_gal = relative_velocity/1e5 - SubhaloVel        (verified: dot(v_gal, los) = v_rest)
  v_r   = v_gal . r_hat   (galactocentric radial; >0 outflow, <0 inflow)
  v_z   = v_gal . n_disk  (perpendicular to the disk plane)
disk-frame + galactocentric position, metallicity (Z/Zsun) and temperature.

Base catalog for the multi-ion / multi-phase abundance, inflow-outflow, metallicity and
ionization figures. Per-ion column-weighting is done downstream.

Usage: python build_absorber_catalog.py <sid>
Output: outputs/disk_ism_velocity/absorber_catalog/absorbers_sid<sid>.parquet
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
from pm_general import get_geometry, get_original_rho, compute_endpoints, CM_PER_KPC  # noqa: E402
import dv_core as dv  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402
import build_sid_rc as B  # noqa: E402

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
MASTER = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
ZSUN = 0.0127

# ion -> ray gas number-density field
IONFIELD = {"HI": "H_p0_number_density", "CII": "C_p1_number_density", "NV": "N_p4_number_density",
            "SiII": "Si_p1_number_density", "SiIII": "Si_p2_number_density", "SiIV": "Si_p3_number_density"}
# per-ion column floor (cm^-2) for keeping a cell (union across ions)
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12}


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MASTER)
    m = m[m.sid == sid].set_index(["mode", "alpha_deg"])
    n_disk = B.disk_normal_from_los(sid)
    e1, e2 = dv.disk_basis(n_disk)
    parts = []
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for ag in sorted(base[f"mode={mode}"].keys(), key=lambda k: int(k.split("=")[1])):
                alpha = int(ag.split("=")[1])
                try:
                    mrow = m.loc[(mode, alpha)]
                except KeyError:
                    continue
                v_ISM = float(mrow.v_ism_primary)
                if not np.isfinite(v_ISM):
                    continue
                g = get_geometry(sid, mode, alpha)
                los, v_sys, sub_vel, center = g["los"], g["v_sys"], g["sub_vel"], g["center_kpc"]
                rho, _, _ = get_original_rho(sid, mode, alpha)
                anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
                grp = base[f"mode={mode}"][ag]
                grid = grp[list(grp.keys())[0]]["original_trident_ray_h5/grid"]
                dl_cm = grid["dl"][()]
                ncell = len(dl_cm)
                Ncol = {ion: (grid[fld][()] * dl_cm if fld in grid else np.zeros(ncell))
                        for ion, fld in IONFIELD.items()}
                sel = np.zeros(ncell, bool)
                for ion in IONFIELD:
                    sel |= Ncol[ion] > FLOOR[ion]
                if not sel.any():
                    continue
                xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC
                vlos = grid["velocity_los"][()] / 1e5
                rv = np.vstack([grid["relative_velocity_x"][()], grid["relative_velocity_y"][()],
                                grid["relative_velocity_z"][()]]).T / 1e5
                T = grid["temperature"][()]
                Z = grid["metallicity"][()]
                v_rest = -vlos - v_sys
                rel = xyz - center
                rel = rel - R.BOX_KPC * np.round(rel / R.BOX_KPC)   # periodic wrap
                r = np.linalg.norm(rel, axis=1)
                rhat = rel / np.clip(r, 1e-6, None)[:, None]
                vgal = rv - sub_vel
                v_r = np.einsum("ij,ij->i", vgal, rhat)
                v_z = vgal @ n_disk
                x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ n_disk
                R_disk = np.hypot(x_d, y_d)
                s = (rel - (anchor - center)) @ los
                d = dict(sid=sid, mode=mode, alpha=alpha, v_ISM=v_ISM, v_mode=str(mrow.v_mode),
                         in_disk=bool(mrow.in_disk), rho_kpc=float(rho),
                         v_rest=v_rest[sel], dv=v_rest[sel] - v_ISM, v_r=v_r[sel], v_z=v_z[sel],
                         R_disk=R_disk[sel], z_disk=z_d[sel], r_gal=r[sel], s=s[sel],
                         Zsolar=Z[sel] / ZSUN, logT=np.log10(np.clip(T[sel], 1.0, None)))
                for ion in IONFIELD:
                    d[f"N_{ion}"] = Ncol[ion][sel]
                parts.append(pd.DataFrame(d))
    if not parts:
        print(f"[SID {sid}] no absorbing cells"); return
    df = pd.concat(parts, ignore_index=True)
    p = OUT / f"absorbers_sid{sid}.parquet"
    df.to_parquet(p, index=False)
    frac = {ion: int((df[f"N_{ion}"] > FLOOR[ion]).sum()) for ion in IONFIELD}
    print(f"[SID {sid}] {len(df)} cells over {df.groupby(['mode','alpha']).ngroups} sightlines -> {p}")
    print(f"  cells above floor per ion: {frac}")


if __name__ == "__main__":
    main()
