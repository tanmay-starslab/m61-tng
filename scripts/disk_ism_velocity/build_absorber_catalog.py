#!/usr/bin/env python3
"""Si II absorbing-gas cell catalog for all 720 sightlines of one SID.

For every ray cell carrying Si II (n_SiII > NSI_FLOOR): the corrected LOS velocity
v_rest and its offset from the sightline disk-ISM velocity (dv = v_rest - v_ISM); the
Si II and H I column carried by the cell; the full galaxy-frame 3D kinematics
  v_gal = relative_velocity/1e5 - SubhaloVel        (verified: dot(v_gal, los) = v_rest)
  v_r   = v_gal . r_hat   (galactocentric radial; >0 outflow, <0 inflow)
  v_z   = v_gal . n_disk  (perpendicular to the disk plane)
disk-frame + galactocentric position, metallicity (Z/Zsun) and temperature. This is the
base catalog for the absorber-abundance and inflow/outflow figures; velocity-component
clustering is done downstream from the combined catalog.

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
NSI_FLOOR = 1e-11   # cm^-3, Si II number-density floor defining 'absorbing gas'
ZSUN = 0.0127       # solar metal mass fraction (GFM_Metallicity is a mass fraction)


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
                nsi = grid["Si_p1_number_density"][()]
                sel = nsi > NSI_FLOOR
                if not sel.any():
                    continue
                xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC
                vlos = grid["velocity_los"][()] / 1e5
                dl_cm = grid["dl"][()]
                rv = np.vstack([grid["relative_velocity_x"][()], grid["relative_velocity_y"][()],
                                grid["relative_velocity_z"][()]]).T / 1e5
                nhi = grid["H_p0_number_density"][()]
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
                NSiII = nsi * dl_cm; NHI = nhi * dl_cm
                parts.append(pd.DataFrame(dict(
                    sid=sid, mode=mode, alpha=alpha, v_ISM=v_ISM, v_mode=str(mrow.v_mode),
                    in_disk=bool(mrow.in_disk), rho_kpc=float(rho),
                    v_rest=v_rest[sel], dv=v_rest[sel] - v_ISM, NSiII=NSiII[sel], NHI=NHI[sel],
                    v_r=v_r[sel], v_z=v_z[sel], R_disk=R_disk[sel], z_disk=z_d[sel], r_gal=r[sel],
                    s=s[sel], Zsolar=Z[sel] / ZSUN, logT=np.log10(np.clip(T[sel], 1.0, None)))))
    if not parts:
        print(f"[SID {sid}] no absorbing cells"); return
    df = pd.concat(parts, ignore_index=True)
    p = OUT / f"absorbers_sid{sid}.parquet"
    df.to_parquet(p, index=False)
    print(f"[SID {sid}] {len(df)} absorbing cells over {df.groupby(['mode','alpha']).ngroups} sightlines "
          f"-> {p}")
    print(f"  dv range [{df.dv.min():.0f},{df.dv.max():.0f}]  v_r [{df.v_r.min():.0f},{df.v_r.max():.0f}]  "
          f"Zsolar median {df.Zsolar.median():.2f}  logNSiII [{np.log10(df.NSiII.clip(1).replace(0,1)).min():.1f},{np.log10(df.NSiII).max():.1f}]")


if __name__ == "__main__":
    main()
