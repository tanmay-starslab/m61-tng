#!/usr/bin/env python3
"""Full multi-ion absorbing-gas cell catalog for all 720 sightlines of one SID.

Reads the COMBINED all_rays_L2Rvir.h5 -- the SAME ray product the v_ISM master was built
on (the per-SID individual original_rays are a DIFFERENT, inconsistent generation, so they
must NOT be used here). Six ions are native to that ray (H I, C II, N V, Si II, Si III,
Si IV); the five others (O VI, C IV, O I, Mg II, Fe II) are RECOMPUTED per ray from the
stored density/temperature/metallicity/redshift with Trident's ionization tables -- each
ray is extracted to a temporary standalone HDF5, yt.load-ed, and add_ion_fields-ed
(verified peak temperatures: O VI ~10^5.5, C IV ~10^5, O I/Mg II/Fe II ~10^4 K).

Per absorbing cell: v_rest and dv = v_rest - v_ISM; every ion's column; galaxy-frame 3D
kinematics (v_r galactocentric radial <0 inflow/>0 outflow, v_z perp to disk); disk-frame +
galactocentric position; metallicity (Z/Zsun) and temperature.

Usage: python build_absorber_catalog.py <sid>
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
import yt  # noqa: E402
import trident  # noqa: E402
yt.set_log_level(50)
from pm_general import get_geometry, get_original_rho, compute_endpoints, CM_PER_KPC  # noqa: E402
import dv_core as dv  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402  (combined_path, BOX_KPC)
import build_sid_rc as B  # noqa: E402

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
MASTER = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
ZSUN = 0.0127
HYPERVEL_KMS = 700.0   # |v_rest| ceiling: above this a cell is a super-escape numerical extreme
RAW = {"HI": "H_p0_number_density", "CII": "C_p1_number_density", "NV": "N_p4_number_density",
       "SiII": "Si_p1_number_density", "SiIII": "Si_p2_number_density", "SiIV": "Si_p3_number_density"}
NEW = {"CIV": ("C IV", "C_p3_number_density"), "OI": ("O I", "O_p0_number_density"),
       "OVI": ("O VI", "O_p5_number_density"), "MgII": ("Mg II", "Mg_p1_number_density"),
       "FeII": ("Fe II", "Fe_p1_number_density")}
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12,
         "CIV": 1e12, "OI": 1e12, "OVI": 3e12, "MgII": 1e11, "FeII": 1e11}


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MASTER); m = m[m.sid == sid].set_index(["mode", "alpha_deg"])
    n_disk = B.disk_normal_from_los(sid); e1, e2 = dv.disk_basis(n_disk)
    tmp = f"/tmp/_combray_{sid}_{os.getpid()}.h5"
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
                grp = base[f"mode={mode}"][ag]
                rayh = grp[list(grp.keys())[0]]["original_trident_ray_h5"]
                gr = rayh["grid"]
                o = np.argsort(gr["l"][()])
                xyz = np.vstack([gr["x"][()], gr["y"][()], gr["z"][()]]).T[o] / CM_PER_KPC
                vlos = gr["velocity_los"][()][o] / 1e5
                dl_cm = gr["dl"][()][o]
                rv = np.vstack([gr["relative_velocity_x"][()], gr["relative_velocity_y"][()],
                                gr["relative_velocity_z"][()]]).T[o] / 1e5
                T = gr["temperature"][()][o]; Z = gr["metallicity"][()][o]
                Ncol = {ion: gr[fld][()][o] * dl_cm for ion, fld in RAW.items()}
                # extract this ray to a standalone file and recompute the 5 other ions
                if os.path.exists(tmp):
                    os.remove(tmp)
                with h5py.File(tmp, "w") as out:
                    for k, v in rayh.attrs.items():
                        out.attrs[k] = v
                    rayh.copy("grid", out)
                ds = yt.load(tmp)
                trident.add_ion_fields(ds, ions=[v[0] for v in NEW.values()], ftype="gas")
                ad = ds.all_data(); oy = np.argsort(np.array(ad["gas", "l"]))
                for ion, (_, fld) in NEW.items():
                    Ncol[ion] = np.array(ad["gas", fld].to("cm**-3"))[oy] * dl_cm
                sel = np.zeros(len(dl_cm), bool)
                for ion in Ncol:
                    sel |= Ncol[ion] > FLOOR[ion]
                if not sel.any():
                    continue
                g = get_geometry(sid, mode, alpha)
                los, v_sys, sub_vel, center = g["los"], g["v_sys"], g["sub_vel"], g["center_kpc"]
                rho, _, _ = get_original_rho(sid, mode, alpha)
                anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
                v_rest = -vlos - v_sys
                # periodic-box-wrap detection: cells whose min-image correction is nonzero on
                # any axis are wrapped across the box (the ray endpoint pokes outside the domain).
                wrapn = np.round((xyz - center) / R.BOX_KPC)
                wrapped = np.any(wrapn != 0, axis=1)
                rel = (xyz - center) - R.BOX_KPC * wrapn
                # hyper-velocity: |v_rest| beyond any plausible halo velocity (super-escape) -> a
                # numerical/wrap-adjacent extreme, not a physical absorber.
                hypervel = np.abs(v_rest) > HYPERVEL_KMS
                r = np.linalg.norm(rel, axis=1); rhat = rel / np.clip(r, 1e-6, None)[:, None]
                vgal = rv - sub_vel
                v_r = np.einsum("ij,ij->i", vgal, rhat); v_z = vgal @ n_disk
                x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ n_disk
                R_disk = np.hypot(x_d, y_d); s = (rel - (anchor - center)) @ los
                d = dict(sid=sid, mode=mode, alpha=alpha, v_ISM=v_ISM, v_mode=str(mrow.v_mode),
                         in_disk=bool(mrow.in_disk), rho_kpc=float(rho),
                         v_rest=v_rest[sel], dv=v_rest[sel] - v_ISM, v_r=v_r[sel], v_z=v_z[sel],
                         R_disk=R_disk[sel], z_disk=z_d[sel], r_gal=r[sel], s=s[sel],
                         Zsolar=Z[sel] / ZSUN, logT=np.log10(np.clip(T[sel], 1.0, None)),
                         wrapped=wrapped[sel], hypervel=hypervel[sel])
                for ion in Ncol:
                    d[f"N_{ion}"] = Ncol[ion][sel]
                parts.append(pd.DataFrame(d))
    if os.path.exists(tmp):
        os.remove(tmp)
    if not parts:
        print(f"[SID {sid}] no absorbing cells"); return
    df = pd.concat(parts, ignore_index=True)
    pth = OUT / f"absorbers_sid{sid}.parquet"; df.to_parquet(pth, index=False)
    nflag = int((df.wrapped | df.hypervel).sum())
    print(f"[SID {sid}] {len(df)} cells over {df.groupby(['mode','alpha']).ngroups} sightlines -> {pth}")
    print(f"  flagged (wrapped|hypervel): {nflag} ({100*nflag/len(df):.2f}%)  "
          f"wrapped={int(df.wrapped.sum())} hypervel={int(df.hypervel.sum())}")
    print("  cells>floor: " + ", ".join(f"{ion} {int((df[f'N_{ion}']>FLOOR[ion]).sum())}" for ion in Ncol))


if __name__ == "__main__":
    main()
