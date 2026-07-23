#!/usr/bin/env python3
"""Stage B: per-sightline disk-ISM velocity for ALL 720 (mode,alpha) of one SID.

Reads the combined all_rays_L2Rvir.h5 (ion grid) + Stage-A rotation curve
(rc_sid<SID>.npz + rc_sid<SID>_cold_gas_1e4.csv for R95). Per sightline computes:
  v_ism_direct_density (gas-density column-weighted v_rest of |z|<2 & R<R95 disk gas),
  v_ism_R95edge (rotation curve at R95, projected), v_ism_primary + in_disk flag,
  Si II / H I weightings, model v_ISM, f_disk(Si II), R_cross, and the Si II spectrum dip.
No plots (use ray_ism_diagnostic.py for plots). Reuses the helpers in ray_ism_diagnostic.

Usage: python stage_b_vism.py <sid>
Output: /scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_sid<sid>.csv
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
from pm_general import get_original_rho, CM_PER_KPC  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402  (reuse helpers: projection, ray_group, r95, consts)

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables")


def compute_one(h5, sid, mode, alpha, rc, R_edge):
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = R.projection(sid, mode, alpha, rc, rho)
    los, v_sys, anchor, center = M["los"], M["v_sys"], M["anchor"], M["center"]
    e1, e2, nd = M["e1"], M["e2"], M["nd"]
    grp = R.ray_group(h5, mode, alpha)
    grid = grp["original_trident_ray_h5/grid"]
    xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC
    vlos = grid["velocity_los"][()] / 1e5
    dl = grid["dl"][()] / CM_PER_KPC
    dens = grid["density"][()]
    n_si = grid[R.SIII][()] if R.SIII in grid else np.zeros(len(vlos))
    n_hi = grid[R.HI][()] if R.HI in grid else np.zeros(len(vlos))

    v_rest = -vlos - v_sys
    rel = xyz - center
    rel = rel - R.BOX_KPC * np.round(rel / R.BOX_KPC)   # periodic wrap
    x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ nd
    R_disk = np.hypot(x_d, y_d)
    disk = (np.abs(z_d) < R.Z0) & (R_disk < R_edge)

    def cw(w):
        ww = w[disk]
        return float(np.average(v_rest[disk], weights=ww)) if (disk.any() and ww.sum() > 0) else np.nan

    w_dens = dens * dl; w_si = n_si * dl; w_hi = n_hi * dl
    v_dens = cw(w_dens); v_si = cw(w_si); v_hi = cw(w_hi)
    f_disk_si = (w_si[disk].sum() / w_si.sum()) if w_si.sum() > 0 else np.nan
    R_cross = float(R_disk[np.abs(z_d).argmin()])
    in_disk = bool(R_cross < R_edge)

    Rc_a = rc["R_center"]; vf_a = rc["v_fid_median"]; fin = np.isfinite(vf_a)
    v_edge = (float(np.interp(R_edge, Rc_a[fin], vf_a[fin])) * M["proj"]
              if (fin.sum() >= 2 and np.isfinite(M["proj"])) else np.nan)
    has_cool = bool(np.isfinite(v_si) and f_disk_si > 0.05)
    v_primary = v_dens if (in_disk and has_cool) else v_edge
    v_mode = "direct" if (in_disk and has_cool) else "R95-edge"

    v_dip = np.nan
    g = grp.get("spectrum_by_line/Si_II_1260/lsf")
    if g is not None:
        vv = R.spectrum_v(g["lambda_A"][()], 1260.4221, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            v_dip = float(vv[w][np.argmin(g["flux"][()][w])])

    return dict(sid=sid, mode=mode, alpha_deg=alpha, rho_kpc=rho, v_sys=v_sys,
                R_edge=R_edge, R_cross=R_cross, in_disk=in_disk, proj_phi=M["proj"],
                v_ism_model=M["v_ism_model"], v_ism_direct_density=v_dens,
                v_ism_SiII=v_si, v_ism_HI=v_hi, v_ism_R95edge=v_edge,
                v_ism_primary=v_primary, v_mode=v_mode, f_disk_SiII=f_disk_si,
                SiII_dip=v_dip)


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    npz = R.RC_DIR / f"rc_sid{sid}.npz"
    if not npz.exists():
        print(f"[SID {sid}] MISSING Stage-A {npz} -- skip"); sys.exit(3)
    rc = dict(np.load(npz))
    R_edge = R.r95_cold_gas(sid)
    rows = []
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for ag in sorted(base[f"mode={mode}"].keys(), key=lambda k: int(k.split("=")[1])):
                alpha = int(ag.split("=")[1])
                try:
                    rows.append(compute_one(h, sid, mode, alpha, rc, R_edge))
                except Exception as e:
                    rows.append(dict(sid=sid, mode=mode, alpha_deg=alpha, error=str(e)[:120]))
    df = pd.DataFrame(rows)
    p = OUT / f"vism_sid{sid}.csv"
    df.to_csv(p, index=False)
    nok = int(df["v_ism_primary"].notna().sum()) if "v_ism_primary" in df else 0
    print(f"[SID {sid}] {len(df)} sightlines, {nok} with v_ISM -> {p}")


if __name__ == "__main__":
    main()
