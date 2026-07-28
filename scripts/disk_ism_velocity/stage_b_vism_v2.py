#!/usr/bin/env python3
"""Stage B v2: supervisor-model disk-ISM velocity as the PRIMARY v_ISM.

v_ISM(sid,mode,alpha) = v_rot(R_anchor) * proj , where
  v_rot   = the 3-tracer (cold gas T<1e4 + SF gas + young stars) average rotation curve
            (rotation_curves_v2/rc_sid<sid>.npz, 1.0 kpc bins),
  R_anchor= galactocentric cylindrical radius of the sightline anchor (~impact parameter),
  proj    = phi_hat . los  (projection of circular rotation onto the line of sight).
This is a galaxy-rest-frame velocity (v_sys already removed via the rotation curve); the
spectrum axis is in the same frame, so v_ISM should land on the ISM absorption.

Per-tracer LOS velocities (cold/SF/young, each * proj) are recorded for diagnostics, as is
whether the anchor lies inside the sampled disk (in_disk_model). The Si II spectrum dip and
the v1 direct cool-gas velocity are merged from the v1 master for a like-for-like comparison
(both are independent of the v_ISM method).

Usage: python stage_b_vism_v2.py            (loops all 20 SIDs; fast, no ray reads)
Output: vism_tables_v2/vism_master_v2.csv
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
from pm_general import get_geometry, get_original_rho, compute_endpoints  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402  (projection(); interp helpers)

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
V1 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2")
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]


def interp(rc, key, Rq):
    Rc = rc["R_center"]; v = rc[key]; fin = np.isfinite(v)
    return float(np.interp(Rq, Rc[fin], v[fin])) if fin.sum() >= 2 else np.nan


def disk_extent(rc):
    """Largest R where the fiducial average is defined (a tracer had >=N_MIN_AVG)."""
    good = np.isfinite(rc["v_fid_median"])
    return float(rc["R_center"][good].max()) if good.any() else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(V1).set_index(["sid", "mode", "alpha_deg"])
    rows = []
    for sid in SIDS:
        npz = RCV2 / f"rc_sid{sid}.npz"
        if not npz.exists():
            print(f"[SID {sid}] MISSING {npz}; skip"); continue
        rc = dict(np.load(npz))
        Rmax_disk = disk_extent(rc)
        for mode in ("flip", "noflip"):
            for alpha in range(360):
                try:
                    rho, _, _ = get_original_rho(sid, mode, alpha)
                except Exception:
                    continue
                M = R.projection(sid, mode, alpha, rc, rho)   # uses v_fid_median from rc
                Rd, proj, v_model = M["R_anchor"], M["proj"], M["v_ism_model"]
                v_cold = interp(rc, "v_cold_gas", Rd) * proj if np.isfinite(proj) else np.nan
                v_sf = interp(rc, "v_sf_gas", Rd) * proj if np.isfinite(proj) else np.nan
                v_young = interp(rc, "v_young_stars", Rd) * proj if np.isfinite(proj) else np.nan
                n_contrib = interp(rc, "n_contrib", Rd)
                in_disk_model = bool(np.isfinite(Rmax_disk) and Rd <= Rmax_disk)
                rec = dict(sid=sid, mode=mode, alpha_deg=alpha, rho_kpc=rho,
                           v_sys=M["v_sys"], R_anchor=Rd, proj_phi=proj,
                           v_ism_model=v_model, v_ism_cold=v_cold, v_ism_sf=v_sf,
                           v_ism_young=v_young, n_contrib_at_R=n_contrib,
                           Rmax_disk=Rmax_disk, in_disk_model=in_disk_model)
                try:
                    r1 = v1.loc[(sid, mode, alpha)]
                    rec["SiII_dip"] = float(r1.SiII_dip)
                    rec["v_ism_direct_cool"] = float(r1.v_ism_direct_cool)
                    rec["v_ism_v1_primary"] = float(r1.v_ism_primary)
                    rec["in_disk_v1"] = bool(r1.in_disk)
                except KeyError:
                    rec["SiII_dip"] = rec["v_ism_direct_cool"] = rec["v_ism_v1_primary"] = np.nan
                rows.append(rec)
        print(f"[SID {sid}] Rmax_disk={Rmax_disk:.1f} done", flush=True)
    df = pd.DataFrame(rows)
    p = OUT / "vism_master_v2.csv"
    df.to_csv(p, index=False)
    ok = df.v_ism_model.notna()
    print(f"\n{len(df)} sightlines -> {p}")
    print(f"  v_ism_model finite: {int(ok.sum())}; in_disk_model: {int(df.in_disk_model.sum())}")
    d = (df.v_ism_model - df.SiII_dip).dropna()
    print(f"  v_ism_model - SiII dip: median {d.median():+.1f}, robust-sigma "
          f"{(d-d.median()).abs().median()*1.4826:.1f} km/s (all sightlines)")
    dd = df[df.in_disk_model]
    d2 = (dd.v_ism_model - dd.SiII_dip).dropna()
    print(f"  in_disk_model only: median {d2.median():+.1f}, sigma "
          f"{(d2-d2.median()).abs().median()*1.4826:.1f} km/s ({len(dd)} sightlines)")


if __name__ == "__main__":
    main()
