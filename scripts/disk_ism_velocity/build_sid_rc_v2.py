#!/usr/bin/env python3
"""Stage A v2: supervisor's 3-tracer ISM rotation curve, cold gas at T < 1e4 K.

FIX vs v1: the v1 fiducial used cold gas at T < 1e3 K, which selects ZERO gas in TNG
(star-forming EoS floors gas at ~1e3.8 K), so the "cold+SF+young" average silently
reduced to SF+young only. Here cold gas = T < 1e4 K, so it actually enters the average
(and dominates the outer disk, where SF/young thin out).

Fiducial v_phi(R) = equal-weight, per-bin average of the median v_phi of
  cold_gas (T<1e4), sf_gas (SFR>0), young_stars (age<300 Myr).
Curve is alpha-independent (galaxy property); the per-orientation LOS projection is Stage B.

Bin-width study: the curve is built at bin_width in {0.5, 1.0, 2.0} kpc so the outer-disk
noise (sparse tracers near rho~26) can be assessed. Production curve = 1.0 kpc (npz).

Disk frame = los-cone axis (disk_normal_from_los; the stored normal_used_hat is wrong by
up to ~49 deg for several SIDs). Velocity: v_rel = v*sqrt(a) - SubhaloVel, positive=recession.

Usage:  python build_sid_rc_v2.py <sid>
Output: rotation_curves_v2/rc_sid<sid>.npz  (+ per-bw per-tracer/average CSVs)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
from pm_general import TNG_H, unit, sid_paths  # noqa: E402
import dv_core as dv  # noqa: E402
from build_sid_rc import disk_normal_from_los  # reuse the validated los-cone disk normal

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
ISM_TRACERS = ("cold_gas", "sf_gas", "young_stars")   # cold gas is now T < 1e4 K
COLD_T = 1e4
BIN_WIDTHS = (0.5, 1.0, 2.0)
PROD_BW = 1.0                     # production bin width stored in the npz
N_MIN = 20                        # min particles for a fixed-bin rotation-curve point
N_MIN_AVG = 5                     # min particles for a tracer to contribute to the average


def tracer_arrays(gal, kind):
    if kind == "cold_gas":
        m = gal["gT"] < COLD_T
        return gal["gpos"][m], gal["gvel"][m], gal["gm"][m]
    if kind == "sf_gas":
        m = gal["gsfr"] > 0
        return gal["gpos"][m], gal["gvel"][m], gal["gm"][m]
    if kind == "young_stars":
        m = gal["sage"] <= 0.3
        return gal["spos"][m], gal["svel"][m], gal["sm"][m]
    raise ValueError(kind)


def build_ism_average(curves, template):
    rows = []
    for i in range(len(template)):
        R = float(template["R_center"].iloc[i])
        meds, mws, p16s, p84s, sigs, contrib, ns = [], [], [], [], [], [], {}
        for t in ISM_TRACERS:
            c = curves.get(t)
            n_i = int(c["n"].iloc[i]) if c is not None else 0
            ns[t] = n_i
            if c is not None and n_i >= N_MIN_AVG and np.isfinite(c["v_phi_median"].iloc[i]):
                meds.append(float(c["v_phi_median"].iloc[i]))
                mws.append(float(c["v_phi_mass_weighted_mean"].iloc[i]))
                p16s.append(float(c["v_phi_p16"].iloc[i]))
                p84s.append(float(c["v_phi_p84"].iloc[i]))
                sigs.append(float(c["sigma_phi"].iloc[i]))
                contrib.append(t)
        row = dict(R_center=R,
                   v_fid_median=float(np.mean(meds)) if meds else np.nan,
                   v_fid_mwmean=float(np.mean(mws)) if mws else np.nan,
                   v_fid_p16=float(np.mean(p16s)) if p16s else np.nan,
                   v_fid_p84=float(np.mean(p84s)) if p84s else np.nan,
                   sigma_fid=float(np.mean(sigs)) if sigs else np.nan,
                   n_contrib=len(contrib), contrib=";".join(contrib) if contrib else "")
        for t in ISM_TRACERS:
            c = curves.get(t)
            row[f"v_{t}"] = float(c["v_phi_median"].iloc[i]) if c is not None else np.nan
            row[f"n_{t}"] = ns[t]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    p = sid_paths(sid)
    orient = json.loads(p["orient_json"].read_text())
    sub = json.loads(p["subhalo_json"].read_text())
    center_ckpch = np.array(orient["center_ckpc_h"])
    sub_vel = np.array(sub["subhalo_vel_kms"])
    n_disk = disk_normal_from_los(sid)
    e1, e2 = dv.disk_basis(n_disk)
    print(f"[SID {sid}] n_disk={n_disk.round(4)} cold_T={COLD_T:g}", flush=True)
    gal = dv.load_galaxy(sid, center_ckpch, sub_vel, R_max=40.)
    print(f"  loaded gas={len(gal['gm'])} stars={len(gal['sm'])}", flush=True)

    prod = None
    for bw in BIN_WIDTHS:
        curves = {}
        for kind in ISM_TRACERS:
            pos, vel, mass = tracer_arrays(gal, kind)
            curves[kind] = (dv.rotation_curve(pos, vel, mass, e1, e2, n_disk,
                                              R_max=30., bin_width=bw, n_min=N_MIN)
                            if len(mass) else None)
        template = next((c for c in (curves.get(t) for t in ISM_TRACERS) if c is not None), None)
        if template is None:
            print(f"[FATAL] all tracers empty at bw={bw}"); sys.exit(1)
        avg = build_ism_average(curves, template)
        tag = f"bw{bw:g}".replace(".", "p")
        for kind, c in curves.items():
            if c is not None:
                c.to_csv(OUT / f"rc_sid{sid}_{tag}_{kind}.csv", index=False)
        avg.to_csv(OUT / f"rc_sid{sid}_{tag}_ism_average.csv", index=False)
        ncold = int((avg["n_cold_gas"] > 0).sum())
        i26 = int(np.clip(np.searchsorted(avg["R_center"].values, 26.0), 0, len(avg) - 1))
        print(f"  bw={bw:g}: cold-gas bins occupied {ncold}/{len(avg)}  "
              f"v_fid@R26={avg['v_fid_median'].iloc[i26]:.1f} "
              f"contrib=[{avg['contrib'].iloc[i26]}]", flush=True)
        if bw == PROD_BW:
            prod = (curves, avg)

    # ---- production npz (interface for Stage B / projection) ----
    curves, avg = prod
    npz = dict(sid=sid, cold_T=COLD_T, bin_width=PROD_BW, n_disk=n_disk, e1=e1, e2=e2,
               center_kpc=center_ckpch / TNG_H,
               R_center=avg["R_center"].values,
               v_fid_median=avg["v_fid_median"].values, v_fid_mwmean=avg["v_fid_mwmean"].values,
               v_fid_p16=avg["v_fid_p16"].values, v_fid_p84=avg["v_fid_p84"].values,
               sigma_fid=avg["sigma_fid"].values, n_contrib=avg["n_contrib"].values)
    for t in ISM_TRACERS:
        c = curves.get(t)
        npz[f"v_{t}"] = c["v_phi_median"].values if c is not None else np.full(len(avg), np.nan)
        npz[f"n_{t}"] = c["n"].values if c is not None else np.zeros(len(avg), int)
    np.savez_compressed(OUT / f"rc_sid{sid}.npz", **npz)
    print(f"[DONE] SID {sid} -> {OUT}/rc_sid{sid}.npz (prod bw={PROD_BW})", flush=True)


if __name__ == "__main__":
    main()
