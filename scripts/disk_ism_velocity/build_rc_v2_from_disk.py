#!/usr/bin/env python3
"""Build the v2 (T<1e4) 3-tracer fiducial rotation curve WITHOUT re-reading cutouts, from the
per-tracer curves already on disk (the v1 run saved cold_gas_1e4, sf_gas, young_stars at
0.5 kpc). Fiducial = equal-weight per-bin mean of the three tracer medians (n>=5). Also emits
coarsened 1 and 2 kpc average curves (count-weighted grouping of the 0.5 kpc curve) for the
bin-sensitivity diagnostic. Interface identical to build_sid_rc_v2.py.

Usage: python build_rc_v2_from_disk.py
Output: rotation_curves_v2/rc_sid<sid>.npz  + rc_sid<sid>_bw{0p5,1,2}_*.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RC1 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]
TRACERS = ("cold_gas", "sf_gas", "young_stars")   # cold_gas <- v1 cold_gas_1e4
V1FILE = {"cold_gas": "cold_gas_1e4", "sf_gas": "sf_gas", "young_stars": "young_stars"}
N_MIN_AVG = 5


def average(curves, R):
    rows = []
    for i in range(len(R)):
        meds, mws, p16, p84, sig, contrib, ns = [], [], [], [], [], [], {}
        for t in TRACERS:
            c = curves[t]
            n_i = int(c["n"].iloc[i]) if c is not None else 0
            ns[t] = n_i
            if c is not None and n_i >= N_MIN_AVG and np.isfinite(c["v_phi_median"].iloc[i]):
                meds.append(c["v_phi_median"].iloc[i]); mws.append(c["v_phi_mass_weighted_mean"].iloc[i])
                p16.append(c["v_phi_p16"].iloc[i]); p84.append(c["v_phi_p84"].iloc[i])
                sig.append(c["sigma_phi"].iloc[i]); contrib.append(t)
        row = dict(R_center=float(R[i]),
                   v_fid_median=float(np.mean(meds)) if meds else np.nan,
                   v_fid_mwmean=float(np.mean(mws)) if mws else np.nan,
                   v_fid_p16=float(np.mean(p16)) if p16 else np.nan,
                   v_fid_p84=float(np.mean(p84)) if p84 else np.nan,
                   sigma_fid=float(np.mean(sig)) if sig else np.nan,
                   n_contrib=len(contrib), contrib=";".join(contrib))
        for t in TRACERS:
            row[f"v_{t}"] = float(curves[t]["v_phi_median"].iloc[i]) if curves[t] is not None else np.nan
            row[f"n_{t}"] = ns[t]
        rows.append(row)
    return pd.DataFrame(rows)


def coarsen(avg, group):
    """Count-weighted coarsening of the fiducial curve by grouping `group` adjacent 0.5 kpc bins."""
    rows = []
    for s in range(0, len(avg), group):
        blk = avg.iloc[s:s + group]
        w = blk[[f"n_{t}" for t in TRACERS]].sum(axis=1).values.astype(float)
        v = blk["v_fid_median"].values
        ok = np.isfinite(v) & (w > 0)
        rows.append(dict(R_center=float(blk["R_center"].mean()),
                         v_fid_median=float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan,
                         n_contrib=float(blk["n_contrib"].max())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in SIDS:
        v1 = dict(np.load(RC1 / f"rc_sid{sid}.npz"))
        curves = {}
        for t in TRACERS:
            f = RC1 / f"rc_sid{sid}_{V1FILE[t]}.csv"
            curves[t] = pd.read_csv(f) if f.exists() else None
        template = next((c for c in curves.values() if c is not None), None)
        R = template["R_center"].values
        avg = average(curves, R)
        # exact 0.5 kpc outputs (per-tracer + average)
        for t in TRACERS:
            if curves[t] is not None:
                curves[t].to_csv(OUT / f"rc_sid{sid}_bw0p5_{t}.csv", index=False)
        avg.to_csv(OUT / f"rc_sid{sid}_bw0p5_ism_average.csv", index=False)
        coarsen(avg, 2).to_csv(OUT / f"rc_sid{sid}_bw1_ism_average.csv", index=False)
        coarsen(avg, 4).to_csv(OUT / f"rc_sid{sid}_bw2_ism_average.csv", index=False)
        # production npz (interface for Stage B / projection)
        npz = dict(sid=sid, cold_T=1e4, bin_width=0.5, n_disk=v1["n_disk"], e1=v1["e1"], e2=v1["e2"],
                   center_kpc=v1["center_kpc"], R_center=avg["R_center"].values,
                   v_fid_median=avg["v_fid_median"].values, v_fid_mwmean=avg["v_fid_mwmean"].values,
                   v_fid_p16=avg["v_fid_p16"].values, v_fid_p84=avg["v_fid_p84"].values,
                   sigma_fid=avg["sigma_fid"].values, n_contrib=avg["n_contrib"].values)
        for t in TRACERS:
            npz[f"v_{t}"] = avg[f"v_{t}"].values
            npz[f"n_{t}"] = avg[f"n_{t}"].values
        np.savez_compressed(OUT / f"rc_sid{sid}.npz", **npz)
        i26 = int(np.clip(np.searchsorted(avg["R_center"].values, 26.0), 0, len(avg) - 1))
        print(f"[SID {sid}] cold-gas bins occupied {int((avg.n_cold_gas>0).sum())}/{len(avg)}  "
              f"v_fid@R26={avg['v_fid_median'].iloc[i26]:.1f} contrib=[{avg['contrib'].iloc[i26]}]")
    print(f"\n[DONE] 20 SIDs -> {OUT}")


if __name__ == "__main__":
    main()
