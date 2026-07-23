#!/usr/bin/env python3
"""Stage A: per-SID multi-tracer rotation curves + 3-tracer ISM average.

Builds v_phi(R) in the galaxy's disk frame for:
  - cold_gas_1e3  : gas with nominal T < 1e3 K   (supervisor's cold-gas definition)
  - sf_gas        : gas with StarFormationRate > 0
  - young_stars   : stars with age < 300 Myr
  - cold_gas_1e4  : gas with T < 1e4 K  (kept ONLY for regression vs prior validated work)

The equal-weight average of the first three = the fiducial disk-ISM rotation curve.
This curve is alpha-INDEPENDENT (a property of the galaxy); the per-orientation LOS
projection happens later in Stage B.

Disk frame = PCA v3 inner-stars normal (`normal_used_hat`), which is the axis the alpha
convention rotates the galaxy about, so the geometry stays consistent with orient_peralpha.

Velocity convention (inherited from dv_core / pm_general):
  v_rel = v_particle*sqrt(a) - SubhaloVel ;  positive = recession ; systemic subtracted once.

Reuses the VALIDATED heavy machinery from disk_velocity_v2/dv_core.py (load_galaxy,
rotation_curve, disk_basis) -- this script adds only the tracer masks, the 3-tracer
average, occupancy flags, and a golden-regression printout.

Usage:  python build_sid_rc.py <sid> [--cold_T 1e3]
Output: /scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves/rc_sid<sid>.{npz,csv}
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
from pm_general import TNG_H, unit, sid_paths  # noqa: E402
import dv_core as dv  # noqa: E402

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
ISM_TRACERS = ("cold_gas_1e3", "sf_gas", "young_stars")
ALL_TRACERS = ("cold_gas_1e3", "cold_gas_1e4", "sf_gas", "young_stars")


def tracer_arrays(gal, kind, cold_T):
    """Return (pos, vel, mass) for a tracer from a loaded galaxy dict."""
    if kind == "cold_gas_1e3":
        m = gal["gT"] < cold_T
        return gal["gpos"][m], gal["gvel"][m], gal["gm"][m]
    if kind == "cold_gas_1e4":
        m = gal["gT"] < 1e4
        return gal["gpos"][m], gal["gvel"][m], gal["gm"][m]
    if kind == "sf_gas":
        m = gal["gsfr"] > 0
        return gal["gpos"][m], gal["gvel"][m], gal["gm"][m]
    if kind == "young_stars":
        m = gal["sage"] <= 0.3
        return gal["spos"][m], gal["svel"][m], gal["sm"][m]
    raise ValueError(kind)


def build_ism_average(curves, template, n_min_bin=5):
    """Equal-weight per-bin average of v_phi across the 3 ISM tracers.

    Only tracers with n >= n_min_bin in a bin contribute; empty tracers drop out
    (this is how young*/SF gracefully vanish at large R while cold gas carries the curve).
    """
    rows = []
    for i in range(len(template)):
        R = float(template["R_center"].iloc[i])
        meds, mws, p16s, p84s, sigs, contrib, ns = [], [], [], [], [], [], {}
        for t in ISM_TRACERS:
            c = curves.get(t)
            n_i = int(c["n"].iloc[i]) if c is not None else 0
            ns[t] = n_i
            if c is not None and n_i >= n_min_bin and np.isfinite(c["v_phi_median"].iloc[i]):
                meds.append(float(c["v_phi_median"].iloc[i]))
                mws.append(float(c["v_phi_mass_weighted_mean"].iloc[i]))
                p16s.append(float(c["v_phi_p16"].iloc[i]))
                p84s.append(float(c["v_phi_p84"].iloc[i]))
                sigs.append(float(c["sigma_phi"].iloc[i]))
                contrib.append(t)
        row = dict(
            R_center=R,
            v_fid_median=float(np.mean(meds)) if meds else np.nan,
            v_fid_mwmean=float(np.mean(mws)) if mws else np.nan,
            v_fid_p16=float(np.mean(p16s)) if p16s else np.nan,
            v_fid_p84=float(np.mean(p84s)) if p84s else np.nan,
            sigma_fid=float(np.mean(sigs)) if sigs else np.nan,
            n_contrib=len(contrib), contrib=";".join(contrib) if contrib else "",
        )
        for t in ISM_TRACERS:
            c = curves.get(t)
            row[f"v_{t}"] = float(c["v_phi_median"].iloc[i]) if c is not None else np.nan
            row[f"n_{t}"] = ns[t]
        row["quality_flag"] = ("OK" if (len(contrib) >= 2 and max([0, *ns.values()]) >= 20)
                               else ("LOW_COUNT" if contrib else "EMPTY"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("usage: build_sid_rc.py <sid> [--cold_T 1e3]"); sys.exit(2)
    sid = int(sys.argv[1])
    cold_T = 1e3
    if "--cold_T" in sys.argv:
        cold_T = float(sys.argv[sys.argv.index("--cold_T") + 1])
    OUT.mkdir(parents=True, exist_ok=True)

    p = sid_paths(sid)
    orient = json.loads(p["orient_json"].read_text())
    sub = json.loads(p["subhalo_json"].read_text())
    center_ckpch = np.array(orient["center_ckpc_h"])
    sub_vel = np.array(sub["subhalo_vel_kms"])
    n_disk = unit(np.array(orient["normal_used_hat"]))
    e1, e2 = dv.disk_basis(n_disk)
    print(f"[SID {sid}] n_disk(normal_used_hat)={n_disk.round(4)}  cold_T={cold_T:g} K", flush=True)

    print("Loading galaxy (heavy cutout read; chunked, filtered to R<40 kpc)...", flush=True)
    gal = dv.load_galaxy(sid, center_ckpch, sub_vel, R_max=40.)
    print(f"  loaded: gas cells={len(gal['gm'])}, star particles={len(gal['sm'])}", flush=True)

    curves = {}
    for kind in ALL_TRACERS:
        pos, vel, mass = tracer_arrays(gal, kind, cold_T)
        rc = (dv.rotation_curve(pos, vel, mass, e1, e2, n_disk, R_max=30., bin_width=0.5, n_min=20)
              if len(mass) else None)
        curves[kind] = rc
        if rc is not None:
            nOK = int((rc["quality_flag"] == "OK").sum())
            print(f"  {kind:14s}: {len(mass):9d} selected, {nOK:3d}/{len(rc)} bins OK", flush=True)
        else:
            print(f"  {kind:14s}: 0 selected -> EMPTY", flush=True)

    template = next((c for c in (curves.get(t) for t in ISM_TRACERS) if c is not None), None)
    if template is None:
        print("[FATAL] all three ISM tracers empty -- cannot build average curve."); sys.exit(1)
    avg = build_ism_average(curves, template)

    # ---- save ----
    for kind, rc in curves.items():
        if rc is not None:
            rc.to_csv(OUT / f"rc_sid{sid}_{kind}.csv", index=False)
    avg.to_csv(OUT / f"rc_sid{sid}_ism_average.csv", index=False)
    npz = dict(sid=sid, cold_T=cold_T, n_disk=n_disk, e1=e1, e2=e2,
               center_kpc=center_ckpch / TNG_H,
               R_center=avg["R_center"].values,
               v_fid_median=avg["v_fid_median"].values, v_fid_mwmean=avg["v_fid_mwmean"].values,
               v_fid_p16=avg["v_fid_p16"].values, v_fid_p84=avg["v_fid_p84"].values,
               sigma_fid=avg["sigma_fid"].values, n_contrib=avg["n_contrib"].values)
    for t in ALL_TRACERS:
        c = curves.get(t)
        npz[f"v_{t}"] = c["v_phi_median"].values if c is not None else np.full(len(avg), np.nan)
        npz[f"n_{t}"] = c["n"].values if c is not None else np.zeros(len(avg), int)
    np.savez_compressed(OUT / f"rc_sid{sid}.npz", **npz)

    # ---- occupancy + golden-regression printout ----
    def at(rc, R):
        if rc is None:
            return (np.nan, 0)
        i = int(np.clip(np.searchsorted(rc["R_center"].values, R), 0, len(rc) - 1))
        return (float(rc["v_phi_median"].iloc[i]), int(rc["n"].iloc[i]))

    print("\n=== Occupancy + v_phi_median at key radii (cells per 0.5 kpc bin) ===", flush=True)
    for kind in ALL_TRACERS:
        v5, n5 = at(curves.get(kind), 5.0)
        v20, n20 = at(curves.get(kind), 20.0)
        v26, n26 = at(curves.get(kind), 26.0)
        print(f"  {kind:14s}: R5 v={v5:8.1f} n={n5:7d} | R20 v={v20:8.1f} n={n20:7d} "
              f"| R26 v={v26:8.1f} n={n26:7d}", flush=True)
    print("  --- fiducial 3-tracer average ---", flush=True)
    for R in (5.0, 20.0, 26.0):
        i = int(np.clip(np.searchsorted(avg["R_center"].values, R), 0, len(avg) - 1))
        r = avg.iloc[i]
        print(f"  ISM avg R={R:4.0f}: v_fid_median={r['v_fid_median']:8.1f}  "
              f"n_contrib={int(r['n_contrib'])}  contrib=[{r['contrib']}]", flush=True)
    print(f"\n[DONE] SID {sid} -> {OUT}/rc_sid{sid}.npz", flush=True)
    print("  (regression check: for SID 342448, cold_gas_1e4 should be ~ -214 @R5, ~ -239 @R20)",
          flush=True)


if __name__ == "__main__":
    main()
