#!/usr/bin/env python3
"""v3: two per-ORIENTATION disk-ISM velocities from the actual cutout gas/stars.

Both average the three supervisor tracers -- cold gas (T<1e4), SF gas (SFR>0), young stars
(age<300 Myr) -- weighted by density (gas) / mass (stars), with v_rest = -(v_pec_rest . los)
(galaxy rest frame; sign matches the ray convention, validated vs the Si II dip). They differ
only in WHICH gas is sampled:

  v3a  ALONG THE SIGHTLINE: tracers in a thin tube (perp < R_TUBE) around the actual line of
       sight, near the disk plane (|z_disk| < Z_THICK, |path| < T_MAX). This is the gas the
       QSO actually shines through -> lands on the absorption; custom per orientation.
  v3b  CENTER -> IMPACT probe-line: tracers within R_AP of the line center + s*d_hat,
       s in [-S_MAX, +S_MAX], d_hat = unit(anchor - center). Literal "line to the impact
       point, extended +-40 kpc". Custom per orientation but averages the impact side with its
       diametric opposite (rotation partly cancels).

Usage: python build_vism_v3.py <sid>
Output: vism_tables_v3/vism_v3_sid<sid>.csv  (v_ism_v3a, v_ism_v3b + components)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
from pm_general import get_geometry, get_original_rho, compute_endpoints, unit, sid_paths  # noqa: E402
import dv_core as dv  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402  (r95_cold_gas)

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
V1 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3")
# v3a (sightline tube)
R_TUBE = 3.0; Z_THICK = 2.0; T_MAX = 40.0
# v3b (center->impact probe-line)
S_MAX = 40.0; R_AP = 5.0
N_MIN = 10; SIGN = 1.0   # v_rest = +(v_pec_rest . los); matches ray/direct convention (calibrated)


def wmean(v, w):
    s = w.sum()
    return float(np.average(v, weights=w)) if (len(v) and s > 0) else np.nan


def avg_tracers(*vals):
    ok = [v for v in vals if np.isfinite(v)]
    return (float(np.mean(ok)), len(ok)) if ok else (np.nan, 0)


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    rc = dict(np.load(RCV2 / f"rc_sid{sid}.npz"))
    center = rc["center_kpc"]; n_disk = rc["n_disk"]; e1 = rc["e1"]; e2 = rc["e2"]
    cc = np.array(json.loads(sid_paths(sid)["orient_json"].read_text())["center_ckpc_h"])
    sv = np.array(json.loads(sid_paths(sid)["subhalo_json"].read_text())["subhalo_vel_kms"])
    gal = dv.load_galaxy(sid, cc, sv, R_max=S_MAX + R_AP + 5.0)
    gpos, gvel, grho, gT, gsfr = gal["gpos"], gal["gvel"], gal["grho"], gal["gT"], gal["gsfr"]
    cold = gT < 1e4; sfg = gsfr > 0
    # pre-filter young stars (small subset) for speed
    ym = gal["sage"] <= 0.3
    ypos = gal["spos"][ym]; yvel = gal["svel"][ym]; ymass = gal["sm"][ym]
    zg = gpos @ n_disk
    R_edge = R.r95_cold_gas(sid)
    v1 = pd.read_csv(V1).set_index(["sid", "mode", "alpha_deg"])

    rows = []
    for mode in ("flip", "noflip"):
        for alpha in range(360):
            try:
                g = get_geometry(sid, mode, alpha)
            except Exception:
                continue
            los = g["los"]; v_sys = g["v_sys"]
            rho, _, _ = get_original_rho(sid, mode, alpha)
            anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
            arel = anchor - center; d_hat = unit(arel)
            vrest_g = SIGN * (gvel @ los)
            vrest_y = SIGN * (yvel @ los)

            # --- v3a: tube around the sightline, near the disk plane ---
            tg = (gpos - arel) @ los
            pg = (gpos - arel) - tg[:, None] * los
            dperp_g = np.sqrt(np.einsum("ij,ij->i", pg, pg))
            a_g = (dperp_g < R_TUBE) & (np.abs(zg) < Z_THICK) & (np.abs(tg) < T_MAX)
            ty = (ypos - arel) @ los
            py = (ypos - arel) - ty[:, None] * los
            dperp_y = np.sqrt(np.einsum("ij,ij->i", py, py))
            zy = ypos @ n_disk
            a_y = (dperp_y < R_TUBE) & (np.abs(zy) < Z_THICK) & (np.abs(ty) < T_MAX)

            # --- v3b: cylinder around center->impact line ---
            sg = gpos @ d_hat
            pgb = gpos - sg[:, None] * d_hat
            dperp_gb = np.sqrt(np.einsum("ij,ij->i", pgb, pgb))
            b_g = (np.abs(sg) <= S_MAX) & (dperp_gb < R_AP)
            sy = ypos @ d_hat
            pyb = ypos - sy[:, None] * d_hat
            dperp_yb = np.sqrt(np.einsum("ij,ij->i", pyb, pyb))
            b_y = (np.abs(sy) <= S_MAX) & (dperp_yb < R_AP)

            def tracer(mask_g, mask_y):
                mc = mask_g & cold; ms = mask_g & sfg
                vc = wmean(vrest_g[mc], grho[mc]) if mc.sum() >= N_MIN else np.nan
                vs = wmean(vrest_g[ms], grho[ms]) if ms.sum() >= N_MIN else np.nan
                vy = wmean(vrest_y[mask_y], ymass[mask_y]) if mask_y.sum() >= N_MIN else np.nan
                v, nt = avg_tracers(vc, vs, vy)
                return v, vc, vs, vy, nt, int(mc.sum()), int(ms.sum()), int(mask_y.sum())

            va, vac, vas, vay, na, nac, nas, nay = tracer(a_g, a_y)
            vb, vbc, vbs, vby, nb, nbc, nbs, nby = tracer(b_g, b_y)
            R_anchor = float(np.hypot(arel @ e1, arel @ e2))
            rec = dict(sid=sid, mode=mode, alpha_deg=alpha, rho_kpc=rho, v_sys=v_sys,
                       R_anchor=R_anchor, R_edge=R_edge,
                       v_ism_v3a=va, v3a_cold=vac, v3a_sf=vas, v3a_young=vay,
                       n_v3a_cold=nac, n_v3a_sf=nas, n_v3a_young=nay, n_v3a_tracers=na,
                       v_ism_v3b=vb, v3b_cold=vbc, v3b_sf=vbs, v3b_young=vby,
                       n_v3b_cold=nbc, n_v3b_sf=nbs, n_v3b_young=nby, n_v3b_tracers=nb)
            try:
                r1 = v1.loc[(sid, mode, alpha)]
                rec["SiII_dip"] = float(r1.SiII_dip)
                rec["v_ism_direct_cool"] = float(r1.v_ism_direct_cool)
                rec["in_disk_v1"] = bool(r1.in_disk)
            except KeyError:
                rec["SiII_dip"] = rec["v_ism_direct_cool"] = np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    p = OUT / f"vism_v3_sid{sid}.csv"; df.to_csv(p, index=False)
    ind = df[df["in_disk_v1"] == True] if "in_disk_v1" in df else df
    for col in ("v_ism_v3a", "v_ism_v3b", "v_ism_direct_cool"):
        for lab, sub in (("all", df), ("in_disk", ind)):
            d = (sub[col] - sub.SiII_dip).dropna()
            if len(d):
                std_a = sub.groupby("mode")[col].std().mean()
                print(f"[SID {sid}] {col:18s} [{lab:7s}] -dip median {d.median():+6.1f} "
                      f"sigma {(d-d.median()).abs().median()*1.4826:6.1f}; std/alpha {std_a:5.1f}; n={len(d)}")
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
