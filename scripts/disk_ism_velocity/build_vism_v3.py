#!/usr/bin/env python3
"""v3 (corrected): two per-ORIENTATION disk-ISM velocities, giving a value for EVERY
orientation and galaxy. Both use the three tracers (cold gas T<1e4, SF gas SFR>0, young stars
age<300 Myr), density(gas)/mass(stars) weighted, with the LOS velocity computed DIRECTLY in the
orientation (no projection): v_rest = +(v_pec_rest . los) (sign matches the ray / Si II dip).

Sky frame per orientation: rhat = unit(anchor-center) (impact-parameter direction, perp to los),
los, uhat = rhat x los. Any particle: s = pos.rhat (impact coord), w = pos.uhat (transverse),
depth = pos.los, z = pos.n_disk.

v3b -- BINNED IMPACT-PARAMETER SLIT: a rectangular slit through the centre along rhat, s in
  [-40,+40] kpc, half-width R_SLIT (in uhat), disk layer |z|<Z_SLIT, depth |depth|<D_MAX. The
  3-tracer density/mass-weighted LOS velocity is computed in DS-kpc bins of s -> a velocity
  profile v(s). v3b = the profile value at the exact impact parameter s=rho (interpolated). If
  rho is beyond the outermost populated bin (disk ends first), v3b = the disk-edge bin value
  (flagged). This is NOT averaged over the slit, so rotation is not cancelled.

v3a -- ALONG THE ACTUAL SIGHTLINE: 3-tracer LOS velocity in a tube of radius R_TUBE around the
  real sightline (impact param rho), disk layer |z|<Z_SLIT. If the tube is empty (sightline
  beyond the disk), v3a falls back to the v3b disk-edge value (flagged). Thicker tube than
  before so more sightlines are populated directly.

Usage: python build_vism_v3.py <sid>
Output: vism_tables_v3/vism_v3_sid<sid>.csv  +  vism_tables_v3/slitprof_sid<sid>.npz
"""
from __future__ import annotations
import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")  # all-NaN-slice from nanmean over empty slit bins

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
from pm_general import get_geometry, get_original_rho, compute_endpoints, unit, sid_paths  # noqa: E402
import dv_core as dv  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
V1 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3")

R_SLIT = 5.0     # slit half-width (uhat) for the v3b binned profile [kpc]
R_TUBE = 1.0     # v3a tube radius around the ACTUAL sightline [kpc] (tight pencil)
Z_SLIT = 3.0     # disk layer half-thickness [kpc]
D_MAX = 40.0     # LOS depth limit [kpc]
S_MIN, S_MAX, DS = -40.0, 40.0, 2.0
N_MIN = 8        # min particles per bin/tube for a tracer to contribute
SIGN = 1.0
SIGMA_FRAC = 0.1   # disk edge = outermost R (beyond peak) with Sigma_cold > SIGMA_FRAC*peak
EDGES = np.arange(S_MIN, S_MAX + DS, DS)
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])
NB = len(CENT)


def wmean(v, w):
    s = w.sum()
    return float(np.average(v, weights=w)) if (len(v) and s > 0) else np.nan


def binned(s, wt, v, mask):
    """Per-s-bin weighted-mean LOS velocity for particles in mask; NaN where n<N_MIN."""
    bi = np.digitize(s[mask], EDGES) - 1
    ok = (bi >= 0) & (bi < NB)
    bi = bi[ok]; wk = wt[mask][ok]; vk = v[mask][ok]
    sw = np.bincount(bi, weights=wk, minlength=NB)
    swv = np.bincount(bi, weights=wk * vk, minlength=NB)
    n = np.bincount(bi, minlength=NB)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where((n >= N_MIN) & (sw > 0), swv / sw, np.nan)
    return prof


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    rc = dict(np.load(RCV2 / f"rc_sid{sid}.npz"))
    center = rc["center_kpc"]; n_disk = rc["n_disk"]; e1 = rc["e1"]; e2 = rc["e2"]
    cc = np.array(json.loads(sid_paths(sid)["orient_json"].read_text())["center_ckpc_h"])
    sv = np.array(json.loads(sid_paths(sid)["subhalo_json"].read_text())["subhalo_vel_kms"])
    gal = dv.load_galaxy(sid, cc, sv, R_max=S_MAX + R_SLIT + 8.0)
    gpos, gvel, grho, gT, gsfr = gal["gpos"], gal["gvel"], gal["grho"], gal["gT"], gal["gsfr"]
    cold = gT < 1e4; sfg = gsfr > 0
    ym = gal["sage"] <= 0.3
    ypos = gal["spos"][ym]; yvel = gal["svel"][ym]; ymass = gal["sm"][ym]
    zg = gpos @ n_disk; zy = ypos @ n_disk

    # ---- disk extent from the cold-gas surface-density profile (referee-proof edge) ----
    # Sigma_cold(R) in the disk plane (|z|<Z_SLIT); disk edge = outermost radius past the peak
    # where Sigma stays above SIGMA_FRAC x peak. R95/R90 of cold-gas mass are also reported but
    # are inflated by sparse low-density tails, so R_disk_sigma is the operational edge.
    Rcyl = np.hypot(gpos @ e1, gpos @ e2)
    dcold = cold & (np.abs(zg) < Z_SLIT)
    Redges = np.arange(0.0, 42.0, 2.0); Rcen = 0.5 * (Redges[:-1] + Redges[1:])
    massR, _ = np.histogram(Rcyl[dcold], bins=Redges, weights=gal["gm"][dcold])
    areaR = np.pi * (Redges[1:] ** 2 - Redges[:-1] ** 2) * (1e3 ** 2)   # pc^2
    sigmaR = massR / areaR
    speak = float(sigmaR.max()) if sigmaR.size and sigmaR.max() > 0 else 0.0
    ipk = int(np.argmax(sigmaR))
    above = np.where(sigmaR[ipk:] >= SIGMA_FRAC * speak)[0]
    R_disk = float(Rcen[ipk + above[-1]]) if len(above) else float(Rcen[ipk])
    cumR = np.cumsum(massR); cumR = cumR / cumR[-1] if cumR[-1] > 0 else cumR
    R90c = float(np.interp(0.90, cumR, Rcen)); R95c = float(np.interp(0.95, cumR, Rcen))

    v1 = pd.read_csv(V1).set_index(["sid", "mode", "alpha_deg"])

    rows = []; profs = []; keys = []
    for mode in ("flip", "noflip"):
        for alpha in range(360):
            try:
                g = get_geometry(sid, mode, alpha)
            except Exception:
                continue
            los = g["los"]; v_sys = g["v_sys"]
            rho, _, _ = get_original_rho(sid, mode, alpha)
            anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
            rhat = unit(anchor - center); uhat = unit(np.cross(rhat, los))
            sg = gpos @ rhat; wg = gpos @ uhat; dg = gpos @ los
            sy = ypos @ rhat; wy = ypos @ uhat; dy = ypos @ los
            vlg = SIGN * (gvel @ los); vly = SIGN * (yvel @ los)

            # ---- v3b: binned slit profile v(s), 3-tracer average ----
            base_g = (np.abs(wg) < R_SLIT) & (np.abs(zg) < Z_SLIT) & (np.abs(dg) < D_MAX)
            base_y = (np.abs(wy) < R_SLIT) & (np.abs(zy) < Z_SLIT) & (np.abs(dy) < D_MAX)
            pc = binned(sg, grho, vlg, base_g & cold)
            ps = binned(sg, grho, vlg, base_g & sfg)
            py = binned(sy, ymass, vly, base_y)
            prof = np.nanmean(np.vstack([pc, ps, py]), axis=0)
            pop = np.isfinite(prof)
            if pop.any():
                sp = CENT[pop]; vp = prof[pop]
                # read at the impact parameter, but not past the disk edge (principled
                # Sigma-based R_disk, further bounded by where the slit actually has gas)
                edge_R = min(R_disk, float(sp.max()))
                s_read = min(rho, edge_R)
                v3b = float(np.interp(s_read, sp, vp))
                v3b_edge = bool(rho > edge_R)
            else:
                v3b = np.nan; v3b_edge = True; edge_R = np.nan

            # ---- v3a: tube around the actual sightline (impact param rho) ----
            pa_g = np.sqrt((sg - rho) ** 2 + wg ** 2)
            tg = (pa_g < R_TUBE) & (np.abs(zg) < Z_SLIT) & (np.abs(dg) < D_MAX)
            pa_y = np.sqrt((sy - rho) ** 2 + wy ** 2)
            ty = (pa_y < R_TUBE) & (np.abs(zy) < Z_SLIT) & (np.abs(dy) < D_MAX)
            vac = wmean(vlg[tg & cold], grho[tg & cold]) if (tg & cold).sum() >= N_MIN else np.nan
            vas = wmean(vlg[tg & sfg], grho[tg & sfg]) if (tg & sfg).sum() >= N_MIN else np.nan
            vay = wmean(vly[ty], ymass[ty]) if ty.sum() >= N_MIN else np.nan
            av = [x for x in (vac, vas, vay) if np.isfinite(x)]
            v3a_tube = float(np.mean(av)) if av else np.nan
            v3a = v3a_tube if np.isfinite(v3a_tube) else v3b
            v3a_edge = not np.isfinite(v3a_tube)

            rows.append(dict(sid=sid, mode=mode, alpha_deg=alpha, rho_kpc=rho, v_sys=v_sys,
                             v_ism_v3a=v3a, v_ism_v3b=v3b, v3a_tube=v3a_tube,
                             v3a_cold=vac, v3a_sf=vas, v3a_young=vay,
                             n_v3a=int((tg & cold).sum() + (tg & sfg).sum() + ty.sum()),
                             v3a_edge_fallback=v3a_edge, v3b_edge_fallback=v3b_edge,
                             disk_edge_R=edge_R, R_disk_sigma=R_disk, R90_cold=R90c,
                             R95_cold=R95c, n_slit_bins=int(pop.sum())))
            profs.append(prof); keys.append((mode, alpha))
    df = pd.DataFrame(rows)
    v1s = v1.reset_index()
    df = df.merge(v1s[["sid", "mode", "alpha_deg", "SiII_dip", "v_ism_direct_cool", "in_disk"]]
                  .rename(columns={"in_disk": "in_disk_v1"}),
                  on=["sid", "mode", "alpha_deg"], how="left")
    df.to_csv(OUT / f"vism_v3_sid{sid}.csv", index=False)
    np.savez_compressed(OUT / f"slitprof_sid{sid}.npz", centers=CENT,
                        modes=np.array([k[0] for k in keys]),
                        alphas=np.array([k[1] for k in keys]),
                        profiles=np.array(profs), Rcen=Rcen, sigmaR=sigmaR, R_disk=R_disk)
    ind = df[df["in_disk_v1"] == True]
    for col in ("v_ism_v3a", "v_ism_v3b"):
        d = (ind[col] - ind.SiII_dip).dropna()
        print(f"[SID {sid}] {col}: finite {int(df[col].notna().sum())}/{len(df)}; in-disk -dip "
              f"median {d.median():+.1f} sigma {(d-d.median()).abs().median()*1.4826:.1f} (n={len(d)})")
    print(f"  v3a edge-fallback {int(df.v3a_edge_fallback.sum())}; v3b edge-fallback "
          f"{int(df.v3b_edge_fallback.sum())}; both finite {int((df.v_ism_v3a.notna()&df.v_ism_v3b.notna()).sum())}/{len(df)}")


if __name__ == "__main__":
    main()
