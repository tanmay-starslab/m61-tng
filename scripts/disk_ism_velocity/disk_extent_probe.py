#!/usr/bin/env python3
"""Probe 'where the disk ends': cold-gas (T<1e4) radial surface-density profile Sigma(R) in the
disk plane (|z|<Z), R90/R95 of the cold-gas mass, the Sigma-threshold radius, and the number of
cold-gas particles in a 1 kpc vs 5 kpc tube around a sightline at the impact parameter -- to
compare a principled disk-extent definition against the slit population and test R_TUBE=1."""
import os, sys, json
from pathlib import Path
import numpy as np
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2")
from pm_general import sid_paths, get_geometry, get_original_rho, compute_endpoints, unit
import dv_core as dv

KPC2PC = 1e3
for sid in [int(x) for x in sys.argv[1:]] or [342448, 143886, 432106]:
    rc = dict(np.load(f"/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2/rc_sid{sid}.npz"))
    center = rc["center_kpc"]; n_disk = rc["n_disk"]; e1 = rc["e1"]; e2 = rc["e2"]
    cc = np.array(json.loads(sid_paths(sid)["orient_json"].read_text())["center_ckpc_h"])
    sv = np.array(json.loads(sid_paths(sid)["subhalo_json"].read_text())["subhalo_vel_kms"])
    gal = dv.load_galaxy(sid, cc, sv, R_max=48.)
    gpos, gm, gT = gal["gpos"], gal["gm"], gal["gT"]
    cold = gT < 1e4
    Rd = np.hypot(gpos @ e1, gpos @ e2); zd = gpos @ n_disk
    disk = cold & (np.abs(zd) < 3.0)
    R = Rd[disk]; m = gm[disk]
    # radial cold-gas mass + surface-density profile
    edges = np.arange(0, 42, 2.0); cen = 0.5 * (edges[:-1] + edges[1:])
    mass, _ = np.histogram(R, bins=edges, weights=m)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * KPC2PC ** 2  # pc^2
    sigma = mass / area  # [mass unit]/pc^2
    cum = np.cumsum(mass); cum = cum / cum[-1]
    R90 = np.interp(0.90, cum, cen); R95 = np.interp(0.95, cum, cen)
    # Sigma threshold: where Sigma drops below 5% of its peak (illustrative)
    speak = np.nanmax(sigma); thr = 0.05 * speak
    below = np.where(sigma < thr)[0]
    R_thr = cen[below[0]] if len(below) and below[0] > np.argmax(sigma) else np.nan
    # particle counts in a tube at the impact parameter (mode flip alpha 0)
    g = get_geometry(sid, "flip", 0); los = g["los"]; rho, _, _ = get_original_rho(sid, "flip", 0)
    anchor = compute_endpoints(sid, "flip", 0, rho, 50.)["anchor_kpc"]
    rhat = unit(anchor - center); uhat = unit(np.cross(rhat, los))
    s = gpos @ rhat; w = gpos @ uhat; perp = np.sqrt((s - rho) ** 2 + w ** 2)
    for Rt in (1.0, 3.0, 5.0):
        n = int((cold & (perp < Rt) & (np.abs(zd) < 3.0)).sum())
        print(f"  SID {sid}: cold particles in R_TUBE={Rt:g} kpc tube @rho={rho:.1f}: n={n}")
    print(f"  SID {sid}: R90={R90:.1f}  R95={R95:.1f}  R(Sigma<5%peak)={R_thr:.1f} kpc  "
          f"(rho={rho:.1f})  peakSigma@R={cen[np.argmax(sigma)]:.0f}")
    print(f"     Sigma(R) [norm to peak]: " +
          " ".join(f"{c:.0f}:{sigma[i]/speak:.2f}" for i, c in enumerate(cen) if c <= 34))
    print()
