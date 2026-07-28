#!/usr/bin/env python3
"""Tier 2b: per-particle-oxygen vs solar-scaled-metallicity audit for O VI columns.

Our recomputed O VI/C IV (trident.add_ion_fields on the extracted ray) scale oxygen by
TOTAL metallicity x Trident's solar abundance pattern. OVI_CGM_compare uses the TNG
per-particle oxygen (GFM_Metals[:,4]). The systematic on log N_OVI is exactly

    Delta log N_OVI = log10[ (O/Z)_TNG,CGM / (O/Z)_sun,Trident ]

(everything else -- density, T, UVB, velocity -- is identical). This script measures
both ratios directly:
  * (O/Z)_sun,Trident : from a recomputed ray, mass ratio n_O*m_O / (rho*Z).
  * (O/Z)_TNG,CGM     : from the cutout GFM_Metals for O VI-phase gas (T 10^5.2-10^5.8).

Usage: python v2b_oxygen_audit.py [sid]   (default 488530, the OVI_CGM_compare overlap)
"""
from __future__ import annotations
import os, sys, glob
from pathlib import Path
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
import yt    # noqa: E402
import trident  # noqa: E402
yt.set_log_level(50)
import ray_ism_diagnostic as R  # noqa: E402

M_P = 1.67262192e-24      # g
M_O = 15.999 * 1.66053907e-24
K_B = 1.380649e-16
GAMMA = 5.0 / 3.0
X_H = 0.76


def trident_solar_OZ(sid):
    """Trident's assumed (O/Z) mass ratio, from add_ion_fields on one extracted ray."""
    tmp = f"/tmp/_oxaudit_{sid}_{os.getpid()}.h5"
    with h5py.File(R.combined_path(sid), "r") as h:
        grp = R.ray_group(h, "flip", 0)
        rayh = grp["original_trident_ray_h5"]
        if os.path.exists(tmp):
            os.remove(tmp)
        with h5py.File(tmp, "w") as out:
            for k, v in rayh.attrs.items():
                out.attrs[k] = v
            rayh.copy("grid", out)
    ds = yt.load(tmp)
    trident.add_ion_fields(ds, ions=["O VI", "C IV"], ftype="gas")
    ad = ds.all_data()
    # Trident exposes only the ion; total oxygen it assumed = n_OVI / ion_fraction.
    nOp5 = np.array(ad["gas", "O_p5_number_density"].to("cm**-3"))
    fOp5 = np.array(ad["gas", "O_p5_ion_fraction"])
    rho = np.array(ad["gas", "density"].to("g/cm**3"))
    Z = np.array(ad["gas", "metallicity"])
    os.remove(tmp)
    m = (Z > 1e-6) & (rho > 0) & (fOp5 > 1e-3)
    nO = nOp5[m] / fOp5[m]
    OZ = (nO * M_O) / (rho[m] * Z[m])   # O mass / total-metal mass
    return float(np.median(OZ)), int(m.sum())


def tng_cgm_OZ(sid):
    """TNG true (O/Z) mass ratio for O VI-phase CGM gas (T 10^5.2-10^5.8 K)."""
    p = glob.glob(f"/data/sborthak/m61/cutouts/out_sub_{sid}/cutout_ALLFIELDS*.hdf5")[0]
    with h5py.File(p, "r") as h:
        g = h["PartType0"]
        Z = g["GFM_Metallicity"][()]
        met = g["GFM_Metals"][()]
        U = g["InternalEnergy"][()].astype(np.float64)     # (km/s)^2
        xe = g["ElectronAbundance"][()].astype(np.float64)
    O = met[:, 4]                                          # oxygen mass fraction
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    T = (GAMMA - 1.0) * (U * 1e10) * mu * M_P / K_B        # K
    logT = np.log10(np.clip(T, 1.0, None))
    phase = (logT > 5.2) & (logT < 5.8) & (Z > 1e-6)
    OZ_all = (O[Z > 1e-6] / Z[Z > 1e-6]).astype(np.float64)
    OZ_ovi = (O[phase] / Z[phase]).astype(np.float64)
    return float(np.median(OZ_ovi)), float(np.median(OZ_all)), int(phase.sum())


def main():
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 488530
    OZ_tri, nray = trident_solar_OZ(sid)
    OZ_ovi, OZ_all, nphase = tng_cgm_OZ(sid)
    print(f"=== Tier 2b oxygen audit  SID {sid} ===")
    print(f"  (O/Z)_sun,Trident (mass)     = {OZ_tri:.3f}   [{nray} ray cells]")
    print(f"  (O/Z)_TNG all metal-gas      = {OZ_all:.3f}")
    print(f"  (O/Z)_TNG O VI-phase gas     = {OZ_ovi:.3f}   [{nphase} particles, logT 5.2-5.8]")
    dex_ovi = np.log10(OZ_ovi / OZ_tri)
    dex_all = np.log10(OZ_all / OZ_tri)
    print(f"  Delta logN_OVI (true - ours) = {dex_ovi:+.3f} dex  (O VI-phase gas)")
    print(f"  Delta logN     (all gas)     = {dex_all:+.3f} dex")
    print(f"  => our metallicity-scaled O VI is {'LOW' if dex_ovi>0 else 'HIGH'} by "
          f"{abs(dex_ovi):.3f} dex vs a true-oxygen calculation.")


if __name__ == "__main__":
    main()
