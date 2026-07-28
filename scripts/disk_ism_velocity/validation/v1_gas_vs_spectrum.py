#!/usr/bin/env python3
"""Tier 1 validation: gas-column-weighted velocity vs mock-spectrum AOD velocity.

For every sightline of one SID and each of the 6 in-band ions (H I, C II, Si II,
Si III, Si IV, N V) we measure:
  * gas_v   = ion-column-weighted v_rest of the ray gas cells (weight = n_ion * dl),
              excluding wrapped / hyper-velocity cells (|v_rest|>VWIN),
  * aod_v   = optical-depth-weighted velocity of the mock spectrum (raw tau),
  * dipflux_v = flux-decrement-weighted centroid (robust to saturation),
both on the SAME corrected axis  v = -(C*(lam/lam0-1)) - v_sys = -velocity_los/1e5 - v_sys.

PHYSICS: in the optically-thin limit tau(v) ∝ (ion column per velocity bin), so
aod_v == gas_v exactly. Saturated lines (H I, strong Si II) bias the AOD centroid
toward the wings -> the match should be tightest for thin lines and degrade with
tau_max, which is the expected, method-validating signature. Because O VI/C IV are
computed from the same gas with the same velocity field, a clean match up through
N V (log T~5.3, adjacent to O VI ~5.5) legitimizes the gas-based O VI velocity.

Usage: python v1_gas_vs_spectrum.py <sid>
Output: validation/tier1/gasvspec_sid<sid>.parquet
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
from pm_general import C_KMS, CM_PER_KPC, get_geometry  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier1")
VWIN = 700.0   # km/s window; also the hyper-velocity ceiling used in the catalog mask

# spectrum line key -> (rest wavelength [A], gas ion number-density field, oscillator strength f)
LINES = {
    "H_I_1216":   (1215.6701, "H_p0_number_density",  0.4164),
    "C_II_1335":  (1334.5323, "C_p1_number_density",  0.1278),
    "Si_III_1206": (1206.500, "Si_p2_number_density", 1.669),
    "N_V_1239":   (1238.821,  "N_p4_number_density",  0.1570),
    "Si_II_1260": (1260.4221, "Si_p1_number_density", 1.180),
    "Si_IV_1403": (1402.770,  "Si_p3_number_density", 0.2553),
}
ION_KEY = {"H_I_1216": "HI", "C_II_1335": "CII", "Si_III_1206": "SiIII",
           "N_V_1239": "NV", "Si_II_1260": "SiII", "Si_IV_1403": "SiIV"}
AOD_CONST = 2.654e-15  # pi e^2 / (m_e c) in cgs, for N = tau / (AOD_CONST*f*lam0) integrated dv


def wmean(v, w):
    s = w.sum()
    return float(np.average(v, weights=w)) if s > 0 else np.nan


def process(sid):
    rows = []
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for ag in sorted(base[f"mode={mode}"].keys(), key=lambda k: int(k.split("=")[1])):
                alpha = int(ag.split("=")[1])
                g = get_geometry(sid, mode, alpha)
                v_sys = g["v_sys"]
                grp = R.ray_group(h, mode, alpha)
                grid = grp["original_trident_ray_h5/grid"]
                vlos = grid["velocity_los"][()] / 1e5
                v_rest = -vlos - v_sys
                dl_cm = grid["dl"][()]
                # gas cells kept for the fair comparison: physical velocity window only
                keep = np.abs(v_rest) < VWIN
                for key, (rest, fld, fval) in LINES.items():
                    if fld not in grid:
                        continue
                    n_ion = grid[fld][()]
                    w = n_ion * dl_cm
                    wk = w * keep
                    gas_v = wmean(v_rest, wk) if wk.sum() > 0 else np.nan
                    N_gas = float((n_ion * dl_cm).sum())
                    # spectrum (raw = intrinsic tau, no LSF)
                    sg = grp.get(f"spectrum_by_line/{key}/raw")
                    aod_v = dip_v = np.nan
                    tau_max = tau_sum = N_aod = np.nan
                    if sg is not None:
                        lam = sg["lambda_A"][()]; tau = sg["tau"][()]; flux = sg["flux"][()]
                        vv = -(C_KMS * (lam / rest - 1.0)) - v_sys
                        m = np.abs(vv) < VWIN
                        if m.any() and tau[m].sum() > 0:
                            aod_v = wmean(vv[m], tau[m])
                            tau_max = float(tau[m].max()); tau_sum = float(tau[m].sum())
                            dv_pix = float(np.abs(np.median(np.diff(vv[m]))))
                            N_aod = float(tau[m].sum() * dv_pix * 1e5 / (AOD_CONST * fval * rest))
                        dec = np.clip(1.0 - flux, 0.0, None)
                        if m.any() and dec[m].sum() > 0:
                            dip_v = wmean(vv[m], dec[m])
                    rows.append(dict(sid=sid, mode=mode, alpha=alpha, ion=ION_KEY[key],
                                     gas_v=gas_v, aod_v=aod_v, dipflux_v=dip_v,
                                     N_gas=N_gas, N_aod=N_aod, tau_max=tau_max, tau_sum=tau_sum,
                                     v_sys=float(v_sys)))
    return pd.DataFrame(rows)


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    df = process(sid)
    p = OUT / f"gasvspec_sid{sid}.parquet"
    df.to_parquet(p, index=False)
    ok = df.dropna(subset=["gas_v", "aod_v"])
    d = ok.aod_v - ok.gas_v
    print(f"[SID {sid}] {len(df)} rows; matched {len(ok)}; "
          f"median(aod-gas)={d.median():+.1f} km/s  MAD={(d - d.median()).abs().median():.1f}")
    for ion in ["HI", "CII", "SiII", "SiIII", "SiIV", "NV"]:
        s = ok[ok.ion == ion]
        if len(s):
            dd = s.aod_v - s.gas_v
            print(f"   {ion:5s} n={len(s):4d}  med(aod-gas)={dd.median():+6.1f}  "
                  f"MAD={(dd - dd.median()).abs().median():5.1f}  <tau_max>={s.tau_max.median():.2f}")


if __name__ == "__main__":
    main()
