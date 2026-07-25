#!/usr/bin/env python3
"""Tier 2a + 2d: recompute-machinery closure test and UVB/ionization-parameter sensitivity.

2a: recompute the 6 NATIVE ions (H I, C II, N V, Si II, Si III, Si IV) with the SAME
    trident.add_ion_fields path we use for O VI/C IV, and compare per-sightline columns to
    the ions stored in the original ray. Agreement proves the recompute reproduces the
    production ionization -> the O VI/C IV columns (made identically) are trustworthy.
2d: recompute O VI / C IV with the gas density scaled x0.5 and x2. At fixed UVB the
    ionization parameter U ~ 1/n, so this brackets a factor-2 UVB-amplitude uncertainty
    (~ HM2012 vs FG2009/FG2020). Report d logN.

Samples every 36 deg (10 alpha x 2 modes = 20 sightlines/SID) for speed.
Usage: python v2ad_recompute_uvb.py <sid>
Output: validation/tier2/recompute_uvb_sid<sid>.parquet
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
import yt    # noqa: E402
import trident  # noqa: E402
yt.set_log_level(50)
import ray_ism_diagnostic as R  # noqa: E402

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier2")
NATIVE = {"HI": ("H I", "H_p0_number_density"), "CII": ("C II", "C_p1_number_density"),
          "NV": ("N V", "N_p4_number_density"), "SiII": ("Si II", "Si_p1_number_density"),
          "SiIII": ("Si III", "Si_p2_number_density"), "SiIV": ("Si IV", "Si_p3_number_density")}
EXTRA = {"OVI": ("O VI", "O_p5_number_density"), "CIV": ("C IV", "C_p3_number_density")}


def write_ray(rayh, tmp, dens_scale=1.0):
    """Copy the ray; optionally scale the hydrogen/density fields that index the Trident
    ion-fraction table by `dens_scale`. Scaling n_H by kappa emulates ionization parameter
    U -> U/kappa, i.e. UVB amplitude x(1/kappa). NOTE this also scales the element amount by
    kappa, so the caller subtracts log10(kappa) to isolate the pure ion-fraction (UVB) shift."""
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(tmp, "w") as out:
        for k, v in rayh.attrs.items():
            out.attrs[k] = v
        rayh.copy("grid", out)
        if dens_scale != 1.0:
            for f in ("density", "H_number_density", "H_nuclei_density"):
                if f"grid/{f}" in out:
                    out[f"grid/{f}"][...] = out[f"grid/{f}"][()] * dens_scale  # in-place: keep attrs
    return tmp


def columns(tmp, ions, fields, dl_cm):
    ds = yt.load(tmp)
    trident.add_ion_fields(ds, ions=ions, ftype="gas")
    ad = ds.all_data()
    return {k: float((np.array(ad["gas", f].to("cm**-3")) * dl_cm).sum())
            for k, f in fields.items()}


def main():
    sid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = f"/tmp/_v2ad_{sid}_{os.getpid()}.h5"
    rows = []
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for alpha in range(0, 360, 36):
                ag = f"alpha={alpha}"
                if ag not in base[f"mode={mode}"]:
                    continue
                grp = base[f"mode={mode}"][ag]
                rayh = grp[list(grp.keys())[0]]["original_trident_ray_h5"]
                gr = rayh["grid"]
                dl_cm = gr["dl"][()]
                stored = {k: float((gr[f][()] * dl_cm).sum()) for k, (_, f) in NATIVE.items()}
                # baseline recompute (native + extra)
                write_ray(rayh, tmp, 1.0)
                allf = {**{k: v[1] for k, v in NATIVE.items()},
                        **{k: v[1] for k, v in EXTRA.items()}}
                allions = [v[0] for v in NATIVE.values()] + [v[0] for v in EXTRA.values()]
                rec = columns(tmp, allions, allf, dl_cm)
                # UVB proxy: density x0.5, x2 (O VI / C IV only)
                exf = {k: v[1] for k, v in EXTRA.items()}
                exion = [v[0] for v in EXTRA.values()]
                write_ray(rayh, tmp, 0.5); lo = columns(tmp, exion, exf, dl_cm)
                write_ray(rayh, tmp, 2.0); hi = columns(tmp, exion, exf, dl_cm)
                row = dict(sid=sid, mode=mode, alpha=alpha)
                for k in NATIVE:
                    row[f"stored_{k}"] = stored[k]; row[f"recomp_{k}"] = rec[k]
                for k in EXTRA:
                    row[f"base_{k}"] = rec[k]; row[f"lo_{k}"] = lo[k]; row[f"hi_{k}"] = hi[k]
                rows.append(row)
    if os.path.exists(tmp):
        os.remove(tmp)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"recompute_uvb_sid{sid}.parquet", index=False)
    # quick console summary
    for k in NATIVE:
        a = np.log10(df[f"stored_{k}"].replace(0, np.nan)); b = np.log10(df[f"recomp_{k}"].replace(0, np.nan))
        d = (b - a).dropna()
        print(f"  {k:5s} recomp-stored dlogN median {d.median():+.3f} MAD {(d-d.median()).abs().median():.3f}")
    L2 = np.log10(2.0)
    for k in EXTRA:
        base = np.log10(df[f"base_{k}"].replace(0, np.nan))
        # fixed-gas ion-fraction shift = dlogN(scaled n_H) - log(scale)
        uvb2 = (np.log10(df[f"lo_{k}"].replace(0, np.nan)) - base + L2).dropna()   # rho x0.5 -> UVB x2
        uvbh = (np.log10(df[f"hi_{k}"].replace(0, np.nan)) - base - L2).dropna()   # rho x2   -> UVB x0.5
        print(f"  {k}: fixed-gas dlogN  (UVB x2) {uvb2.median():+.2f}  (UVB x0.5) {uvbh.median():+.2f}")


if __name__ == "__main__":
    main()
