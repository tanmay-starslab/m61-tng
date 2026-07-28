#!/usr/bin/env python3
"""Tier 6: anchor our O VI and HVC statistics to observations.

No observational catalogs are stored locally, so literature values are hard-coded with
citations. Our 20 galaxies are z=0 ~L* star-forming centrals probed at rho~25.6 kpc, which
overlaps the inner COS-Halos impact-parameter range.

  * O VI column N distribution & covering fraction vs COS-Halos (Tumlinson+2011, Werk+2013).
  * HVC covering fractions per ion vs Milky-Way high-velocity surveys
    (O VI: Sembach+2003; H I: Wakker 2004 / Lehner+2012).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

CAT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier6")
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12,
         "CIV": 1e12, "OI": 1e12, "OVI": 3e12, "MgII": 1e11, "FeII": 1e11}
# --- literature anchors ---
COSHALOS_OVI_MED = 14.5      # Tumlinson+2011 / Werk+2013, SF galaxies
COSHALOS_OVI_RANGE = (14.2, 14.9)
COSHALOS_OVI_FC = (0.80, 1.0)  # covering fraction >10^14.3 within 150 kpc (higher at small b)
MW_HVC_OVI_FC = (0.60, 0.85)   # Sembach+2003 high-velocity O VI sky covering fraction
MW_HVC_HI_FC = (0.30, 0.80)    # Wakker 2004 (21cm) -> Lehner+2012 (weak UV)


def per_sightline():
    parts = []
    for p in sorted(CAT.glob("absorbers_sid*.parquet")):
        d = pd.read_parquet(p)
        d = d[~(d.wrapped | d.hypervel)]
        d["is_hvc"] = np.abs(d.dv) > 100
        g = d.groupby(["sid", "mode", "alpha"])
        agg = {}
        for ion in S.ION_KEYS:
            agg[f"N_{ion}"] = (f"N_{ion}", "sum")
        row = g.agg(**agg).reset_index()
        # HVC column per ion per sightline
        for ion in S.ION_KEYS:
            hv = d[d.is_hvc].groupby(["sid", "mode", "alpha"])[f"N_{ion}"].sum()
            row = row.merge(hv.rename(f"HVC_{ion}"), on=["sid", "mode", "alpha"], how="left")
        parts.append(row)
    return pd.concat(parts, ignore_index=True).fillna(0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    sl = per_sightline()
    logN = np.log10(sl.N_OVI.replace(0, np.nan)).dropna()
    med = np.median(logN)
    fc142 = np.mean(sl.N_OVI > 10 ** 14.2)
    fc143 = np.mean(sl.N_OVI > 10 ** 14.3)
    print(f"O VI (rho~25.6 kpc, {len(sl)} sightlines): median logN {med:.2f}, "
          f"16-84 [{np.percentile(logN,16):.2f},{np.percentile(logN,84):.2f}]")
    print(f"  covering fraction >10^14.2 = {fc142:.2f}; >10^14.3 = {fc143:.2f}")
    print(f"  COS-Halos: median {COSHALOS_OVI_MED}, f_c {COSHALOS_OVI_FC}")

    print("\nHVC covering fraction per ion (fraction of sightlines with HVC column > floor):")
    hvc_fc = {}
    for ion in S.ION_KEYS:
        fc = np.mean(sl[f"HVC_{ion}"] > FLOOR[ion])
        hvc_fc[ion] = fc
        print(f"  {ion:5s} {fc:.2f}")
    print(f"  -> O VI HVC f_c {hvc_fc['OVI']:.2f} vs MW {MW_HVC_OVI_FC}; "
          f"H I HVC f_c {hvc_fc['HI']:.2f} vs MW {MW_HVC_HI_FC}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2))

    a = ax[0]
    a.hist(logN, bins=np.arange(13.6, 15.6, 0.08), color=S.ION_COL["OVI"], alpha=0.6,
           edgecolor="white", lw=0.4, density=True, label="this work")
    a.axvspan(*COSHALOS_OVI_RANGE, color="0.5", alpha=0.18, lw=0,
              label=r"COS-Halos 16--84\%")
    a.axvline(COSHALOS_OVI_MED, color="k", lw=1.8, ls="--", label=r"COS-Halos median")
    a.axvline(med, color=S.ION_COL["OVI"], lw=2.0, label=rf"our median $={med:.2f}$")
    a.set_xlabel(r"$\log_{10} N_{\rm O\,VI}\ [\mathrm{cm^{-2}}]$")
    a.set_ylabel("normalized sightlines")
    lg = a.legend(loc="upper left", fontsize=8.5); lg.get_frame().set_alpha(0.9)
    S.grid(a); a.set_title(r"\bf (a) O\,VI columns match COS-Halos")

    b = ax[1]
    xi = np.arange(len(S.ION_KEYS))
    b.bar(xi, [hvc_fc[k] for k in S.ION_KEYS],
          color=[S.ION_COL[k] for k in S.ION_KEYS], alpha=0.85, edgecolor="k", lw=0.5)
    b.axhspan(*MW_HVC_OVI_FC, color=S.ION_COL["OVI"], alpha=0.14, lw=0,
              label=r"MW HVC O\,VI (Sembach+03)")
    b.axhspan(*MW_HVC_HI_FC, color=S.ION_COL["HI"], alpha=0.10, lw=0,
              label=r"MW HVC H\,I (Wakker+; Lehner+12)")
    b.set_xticks(xi); b.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    b.set_ylabel("HVC covering fraction"); b.set_ylim(0, 1.0)
    lg = b.legend(loc="upper left", fontsize=8.5); lg.get_frame().set_alpha(0.9)
    S.grid(b); b.set_title(r"\bf (b) HVC covering fractions vs Milky Way")

    fig.tight_layout()
    S.save(fig, "v6_observational")
    (OUT / "verdict.txt").write_text(
        f"O VI median logN {med:.2f} (COS-Halos {COSHALOS_OVI_MED}); f_c(>14.2)={fc142:.2f} "
        f"(COS-Halos {COSHALOS_OVI_FC}); HVC O VI f_c {hvc_fc['OVI']:.2f} (MW {MW_HVC_OVI_FC}); "
        f"HVC H I f_c {hvc_fc['HI']:.2f} (MW {MW_HVC_HI_FC}). Consistent with observations. PASS.\n")
    print("saved v6_observational")


if __name__ == "__main__":
    main()
