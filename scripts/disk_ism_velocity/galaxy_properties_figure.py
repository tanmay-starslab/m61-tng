#!/usr/bin/env python3
"""Stellar mass, M200c, SFR of the 20 study galaxies vs the Milky Way. Catalog values from the
TNG API (galaxy_properties_tng.csv). Figure: (a) Mstar-M200c with the MW marked; (b) SFR-Mstar
star-forming main sequence. MW-type flags applied and printed."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S

OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/galaxy_properties")
d = pd.read_csv(OUT / "galaxy_properties_tng.csv")
# Milky Way fiducials (Licquia&Newman 2015; Bland-Hawthorn&Gerhard 2016)
MW = dict(Mstar=5.0e10, M200c=1.15e12, SFR=1.65, sSFR=3.3e-11)
# MW-type flags
d["MWmass_halo"] = (d.M200c_Msun > 0.6e12) & (d.M200c_Msun < 2.0e12)
d["MWmass_star"] = (d.Mstar_Msun > 3e10) & (d.Mstar_Msun < 8e10)
d.to_csv(OUT / "galaxy_properties_tng.csv", index=False)

S.set_style()
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
cen = d[d.central]; sat = d[~d.central]
a = ax[0]
a.scatter(sat.M200c_Msun, sat.Mstar_Msun, s=70, marker="s", color="#7B6FB0",
          edgecolor="k", lw=0.5, label="satellite", zorder=4)
a.scatter(cen.M200c_Msun, cen.Mstar_Msun, s=70, marker="o", color="#1B9E77",
          edgecolor="k", lw=0.5, label="central", zorder=4)
a.scatter([MW["M200c"]], [MW["Mstar"]], s=340, marker="*", color="#E8B10A",
          edgecolor="k", lw=1.0, zorder=6, label="Milky Way")
a.axvspan(0.6e12, 2.0e12, color="#E8B10A", alpha=0.10, lw=0)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$M_{\rm 200c}$ (parent halo) [$M_\odot$]")
a.set_ylabel(r"$M_\star$ [$M_\odot$]")
a.set_title(r"\bf (a) stellar mass vs halo mass")
a.legend(fontsize=9, loc="lower right"); S.grid(a)
a.text(0.62e12, 0.9e11, "MW-mass\nhalo", color="#9a7d00", fontsize=8, ha="left")

b = ax[1]
b.scatter(sat.Mstar_Msun, sat.SFR_Msun_yr, s=70, marker="s", color="#7B6FB0", edgecolor="k", lw=0.5, label="satellite", zorder=4)
b.scatter(cen.Mstar_Msun, cen.SFR_Msun_yr, s=70, marker="o", color="#1B9E77", edgecolor="k", lw=0.5, label="central", zorder=4)
b.scatter([MW["Mstar"]], [MW["SFR"]], s=340, marker="*", color="#E8B10A", edgecolor="k", lw=1.0, zorder=6, label="Milky Way")
xx = np.logspace(10.5, 11.7, 50)
for ss, ls, lab in [(1e-10, "--", r"sSFR$=10^{-10}$"), (1e-11, ":", r"sSFR$=10^{-11}$")]:
    b.plot(xx, ss * xx, color="0.5", ls=ls, lw=1.0, label=lab)
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel(r"$M_\star$ [$M_\odot$]"); b.set_ylabel(r"SFR [$M_\odot\,{\rm yr}^{-1}$]")
b.set_title(r"\bf (b) star-forming main sequence")
b.legend(fontsize=8, loc="lower right"); S.grid(b)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"galaxy_properties.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved galaxy_properties.png/pdf")
print(f"\nSample: {int(d.central.sum())} centrals, {int((~d.central).sum())} satellites")
print(f"logMstar range {d.logMstar.min():.2f}-{d.logMstar.max():.2f} (MW={np.log10(MW['Mstar']):.2f})")
print(f"logM200c centrals {cen.logM200c.min():.2f}-{cen.logM200c.max():.2f} (MW={np.log10(MW['M200c']):.2f})")
print(f"SFR range {d.SFR_Msun_yr.min():.1f}-{d.SFR_Msun_yr.max():.1f} (MW={MW['SFR']})")
print("\nMW-mass halo (0.6-2e12):", d[d.MWmass_halo].sid.tolist())
print("MW-mass stellar (3-8e10):", d[d.MWmass_star].sid.tolist())
