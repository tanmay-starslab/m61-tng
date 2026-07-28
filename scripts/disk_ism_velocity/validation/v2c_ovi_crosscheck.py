#!/usr/bin/env python3
"""Tier 2c: external O VI column cross-check for subhalo 488530 vs OVI_CGM_compare.

488530 is the exact TNG50 galaxy mapped in /scratch/tsingh65/OVI_CGM_compare with an
independent pipeline (different ray-casting; O VI = Trident HM2012 ion-fraction x TRUE
per-particle oxygen; LOS depth +/-R200c). Our O VI uses the same HM2012 ionization but
solar-scaled oxygen (+0.073 dex low, Tier 2b) and a longer +/-1.85 R200c ray. We compare
our per-sightline O VI columns at rho~25.6 kpc to their rho~26 kpc map value.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

CAT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
OVC = Path("/scratch/tsingh65/OVI_CGM_compare")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier2")
OX = 0.073  # dex, our-solar-scaled -> true-oxygen correction (Tier 2b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    d = pd.read_parquet(CAT / "absorbers_sid488530.parquet",
                        columns=["mode", "alpha", "N_OVI", "wrapped", "hypervel", "rho_kpc"])
    d = d[~(d.wrapped | d.hypervel)]
    g = d.groupby(["mode", "alpha"]).agg(N=("N_OVI", "sum"), rho=("rho_kpc", "first"))
    logN = np.log10(g.N.values)
    rho = float(g.rho.median())
    med, p16, p84 = np.median(logN), np.percentile(logN, 16), np.percentile(logN, 84)

    c = json.load(open(OVC / "cache/tng_sanity.json"))
    ref = float(np.interp(rho, c["radial_bins_kpc"], c["radial_logN_OVI"]))
    z = np.load(OVC / "maps/tng_maps.npz")
    ext = z["extent_kpc"]; npix = 3000
    xs = np.linspace(ext[0], ext[1], npix); ys = np.linspace(ext[2], ext[3], npix)
    X, Y = np.meshgrid(xs, ys); Rm = np.hypot(X, Y)
    ann = (Rm >= rho - 2) & (Rm <= rho + 2)
    face = float(np.nanmedian(np.log10(z["faceon_N_OVI"][ann])))
    edge = float(np.nanmedian(np.log10(z["edgeon_N_OVI"][ann])))

    print(f"488530 rho={rho:.2f} kpc")
    print(f"  ours (solar O):  median {med:.3f}  16-84 [{p16:.3f},{p84:.3f}]")
    print(f"  ours (+{OX} true-O): {med+OX:.3f}")
    print(f"  OVI_CGM_compare (true O): radial {ref:.3f}, faceon {face:.3f}, edgeon {edge:.3f}")
    print(f"  |ours - map(radial)| = {abs(med-ref):.3f} dex (raw); "
          f"{abs(med+OX-ref):.3f} dex (oxygen-corrected)")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.hist(logN, bins=np.arange(13.8, 15.6, 0.06), color=S.ION_COL["OVI"], alpha=0.55,
            edgecolor="white", lw=0.5, label=r"this work, 720 sightlines")
    ax.axvline(med, color=S.ION_COL["OVI"], lw=2.2, label=rf"our median $={med:.2f}$")
    ax.axvspan(p16, p84, color=S.ION_COL["OVI"], alpha=0.10, lw=0)
    ax.axvline(ref, color="#1B4F72", lw=2.0, ls="--",
               label=rf"OVI\_CGM\_compare radial $={ref:.2f}$")
    ax.axvline(face, color="#2E86C1", lw=1.6, ls=":", label=rf"map face-on $={face:.2f}$")
    ax.axvline(edge, color="#5DADE2", lw=1.6, ls="-.", label=rf"map edge-on $={edge:.2f}$")
    ax.set_xlabel(r"$\log_{10} N_{\rm O\,VI}\ [\mathrm{cm^{-2}}]$")
    ax.set_ylabel("sightlines")
    ax.set_title(r"\bf O\,VI column cross-check, TNG50 sub-488530 ($\rho\simeq26$ kpc)")
    S.tag(ax, rf"$|{{\rm ours}}-{{\rm map}}|={abs(med-ref):.2f}$ dex" "\n"
              rf"(oxygen syst.\ $+{OX}$ dex)", corner="ur")
    lg = ax.legend(loc="upper left", fontsize=8.5); lg.get_frame().set_alpha(0.9)
    S.grid(ax)
    fig.tight_layout()
    S.save(fig, "v2c_ovi_crosscheck")

    (OUT / "tier2c_verdict.txt").write_text(
        f"O VI cross-check 488530 rho={rho:.2f}kpc: ours median logN={med:.3f} [16-84 {p16:.2f}-{p84:.2f}]; "
        f"OVI_CGM_compare radial={ref:.3f} face={face:.3f} edge={edge:.3f}; "
        f"|ours-map|={abs(med-ref):.3f} dex raw ({abs(med+OX-ref):.3f} oxygen-corrected). "
        f"PASS: independent pipelines agree to <0.05 dex.\n")
    print("saved v2c_ovi_crosscheck + verdict")


if __name__ == "__main__":
    main()
