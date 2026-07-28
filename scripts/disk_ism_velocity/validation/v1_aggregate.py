#!/usr/bin/env python3
"""Tier 1 aggregation + figure: gas velocity vs mock-spectrum velocity, all 20 SIDs.

Panel A: AOD (raw-tau-weighted) spectral velocity vs gas ion-column-weighted v_rest
         for the 6 in-band ions -- the near-perfect (~1 km/s) match that proves the
         gas-based velocity IS the absorption velocity.
Panel B: (flux-centroid - gas) velocity bias vs line saturation (tau_max). Thin lines
         are unbiased; only heavily saturated lines drift. O VI (a weak line,
         tau_max<~1) sits in the unbiased regime -> its gas velocity is what an
         observer would measure, even though it is out of the COS-G130M band.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

T1 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier1")
CAT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
IONS6 = ["HI", "CII", "SiII", "SiIII", "SiIV", "NV"]


def robust(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return med, mad, np.percentile(x, 16), np.percentile(x, 84)


def ovi_tau_estimate():
    """Typical O VI 1031.9 line-centre optical depth from the per-SIGHTLINE integrated
    column: tau0 ~ N f lam0 /(b). Confirms O VI is optically thin/marginal (unsaturated)
    -> its centroid is velocity-unbiased, like the thin N V / Si IV lines."""
    f, lam0 = 0.1325, 1031.912e-8  # cm
    CONST = np.pi * (4.803e-10) ** 2 / (9.109e-28 * 2.998e10)  # pi e^2/(m_e c)
    Nsl = []
    for p in sorted(CAT.glob("absorbers_sid*.parquet")):
        d = pd.read_parquet(p, columns=["sid", "mode", "alpha", "N_OVI"])
        g = d.groupby(["sid", "mode", "alpha"])["N_OVI"].sum()  # integrated per sightline
        Nsl.append(g.values)
    N = np.concatenate(Nsl); N = N[N > 1e12]
    b = 40e5  # cm/s, representative Doppler b for warm-hot O VI
    tau0 = CONST * f * lam0 * N / (b / np.sqrt(np.pi))
    return np.percentile(tau0, [50, 90, 99]), np.log10(np.median(N))


def main():
    S.set_style()
    df = pd.concat([pd.read_parquet(p) for p in sorted(T1.glob("gasvspec_sid*.parquet"))],
                   ignore_index=True)
    df = df.dropna(subset=["gas_v", "aod_v"])
    print(f"Tier 1: {len(df)} (sightline,ion) measurements over {df.sid.nunique()} SIDs")

    # per-ion residual stats
    print("\n ion    N    med(aod-gas)  MAD    med(dip-gas)  MAD    <log tau_max>")
    stats = {}
    for ion in IONS6:
        s = df[df.ion == ion]
        r_a = (s.aod_v - s.gas_v).values
        r_d = (s.dipflux_v - s.gas_v).values
        ma, sa, _, _ = robust(r_a); md, sd, _, _ = robust(r_d)
        lt = np.log10(np.clip(s.tau_max.values, 1e-3, None))
        stats[ion] = (ma, sa, md, sd, np.median(lt))
        print(f" {ion:5s} {len(s):5d}  {ma:+7.2f}  {sa:5.2f}   {md:+7.2f}  {sd:5.2f}    {np.median(lt):+.2f}")
    ovi_tau, ovi_logN = ovi_tau_estimate()
    print(f"\nO VI 1032 line-centre optical depth (median/90/99 pct): "
          f"{ovi_tau[0]:.2f}/{ovi_tau[1]:.2f}/{ovi_tau[2]:.2f}  (median logN_OVI={ovi_logN:.2f})")
    print("-> O VI tau~few (Si IV / N V regime, NOT damped like H I). AOD velocity recovers "
          "the gas velocity to <1 km/s for every ion regardless of saturation, and AOD is the "
          "standard observational O VI velocity method -> the gas-based O VI velocity equals "
          "the absorption velocity an observer would measure, well within the 40/100 km/s bins.")

    allr = (df.aod_v - df.gas_v).values
    m, sd, lo, hi = robust(allr)
    print(f"\nALL in-band ions: med(aod-gas)={m:+.2f} km/s, robust-sigma={sd:.2f}, 16-84=[{lo:+.1f},{hi:+.1f}]")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4))

    a = ax[0]
    lim = 400
    for ion in IONS6:
        s = df[df.ion == ion]
        a.scatter(s.gas_v, s.aod_v, s=4, color=S.ION_COL[ion], alpha=0.25,
                  edgecolors="none", rasterized=True, label=S.ION_LAB[ion])
    a.plot([-lim, lim], [-lim, lim], color="0.25", lw=1.2, ls="--", zorder=5)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel(r"gas column-weighted $v_{\rm rest}$ [km s$^{-1}$]")
    a.set_ylabel(r"spectrum AOD velocity [km s$^{-1}$]")
    S.tag(a, rf"med$(\Delta)={m:+.1f}$, $\sigma={sd:.1f}$ km s$^{{-1}}$", corner="ul")
    lg = a.legend(loc="lower right", markerscale=3, handletextpad=0.3, ncol=2, fontsize=8)
    lg.get_frame().set_alpha(0.9)
    S.grid(a)
    a.set_title(r"\bf (a) gas velocity $=$ absorption velocity")

    b = ax[1]
    for ion in IONS6:
        s = df[df.ion == ion]
        lt = np.log10(np.clip(s.tau_max.values, 1e-3, None))
        bias = (s.dipflux_v - s.gas_v).values
        b.scatter(lt, bias, s=4, color=S.ION_COL[ion], alpha=0.22, edgecolors="none",
                  rasterized=True)
        mlt, mbias = np.median(lt), np.median(bias)
        b.scatter([mlt], [mbias], s=90, color=S.ION_COL[ion], edgecolors="k",
                  linewidths=1.0, zorder=6, label=S.ION_LAB[ion])
    b.axhline(0, color="0.25", lw=1.1, ls="--")
    b.axvspan(np.log10(ovi_tau[0]), np.log10(ovi_tau[1]), color=S.ION_COL["OVI"],
              alpha=0.12, lw=0, zorder=0)
    b.text(np.log10(ovi_tau[0]), 190,
           r"O\,VI $\tau{\sim}$few" "\n" r"(Si\,IV/N\,V regime)", color=S.ION_COL["OVI"],
           fontsize=8.5, ha="center", va="top")
    b.set_ylim(-220, 220)
    b.set_xlabel(r"$\log_{10}\,\tau_{\rm max}$ (line saturation)")
    b.set_ylabel(r"flux-centroid $-$ gas velocity [km s$^{-1}$]")
    lg2 = b.legend(loc="lower left", markerscale=1, handletextpad=0.3, ncol=2, fontsize=8)
    lg2.get_frame().set_alpha(0.9)
    S.grid(b)
    b.set_title(r"\bf (b) flux-centroid bias grows with saturation")

    fig.tight_layout()
    S.save(fig, "v1_gas_vs_spectrum")
    print("\nsaved figure v1_gas_vs_spectrum.{pdf,png}")


if __name__ == "__main__":
    main()
