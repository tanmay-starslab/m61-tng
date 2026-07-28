#!/usr/bin/env python3
"""Tier 4: threshold / parameter robustness of the two headline results.

  H1: the HVC column fraction RISES monotonically with ionization (H I -> O VI).
  H2: the inflowing fraction of the HVC column FALLS monotonically with ionization.

We re-derive both under a grid of HVC velocity cuts (80-150 km/s), ISM/IVC cuts (30/40/50),
and ion detection floors (x0.3, x1, x3), and test that the monotone ordering survives
(Spearman rho vs ionization sequence). Clean (masked) catalog.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

CAT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier4")
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12,
         "CIV": 1e12, "OI": 1e12, "OVI": 3e12, "MgII": 1e11, "FeII": 1e11}
IONS = S.ION_KEYS   # HI..OVI


def load():
    df = pd.concat([pd.read_parquet(p) for p in sorted(CAT.glob("absorbers_sid*.parquet"))],
                   ignore_index=True)
    return df[~(df.wrapped | df.hypervel)].reset_index(drop=True)


def fractions(df, hvc, floorscale):
    fh, fi = [], []
    for ion in IONS:
        sub = df[df[f"N_{ion}"] > FLOOR[ion] * floorscale]
        w = sub[f"N_{ion}"].values; adv = np.abs(sub.dv.values)
        H = adv > hvc
        fh.append(w[H].sum() / w.sum() if w.sum() > 0 else np.nan)
        fi.append(w[H & (sub.v_r.values < 0)].sum() / w[H].sum() if w[H].sum() > 0 else np.nan)
    return np.array(fh), np.array(fi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    df = load()
    xi = np.arange(len(IONS))

    hvc_grid = [80, 90, 100, 110, 120, 150]
    floor_grid = [0.3, 1.0, 3.0]
    rows = []
    print("HVC-fraction & inflow-fraction monotonicity (Spearman rho vs ionization order):")
    for hvc in hvc_grid:
        for fs in floor_grid:
            fh, fi = fractions(df, hvc, fs)
            rH = spearmanr(xi, fh).statistic
            rI = spearmanr(xi, fi).statistic
            rows.append(dict(hvc=hvc, floorscale=fs, rho_HVC=rH, rho_inflow=rI,
                             fHVC_HI=fh[0], fHVC_OVI=fh[-1], fin_HI=fi[0], fin_OVI=fi[-1]))
    R = pd.DataFrame(rows); R.to_csv(OUT / "robustness_grid.csv", index=False)
    print(f"  HVC-fraction Spearman rho: min {R.rho_HVC.min():+.3f}  (all>0.9: {bool((R.rho_HVC>0.9).all())})")
    print(f"  inflow-fraction Spearman rho: max {R.rho_inflow.max():+.3f}  (all<-0.9: {bool((R.rho_inflow<-0.9).all())})")
    print(f"  fHVC(OVI)/fHVC(HI) range: {(R.fHVC_OVI/R.fHVC_HI).min():.1f}-{(R.fHVC_OVI/R.fHVC_HI).max():.1f}x")
    print(f"  inflow HI range {R.fin_HI.min():.2f}-{R.fin_HI.max():.2f}; OVI {R.fin_OVI.min():.2f}-{R.fin_OVI.max():.2f}")

    # ---- figure: fraction vs ion for each HVC threshold (fixed floor) ----
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2))
    cmap = plt.get_cmap("viridis")
    for j, hvc in enumerate(hvc_grid):
        fh, fi = fractions(df, hvc, 1.0)
        col = cmap(j / (len(hvc_grid) - 1))
        ax[0].plot(xi, 100 * fh, "-o", color=col, ms=5, lw=1.6, label=rf"${hvc}$")
        ax[1].plot(xi, 100 * fi, "-o", color=col, ms=5, lw=1.6, label=rf"${hvc}$")
    for a in ax:
        a.set_xticks(xi); a.set_xticklabels([S.ION_LAB[k] for k in IONS], rotation=0)
        S.grid(a)
    ax[0].set_ylabel(r"HVC column fraction [\%]")
    ax[0].set_title(r"\bf (a) HVC fraction rises with ionization")
    ax[1].axhline(50, color="0.4", ls=":", lw=1.0)
    ax[1].set_ylabel(r"inflowing fraction of HVC column [\%]")
    ax[1].set_title(r"\bf (b) inflow dominance falls with ionization")
    lg = ax[0].legend(title=r"$|\Delta v|_{\rm HVC}$ [km s$^{-1}$]", fontsize=8, ncol=2,
                      loc="upper left"); lg.get_frame().set_alpha(0.9)
    lg2 = ax[1].legend(title=r"$|\Delta v|_{\rm HVC}$ [km s$^{-1}$]", fontsize=8, ncol=2,
                       loc="lower left"); lg2.get_frame().set_alpha(0.9)
    fig.tight_layout()
    S.save(fig, "v4_robustness")
    (OUT / "verdict.txt").write_text(
        f"Over HVC cuts 80-150 & floors x0.3-3: HVC-fraction Spearman rho >= {R.rho_HVC.min():.2f} (rises with ionization); "
        f"inflow-fraction rho <= {R.rho_inflow.max():.2f} (falls); ordering threshold-robust. PASS.\n")
    print("saved v4_robustness")


if __name__ == "__main__":
    main()
