#!/usr/bin/env python3
"""Two quick-turnaround covering-fraction figures, exactly as specified.

  hvc_detection_fraction_vs_velocity : unweighted fraction of sightlines with >=1 detected
      Voigt component per signed-dv bin (20 km/s over [-500,500]), one curve per line.
  hvc_column_vs_velocity             : where in velocity the HVC COLUMN lives -- summed
      fitted column per signed-dv bin, the HVC differential, and the cumulative fraction
      of each line's HVC column above |dv|.

Raw components -- no b_at_ceiling / beyond_common_window cleaning applied (as specified).
Detection = uplim == False, which is already the default of V.load_components().

Usage: python fig_quick_covfrac.py <v3a|v3b>     (default v3b)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402
import m61_voigt as V  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v3b"
assert VARIANT in ("v3a", "v3b")
S.FIGDIR = Path(f"/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures_{VARIANT}")

EDGES = np.arange(-500, 501, 20.0)
LOGN_GRID = np.arange(11.5, 15.51, 0.05)


def plot1(comp, stat):
    xc = 0.5 * (EDGES[:-1] + EDGES[1:])
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    S.shade_classes(ax, xmax=500, alpha=0.05)
    for L in V.LINES:
        k, n = V.cf_vs_velocity(comp, stat, VARIANT, L.key, EDGES)
        ax.step(xc, 100 * k / n, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)
        print(f"  {L.key:13s} n={n:5d}  peak {100 * (k / n).max():5.1f}%  "
              f"at dv={xc[np.argmax(k)]:+6.0f}")
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    ax.set_xlim(-500, 500)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\Delta v = v_{\rm rest} - v_{\rm ISM}\ \ [\mathrm{km\,s^{-1}}]$")
    ax.set_ylabel(r"sightlines with a detected component [\%]"
                  if plt.rcParams["text.usetex"] else
                  "sightlines with a detected component [%]")
    ax.legend(ncol=2, loc="upper right", fontsize=8.5, handlelength=2.4)
    S.grid(ax)
    fig.tight_layout()
    return S.save(fig, "hvc_detection_fraction_vs_velocity")


def plot2(comp, stat):
    """WHERE IN VELOCITY THE HVC COLUMN LIVES.

    (a) fitted column summed per signed-dv bin (per sightline), one curve per line --
        the whole velocity distribution of the absorbing column, ISM/IVC/HVC shaded.
    (b) HVC only (|dv| >= 100): the fraction of each line's total HVC column in each
        |dv| bin, so the peak velocity bin can be read straight off.

    Column is summed as N = 10**logN over DETECTED components. Note that saturated,
    strong transitions under-report N -- so compare the SHAPE across lines, and read
    absolute columns from the weak transitions.
    """
    edges = np.arange(-500, 501, 20.0)
    xc = 0.5 * (edges[:-1] + edges[1:])
    # 40 km/s for the HVC panels: column-weighted histograms are dominated by rare
    # high-N (often saturated) components, so 20 km/s bins are single-component spiky.
    hedges = np.arange(100, 501, 40.0)
    hxc = 0.5 * (hedges[:-1] + hedges[1:])

    fig, ax = plt.subplots(1, 3, figsize=(17.6, 5.2))
    S.shade_classes(ax[0], xmax=500, alpha=0.05)
    for L in V.LINES:
        d = comp[comp.line == L.key]
        dv = d[f"dv_{VARIANT}"].to_numpy()
        N = np.power(10.0, d["logN"].to_numpy())
        n_sl = V.line_denominator(stat, L.key)

        h, _ = np.histogram(dv, bins=edges, weights=N)
        ax[0].step(xc, h / n_sl, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        m = np.abs(dv) >= V.HVC
        hh, _ = np.histogram(np.abs(dv[m]), bins=hedges, weights=N[m])
        frac = hh / hh.sum() if hh.sum() > 0 else hh
        ax[1].step(hxc, 100 * frac, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        # cumulative: fraction of the line's HVC column ABOVE |dv|
        cum_above = 1.0 - np.cumsum(hh) / hh.sum()
        ax[2].step(hedges[1:], 100 * cum_above, where="pre",
                   color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        # where the HVC column actually is
        peak = hxc[np.argmax(hh)]
        cum = np.cumsum(hh) / hh.sum()
        med = np.interp(0.5, cum, hxc)
        f100_200 = hh[hxc < 200].sum() / hh.sum()
        print(f"  {L.key:13s} peak |dv| bin {peak:5.0f}  median |dv| of HVC column "
              f"{med:6.1f}  frac in 100-200: {f100_200:5.2f}")

    ax[0].set_yscale("log")
    ax[0].set_xlim(-500, 500)
    ax[0].axvline(0, color="0.35", ls=":", lw=1.0)
    ax[0].set_xlabel(r"$\Delta v = v_{\rm rest} - v_{\rm ISM}\ \ [\mathrm{km\,s^{-1}}]$")
    ax[0].set_ylabel(r"$\Sigma\,N$ per sightline $[\mathrm{cm^{-2}}/\mathrm{bin}]$")
    ax[0].legend(ncol=2, loc="upper right", fontsize=8.0, handlelength=2.4)
    S.grid(ax[0])
    S.panel_label(ax[0], "(a)")

    ax[1].set_xlim(100, 500)
    ax[1].set_xlabel(r"$|\Delta v|\ \ [\mathrm{km\,s^{-1}}]$")
    ax[1].set_ylabel(r"fraction of the line's HVC column per bin [\%]"
                     if plt.rcParams["text.usetex"] else
                     "fraction of the line's HVC column per bin [%]")
    S.grid(ax[1])
    S.panel_label(ax[1], "(b)")

    ax[2].axhline(50, color="0.35", ls="--", lw=1.0)
    ax[2].set_xlim(100, 500)
    ax[2].set_ylim(0, 100)
    ax[2].set_xlabel(r"$|\Delta v|\ \ [\mathrm{km\,s^{-1}}]$")
    ax[2].set_ylabel(r"HVC column above $|\Delta v|$ [\%]"
                     if plt.rcParams["text.usetex"] else
                     "HVC column above $|\\Delta v|$ [%]")
    ax[2].legend(ncol=2, loc="upper right", fontsize=8.0, handlelength=2.4)
    S.grid(ax[2])
    S.panel_label(ax[2], "(c)")

    fig.tight_layout()
    return S.save(fig, "hvc_column_vs_velocity")


def main():
    S.set_style()
    comp = V.load_components()          # detections only (uplim dropped)
    stat = V.load_line_status()
    print(f"[{VARIANT}] plot 1 -- detection fraction vs signed dv")
    p1 = plot1(comp, stat)
    print(f"[{VARIANT}] plot 2 -- where the HVC column lives in velocity")
    p2 = plot2(comp, stat)
    print(f"saved {p1}\nsaved {p2}")


if __name__ == "__main__":
    main()
