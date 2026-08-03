#!/usr/bin/env python3
"""Two quick-turnaround covering-fraction figures, exactly as specified.

  hvc_detection_fraction_vs_velocity : unweighted fraction of sightlines with >=1 detected
      Voigt component per signed-dv bin (20 km/s over [-500,500]), one curve per line.
  hvc_column_vs_velocity             : where in velocity the HVC COLUMN lives -- summed
      fitted column per signed-dv bin, the HVC differential, and the cumulative fraction
      of each line's HVC column above |dv|.

SAMPLE (stated on both figures): RAW components -- the b_at_ceiling / beyond_common_window
cleaning of the default science sample is deliberately NOT applied here.  H I 1216 is
therefore NOT comparable to the metals: 59.5 per cent of its components sit at the fitter's
b ceiling and 60.9 per cent lie beyond |dv| > 500 km/s, which inflates its raw HVC covering
fraction from 0.486 (clean) to 0.930.  Read the H I curves as an upper envelope only.

Detection = uplim == False, which is already the default of V.load_components().

VELOCITY RANGE.  The signed-dv panels are histogrammed over [-500, +500] km/s; components
outside that range are not plotted.  For the metals this loses 0.9-1.6 per cent of the
components; for H I it loses 60.9 per cent.  The HVC panels (b)/(c) therefore carry an
explicit >=500 km/s overflow term in their denominator so that "fraction of the line's HVC
column" and "HVC column above |dv|" are fractions of the FULL HVC column, not of the part
that happens to fall inside the plotted window.

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

# every panel is tagged with the sample actually used (raw = no b/window cleaning).
# Evaluated lazily: the style (and hence text.usetex) is only set inside main().
def sample_tag():
    if plt.rcParams["text.usetex"]:
        return (r"raw sample: no $b$-ceiling / $|\Delta v|<500$ cuts" + "\n"
                + r"$\Rightarrow$ $\mathrm{H\,I}$ artefact-dominated ($f_{\rm c}=0.93$ vs "
                + r"$0.49$ clean)")
    return ("raw sample: no b-ceiling / |dv|<500 cuts\n"
            "=> H I artefact-dominated (f_c=0.93 vs 0.49 clean)")


def plot1(comp, stat):
    xc = 0.5 * (EDGES[:-1] + EDGES[1:])
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    S.shade_classes(ax, xmax=500, alpha=0.05)
    for L in V.LINES:
        k, n = V.cf_vs_velocity(comp, stat, VARIANT, L.key, EDGES)
        ax.step(xc, 100 * k / n, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)
        d = comp[comp.line == L.key]
        out = float((d[f"dv_{VARIANT}"].abs() > 500).mean())
        print(f"  {L.key:13s} n={n:5d}  peak {100 * (k / n).max():5.1f}%  "
              f"at dv={xc[np.argmax(k)]:+6.0f}   (components outside +-500 km/s, "
              f"not plotted: {100 * out:4.1f}%)")
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    S.tag(ax, sample_tag(), "ul", fs=7.4)
    ax.set_xlim(-500, 500)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\Delta v = v_{\rm rest} - v_{\rm ISM}\ \ [\mathrm{km\,s^{-1}}]$")
    ax.set_ylabel(r"HVC covering fraction per $\Delta v$ bin [\%]"
                  if plt.rcParams["text.usetex"] else
                  "covering fraction per dv bin [%]")
    ax.legend(ncol=2, loc="upper right", fontsize=8.5, handlelength=2.4)
    S.grid(ax)
    fig.tight_layout()
    return S.save(fig, "hvc_detection_fraction_vs_velocity")


def plot2(comp, stat):
    """WHERE IN VELOCITY THE HVC COLUMN LIVES.

    (a) fitted column summed per signed-dv bin (per sightline), one curve per line --
        the whole velocity distribution of the absorbing column, ISM/IVC/HVC shaded.
        Components outside +-500 km/s are not plotted (61 per cent of H I's; ~1 per cent
        of each metal's) -- the fraction omitted is printed per line.
    (b) HVC only (|dv| >= 100): the fraction of each line's total HVC column in each
        |dv| bin, so the peak velocity bin can be read straight off.
    (c) survival curve: fraction of the line's HVC column above |dv|.

    DENOMINATOR.  (b) and (c) divide by the FULL HVC column of the line, including the
    part at |dv| >= 500 km/s that lies outside the plotted range.  That overflow is
    3-5 per cent for the metals but 22.7 per cent for H I, so normalising to the in-window
    column alone (as a naive histogram would) would overstate every H I bin by ~1.3x and
    force the survival curve to zero at 500 km/s when in reality ~23 per cent of the H I
    HVC column lies beyond it.  The residual at the right-hand edge of (c) IS that overflow.

    Column is summed as N = 10**logN over DETECTED components. Note that saturated,
    strong transitions under-report N -- so compare the SHAPE across lines, and read
    absolute columns from the weak transitions.
    """
    edges = np.arange(-500, 501, 20.0)
    xc = 0.5 * (edges[:-1] + edges[1:])
    # 50 km/s for the HVC panels: column-weighted histograms are dominated by rare
    # high-N (often saturated) components, so 20 km/s bins are single-component spiky.
    # 50 (not 40) also makes 200 km/s a genuine bin EDGE -- with 40 km/s bins the old
    # "frac in 100-200" print selected on bin CENTRES and really measured 100-180.
    hedges = np.arange(100, 501, 50.0)
    hxc = 0.5 * (hedges[:-1] + hedges[1:])
    OVER_X = 550.0          # drawing position of the OPEN-ENDED >= 500 km/s bin

    fig, ax = plt.subplots(1, 3, figsize=(17.6, 5.2))
    S.shade_classes(ax[0], xmax=500, alpha=0.05)
    print(f"  {'line':13s} {'peak|dv|':>8s} {'med|dv|':>8s} {'100-200':>8s} "
          f"{'>=500 overflow':>15s} {'comps outside +-500':>20s}")
    for L in V.LINES:
        d = comp[comp.line == L.key]
        dv = d[f"dv_{VARIANT}"].to_numpy()
        N = np.power(10.0, d["logN"].to_numpy())
        n_sl = V.line_denominator(stat, L.key)

        h, _ = np.histogram(dv, bins=edges, weights=N)
        ax[0].step(xc, h / n_sl, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        m = np.abs(dv) >= V.HVC
        hh, _ = np.histogram(np.abs(dv[m]), bins=hedges, weights=N[m])
        over = float(N[m & (np.abs(dv) >= hedges[-1])].sum())    # HVC column beyond 500
        tot = hh.sum() + over                                    # FULL HVC column
        if tot <= 0:
            continue
        # last bin is OPEN-ENDED (>= 500 km/s) and is drawn at OVER_X
        ax[1].step(np.append(hxc, OVER_X), 100 * np.append(hh, over) / tot, where="mid",
                   color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        # survival curve: fraction of the line's FULL HVC column at or above each edge.
        # y[i] belongs at hedges[i] and is constant until hedges[i+1] -> where="post".
        surv = 1.0 - np.concatenate([[0.0], np.cumsum(hh)]) / tot   # len == len(hedges)
        ax[2].step(np.append(hedges, OVER_X), 100 * np.append(surv, surv[-1]), where="post",
                   color=L.color, ls=L.ls, lw=L.lw, label=L.label)

        # where the HVC column actually is (medians on the bin EDGES, not the centres)
        peak = hxc[np.argmax(hh)]
        med = np.interp(0.5, 1.0 - surv, hedges) if surv[-1] < 0.5 else np.nan
        f100_200 = hh[hxc < 200].sum() / tot        # edges 100-150-200 -> a true <200 cut
        print(f"  {L.key:13s} {peak:8.0f} {med:8.1f} {f100_200:8.2f} "
              f"{over / tot:15.3f} {float((np.abs(dv) > 500).mean()):20.3f}")

    ax[0].set_yscale("log")
    ax[0].set_xlim(-500, 500)
    ax[0].axvline(0, color="0.35", ls=":", lw=1.0)
    ax[0].set_xlabel(r"$\Delta v = v_{\rm rest} - v_{\rm ISM}\ \ [\mathrm{km\,s^{-1}}]$")
    ax[0].set_ylabel(r"$\Sigma\,N$ per sightline $[\mathrm{cm^{-2}}/\mathrm{bin}]$")
    ax[0].legend(ncol=2, loc="upper right", fontsize=8.0, handlelength=2.4)
    S.grid(ax[0])
    S.panel_label(ax[0], "(a)")
    S.tag(ax[0], sample_tag(), "ll", fs=7.0)

    for axi in (ax[1], ax[2]):
        axi.axvline(500, color="0.55", ls="-", lw=0.9, zorder=1)
        axi.set_xlim(100, OVER_X + 25)
        axi.set_xticks(list(np.arange(100, 501, 100)) + [OVER_X])
        axi.set_xticklabels([f"{v:.0f}" for v in np.arange(100, 501, 100)]
                            + [r"$\geq500$" if plt.rcParams["text.usetex"] else ">=500"])
    ax[1].set_xlabel(r"$|\Delta v|\ \ [\mathrm{km\,s^{-1}}]$")
    ax[1].set_ylabel(r"fraction of the line's HVC column per bin [\%]"
                     if plt.rcParams["text.usetex"] else
                     "fraction of the line's HVC column per bin [%]")
    yl = ax[1].get_ylim()
    ax[1].set_ylim(0, yl[1] * 1.34)          # headroom so the curves clear the "(b)" label
    ax[1].legend(ncol=2, loc="upper right", fontsize=7.4, handlelength=2.4)
    S.grid(ax[1])
    S.panel_label(ax[1], "(b)")
    S.tag(ax[1], "denominator = full HVC column;\nlast bin is OPEN-ENDED ("
          + (r"$\geq500$" if plt.rcParams["text.usetex"] else ">=500") + " km/s)",
          "ll", fs=7.0)

    ax[2].axhline(50, color="0.35", ls="--", lw=1.0)
    ax[2].set_ylim(0, 100)
    ax[2].set_xlabel(r"$|\Delta v|\ \ [\mathrm{km\,s^{-1}}]$")
    ax[2].set_ylabel(r"HVC column above $|\Delta v|$ [\%]"
                     if plt.rcParams["text.usetex"] else
                     "HVC column above $|\\Delta v|$ [%]")
    ax[2].legend(ncol=2, loc="upper right", fontsize=8.0, handlelength=2.4)
    S.grid(ax[2])
    S.panel_label(ax[2], "(c)")
    S.tag(ax[2], (r"plateau beyond 500 km s$^{-1}$ $=$ the open-ended"
                  if plt.rcParams["text.usetex"] else
                  "plateau beyond 500 km/s = the open-ended")
          + "\nbin: HVC column outside the fitted window", "ll", fs=7.0)

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
