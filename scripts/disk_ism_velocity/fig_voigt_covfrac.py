#!/usr/bin/env python3
r"""Voigt-fit covering-fraction figures: velocity, column, and the two together.

Everything here is PURE SIGHTLINE COUNTING on the pipeline's own detections:

    covering fraction f_c = (# sightlines with >=1 DETECTED Voigt component satisfying
                             the condition) / (# sightlines on which that line was fit)

A DETECTION is a fitted component with UpLim == False -- that definition comes from the
pyGad individual-line fits (S/N=10, bin=3), not from anything invented here. No column
weighting, no EW weighting, no oscillator-strength / curve-of-growth thresholding: the
weak transitions are "weak" here only because the fitter genuinely failed to recover
components in them, which is exactly the observational statement we want.

Denominators come from voigt_line_status.parquet via m61_voigt.line_denominator: a
sightline where a line was fit and nothing was found is a genuine zero, not a gap.

SCIENCE SAMPLE (see the m61_voigt caveats): components are cleaned with
    ~b_at_ceiling & ~beyond_common_window
i.e. the Doppler parameter must not be pinned at the fitter's upper bound and the
component must lie inside the +-500 km/s velocity window common to all nine transitions.
This matters ONLY for H I 1216, which was fit over +-1200 km/s with b_max = 150 km/s
(the metals used +-800 and b_max 50-100): most of its nominal "high-velocity" components
are multi-component fits to the damped, saturated Ly-alpha profile, not clouds, and they
alone drive its raw ~0.93 HVC covering fraction. The raw H I curve is over-plotted
everywhere it matters so the size of that artefact is visible, and the metals (~1-1.6%
beyond 500 km/s) barely move. `sat`/`Chisq` are PER-FIT-REGION flags, not per-component,
and are deliberately not used here.

Figures
  fig11_hvc_detection_vs_velocity
      (a) detection fraction vs SIGNED dv (20 km/s bins), one curve per transition
      (b) cumulative f_c(|dv| >= v_thr) -- read off the HVC covering fraction at any cut
      (c) HVC (|dv|>=100) detection fraction per transition, blueshifted / redshifted split
  fig12_covfrac_vs_column
      (a) cumulative f_c(> logN) of HVC components (+ thin ISM / IVC context curves)
      (b) the same, split blueshifted vs redshifted (mirrored)
      (c) differential: where the STRONGEST HVC component of each sightline lands in logN
  fig13_covfrac_velocity_column_plane
      f_c(|dv| >= v AND logN >= N) over the (|dv|, logN) plane, four contrasting lines

HONESTY NOTE (enforced in the panel text): the Voigt fits carry no galactocentric radial
velocity, so blueshifted/redshifted is an OBSERVABLE PROXY, not an inflow/outflow
measurement. The physical decomposition lives in the companion gas-based figure.

Usage:
    python fig_voigt_covfrac.py <v3a|v3b|both>      (default v3b)
    python fig_voigt_covfrac.py v3b --partial       dry run on whatever per-SID
                                                    catalogs exist; writes to scratch
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S      # noqa: E402
import m61_voigt as V      # noqa: E402

OUTROOT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity")
DRYDIR = Path("/tmp/claude-2760262/-scratch-tsingh65/"
              "a6c3c51d-fb27-4c28-9534-54a7f5da29f6/scratchpad/dryrun_figs")

# grids
DV_EDGES = np.arange(-500.0, 500.1, 20.0)          # signed dv bins, fig11a
VTHR = np.arange(0.0, 500.1, 5.0)                  # |dv| thresholds, fig11b
LOGN_GRID = np.arange(11.5, 20.01, 0.10)           # cumulative logN thresholds, fig12ab
LOGN_EDGES = np.arange(11.5, 20.01, 0.25)          # differential logN bins, fig12c
V2D = np.arange(0.0, 400.1, 10.0)                  # fig13 |dv| thresholds
N2D = np.arange(11.5, 15.51, 0.10)                 # fig13 logN thresholds
PLANE_LINES = ["Si_II_1260", "Si_II_1190", "Si_III_1206", "N_V_1239"]

PCT = "%"          # replaced by \% when LaTeX is on (set in main)
HI = "H_I_1216"
RAW_KW = dict(color=V.LINE_BY_KEY[HI].color, ls=(0, (5, 1.6)), lw=1.4, alpha=0.55)
RAW_LAB = r"$\mathrm{H\,I}\ \lambda1216$ (raw, uncleaned)"
CLEAN_NOTE = (r"components cleaned: $b$ not at the fitter ceiling, $|\Delta v|\leq500$"
              "\n" r"(the window common to all nine transitions)")
HI_NOTE = (r"$\mathrm{H\,I}$: raw high-velocity components are multi-component fits"
           "\n" r"to the damped Ly$\alpha$ profile, not clouds -- see the raw curve.")

# S II 1259.519 falls inside the Si II 1260.422 fit window. Fit with the Si II rest
# wavelength it gives v_obs = -215 km/s, and because v_rest = -v_obs - v_sys it lands at
# dv = +215 km/s -- i.e. it masquerades as REDSHIFTED Si II 1260 absorption. Measured:
# in |dv| = 115-315 km/s the Si II 1260 red/blue component ratio is 1.74, against 1.07
# for 1193 and 1190, so most of 1260's redshifted excess is this blend, not gas motion.
DV_SII_BLEND = -299792.458 * (1259.519 / 1260.422 - 1.0)      # +214.8 km/s


def clean_sample(comp):
    """The default science sample: drop b-at-ceiling and out-of-common-window components."""
    m = ~comp["b_at_ceiling"].to_numpy() & ~comp["beyond_common_window"].to_numpy()
    out = comp[m].reset_index(drop=True)
    print(f"[clean] {len(comp)} -> {len(out)} components "
          f"(dropped {int((~m).sum())} = {100*(~m).mean():.1f}%)")
    print(f"  {'line':14s} {'n_comp':>7s} {'b_ceil%':>8s} {'>500%':>7s} {'kept%':>7s}")
    for L in V.LINES:
        c = comp[comp.line == L.key]
        if not len(c):
            continue
        k = (~c["b_at_ceiling"] & ~c["beyond_common_window"]).mean()
        print(f"  {L.key:14s} {len(c):7d} {100*c['b_at_ceiling'].mean():8.1f} "
              f"{100*c['beyond_common_window'].mean():7.1f} {100*k:7.1f}")
    return out


# ── sightline-counting helpers ──────────────────────────────────────────────────
# These mirror m61_voigt's helpers exactly (same numerator = unique sightlines with
# >=1 detected component satisfying the cut, same denominator = line_denominator);
# they only add the vmax window / speed that m61_voigt.cf_vs_logN does not expose.
# _selfcheck() below asserts they agree with m61_voigt where the two overlap.
def _sl_code(c):
    """Integer code per (sid, mode, alpha) sightline."""
    key = (c["sid"].to_numpy().astype(np.int64) * 1_000_000
           + (c["mode"].to_numpy() == "flip").astype(np.int64) * 1000
           + c["alpha"].to_numpy().astype(np.int64))
    return pd.factorize(key)[0]


def _best_logN(c):
    """Per-sightline maximum fitted logN over the selected components (sorted array)."""
    if len(c) == 0:
        return np.empty(0)
    code = _sl_code(c)
    lN = c["logN"].to_numpy(float)
    ok = np.isfinite(lN)
    if not ok.any():
        return np.empty(0)
    best = np.full(code.max() + 1, -np.inf)
    np.maximum.at(best, code[ok], lN[ok])
    return np.sort(best[np.isfinite(best)])


def _window(comp, variant, line, vmin, vmax=np.inf, signed=None):
    c = comp[comp.line == line]
    dv = c[f"dv_{variant}"]
    m = (dv.abs() >= vmin) & (dv.abs() < vmax)
    if signed == "blue":
        m &= dv < 0
    elif signed == "red":
        m &= dv > 0
    return c[m]


def cf_logN_cum(comp, stat, variant, line, grid, vmin=V.HVC, vmax=np.inf, signed=None):
    """Cumulative f_c(> logN) inside a |dv| window. Returns (k, n) sightline counts.

    Identical counting to V.cf_vs_logN (a sightline enters the numerator iff at least one
    of its detected components in the window has logN >= grid[i]), extended with vmax."""
    n = V.line_denominator(stat, line)
    b = _best_logN(_window(comp, variant, line, vmin, vmax, signed))
    if len(b) == 0:
        return np.zeros(len(grid)), n
    k = len(b) - np.searchsorted(b, grid, side="left")
    return k.astype(float), n


def strongest_logN_hist(comp, stat, variant, line, edges, vmin=V.HVC, vmax=np.inf):
    """Differential: # sightlines whose STRONGEST component in the window falls in
    each logN bin, and the denominator."""
    n = V.line_denominator(stat, line)
    b = _best_logN(_window(comp, variant, line, vmin, vmax))
    k, _ = np.histogram(b, bins=edges)
    return k.astype(float), n


def cf_2d(comp, stat, variant, line, vgrid, ngrid):
    """f_c(|dv| >= v AND logN >= N): counts on the (velocity, column) threshold plane."""
    n = V.line_denominator(stat, line)
    c = comp[comp.line == line]
    adv = c[f"dv_{variant}"].abs().to_numpy()
    k = np.zeros((len(vgrid), len(ngrid)))
    for i, v in enumerate(vgrid):
        sub = c[adv >= v]
        b = _best_logN(sub)
        if len(b) == 0:
            continue
        k[i] = len(b) - np.searchsorted(b, ngrid, side="left")
    return k, n


def half_logN(grid, f):
    """logN at which the cumulative covering fraction has fallen to half its value at
    the lowest column on the grid. NaN if the curve never gets there."""
    f = np.asarray(f, float)
    if f[0] <= 0:
        return np.nan
    tgt = 0.5 * f[0]
    if f[-1] > tgt:
        return np.nan
    return float(np.interp(-tgt, -f, grid))    # f decreasing -> negate to interpolate


def _selfcheck(comp, stat, variant):
    """Prove the local helpers reproduce m61_voigt's counting before plotting."""
    g = np.arange(11.5, 16.01, 0.5)
    for line in ("Si_II_1260", "H_I_1216", "N_V_1239"):
        for signed in (None, "blue", "red"):
            ka, na = V.cf_vs_logN(comp, stat, variant, line, g, vmin=V.HVC, signed=signed)
            kb, nb = cf_logN_cum(comp, stat, variant, line, g, vmin=V.HVC, signed=signed)
            if na != nb or not np.array_equal(np.asarray(ka, float), kb):
                raise AssertionError(
                    f"counting mismatch vs m61_voigt for {line} signed={signed}: "
                    f"{ka} (n={na}) vs {kb} (n={nb})")
        f, k, n = V.covering_fraction(comp, stat, variant, line)
        k2, n2 = cf_logN_cum(comp, stat, variant, line, np.array([-np.inf]))
        if (k, n) != (int(k2[0]), n2):
            raise AssertionError(f"HVC total mismatch for {line}: {(k, n)} vs {(k2[0], n2)}")
    print("[selfcheck] local counting == m61_voigt counting (3 lines x 3 sign cuts)")


# ── fig 11 ──────────────────────────────────────────────────────────────────────
def fig11(comp, raw, stat, variant):
    fig, ax = plt.subplots(1, 3, figsize=(19.8, 5.6))

    # (a) signed dv, per-bin detection fraction
    a = ax[0]
    xc = 0.5 * (DV_EDGES[:-1] + DV_EDGES[1:])
    floor = np.inf
    for L in V.LINES:
        k, n = V.cf_vs_velocity(comp, stat, variant, L.key, DV_EDGES)
        f = 100.0 * k / n
        lo, hi = V.wilson(k, n)
        f = np.where(k > 0, f, np.nan)
        a.fill_between(xc, 100 * lo, 100 * hi, step="mid", color=L.color, alpha=0.11, lw=0)
        a.step(xc, f, where="mid", color=L.color, ls=L.ls, lw=L.lw, label=L.label)
        if np.isfinite(f).any():
            floor = min(floor, np.nanmin(f))
    kr, nr = V.cf_vs_velocity(raw, stat, variant, HI, DV_EDGES)
    a.step(xc, np.where(kr > 0, 100.0 * kr / nr, np.nan), where="mid",
           label=RAW_LAB, zorder=4, **RAW_KW)
    S.shade_classes(a)
    a.axvline(0.0, color="0.35", ls=":", lw=1.1)
    a.axvline(DV_SII_BLEND, color=V.LINE_BY_KEY["Si_II_1260"].color, ls=(0, (1, 2)),
              lw=1.3, alpha=0.95, zorder=5)
    a.text(DV_SII_BLEND + 12, 7.0, r"$\mathrm{S\,II}\ \lambda1259.5$ blend", rotation=90,
           ha="left", va="bottom", fontsize=8.0,
           color=V.LINE_BY_KEY["Si_II_1260"].color, zorder=6)
    a.set_xlim(-500, 500)
    a.set_yscale("log")
    a.set_ylim(0.6 * floor, 160)
    a.set_xlabel(r"$\Delta v = v_{\mathrm{rest}} - v_{\mathrm{ISM}}\ \ [\mathrm{km\,s^{-1}}]$")
    a.set_ylabel(rf"detection fraction per $20\,\mathrm{{km\,s^{{-1}}}}$ bin  $[{PCT}]$")
    a.legend(ncol=2, loc="lower center", fontsize=8.5, framealpha=0.92)
    a.text(0.02, 0.905, CLEAN_NOTE + "\n" + HI_NOTE, transform=a.transAxes,
           ha="left", va="top", fontsize=8.0, color="0.25")
    S.grid(a)
    S.panel_label(a, "(a)")
    for cls, x in (("ISM", 0.0), ("IVC", 70.0), ("HVC", 300.0)):
        a.text(x, 115, cls, color=S.CLASS[cls], ha="center", va="top",
               fontsize=9, fontweight="bold")

    # (b) cumulative: |dv| >= v_thr
    b = ax[1]
    fhvc = {}
    for L in V.LINES:
        k, n = V.cf_vs_threshold(comp, stat, variant, L.key, VTHR)
        f = 100.0 * k / n
        lo, hi = V.wilson(k, n)
        b.fill_between(VTHR, 100 * lo, 100 * hi, color=L.color, alpha=0.12, lw=0)
        b.plot(VTHR, np.where(k > 0, f, np.nan), color=L.color, ls=L.ls, lw=L.lw,
               label=L.label)
        fhvc[L.key] = float(np.interp(V.HVC, VTHR, f))
    kr, nr = V.cf_vs_threshold(raw, stat, variant, HI, VTHR)
    b.plot(VTHR, np.where(kr > 0, 100.0 * kr / nr, np.nan), label=RAW_LAB,
           zorder=4, **RAW_KW)
    for x, cls in ((V.IVC, "IVC"), (V.HVC, "HVC")):
        b.axvline(x, color=S.CLASS[cls], ls="--", lw=1.2, alpha=0.85)
    b.text(V.HVC + 8, 0.1, r"HVC cut", color=S.CLASS["HVC"], rotation=90,
           va="bottom", ha="left", fontsize=9)
    b.set_xlim(0, 500)
    b.set_yscale("log")
    b.set_ylim(0.05, 160)
    b.set_xlabel(r"velocity cut $v_{\mathrm{thr}}\ \ [\mathrm{km\,s^{-1}}]$")
    b.set_ylabel(rf"sightlines with $|\Delta v| \geq v_{{\mathrm{{thr}}}}$  $[{PCT}]$")
    b.legend(ncol=2, loc="lower left", fontsize=8.5, framealpha=0.92)
    S.grid(b)
    S.panel_label(b, "(b)")

    # (c) HVC detection fraction per line, blue / red split (raw shown as open bars)
    c = ax[2]
    xk = np.arange(len(V.LINES))
    rows = []
    for i, L in enumerate(V.LINES):
        ft, kt, nt = V.covering_fraction(comp, stat, variant, L.key)
        fb, kb, _ = V.covering_fraction(comp, stat, variant, L.key, signed="blue")
        fr, kr, _ = V.covering_fraction(comp, stat, variant, L.key, signed="red")
        gt, jt, _ = V.covering_fraction(raw, stat, variant, L.key)
        gb, jb, _ = V.covering_fraction(raw, stat, variant, L.key, signed="blue")
        gr, jr, _ = V.covering_fraction(raw, stat, variant, L.key, signed="red")
        rows.append((L, ft, kt, nt, fb, kb, fr, kr, gt, jt, gb, gr))
        for dx, kk, gg, col, lab in ((-0.19, kb, gb, S.INFLOW, "blueshifted"),
                                     (+0.19, kr, gr, S.OUTFLOW, "redshifted")):
            lo, hi = V.wilson(kk, nt)
            f = 100.0 * kk / nt
            c.bar(i + dx, 100.0 * gg, 0.36, facecolor="none", edgecolor="0.35", lw=0.9,
                  ls="--", zorder=2,
                  label="raw (uncleaned)" if (i == 0 and dx < 0) else None)
            c.bar(i + dx, f, 0.36, color=col, alpha=0.85, lw=0.6, edgecolor="0.25",
                  label=lab if i == 0 else None, zorder=3)
            c.errorbar(i + dx, f, yerr=[[f - 100 * lo], [100 * hi - f]], fmt="none",
                       ecolor="0.15", elinewidth=1.1, capsize=2.5, zorder=4)
        lo, hi = V.wilson(kt, nt)
        c.errorbar(i, 100.0 * ft, yerr=[[100 * ft - 100 * lo], [100 * hi - 100 * ft]],
                   fmt="_", ms=22, mew=1.8, color="0.12", ecolor="0.12", elinewidth=1.1,
                   capsize=2.5, zorder=5, label="either sign" if i == 0 else None)
    c.annotate(r"raw $\mathrm{H\,I}$: damping-wing fits", xy=(-0.19, 100 * rows[0][10]),
               xytext=(0.55, 72.0), fontsize=8.5, color="0.25",
               arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0), zorder=6)
    c.set_xticks(xk)
    c.set_xticklabels([L.label for L in V.LINES], rotation=32, ha="right", fontsize=9)
    c.set_ylim(0, 115)
    c.set_ylabel(rf"HVC detection fraction, $|\Delta v|\geq100\ \mathrm{{km\,s^{{-1}}}}$  $[{PCT}]$")
    c.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    S.grid(c)
    S.panel_label(c, "(c)")

    f60, f93, f90 = (fhvc["Si_II_1260"], fhvc["Si_II_1193"], fhvc["Si_II_1190"])
    ratio = f60 / f90 if f90 > 0 else np.nan
    c.text(0.985, 0.62, "\n".join([
        r"$\mathrm{Si\,II}$ triplet -- same gas, three independent fits:",
        rf"$\lambda1260 = {f60:.1f}{PCT}$,  $\lambda1193 = {f93:.1f}{PCT}$,  "
        rf"$\lambda1190 = {f90:.1f}{PCT}$",
        rf"$f_{{1260}}/f_{{1190}} = {ratio:.2f}$",
        r"($\lambda1260$'s redshifted excess is partly the",
        r"$\mathrm{S\,II}\ \lambda1259.5$ blend at $\Delta v \simeq +215$)"]),
        transform=c.transAxes, ha="right", va="top", fontsize=8.5, zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92))
    fig.tight_layout()
    S.save(fig, "fig11_hvc_detection_vs_velocity")
    return rows, (f60, f93, f90, ratio)


# ── fig 12 ──────────────────────────────────────────────────────────────────────
def fig12(comp, raw, stat, variant):
    fig, ax = plt.subplots(1, 3, figsize=(19.8, 5.6))
    xlab = r"fitted column threshold  $\log_{10} N\ \ [\mathrm{cm^{-2}}]$"

    # (a) cumulative HVC f_c(>logN), with ISM / IVC context
    a = ax[0]
    halves = {}
    for L in V.LINES:
        for vmin, vmax, alpha, lw in ((0.0, V.IVC, 0.20, 0.9),
                                      (V.IVC, V.HVC, 0.42, 1.0)):
            k, n = cf_logN_cum(comp, stat, variant, L.key, LOGN_GRID, vmin=vmin, vmax=vmax)
            a.plot(LOGN_GRID, np.where(k > 0, k / n, np.nan), color=L.color, ls=L.ls,
                   lw=lw, alpha=alpha, zorder=1)
        k, n = V.cf_vs_logN(comp, stat, variant, L.key, LOGN_GRID, vmin=V.HVC)
        f = k / n
        lo, hi = V.wilson(k, n)
        a.fill_between(LOGN_GRID, lo, hi, color=L.color, alpha=0.12, lw=0, zorder=2)
        a.plot(LOGN_GRID, np.where(k > 0, f, np.nan), color=L.color, ls=L.ls, lw=L.lw,
               label=L.label, zorder=3)
        halves[L.key] = half_logN(LOGN_GRID, f)
    kr, nr = V.cf_vs_logN(raw, stat, variant, HI, LOGN_GRID, vmin=V.HVC)
    a.plot(LOGN_GRID, np.where(kr > 0, kr / nr, np.nan), label=RAW_LAB, zorder=4, **RAW_KW)
    halves["H_I_1216_raw"] = half_logN(LOGN_GRID, kr / nr)
    a.set_yscale("log")
    a.set_xlim(LOGN_GRID[0], 20.0)
    a.set_ylim(3e-4, 6.0)
    a.set_xlabel(xlab)
    a.set_ylabel(r"covering fraction  $f_c(>\log N)$")
    leg = a.legend(ncol=2, loc="lower left", fontsize=8.5, framealpha=0.92)
    a.add_artist(leg)
    ctx = [Line2D([], [], color="0.25", lw=2.0, label=r"HVC $|\Delta v|\geq100$"),
           Line2D([], [], color="0.25", lw=1.0, alpha=0.42, label=r"IVC $40\!-\!100$"),
           Line2D([], [], color="0.25", lw=0.9, alpha=0.20, label=r"ISM $|\Delta v|<40$")]
    a.legend(handles=ctx, loc="center right", fontsize=8.5, framealpha=0.92)
    a.text(0.245, 0.962, CLEAN_NOTE + "\n" + HI_NOTE, transform=a.transAxes,
           ha="left", va="top", fontsize=8.0, color="0.25")
    S.grid(a)
    S.panel_label(a, "(a)")

    # (b) blueshifted vs redshifted, mirrored
    b = ax[1]
    asym = {}
    for L in V.LINES:
        kb, n = V.cf_vs_logN(comp, stat, variant, L.key, LOGN_GRID, vmin=V.HVC, signed="blue")
        kr, _ = V.cf_vs_logN(comp, stat, variant, L.key, LOGN_GRID, vmin=V.HVC, signed="red")
        b.plot(LOGN_GRID, np.where(kb > 0, +100.0 * kb / n, np.nan),
               color=L.color, ls=L.ls, lw=L.lw)
        b.plot(LOGN_GRID, np.where(kr > 0, -100.0 * kr / n, np.nan),
               color=L.color, ls=L.ls, lw=L.lw)
        asym[L.key] = (kb[0] / n, kr[0] / n)
    b.axhline(0.0, color="0.3", lw=1.1)
    b.set_yscale("symlog", linthresh=0.05, linscale=0.45)
    yt = np.array([-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100], float)
    b.set_yticks(yt)
    b.set_yticklabels([("0" if t == 0 else (f"{abs(t):g}")) for t in yt])
    b.set_ylim(-260, 260)
    b.set_xlim(LOGN_GRID[0], 20.0)
    b.set_xlabel(xlab)
    b.set_ylabel(rf"HVC covering fraction  $f_c(>\log N)$  $[{PCT}]$")
    b.text(0.975, 0.955, r"blueshifted  ($\Delta v \leq -100\ \mathrm{km\,s^{-1}}$)",
           transform=b.transAxes, ha="right", va="top", color=S.INFLOW, fontsize=10)
    b.text(0.975, 0.045, r"redshifted  ($\Delta v \geq +100\ \mathrm{km\,s^{-1}}$)",
           transform=b.transAxes, ha="right", va="bottom", color=S.OUTFLOW, fontsize=10)
    b.text(0.985, 0.5, "\n".join([
        r"blue/red is an \emph{observable} proxy:" if plt.rcParams["text.usetex"]
        else "blue/red is an observable proxy:",
        r"no galactocentric $v_r$ in the fits.",
        r"Physical inflow/outflow:",
        r"companion gas-based figure."]),
        transform=b.transAxes, ha="right", va="center", fontsize=8.5, zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92))
    S.grid(b)
    S.panel_label(b, "(b)")

    # (c) differential: strongest HVC component per sightline
    c = ax[2]
    xc = 0.5 * (LOGN_EDGES[:-1] + LOGN_EDGES[1:])
    for L in V.LINES:
        k, n = strongest_logN_hist(comp, stat, variant, L.key, LOGN_EDGES)
        c.step(xc, np.where(k > 0, 100.0 * k / n, np.nan), where="mid",
               color=L.color, ls=L.ls, lw=L.lw, label=L.label)
    c.set_yscale("log")
    c.set_xlim(LOGN_EDGES[0], 20.0)
    c.set_ylim(0.02, 60)
    c.set_xlabel(r"$\log_{10} N$ of the strongest HVC component  $[\mathrm{cm^{-2}}]$")
    c.set_ylabel(rf"sightlines per $0.25\,$dex bin  $[{PCT}]$")
    S.tag(c, r"colours as in (a)", corner="ur", fs=9.0)
    S.grid(c)
    S.panel_label(c, "(c)")

    fig.tight_layout()
    S.save(fig, "fig12_covfrac_vs_column")
    return halves, asym


# ── fig 13 ──────────────────────────────────────────────────────────────────────
def fig13(comp, stat, variant):
    ks, ns = {}, {}
    for key in PLANE_LINES:
        ks[key], ns[key] = cf_2d(comp, stat, variant, key, V2D, N2D)
    fmax = max(float((ks[k] / ns[k]).max()) for k in PLANE_LINES)
    fmin = 0.5 / max(ns.values())
    norm = mcolors.LogNorm(vmin=max(fmin, 1e-4), vmax=min(1.0, 1.35 * fmax))
    cmap = plt.get_cmap(S.SEQCMAP).copy()
    cmap.set_bad("white")

    dv, dn = V2D[1] - V2D[0], N2D[1] - N2D[0]
    ve = np.append(V2D, V2D[-1] + dv) - 0.5 * dv
    ne = np.append(N2D, N2D[-1] + dn) - 0.5 * dn

    fig, ax = plt.subplots(1, len(PLANE_LINES), figsize=(20.4, 5.2),
                           sharex=True, sharey=True, constrained_layout=True)
    for i, key in enumerate(PLANE_LINES):
        L = V.LINE_BY_KEY[key]
        a = ax[i]
        f = ks[key] / ns[key]
        im = a.pcolormesh(ve, ne, np.ma.masked_where(ks[key] == 0, f).T,
                          cmap=cmap, norm=norm, shading="flat", rasterized=True)
        cs = a.contour(V2D, N2D, f.T, levels=[0.001, 0.01, 0.05, 0.20],
                       colors="w", linewidths=0.9, alpha=0.75)
        a.clabel(cs, fmt=lambda v: (rf"{100*v:g}\%" if plt.rcParams["text.usetex"]
                                    else f"{100*v:g}%"), fontsize=7.5, inline=True)
        a.axvline(V.HVC, color=S.CLASS["HVC"], ls="--", lw=1.3)
        a.set_xlabel(r"velocity cut $|\Delta v| \geq v\ \ [\mathrm{km\,s^{-1}}]$")
        if i == 0:
            a.set_ylabel(r"column cut $\log_{10} N \geq N_{\mathrm{thr}}\ \ [\mathrm{cm^{-2}}]$")
        S.tag(a, L.label + rf"  ($n = {ns[key]}$)", corner="ur", fs=9.5)
        S.panel_label(a, "(" + "abcd"[i] + ")")
        if i == 0:
            a.text(0.965, 0.845, CLEAN_NOTE, transform=a.transAxes, ha="right",
                   va="top", fontsize=8.0, color="w")
        a.set_xlim(V2D[0] - 0.5 * dv, V2D[-1] + 0.5 * dv)
        a.set_ylim(N2D[0] - 0.5 * dn, N2D[-1] + 0.5 * dn)
    cb = fig.colorbar(im, ax=list(ax), fraction=0.035, pad=0.012)
    cb.set_label(r"covering fraction  $f_c\,(|\Delta v| \geq v \ \wedge\ \log N \geq N_{\mathrm{thr}})$")
    S.save(fig, "fig13_covfrac_velocity_column_plane")
    return {k: float((ks[k] / ns[k])[V2D == V.HVC][0].max()) for k in PLANE_LINES}


# ── driver ──────────────────────────────────────────────────────────────────────
def wait_for_catalog(timeout_s=1800, poll=60):
    f = V.VOIGT_DIR / "voigt_components.parquet"
    g = V.VOIGT_DIR / "voigt_line_status.parquet"
    t0 = time.time()
    while not (f.exists() and g.exists()):
        el = time.time() - t0
        if el > timeout_s:
            raise TimeoutError(f"{f} still missing after {el/60:.1f} min -- "
                               "build_voigt_catalog.py --combine has not run")
        n = len(list(V.VOIGT_DIR.glob("voigt_components_sid*.parquet")))
        print(f"[wait] combined catalog not ready ({n} per-SID files, "
              f"{el/60:.1f} min elapsed) -- sleeping {poll}s", flush=True)
        time.sleep(poll)
    return f


def load(partial=False):
    if partial:
        cf = sorted(V.VOIGT_DIR.glob("voigt_components_sid*.parquet"))
        sf = sorted(V.VOIGT_DIR.glob("voigt_linestatus_sid*.parquet"))
        if not cf:
            raise FileNotFoundError("no per-SID catalogs to dry-run on")
        comp = pd.concat([pd.read_parquet(p) for p in cf], ignore_index=True)
        stat = pd.concat([pd.read_parquet(p) for p in sf], ignore_index=True)
        comp = comp[~comp.uplim].reset_index(drop=True)
        print(f"[PARTIAL DRY RUN] {len(cf)} SIDs -- NOT the publication sample")
    else:
        wait_for_catalog()
        comp = V.load_components()
        stat = V.load_line_status()
    missing = [c for c in ("b_at_ceiling", "beyond_common_window") if c not in comp]
    if missing:
        raise KeyError(f"catalog is missing the cleaning flags {missing} -- "
                       "rebuild with the current build_voigt_catalog.py")
    n_sl = stat[V.SLKEY].drop_duplicates().shape[0]
    print(f"[load] {len(comp)} detected components, {len(stat)} (sightline,line) fits, "
          f"{n_sl} sightlines, {comp.sid.nunique()} galaxies")
    return comp, stat


def report(variant, rows, siii, halves, asym, plane):
    print(f"\n{'='*104}\nVOIGT COVERING FRACTIONS -- variant {variant}\n{'='*104}")
    print("HVC = |dv| >= 100 km/s. 'cleaned' = ~b_at_ceiling & ~beyond_common_window "
          "(the science sample); 'raw' = every detected component.")
    print(f"{'line':14s} {'n_fit':>6s} {'HVC f_c':>8s} {'68% Wilson':>17s} "
          f"{'raw f_c':>8s} {'blue':>7s} {'red':>7s} {'blue-red':>9s} {'logN_1/2':>9s}")
    for (L, ft, kt, nt, fb, kb, fr, kr, gt, jt, gb, gr) in rows:
        lo, hi = V.wilson(kt, nt)
        h = halves.get(L.key, np.nan)
        print(f"{L.key:14s} {nt:6d} {ft:8.4f} [{float(lo):.4f},{float(hi):.4f}] "
              f"{gt:8.4f} {fb:7.4f} {fr:7.4f} {fb-fr:+9.4f} "
              f"{('%9.2f' % h) if np.isfinite(h) else '      n/a'}")
    f60, f93, f90, ratio = siii
    order = "DECREASING (as expected)" if f60 >= f93 >= f90 else "*** NOT MONOTONIC ***"
    print(f"\nSi II triplet HVC f_c [{PCT}]: 1260={f60:.2f}  1193={f93:.2f}  1190={f90:.2f}"
          f"   ratio 1260/1190 = {ratio:.2f}   -> {order}")
    hraw = halves.get("H_I_1216_raw", np.nan)
    print(f"H I logN_1/2: cleaned {halves.get('H_I_1216', np.nan):.2f}  raw {hraw:.2f}")
    print("plane (fig13) f_c at |dv|>=100, lowest column cut: "
          + "  ".join(f"{k}={v:.4f}" for k, v in plane.items()))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    partial = "--partial" in sys.argv
    which = args[0] if args else "v3b"
    variants = ["v3a", "v3b"] if which == "both" else [which]
    for v in variants:
        if v not in ("v3a", "v3b"):
            raise SystemExit(f"variant must be v3a, v3b or both (got {v!r})")

    usetex = S.set_style()
    global PCT
    PCT = r"\%" if usetex else "%"
    print(f"[style] usetex={usetex}")

    raw, stat = load(partial=partial)
    comp = clean_sample(raw)
    for v in variants:
        S.FIGDIR = (DRYDIR / v) if partial else (OUTROOT / f"paper_figures_{v}")
        S.FIGDIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[{v}] FIGDIR = {S.FIGDIR}")
        _selfcheck(comp, stat, v)
        t0 = time.time()
        rows, siii = fig11(comp, raw, stat, v)
        print(f"  fig11 done ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        halves, asym = fig12(comp, raw, stat, v)
        print(f"  fig12 done ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        plane = fig13(comp, stat, v)
        print(f"  fig13 done ({time.time()-t0:.0f}s)", flush=True)
        report(v, rows, siii, halves, asym, plane)


if __name__ == "__main__":
    main()
