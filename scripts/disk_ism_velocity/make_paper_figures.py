#!/usr/bin/env python3
"""Multi-ion / multi-phase publication figures for the m61-tng disk-ISM / HVC study.

Reads the combined multi-ion absorbing-gas catalog (build_absorber_catalog.py) + the
v_ISM master, and makes column-weighted, per-ion figures across the ionization sequence
H I -> C II -> Si II -> Si III -> Si IV -> N V (neutral/cool -> warm-hot):

  fig1_abundance_multiion       Sigma N_ion / sightline vs dv = v_los - v_ISM (abs + shape)
  fig2_hvc_multiion             HVC (|dv|>=100): per-ion abundance + column fraction in HVC
  fig3_inflow_outflow_multiion  HVC radial motion by ion: inflow fraction + v_r PDFs
  fig4_metallicity_multiion     ion-weighted Z/Zsun (H I reveals low-Z accretion) + by phase
  fig5_ionization_temperature   ion-weighted temperature (the ionization sequence) ISM vs HVC
  fig6_column_budget_multiion   fraction of each ion's column in ISM/IVC/HVC
  fig7_ionratio_vs_dv           ionization state (N_SiIV/N_SiII, N_NV/N_SiII) vs dv

Usage: python make_paper_figures.py
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import m61_style as S

CATDIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
MASTER = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
HVC = 100.0
IVC = 40.0
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12,
         "CIV": 1e12, "OI": 1e12, "OVI": 3e12, "MgII": 1e11, "FeII": 1e11}


def load():
    files = sorted(glob.glob(str(CATDIR / "absorbers_sid*.parquet")))
    cells = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m = pd.read_csv(MASTER)
    n_sl = int(m["v_ism_primary"].notna().sum())   # sightlines probed
    return cells, n_sl


def wmedian(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w)
    return float(np.interp(0.5 * c[-1], c, x))


def colw_hist(cells, ion, bins, absval=False):
    x = cells["dv"].abs().to_numpy() if absval else cells["dv"].to_numpy()
    h, _ = np.histogram(x, bins=bins, weights=cells[f"N_{ion}"].to_numpy())
    return h


# ── fig1 ─────────────────────────────────────────────────────────────────────────
def fig1_abundance(cells, n_sl):
    bins = np.arange(-500, 501, 20.0); xc = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    for a in ax:
        S.shade_classes(a)
        a.axvline(0, color="0.35", ls=":", lw=1.2)
        a.set_xlim(-500, 500)
        a.set_xlabel(r"$v_{\mathrm{los}} - v_{\mathrm{ISM}}\ \ [\mathrm{km\,s^{-1}}]$")
    for k in S.ION_KEYS:
        h = colw_hist(cells, k, bins) / n_sl
        ax[0].step(xc, h, where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
        hp = h / h.max() if h.max() > 0 else h
        ax[1].step(xc, hp, where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\Sigma\,N_{\rm ion}$ per sightline  $[\mathrm{cm^{-2}}/\mathrm{bin}]$")
    ax[0].legend(ncol=2, loc="upper left", fontsize=9); S.grid(ax[0]); S.panel_label(ax[0], "(a)")
    ax[1].set_ylabel(r"column-weighted PDF (peak-normalised)"); ax[1].set_ylim(0, 1.15)
    S.grid(ax[1]); S.panel_label(ax[1], "(b)")
    for c, x in (("ISM", 0), ("IVC", 72), ("HVC", 320)):
        ax[1].text(x, 1.08, c, color=S.CLASS[c], ha="center", va="top", fontsize=9, fontweight="bold")
    fig.suptitle(r"Multi-ion absorber abundance vs.\ velocity offset from the disk ISM"
                 if plt.rcParams["text.usetex"] else "Multi-ion absorber abundance vs. velocity offset from the disk ISM",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig1_abundance_multiion")


# ── fig2 ─────────────────────────────────────────────────────────────────────────
def fig2_hvc(cells, n_sl):
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    bins = np.arange(100, 561, 20.0); xc = 0.5 * (bins[:-1] + bins[1:])
    a = ax[0]
    for k in S.ION_KEYS:
        h = colw_hist(cells, k, bins, absval=True) / n_sl
        a.step(xc, h, where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
    a.set_yscale("log"); a.set_xlim(100, 560)
    a.set_xlabel(r"$|v_{\mathrm{los}} - v_{\mathrm{ISM}}|\ \ [\mathrm{km\,s^{-1}}]$")
    a.set_ylabel(r"$\Sigma\,N_{\rm ion}$ per sightline  $[\mathrm{cm^{-2}}/\mathrm{bin}]$")
    a.legend(ncol=2, loc="upper right", fontsize=9); S.grid(a); S.panel_label(a, "(a)")
    # (b) fraction of each ion's total column in IVC and HVC
    b = ax[1]
    fh, fi = [], []
    for k in S.ION_KEYS:
        N = cells[f"N_{k}"]
        tot = N.sum()
        fh.append((N[cells.dv.abs() > HVC].sum() / tot) if tot > 0 else 0)
        fi.append((N[(cells.dv.abs() > IVC) & (cells.dv.abs() <= HVC)].sum() / tot) if tot > 0 else 0)
    xk = np.arange(len(S.ION_KEYS))
    b.bar(xk - 0.19, fi, 0.36, color=S.CLASS["IVC"], label=r"IVC ($40\!-\!100$)")
    b.bar(xk + 0.19, fh, 0.36, color=S.CLASS["HVC"], label=r"HVC ($>100$)")
    b.set_xticks(xk); b.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    b.set_ylabel(r"fraction of ion column at $|\Delta v|>40$")
    b.legend(loc="upper left"); S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle(r"High-velocity absorption by ion ($|\Delta v|\geq100\ \mathrm{km\,s^{-1}}$)",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig2_hvc_multiion")


# ── fig3 ─────────────────────────────────────────────────────────────────────────
def fig3_inflow_outflow(cells):
    hv = cells[cells.dv.abs() > HVC]
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    a = ax[0]
    finflow = []
    for k in S.ION_KEYS:
        N = hv[f"N_{k}"]
        tot = N.sum()
        finflow.append((N[hv.v_r < 0].sum() / tot) if tot > 0 else np.nan)
    xk = np.arange(len(S.ION_KEYS))
    cols = [S.INFLOW if f > 0.5 else S.OUTFLOW for f in finflow]
    a.bar(xk, finflow, 0.6, color=cols, alpha=0.85)
    a.axhline(0.5, color="0.3", ls="--", lw=1.2)
    a.set_xticks(xk); a.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    a.set_ylabel(r"inflowing fraction of HVC column ($v_r<0$)"); a.set_ylim(0, 1)
    a.text(0.975, 0.96, "inflow-dominated", transform=a.transAxes, color=S.INFLOW, ha="right", va="top", fontsize=9)
    a.text(0.975, 0.045, "outflow-dominated", transform=a.transAxes, color=S.OUTFLOW, ha="right", va="bottom", fontsize=9)
    S.grid(a); S.panel_label(a, "(a)")
    b = ax[1]
    vb = np.arange(-400, 401, 25.0); vc = 0.5 * (vb[:-1] + vb[1:])
    for k in S.ION_KEYS:
        h, _ = np.histogram(hv["v_r"], bins=vb, weights=hv[f"N_{k}"])
        if h.sum() > 0:
            b.step(vc, h / h.sum(), where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
    b.axvline(0, color="0.3", ls=":", lw=1.2)
    b.set_xlabel(r"$v_r$ (galactocentric radial)  $[\mathrm{km\,s^{-1}}]$")
    b.set_ylabel(r"column-weighted PDF"); b.legend(ncol=2, fontsize=9)
    S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle(r"HVC inflow vs.\ outflow across the ionization sequence"
                 if plt.rcParams["text.usetex"] else "HVC inflow vs. outflow across the ionization sequence",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig3_inflow_outflow_multiion")


# ── fig4 ─────────────────────────────────────────────────────────────────────────
def fig4_metallicity(cells):
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    a = ax[0]
    lb = np.linspace(-3, 1.0, 46); lc = 0.5 * (lb[:-1] + lb[1:])
    logZ = np.log10(np.clip(cells["Zsolar"].to_numpy(), 1e-4, None))
    for k in S.ION_KEYS:
        h, _ = np.histogram(logZ, bins=lb, weights=cells[f"N_{k}"].to_numpy())
        if h.sum() > 0:
            a.step(lc, h / h.sum(), where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
    a.set_xlabel(r"$\log_{10}(Z/Z_\odot)$"); a.set_ylabel(r"column-weighted PDF")
    a.legend(ncol=2, fontsize=9); S.grid(a); S.panel_label(a, "(a)")
    # (b) HVC ion-weighted median Z, inflow vs outflow
    b = ax[1]
    hv = cells[cells.dv.abs() > HVC]
    xk = np.arange(len(S.ION_KEYS))
    zin = [wmedian(hv[hv.v_r < 0]["Zsolar"], hv[hv.v_r < 0][f"N_{k}"]) for k in S.ION_KEYS]
    zout = [wmedian(hv[hv.v_r > 0]["Zsolar"], hv[hv.v_r > 0][f"N_{k}"]) for k in S.ION_KEYS]
    b.plot(xk, np.log10(zin), "o-", color=S.INFLOW, lw=2, label="inflow")
    b.plot(xk, np.log10(zout), "s--", color=S.OUTFLOW, lw=2, label="outflow")
    b.set_xticks(xk); b.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    b.set_ylabel(r"median $\log_{10}(Z/Z_\odot)$ of HVC gas"); b.legend()
    S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle("Metallicity of the multi-phase absorbing gas", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig4_metallicity_multiion")


# ── fig5 ─────────────────────────────────────────────────────────────────────────
def fig5_ionization_temperature(cells):
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    a = ax[0]
    tb = np.linspace(3.6, 6.2, 46); tc = 0.5 * (tb[:-1] + tb[1:])
    for k in S.ION_KEYS:
        h, _ = np.histogram(cells["logT"].to_numpy(), bins=tb, weights=cells[f"N_{k}"].to_numpy())
        if h.sum() > 0:
            a.step(tc, h / h.sum(), where="mid", color=S.ION_COL[k], lw=1.9, label=S.ION_LAB[k])
    a.set_xlabel(r"$\log_{10}(T/\mathrm{K})$"); a.set_ylabel(r"column-weighted PDF")
    a.legend(ncol=2, fontsize=9); S.grid(a); S.panel_label(a, "(a)")
    # (b) median logT per ion, ISM vs HVC
    b = ax[1]
    xk = np.arange(len(S.ION_KEYS))
    ism = cells[cells.dv.abs() < IVC]; hv = cells[cells.dv.abs() > HVC]
    tism = [wmedian(ism["logT"], ism[f"N_{k}"]) for k in S.ION_KEYS]
    thv = [wmedian(hv["logT"], hv[f"N_{k}"]) for k in S.ION_KEYS]
    b.plot(xk, tism, "o-", color=S.CLASS["ISM"], lw=2, label="ISM gas")
    b.plot(xk, thv, "^--", color=S.CLASS["HVC"], lw=2, label="HVC gas")
    b.set_xticks(xk); b.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    b.set_ylabel(r"median $\log_{10}(T/\mathrm{K})$"); b.legend()
    S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle("Temperature of the gas traced by each ion (ionization sequence)",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig5_ionization_temperature")


# ── fig6 ─────────────────────────────────────────────────────────────────────────
def fig6_budget(cells):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    xk = np.arange(len(S.ION_KEYS))
    ism, ivc, hvc = [], [], []
    ad = cells.dv.abs()
    for k in S.ION_KEYS:
        N = cells[f"N_{k}"]; tot = N.sum()
        ism.append(N[ad < IVC].sum() / tot); ivc.append(N[(ad >= IVC) & (ad < HVC)].sum() / tot)
        hvc.append(N[ad >= HVC].sum() / tot)
    ism = np.array(ism); ivc = np.array(ivc); hvc = np.array(hvc)
    ax.bar(xk, ism, 0.62, color=S.CLASS["ISM"], label=S.CLASS_LABEL["ISM"])
    ax.bar(xk, ivc, 0.62, bottom=ism, color=S.CLASS["IVC"], label=S.CLASS_LABEL["IVC"])
    ax.bar(xk, hvc, 0.62, bottom=ism + ivc, color=S.CLASS["HVC"], label=S.CLASS_LABEL["HVC"])
    for i in xk:
        ax.text(i, ism[i] + ivc[i] + hvc[i] + 0.01, f"{hvc[i]*100:.0f}\\%" if plt.rcParams["text.usetex"]
                else f"{hvc[i]*100:.0f}%", ha="center", fontsize=9, color=S.CLASS["HVC"])
    ax.set_xticks(xk); ax.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    ax.set_ylabel(r"fraction of ion column"); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
    S.grid(ax)
    fig.tight_layout()
    return S.save(fig, "fig6_column_budget_multiion")


# ── fig7 ─────────────────────────────────────────────────────────────────────────
def fig7_ionratio(cells):
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    for a, (num, den, lab) in zip(ax, [("SiIV", "SiII", r"$\log_{10}\,N_{\mathrm{Si\,IV}}/N_{\mathrm{Si\,II}}$"),
                                       ("CIV", "SiII", r"$\log_{10}\,N_{\mathrm{C\,IV}}/N_{\mathrm{Si\,II}}$")]):
        ok = (cells[f"N_{num}"] > FLOOR[num]) & (cells[f"N_{den}"] > FLOOR[den])
        c = cells[ok]
        ratio = np.log10(c[f"N_{num}"] / c[f"N_{den}"])
        hb = a.hexbin(c["dv"], ratio, gridsize=34, cmap=S.DENSCMAP, bins="log", mincnt=1)
        a.axvline(0, color="w", ls=":", lw=1.0)
        a.set_xlabel(r"$\Delta v\ [\mathrm{km\,s^{-1}}]$"); a.set_ylabel(lab)
        a.set_xlim(-500, 500)
        cb = fig.colorbar(hb, ax=a, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm cells}$")
    S.panel_label(ax[0], "(a)"); S.panel_label(ax[1], "(b)")
    fig.suptitle(r"Ionization state vs.\ velocity offset (higher $=$ more ionized)"
                 if plt.rcParams["text.usetex"] else "Ionization state vs. velocity offset (higher = more ionized)",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig7_ionratio_vs_dv")


# ── fig8 ─────────────────────────────────────────────────────────────────────────
def fig8_detection(cells, n_sl):
    """HVC detection rate: covering fraction per ion and per galaxy."""
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    ad = cells.dv.abs()
    a = ax[0]
    fcov = []
    for k in S.ION_KEYS:
        hv = cells[(ad > HVC) & (cells[f"N_{k}"] > FLOOR[k])]
        fcov.append(hv.groupby(["sid", "mode", "alpha"]).ngroups / n_sl)
    xk = np.arange(len(S.ION_KEYS))
    a.bar(xk, fcov, 0.62, color=[S.ION_COL[k] for k in S.ION_KEYS], alpha=0.9)
    for i, f in enumerate(fcov):
        a.text(i, f + 0.005, f"{f:.2f}", ha="center", fontsize=9)
    a.set_xticks(xk); a.set_xticklabels([S.ION_LAB[k] for k in S.ION_KEYS])
    a.set_ylabel(r"HVC covering fraction (per sightline)")
    S.grid(a); S.panel_label(a, "(a)")
    b = ax[1]
    rows = []
    for sid, g in cells.groupby("sid"):
        rows.append((sid, g[g.dv.abs() > HVC].groupby(["mode", "alpha"]).ngroups / 720.0))
    r = pd.DataFrame(rows, columns=["sid", "fcov"]).sort_values("fcov")
    b.barh(np.arange(len(r)), r.fcov, color=S.CLASS["HVC"], alpha=0.85)
    b.set_yticks(np.arange(len(r))); b.set_yticklabels(r.sid.astype(str), fontsize=8)
    b.axvline(r.fcov.mean(), color="0.2", ls="--", lw=1.4)
    b.set_xlabel(r"HVC covering fraction (any ion)"); b.set_ylabel("subhalo ID")
    S.tag(b, r"$\langle f\rangle=%.2f$" % r.fcov.mean(), "lr")
    S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle("HVC detection rate: by ion and by galaxy", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig8_detection_rate")


# ── fig9 ─────────────────────────────────────────────────────────────────────────
def fig9_phase(cells):
    """HVC spatial distribution and radial kinematics."""
    hv = cells[cells.dv.abs() > HVC]
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))
    a = ax[0]
    inner = hv[hv.R_disk < 120]
    hb = a.hexbin(inner.R_disk, inner.z_disk, C=inner.v_r, reduce_C_function=np.mean,
                  gridsize=38, cmap=S.DIVCMAP, vmin=-200, vmax=200, mincnt=1)
    a.axhline(0, color="0.35", ls=":", lw=1.0)
    a.set_xlabel(r"$R_{\mathrm{disk}}\ [\mathrm{kpc}]$"); a.set_ylabel(r"$z_{\mathrm{disk}}\ [\mathrm{kpc}]$")
    cb = fig.colorbar(hb, ax=a, fraction=0.05, pad=0.02); cb.set_label(r"$\langle v_r\rangle\ [\mathrm{km\,s^{-1}}]$")
    S.panel_label(a, "(a)")
    b = ax[1]
    hb = b.hexbin(hv.r_gal, hv.v_r, gridsize=34, cmap=S.DENSCMAP, bins="log", mincnt=1)
    b.axhline(0, color="w", ls=":", lw=1.0)
    b.set_xlabel(r"$r_{\mathrm{gal}}\ [\mathrm{kpc}]$"); b.set_ylabel(r"$v_r\ [\mathrm{km\,s^{-1}}]$")
    cb = fig.colorbar(hb, ax=b, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm cells}$")
    S.panel_label(b, "(b)")
    fig.suptitle("HVC spatial distribution and radial kinematics", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig9_hvc_phase_space")


def main():
    usetex = S.set_style()
    print(f"usetex={usetex}")
    cells, n_sl = load()
    print(f"{len(cells)} multi-ion absorbing cells over {n_sl} sightlines")
    for fn in (fig1_abundance, fig2_hvc, fig3_inflow_outflow, fig4_metallicity,
               fig5_ionization_temperature, fig6_budget, fig7_ionratio, fig8_detection, fig9_phase):
        try:
            p = fn(cells, n_sl) if fn in (fig1_abundance, fig2_hvc, fig8_detection) else fn(cells)
            print(f"  saved {p.name}")
        except Exception as e:
            import traceback
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
