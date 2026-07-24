#!/usr/bin/env python3
"""Publication figures for the m61-tng disk-ISM / HVC study.

Reads the combined Si II absorbing-gas catalog (build_absorber_catalog.py) + the v_ISM
master table, clusters each sightline's cells into velocity components (= absorbers), and
makes the paper figures:

  fig1_absorber_abundance_dv        abundance / detection rate vs dv = v_los - v_ISM (full)
  fig2_hvc_abundance_ge100          HVC abundance vs |dv| for |dv|>=100 (x starts at 100)
  fig3_hvc_inflow_outflow           HVC radial motion: inflow vs outflow (+ dv-v_r plane)
  fig4_metallicity_accretion_outflow  Z/Zsun of inflowing vs outflowing HVCs
  fig5_hvc_phase_space              where HVCs are: (R,z) and v_r vs r_gal
  fig6_detection_rate               HVC covering fraction per galaxy + Si II budget by class
  fig7_column_temperature_vs_dv     Si II column & temperature vs dv

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
COMP_MIN_NSIII = 10 ** 12.5   # cm^-2, component Si II column detection floor
VGAP = 25.0                   # km/s, velocity gap that separates components
HVC = 100.0                   # km/s, |dv| threshold for HVC
IVC = 40.0


# ── load + build absorber components ─────────────────────────────────────────────
def load_cells():
    files = sorted(glob.glob(str(CATDIR / "absorbers_sid*.parquet")))
    if not files:
        raise SystemExit("no absorber catalog parquet files found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def build_components(cells):
    """Cluster each sightline's absorbing cells into Si II velocity components."""
    out = []
    keys = ["sid", "mode", "alpha"]
    for (sid, mode, alpha), c in cells.groupby(keys, sort=False):
        c = c.sort_values("v_rest")
        v = c["v_rest"].to_numpy()
        if len(v) == 0:
            continue
        brk = np.where(np.diff(v) > VGAP)[0]
        for g in np.split(np.arange(len(c)), brk + 1):
            cc = c.iloc[g]
            w = cc["NSiII"].to_numpy()
            W = float(w.sum())
            if W < COMP_MIN_NSIII:
                continue
            def cw(col):
                return float(np.average(cc[col].to_numpy(), weights=w))
            out.append(dict(
                sid=sid, mode=mode, alpha=alpha, v_ISM=float(cc["v_ISM"].iloc[0]),
                in_disk=bool(cc["in_disk"].iloc[0]), rho_kpc=float(cc["rho_kpc"].iloc[0]),
                v_rest=cw("v_rest"), dv=cw("dv"), NSiII=W, NHI=float(cc["NHI"].sum()),
                v_r=cw("v_r"), v_z=cw("v_z"), R_disk=cw("R_disk"), z_disk=cw("z_disk"),
                r_gal=cw("r_gal"), Zsolar=cw("Zsolar"), logT=cw("logT"), ncell=len(cc)))
    comp = pd.DataFrame(out)
    comp["cls"] = S.classify(comp["dv"])
    comp["logNSiII"] = np.log10(comp["NSiII"])
    return comp


# ── figures ──────────────────────────────────────────────────────────────────────
def fig1_abundance(comp, cells):
    """Absorber abundance / detection rate vs dv = v_los - v_ISM."""
    n_sl = comp.groupby(["sid", "mode", "alpha"]).ngroups
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    bins = np.arange(-500, 501, 20.0)
    xc = 0.5 * (bins[:-1] + bins[1:])
    # (a) discrete absorber counts per sightline
    a = ax[0]
    S.shade_classes(a)
    h, _ = np.histogram(comp["dv"], bins=bins)
    a.step(xc, h / n_sl, where="mid", color="#1B3A5B", lw=2.0, zorder=4)
    a.fill_between(xc, h / n_sl, step="mid", color="#1B3A5B", alpha=0.12)
    a.axvline(0, color="0.35", ls=":", lw=1.3)
    for xb in (IVC, HVC):
        for s in (1, -1):
            a.axvline(s * xb, color="0.5", ls="--", lw=0.7, alpha=0.6)
    a.set_xlim(-500, 500)
    a.set_xlabel(r"$v_{\mathrm{los}} - v_{\mathrm{ISM}}\ \ [\mathrm{km\,s^{-1}}]$")
    a.set_ylabel(r"absorbers per sightline per bin")
    S.grid(a); S.panel_label(a, "(a)")
    # (b) Si II column-weighted abundance (all absorbing gas)
    b = ax[1]
    S.shade_classes(b)
    hw, _ = np.histogram(cells["dv"], bins=bins, weights=cells["NSiII"])
    b.step(xc, hw / n_sl, where="mid", color=S.ACCENT, lw=2.0, zorder=4)
    b.fill_between(xc, hw / n_sl, step="mid", color=S.ACCENT, alpha=0.12)
    b.axvline(0, color="0.35", ls=":", lw=1.3)
    b.set_xlim(-500, 500); b.set_yscale("log")
    b.set_xlabel(r"$v_{\mathrm{los}} - v_{\mathrm{ISM}}\ \ [\mathrm{km\,s^{-1}}]$")
    b.set_ylabel(r"$\Sigma\,N_{\mathrm{Si\,II}}$ per sightline  $[\mathrm{cm^{-2}}\,/\,\mathrm{bin}]$")
    S.grid(b); S.panel_label(b, "(b)")
    # class labels (x in data coords, y in axes fraction)
    tr = ax[0].get_xaxis_transform()
    for c, x in (("ISM", 0), ("IVC", 70), ("HVC", 300)):
        ax[0].text(x, 0.96, c, transform=tr, color=S.CLASS[c], ha="center",
                   va="top", fontsize=plt.rcParams["legend.fontsize"], fontweight="bold")
    fig.text(0.5, 0.995, r"Absorber abundance vs.\ velocity offset from the disk ISM"
             if plt.rcParams["text.usetex"] else "Absorber abundance vs. velocity offset from the disk ISM",
             ha="center", va="top", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig1_absorber_abundance_dv")


def fig2_hvc_ge100(comp):
    """HVC abundance vs |dv| for |dv| >= 100 (x-axis starts at 100)."""
    hv = comp[comp["dv"].abs() >= HVC]
    n_sl = comp.groupby(["sid", "mode", "alpha"]).ngroups
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    bins = np.arange(100, 561, 20.0)
    xc = 0.5 * (bins[:-1] + bins[1:])
    appr = hv[hv["dv"] < 0]; rec = hv[hv["dv"] > 0]
    for sub, col, lab in ((appr, S.INFLOW, r"approaching ($\Delta v<0$)"),
                          (rec, S.OUTFLOW, r"receding ($\Delta v>0$)")):
        h, _ = np.histogram(sub["dv"].abs(), bins=bins)
        ax.step(xc, h / n_sl, where="mid", color=col, lw=2.2, label=lab, zorder=4)
        ax.fill_between(xc, h / n_sl, step="mid", color=col, alpha=0.10)
    hall, _ = np.histogram(hv["dv"].abs(), bins=bins)
    ax.step(xc, hall / n_sl, where="mid", color="0.2", lw=1.4, ls="--", label="all HVC", zorder=3)
    ax.set_xlim(100, 560); ax.set_yscale("log")
    ax.set_xlabel(r"$|v_{\mathrm{los}} - v_{\mathrm{ISM}}|\ \ [\mathrm{km\,s^{-1}}]$")
    ax.set_ylabel(r"HVC absorbers per sightline  $dN/d|\Delta v|$")
    ax.legend(loc="upper right")
    S.grid(ax)
    S.tag(ax, (r"$N_{\rm HVC}=%d$" % len(hv)) + "\n" + (r"$f_{\rm HVC}=%.2f$/sl" % (len(hv) / n_sl)), "ll")
    fig.tight_layout()
    return S.save(fig, "fig2_hvc_abundance_ge100")


def fig3_inflow_outflow(comp):
    """HVC radial motion: inflow vs outflow, and the dv - v_r plane."""
    hv = comp[comp["dv"].abs() >= HVC].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))
    a = ax[0]
    bins = np.arange(-400, 401, 25.0)
    inflow = hv[hv["v_r"] < 0]; outflow = hv[hv["v_r"] > 0]
    a.hist(inflow["v_r"], bins=bins, color=S.INFLOW, alpha=0.75, label="inflow ($v_r<0$)")
    a.hist(outflow["v_r"], bins=bins, color=S.OUTFLOW, alpha=0.75, label="outflow ($v_r>0$)")
    a.axvline(0, color="0.3", ls=":", lw=1.3)
    a.set_xlabel(r"$v_r$ (galactocentric radial)  $[\mathrm{km\,s^{-1}}]$")
    a.set_ylabel("number of HVC absorbers")
    a.legend(loc="upper left"); S.grid(a); S.panel_label(a, "(a)")
    fout = (hv["v_r"] > 0).mean()
    S.tag(a, r"outflow %.0f\%%" % (100 * fout) if plt.rcParams["text.usetex"]
          else "outflow %.0f%%" % (100 * fout), "ur")
    # (b) dv vs v_r 2D
    b = ax[1]
    hb = b.hexbin(hv["dv"], hv["v_r"], gridsize=34, cmap=S.DENSCMAP, bins="log", mincnt=1)
    b.axhline(0, color="w", ls=":", lw=1.0); b.axvline(0, color="w", ls=":", lw=1.0)
    b.set_xlabel(r"$\Delta v = v_{\mathrm{los}}-v_{\mathrm{ISM}}\ [\mathrm{km\,s^{-1}}]$")
    b.set_ylabel(r"$v_r\ [\mathrm{km\,s^{-1}}]$")
    cb = fig.colorbar(hb, ax=b, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm HVC}$")
    S.panel_label(b, "(b)")
    fig.suptitle("High-velocity cloud kinematics: inflow vs.\\ outflow" if plt.rcParams["text.usetex"]
                 else "High-velocity cloud kinematics: inflow vs. outflow",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig3_hvc_inflow_outflow")


def fig4_metallicity(comp):
    """Metallicity of inflowing (accretion) vs outflowing HVCs."""
    hv = comp[comp["dv"].abs() >= HVC].copy()
    hv = hv[np.isfinite(hv["Zsolar"]) & (hv["Zsolar"] > 0)]
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))
    a = ax[0]
    lb = np.linspace(-3, 1.0, 40)
    for sub, col, lab in ((hv[hv.v_r < 0], S.INFLOW, "inflow"), (hv[hv.v_r > 0], S.OUTFLOW, "outflow")):
        a.hist(np.log10(sub["Zsolar"]), bins=lb, color=col, alpha=0.7, density=True, label=lab)
        a.axvline(np.log10(sub["Zsolar"].median()), color=col, ls="--", lw=1.6)
    a.set_xlabel(r"$\log_{10}(Z/Z_\odot)$"); a.set_ylabel("PDF (HVC absorbers)")
    a.legend(loc="upper left"); S.grid(a); S.panel_label(a, "(a)")
    b = ax[1]
    hb = b.hexbin(hv["v_r"], np.log10(hv["Zsolar"]), gridsize=32, cmap=S.DENSCMAP, bins="log", mincnt=1)
    b.axvline(0, color="w", ls=":", lw=1.0)
    b.set_xlabel(r"$v_r\ [\mathrm{km\,s^{-1}}]$"); b.set_ylabel(r"$\log_{10}(Z/Z_\odot)$")
    cb = fig.colorbar(hb, ax=b, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm HVC}$")
    S.panel_label(b, "(b)")
    fig.suptitle("Metallicity of inflowing vs.\\ outflowing HVCs" if plt.rcParams["text.usetex"]
                 else "Metallicity of inflowing vs. outflowing HVCs", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig4_metallicity_accretion_outflow")


def fig5_phase_space(comp):
    """Where the HVCs are: disk-frame (R,z) and v_r vs galactocentric r."""
    hv = comp[comp["dv"].abs() >= HVC].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))
    a = ax[0]
    inner = hv[hv["R_disk"] < 120]
    hb0 = a.hexbin(inner["R_disk"], inner["z_disk"], C=inner["v_r"], reduce_C_function=np.mean,
                   gridsize=38, cmap=S.DIVCMAP, vmin=-200, vmax=200, mincnt=1)
    a.axhline(0, color="0.35", ls=":", lw=1.0)
    a.set_xlabel(r"$R_{\mathrm{disk}}\ [\mathrm{kpc}]$"); a.set_ylabel(r"$z_{\mathrm{disk}}\ [\mathrm{kpc}]$")
    cb = fig.colorbar(hb0, ax=a, fraction=0.05, pad=0.02)
    cb.set_label(r"$\langle v_r\rangle\ [\mathrm{km\,s^{-1}}]$")
    S.panel_label(a, "(a)")
    b = ax[1]
    hb = b.hexbin(hv["r_gal"], hv["v_r"], gridsize=32, cmap=S.DENSCMAP, bins="log", mincnt=1)
    b.axhline(0, color="w", ls=":", lw=1.0)
    b.set_xlabel(r"$r_{\mathrm{gal}}\ [\mathrm{kpc}]$"); b.set_ylabel(r"$v_r\ [\mathrm{km\,s^{-1}}]$")
    cb = fig.colorbar(hb, ax=b, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm HVC}$")
    S.panel_label(b, "(b)")
    fig.suptitle("HVC spatial distribution and radial kinematics", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig5_hvc_phase_space")


def fig6_detection_rate(comp):
    """HVC covering fraction per galaxy + Si II column budget by kinematic class."""
    n_sl_tot = comp.groupby(["sid", "mode", "alpha"]).ngroups
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    # (a) covering fraction of >=1 HVC per sightline, per galaxy
    a = ax[0]
    rows = []
    for sid, g in comp.groupby("sid"):
        nsl = g.groupby(["mode", "alpha"]).ngroups
        hvsl = g[g.dv.abs() >= HVC].groupby(["mode", "alpha"]).ngroups
        rows.append((sid, hvsl / nsl))
    r = pd.DataFrame(rows, columns=["sid", "fcov"]).sort_values("fcov")
    a.barh(np.arange(len(r)), r["fcov"], color=S.CLASS["HVC"], alpha=0.85)
    a.set_yticks(np.arange(len(r))); a.set_yticklabels(r["sid"].astype(str), fontsize=8)
    a.set_xlabel(r"HVC covering fraction  $f(\geq\!1\ \mathrm{HVC})$"); a.set_ylabel("subhalo ID")
    a.axvline(r["fcov"].mean(), color="0.2", ls="--", lw=1.4)
    S.grid(a); S.panel_label(a, "(a)")
    S.tag(a, r"$\langle f\rangle=%.2f$" % r["fcov"].mean(), "lr")
    # (b) Si II column budget by class (fraction of total Si II column)
    b = ax[1]
    tot = comp.groupby("cls")["NSiII"].sum()
    frac = (tot / tot.sum()).reindex(["ISM", "IVC", "HVC"])
    bars = b.bar(range(3), frac.values, color=[S.CLASS[c] for c in ["ISM", "IVC", "HVC"]], alpha=0.9)
    b.set_xticks(range(3)); b.set_xticklabels([S.CLASS_LABEL[c] for c in ["ISM", "IVC", "HVC"]], fontsize=9)
    b.set_ylabel(r"fraction of total Si\,II column")
    for rect, f in zip(bars, frac.values):
        b.text(rect.get_x() + rect.get_width() / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=10)
    S.grid(b); S.panel_label(b, "(b)")
    fig.suptitle("HVC detection rate and Si II column budget", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig6_detection_rate")


def fig7_column_temp(comp):
    """Si II column and temperature as a function of dv."""
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    a = ax[0]
    hb = a.hexbin(comp["dv"], comp["logNSiII"], gridsize=36, cmap=S.DENSCMAP, bins="log", mincnt=1)
    a.axvline(0, color="w", ls=":", lw=1.0)
    a.set_xlabel(r"$\Delta v\ [\mathrm{km\,s^{-1}}]$"); a.set_ylabel(r"$\log_{10} N_{\mathrm{Si\,II}}\ [\mathrm{cm^{-2}}]$")
    cb = fig.colorbar(hb, ax=a, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm abs}$")
    a.set_xlim(-500, 500); S.panel_label(a, "(a)")
    b = ax[1]
    hb = b.hexbin(comp["dv"], comp["logT"], gridsize=36, cmap=S.DENSCMAP, bins="log", mincnt=1)
    b.axvline(0, color="w", ls=":", lw=1.0)
    b.set_xlabel(r"$\Delta v\ [\mathrm{km\,s^{-1}}]$"); b.set_ylabel(r"$\log_{10}(T/\mathrm{K})$")
    cb = fig.colorbar(hb, ax=b, fraction=0.05, pad=0.02); cb.set_label(r"$N_{\rm abs}$")
    b.set_xlim(-500, 500); S.panel_label(b, "(b)")
    fig.suptitle("Absorber column and temperature vs.\\ velocity offset" if plt.rcParams["text.usetex"]
                 else "Absorber column and temperature vs. velocity offset", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return S.save(fig, "fig7_column_temperature_vs_dv")


def main():
    usetex = S.set_style()
    print(f"usetex={usetex}")
    cells = load_cells()
    print(f"loaded {len(cells)} absorbing cells")
    comp = build_components(cells)
    comp.to_parquet(S.FIGDIR.parent / "absorber_catalog" / "absorber_components.parquet", index=False)
    n_sl = comp.groupby(["sid", "mode", "alpha"]).ngroups
    print(f"{len(comp)} absorber components over {n_sl} sightlines "
          f"({(comp.cls.value_counts(normalize=True)*100).round(1).to_dict()})")
    for fn in (fig1_abundance, fig2_hvc_ge100, fig3_inflow_outflow, fig4_metallicity,
               fig5_phase_space, fig6_detection_rate, fig7_column_temp):
        try:
            if fn is fig1_abundance:
                p = fn(comp, cells)
            else:
                p = fn(comp)
            print(f"  saved {p.name}")
        except Exception as e:
            import traceback
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
