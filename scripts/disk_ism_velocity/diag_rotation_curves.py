#!/usr/bin/env python3
"""Diagnostic: full 3-tracer rotation curves per galaxy, with occupancy, bin-width overlay,
and the radius range the sightlines actually sample.

For each SID: v_phi(R) for cold gas (T<1e4), SF gas, young stars (median + 16-84 band) and
their equal-weight average (the fiducial v_ISM curve), 0-30 kpc at 1 kpc bins; the 0.5 and
2 kpc average curves are overlaid to show bin sensitivity; per-tracer counts are shown below;
the R_anchor range spanned by the 720 sightlines and rho~=25.6 kpc are marked.

Outputs diagnostics_v2/rotation_curves/: one detail figure per SID + a 20-panel overview.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
V2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2/rotation_curves")
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]
TCOL = {"cold_gas": "#1d3557", "sf_gas": "#2a9d8f", "young_stars": "#e76f51"}
TLAB = {"cold_gas": r"cold gas ($T<10^4$)", "sf_gas": "SF gas", "young_stars": "young stars"}


def load(sid, bw):
    tag = f"bw{bw:g}".replace(".", "p")
    avg = pd.read_csv(RCV2 / f"rc_sid{sid}_{tag}_ism_average.csv")
    tracers = {}
    for t in TCOL:
        f = RCV2 / f"rc_sid{sid}_{tag}_{t}.csv"
        tracers[t] = pd.read_csv(f) if f.exists() else None
    return avg, tracers


def rspan(sid, m):
    d = m[m.sid == sid]
    return (float(d.R_anchor.quantile(0.05)), float(d.R_anchor.quantile(0.95)),
            float(d.rho_kpc.median()))


def draw(ax, sid, m, legend=False, counts_ax=None):
    avg1, tr1 = load(sid, 1.0)
    for t, c in tr1.items():
        if c is None:
            continue
        ok = c.n >= 20
        ax.plot(c.R_center[ok], c.v_phi_median[ok], color=TCOL[t], lw=1.6, label=TLAB[t])
        ax.fill_between(c.R_center[ok], c.v_phi_p16[ok], c.v_phi_p84[ok],
                        color=TCOL[t], alpha=0.12, lw=0)
    ax.plot(avg1.R_center, avg1.v_fid_median, color="k", lw=2.4, label="3-tracer avg (1 kpc)")
    for bw, ls in [(0.5, ":"), (2.0, "--")]:
        a, _ = load(sid, bw)
        ax.plot(a.R_center, a.v_fid_median, color="0.45", lw=1.1, ls=ls, label=f"avg ({bw:g} kpc)")
    r5, r95, rho = rspan(sid, m)
    ax.axvspan(r5, r95, color="#ffd166", alpha=0.25, lw=0, zorder=0)
    ax.axvline(rho, color="0.4", ls=":", lw=1.0)
    ax.set_xlim(0, 30)
    ax.set_title(rf"SID {sid}", fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc="lower right", ncol=1)
    if counts_ax is not None:
        for t, c in tr1.items():
            if c is not None:
                counts_ax.plot(c.R_center, np.clip(c.n, 0.5, None), color=TCOL[t], lw=1.2)
        counts_ax.axvspan(r5, r95, color="#ffd166", alpha=0.25, lw=0)
        counts_ax.set_yscale("log"); counts_ax.set_xlim(0, 30)
        counts_ax.set_ylabel("N/bin"); counts_ax.set_xlabel(r"$R$ [kpc]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    m = pd.read_csv(V2)

    # per-SID detail (curve + counts)
    for sid in SIDS:
        try:
            fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.2), height_ratios=[3, 1], sharex=True)
            draw(ax[0], sid, m, legend=True, counts_ax=ax[1])
            ax[0].set_ylabel(r"$v_\phi$ [km s$^{-1}$]"); S.grid(ax[0]); S.grid(ax[1])
            fig.tight_layout(); fig.savefig(OUT / f"rc_sid{sid}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[SID {sid}] detail failed: {e}")

    # 20-panel overview
    fig, ax = plt.subplots(4, 5, figsize=(22, 15), sharex=True)
    for k, sid in enumerate(SIDS):
        a = ax[k // 5][k % 5]
        try:
            draw(a, sid, m, legend=(k == 0))
            a.set_ylabel(r"$v_\phi$" if k % 5 == 0 else "")
        except Exception as e:
            a.text(0.5, 0.5, f"{sid}\n{e}", transform=a.transAxes, ha="center", fontsize=7)
        S.grid(a)
    fig.suptitle(r"3-tracer disk rotation curves (cold gas $T<10^4$ + SF gas + young stars); "
                 r"shaded = sightline $R_{\rm anchor}$ range", fontsize=14)
    fig.supxlabel(r"$R$ [kpc]")
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(OUT / "overview_all_sids.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved rotation-curve diagnostics -> {OUT}")


if __name__ == "__main__":
    main()
