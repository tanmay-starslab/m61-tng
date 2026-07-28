#!/usr/bin/env python3
"""Diagnostics: (A) systemic-velocity behaviour, (B) bin-width sensitivity of the fiducial
curve and of v_ISM, (C) v_ISM(model) vs the Si II dip and the v1 direct method.

(A) v_sys = SubhaloVel . los(alpha): shown vs alpha per galaxy (it varies as the fixed
    galaxy velocity vector projects onto the rotating sightline).
(B) v_fid(R) at the sampled radius under 0.5/1/2 kpc bins, and the induced spread in the
    projected v_ISM -> shows how much binning noise propagates.
(C) v_ISM(model) vs Si II spectrum dip and vs the v1 direct cool-gas velocity.
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
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2")
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]


def vfid_at(sid, bw, R):
    tag = f"bw{bw:g}".replace(".", "p")
    a = pd.read_csv(RCV2 / f"rc_sid{sid}_{tag}_ism_average.csv")
    ok = np.isfinite(a.v_fid_median)
    return float(np.interp(R, a.R_center[ok], a.v_fid_median[ok])) if ok.sum() >= 2 else np.nan


def robust(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.median(x), np.median(np.abs(x - np.median(x))) * 1.4826) if len(x) else (np.nan, np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    m = pd.read_csv(V2)

    # ---------- (A) v_sys ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("turbo")
    for i, sid in enumerate(SIDS):
        d = m[(m.sid == sid) & (m["mode"] == "flip")].sort_values("alpha_deg")
        ax[0].plot(d.alpha_deg, d.v_sys, color=cmap(i / 19), lw=1.1)
    ax[0].set_xlabel(r"$\alpha$ [deg]"); ax[0].set_ylabel(r"$v_{\rm sys}=\mathrm{SubhaloVel}\cdot\hat{n}_{\rm los}$ [km s$^{-1}$]")
    ax[0].set_title(r"\bf (A) systemic LOS velocity vs orientation"); S.grid(ax[0])
    amp = m.groupby("sid").v_sys.agg(lambda s: s.max() - s.min())
    ax[1].bar(range(len(SIDS)), [amp[s] for s in SIDS], color=S.ACCENT, alpha=0.8, edgecolor="k", lw=0.4)
    ax[1].set_xticks(range(len(SIDS))); ax[1].set_xticklabels(SIDS, rotation=90, fontsize=6)
    ax[1].set_ylabel(r"$v_{\rm sys}$ peak-to-peak over $\alpha$ [km s$^{-1}$]")
    ax[1].set_title(r"\bf per-galaxy $v_{\rm sys}$ amplitude"); S.grid(ax[1])
    S.tag(ax[0], r"$v_{\rm sys}$ = galaxy bulk velocity projected on the sightline;"
                 "\n" r"subtracted identically from gas \& spectrum", corner="ll", fs=8)
    fig.tight_layout(); fig.savefig(OUT / "diag_vsys.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- (B) bin sensitivity ----------
    rows = []
    for sid in SIDS:
        Rmed = float(m[m.sid == sid].R_anchor.median())
        v = {bw: vfid_at(sid, bw, Rmed) for bw in (0.5, 1.0, 2.0)}
        proj = float(m[m.sid == sid].proj_phi.median())
        rows.append(dict(sid=sid, R=Rmed, v05=v[0.5], v10=v[1.0], v20=v[2.0],
                         dvism_05_20=abs(v[0.5] - v[2.0]) * abs(proj)))
    b = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(SIDS))
    for col, lab, mk in [("v05", "0.5 kpc", "o"), ("v10", "1 kpc", "s"), ("v20", "2 kpc", "^")]:
        ax[0].plot(x, b[col], mk, ms=5, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(SIDS, rotation=90, fontsize=6)
    ax[0].set_ylabel(r"$v_\phi$ at sampled $R$ [km s$^{-1}$]")
    ax[0].set_title(r"\bf (B) fiducial $v_\phi$ vs bin width"); ax[0].legend(fontsize=8); S.grid(ax[0])
    ax[1].bar(x, b.dvism_05_20, color="#B02418", alpha=0.8, edgecolor="k", lw=0.4)
    ax[1].axhline(10, color="0.3", ls="--", lw=1.0, label="10 km/s")
    ax[1].set_xticks(x); ax[1].set_xticklabels(SIDS, rotation=90, fontsize=6)
    ax[1].set_ylabel(r"$|\Delta v_{\rm ISM}|$ (0.5 vs 2 kpc) [km s$^{-1}$]")
    ax[1].set_title(r"\bf binning-induced $v_{\rm ISM}$ spread"); ax[1].legend(fontsize=8); S.grid(ax[1])
    fig.tight_layout(); fig.savefig(OUT / "diag_bin_sensitivity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- (C) comparison ----------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ind = m[m.in_disk_v1.astype(bool)]
    lim = 340
    for a, (xcol, xlab) in zip(ax[:2], [("SiII_dip", "Si II 1260 dip"),
                                        ("v_ism_direct_cool", "v1 direct cool-gas")]):
        a.scatter(ind[xcol], ind.v_ism_model, s=5, color=S.ACCENT, alpha=0.3, edgecolors="none",
                  rasterized=True, label="in-disk")
        out = m[~m.in_disk_v1.astype(bool)]
        a.scatter(out[xcol], out.v_ism_model, s=5, color="0.7", alpha=0.2, edgecolors="none",
                  rasterized=True, label="beyond disk")
        a.plot([-lim, lim], [-lim, lim], "k--", lw=1.1)
        med, sig = robust(ind.v_ism_model - ind[xcol])
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel(xlab); a.set_ylabel(r"$v_{\rm ISM}$ model")
        S.tag(a, rf"in-disk: med$={med:+.0f}$, $\sigma={sig:.0f}$", corner="ul")
        a.legend(fontsize=8, loc="lower right"); S.grid(a)
    res = (ind.v_ism_model - ind.SiII_dip).dropna()
    ax[2].hist(res, bins=np.linspace(-150, 150, 61), color=S.ACCENT, alpha=0.75, edgecolor="white", lw=0.4)
    med, sig = robust(res)
    ax[2].axvline(med, color="#B02418", lw=1.8, label=rf"median $={med:+.0f}$")
    ax[2].set_xlabel(r"$v_{\rm ISM}$ model $-$ Si II dip [km s$^{-1}$]"); ax[2].set_ylabel("sightlines")
    ax[2].legend(fontsize=9); S.grid(ax[2])
    ax[0].set_title(r"\bf (C) model vs observable Si II dip")
    ax[1].set_title(r"\bf model vs v1 direct method")
    ax[2].set_title(rf"\bf residual (in-disk): $\sigma={sig:.0f}$ km s$^{{-1}}$")
    fig.tight_layout(); fig.savefig(OUT / "diag_model_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print("saved diag_vsys, diag_bin_sensitivity, diag_model_comparison")
    print(f"  v_sys peak-to-peak median over SIDs: {amp.median():.0f} km/s")
    print(f"  binning |Δv_ISM|(0.5 vs 2kpc) median: {b.dvism_05_20.median():.1f} km/s")
    med, sig = robust((ind.v_ism_model - ind.SiII_dip))
    print(f"  v_ISM model - SiII dip (in-disk): median {med:+.1f}, sigma {sig:.1f} km/s")


if __name__ == "__main__":
    main()
