#!/usr/bin/env python3
"""Per-sightline ray diagnostics for the v3 velocities. Marks v3a (3-tracer along sightline),
v3b (centre->impact line), v2 (rotation curve), v1 direct, and the Si II dip on the spectra and
the ray velocity profile, so each per-orientation value can be audited against the absorption.
All velocities are read from the v3 master (no cutout needed); the ray gives the spectra + gas.

Usage: python ray_diagnostic_v3.py <sid> [alpha_step]   (default step=20 -> 36/SID)
Output: diagnostics_v3/ray_diagnostics_v3/raydiagv3_sid<sid>_<mode>_alpha<aaa>.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
from pm_general import get_original_rho  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
M3 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3/vism_master_v3.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v3/ray_diagnostics_v3")
MARK = [("v_ism_v3a", "v3a along-sightline", "#1B9E77", "-", 3.0),
        ("v_ism_v3b", "v3b centre->impact", "#B02418", "--", 1.6),
        ("v_ism_v2", "v2 rot-curve", "#7B6FB0", "-.", 1.4),
        ("v_ism_direct_cool", "v1 direct", "#E19A3C", ":", 1.6)]


def diagnose(sid, mode, alpha, rc, m3):
    try:
        row = m3.loc[(mode, alpha)]
    except KeyError:
        return None
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = R.projection(sid, mode, alpha, rc, rho)
    v_sys = M["v_sys"]; R_edge = R.r95_cold_gas(sid)
    with h5py.File(R.combined_path(sid), "r") as h:
        r = R.ray_group(h, mode, alpha)
        grid = r["original_trident_ray_h5/grid"]
        res = R.compute_vism_fields(grid, M, R_edge, rc)
        ion_n = {fld: (grid[fld][()] if fld in grid else np.zeros(len(res["v_rest"])))
                 for _, _, fld, _, _ in R.IONS}
        specs = {}
        for key, rest, fld, lab, col in R.IONS:
            g = r.get(f"spectrum_by_line/{key}/lsf")
            if g is not None:
                specs[key] = (g["lambda_A"][()], g["flux"][()], rest, lab, col)
    v_rest, s, z_d, R_disk = res["v_rest"], res["s"], res["z_d"], res["R_disk"]
    disk = res["disk"]
    vdip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = R.spectrum_v(lam, rest, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            vdip = float(vv[w][np.argmin(flux[w])])
    v3a = float(row.v_ism_v3a)
    off = v3a - vdip
    verdict = "LANDS" if abs(off) < 30 else ("NEAR" if abs(off) < 60 else "OFF")

    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5))
    fig.suptitle(f"SID {sid} · {mode} · α={alpha}° · ρ={rho:.1f} · in_disk={bool(row.in_disk_v1)} · "
                 f"v3a={v3a:.0f} (dip {vdip:.0f}, off {off:+.0f} [{verdict}]) · v_sys={v_sys:.0f}",
                 fontsize=12)

    def marks(a, axis="v"):
        for col, lab, c, ls, lw in MARK:
            v = float(row[col]) if col in row and np.isfinite(row[col]) else np.nan
            if np.isfinite(v):
                (a.axvline if axis == "v" else a.axhline)(v, color=c, ls=ls, lw=lw, label=f"{lab}={v:.0f}")
        if np.isfinite(vdip):
            (a.axvline if axis == "v" else a.axhline)(vdip, color="k", ls=":", lw=1.4, label=f"Si II dip={vdip:.0f}")

    a = ax[0, 0]
    for key, rest, fld, lab, col in R.IONS:
        if key in specs:
            lam, flux, rest, lab, col = specs[key]
            vv = R.spectrum_v(lam, rest, v_sys); mm = (vv > -700) & (vv < 700); o = np.argsort(vv[mm])
            a.plot(vv[mm][o], flux[mm][o], color=col, lw=1.2, label=lab)
    a.axvline(0, color="0.5", ls=":", lw=0.8); marks(a)
    a.set_xlim(-700, 700); a.set_ylim(-0.05, 1.15)
    a.set_xlabel(r"$v_{\rm rest}$ [km/s]"); a.set_ylabel("flux")
    a.set_title("(A) spectra + all v_ISM estimates"); a.legend(fontsize=6.5, ncol=2, loc="lower left")

    a = ax[0, 1]
    c = np.log10(np.clip(ion_n[R.SIII], 1e-14, None))
    a.scatter(s, v_rest, c=c, s=8 + 260 * (res["w_si"] / (res["w_si"].max() + 1e-30)),
              cmap="viridis", alpha=0.7, edgecolors="none")
    ds = disk & (s > s.min())
    if ds.any():
        a.axvspan(s[ds].min(), s[ds].max(), color="#ffd166", alpha=0.3, label="disk crossing")
    marks(a, axis="h"); a.axhline(0, color="0.5", ls=":", lw=0.8)
    a.set_xlabel("s from impact point [kpc]"); a.set_ylabel(r"$v_{\rm rest}$")
    a.set_title("(B) ray velocity profile — where v3a is measured"); a.legend(fontsize=7, loc="upper right")

    a = ax[1, 0]
    bins = np.linspace(-700, 700, 141)
    for key, rest, fld, lab, col in R.IONS:
        w = ion_n[fld] * res["dl"]
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(v_rest, bins=bins, weights=w)
        if hist.max() > 0:
            hist = hist / hist.max()
        a.step(0.5 * (bins[:-1] + bins[1:]), hist, where="mid", color=col, lw=1.2, label=lab)
    a.axvline(0, color="0.5", ls=":", lw=0.8); marks(a)
    a.set_xlim(-700, 700); a.set_xlabel(r"$v_{\rm rest}$"); a.set_ylabel("column-wt (norm.)")
    a.set_title("(C) ion column-weighted velocity"); a.legend(fontsize=6.5, ncol=2)

    a = ax[1, 1]; a.axis("off")
    lines = ["AUDIT — v3 velocities (per orientation)", "",
             f"rho = {rho:.1f} kpc   R_anchor = {float(row.R_anchor):.1f}   R_edge = {float(row.R_edge):.1f}",
             f"in_disk (v1) = {bool(row.in_disk_v1)}", ""]
    for col, lab, _, _, _ in MARK:
        v = float(row[col]) if np.isfinite(row[col]) else np.nan
        lines.append(f"{lab:22s} = {v:7.1f}   (- dip = {v - vdip:+.1f})")
    lines += ["", f"Si II 1260 dip         = {vdip:7.1f}", "",
              f"v3a tracers used: cold n={int(row.n_v3a_cold)} / SF n={int(row.n_v3a_sf)} / young n={int(row.n_v3a_young)}",
              f"v3a components: cold {row.v3a_cold:.0f} / SF {row.v3a_sf:.0f} / young {row.v3a_young:.0f}",
              "", f"VERDICT (v3a vs dip): {verdict}"]
    a.text(0.0, 1.0, "\n".join(lines), transform=a.transAxes, va="top", ha="left",
           fontsize=10.5, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"raydiagv3_sid{sid}_{mode}_alpha{alpha:03d}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
    return dict(sid=sid, mode=mode, alpha=alpha, v3a=v3a, dip=vdip, off=off, verdict=verdict,
                in_disk=bool(row.in_disk_v1))


def main():
    sid = int(sys.argv[1]); step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rc = dict(np.load(RCV2 / f"rc_sid{sid}.npz"))
    m3 = pd.read_csv(M3); m3 = m3[m3.sid == sid].set_index(["mode", "alpha_deg"])
    rows = []
    for mode in ("flip", "noflip"):
        for alpha in range(0, 360, step):
            try:
                r = diagnose(sid, mode, alpha, rc, m3)
                if r:
                    rows.append(r)
            except Exception as ex:
                print(f"  {mode} a{alpha} FAIL: {type(ex).__name__}: {ex}")
    if rows:
        df = pd.DataFrame(rows); df.to_csv(OUT / f"summary_sid{sid}.csv", index=False)
        ind = df[df.in_disk]
        print(f"[SID {sid}] {len(df)} plots; in-disk v3a lands<30 "
              f"{(ind.verdict == 'LANDS').mean() if len(ind) else float('nan'):.2f} (n={len(ind)})")


if __name__ == "__main__":
    main()
