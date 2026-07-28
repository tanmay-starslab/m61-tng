#!/usr/bin/env python3
"""Per-sightline ray diagnostics for the corrected v3 velocities. Six panels so each
per-orientation value can be audited against the absorption:
  (A) multi-ion spectra with v3a, v3b, v2, v1-direct and the Si II dip marked;
  (B) ray velocity profile v_rest vs path s, disk crossing shaded;
  (C) the v3b SLIT PROFILE v(s) along the impact-parameter axis, with s=rho (the read point)
      and the disk edge marked -- shows how v3b is read at the impact parameter, not averaged;
  (D) ion column-weighted velocity;
  (E) sightline geometry;
  (F) audit text with every number, tracer counts, and a lands/near/off verdict vs the dip.

Usage: python ray_diagnostic_v3.py <sid> [alpha_step]   (default step=20)
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
VT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v3/ray_diagnostics_v3")
MARK = [("v_ism_v3a", "v3a along-sightline", "#1B9E77", "-", 3.0),
        ("v_ism_v3b", "v3b slit@impact", "#B02418", "--", 1.8),
        ("v_ism_v2", "v2 rot-curve", "#7B6FB0", "-.", 1.3),
        ("v_ism_direct_cool", "v1 direct", "#E19A3C", ":", 1.5)]


def diagnose(sid, mode, alpha, rc, m3, cent, prof):
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
            gg = r.get(f"spectrum_by_line/{key}/lsf")
            if gg is not None:
                specs[key] = (gg["lambda_A"][()], gg["flux"][()], rest, lab, col)
    v_rest, s, z_d, R_disk, disk = res["v_rest"], res["s"], res["z_d"], res["R_disk"], res["disk"]
    vdip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = R.spectrum_v(lam, rest, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            vdip = float(vv[w][np.argmin(flux[w])])
    v3a = float(row.v_ism_v3a); off = v3a - vdip
    verdict = "LANDS" if abs(off) < 30 else ("NEAR" if abs(off) < 60 else "OFF")

    fig, ax = plt.subplots(2, 3, figsize=(21, 11))
    fig.suptitle(f"SID {sid} · {mode} · α={alpha}° · ρ={rho:.1f} · in_disk={bool(row.in_disk_v1)} · "
                 f"v3a={v3a:.0f} v3b={float(row.v_ism_v3b):.0f} (dip {vdip:.0f}, off {off:+.0f} "
                 f"[{verdict}]) · v_sys={v_sys:.0f}", fontsize=12)

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
    a.set_title("(B) ray velocity profile"); a.legend(fontsize=7, loc="upper right")

    # (C) slit profile v(s) with the read point at rho
    a = ax[0, 2]
    if prof is not None and np.isfinite(prof).any():
        a.plot(cent, prof, "-o", color="#B02418", ms=3, lw=1.4, label="v3b slit profile $v(s)$")
        sp = cent[np.isfinite(prof)]
        a.axvline(rho, color="0.3", ls="--", lw=1.4, label=rf"impact $s=\rho={rho:.0f}$")
        a.plot([rho], [float(row.v_ism_v3b)], "*", color="k", ms=15, zorder=6,
               label=f"v3b read = {float(row.v_ism_v3b):.0f}")
        if bool(row.v3b_edge_fallback) and len(sp):
            a.axvline(sp.max(), color="#7B6FB0", ls=":", lw=1.4, label=f"disk edge R={sp.max():.0f}")
    a.axhline(0, color="0.5", ls=":", lw=0.8); a.axvline(0, color="0.5", ls=":", lw=0.8)
    a.set_xlabel("s along impact axis [kpc]"); a.set_ylabel(r"$v_{\rm rest}$ [km/s]")
    a.set_title("(C) v3b binned slit profile — read at ρ"); a.legend(fontsize=7.5, loc="best")

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
    a.set_title("(D) ion column-weighted velocity"); a.legend(fontsize=6.5, ncol=2)

    a = ax[1, 1]
    a.plot(s, z_d, color="#e76f51", lw=1.3, label=r"$z_{\rm disk}$")
    a.plot(s, R_disk, color="#264653", lw=1.3, ls="--", label=r"$R_{\rm disk}$")
    a.axhspan(-R.Z0, R.Z0, color="#ffd166", alpha=0.3); a.axhline(0, color="0.5", ls=":", lw=0.8)
    a.set_xlabel("s [kpc]"); a.set_ylabel("disk-frame [kpc]")
    a.set_title("(E) sightline geometry"); a.legend(fontsize=8)

    a = ax[1, 2]; a.axis("off")
    lines = ["AUDIT — v3 velocities (per orientation)", "",
             f"rho = {rho:.1f} kpc   disk_edge_R = {float(row.disk_edge_R):.1f}   in_disk(v1) = {bool(row.in_disk_v1)}",
             f"v3a edge-fallback = {bool(row.v3a_edge_fallback)}   v3b edge-fallback = {bool(row.v3b_edge_fallback)}", ""]
    for col, lab, _, _, _ in MARK:
        v = float(row[col]) if np.isfinite(row[col]) else np.nan
        lines.append(f"{lab:22s} = {v:7.1f}   (- dip = {v - vdip:+.1f})")
    lines += ["", f"Si II 1260 dip         = {vdip:7.1f}", "",
              f"v3a tube tracers: cold {row.v3a_cold:.0f} / SF {row.v3a_sf:.0f} / young {row.v3a_young:.0f}",
              f"slit bins populated = {int(row.n_slit_bins)}", "",
              f"VERDICT (v3a vs dip): {verdict}"]
    a.text(0.0, 1.0, "\n".join(lines), transform=a.transAxes, va="top", ha="left",
           fontsize=10.5, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"raydiagv3_sid{sid}_{mode}_alpha{alpha:03d}.png"
    fig.savefig(p, dpi=112, bbox_inches="tight"); plt.close(fig)
    return dict(sid=sid, mode=mode, alpha=alpha, v3a=v3a, v3b=float(row.v_ism_v3b),
                dip=vdip, off=off, verdict=verdict, in_disk=bool(row.in_disk_v1))


def main():
    sid = int(sys.argv[1]); step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rc = dict(np.load(RCV2 / f"rc_sid{sid}.npz"))
    m3 = pd.read_csv(VT / "vism_master_v3.csv"); m3 = m3[m3.sid == sid].set_index(["mode", "alpha_deg"])
    sp = np.load(VT / f"slitprof_sid{sid}.npz", allow_pickle=True)
    cent = sp["centers"]; pmodes = sp["modes"]; palphas = sp["alphas"]; profs = sp["profiles"]
    pidx = {(str(pmodes[i]), int(palphas[i])): i for i in range(len(palphas))}
    rows = []
    for mode in ("flip", "noflip"):
        for alpha in range(0, 360, step):
            prof = profs[pidx[(mode, alpha)]] if (mode, alpha) in pidx else None
            try:
                res = diagnose(sid, mode, alpha, rc, m3, cent, prof)
                if res:
                    rows.append(res)
            except Exception as ex:
                print(f"  {mode} a{alpha} FAIL: {type(ex).__name__}: {ex}")
    if rows:
        df = pd.DataFrame(rows); df.to_csv(OUT / f"summary_sid{sid}.csv", index=False)
        ind = df[df.in_disk]
        print(f"[SID {sid}] {len(df)} plots; in-disk v3a lands<30 "
              f"{(ind.verdict == 'LANDS').mean() if len(ind) else float('nan'):.2f} (n={len(ind)})")


if __name__ == "__main__":
    main()
