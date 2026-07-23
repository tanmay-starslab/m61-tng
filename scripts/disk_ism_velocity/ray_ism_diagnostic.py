#!/usr/bin/env python3
"""Ray-level ISM diagnostic + DIRECT density-weighted v_ISM.

For each (sid, mode, alpha): from the combined all_rays_L2Rvir.h5 ray (ion grid +
per-line spectra) compute the corrected v_rest along the sightline, locate the disk
plane (|z_disk| < Z0 kpc AND R_disk < R_edge), and measure v_ISM as the GAS-DENSITY
(mass) column-weighted v_rest of that gas. Si II and H I weightings are reported as
cross-checks. R_edge = R95 of the cold-gas (T<1e4) mass profile, per galaxy.

Velocity: v_rest = -velocity_los/1e5 - v_sys ; positive = recession ; v_sys = SubhaloVel.los.

Usage: python ray_ism_diagnostic.py <sid> <mode> <alpha> [<sid> <mode> <alpha> ...]
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
import h5py  # noqa: E402
from pm_general import (C_KMS, CM_PER_KPC, TNG_H, get_geometry, get_original_rho,  # noqa: E402
                        compute_endpoints)

BOX_KPC = 35000.0 / TNG_H   # TNG50 box (35 Mpc/h) in physical kpc -> periodic wrapping

RC_DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/ray_diagnostics")
Z0 = 2.0   # kpc, disk-plane half-thickness

def combined_path(sid):
    return Path(f"/scratch/tsingh65/m61-tng/outputs/sid{sid}"
                f"/rays_and_spectra_sid{sid}_snap99_L2Rvir/combined/all_rays_L2Rvir.h5")

IONS = [
    ("H_I_1216",   1215.6701, "H_p0_number_density",  "H I 1216",   "#444444"),
    ("Si_II_1260", 1260.4221, "Si_p1_number_density", "Si II 1260", "#1d3557"),
    ("C_II_1335",  1334.5323, "C_p1_number_density",  "C II 1335",  "#2a9d8f"),
    ("Si_IV_1403", 1402.770,  "Si_p3_number_density", "Si IV 1403", "#e63946"),
    ("N_V_1239",   1238.821,  "N_p4_number_density",  "N V 1239",   "#9d4edd"),
]
SIII = "Si_p1_number_density"
HI = "H_p0_number_density"


def r95_cold_gas(sid):
    """R95 of the cold-gas (T<1e4) disk-plane mass profile -> disk radial edge."""
    f = RC_DIR / f"rc_sid{sid}_cold_gas_1e4.csv"
    if not f.exists():
        return np.inf
    df = pd.read_csv(f)
    m = np.nan_to_num(df["mass"].values); cum = np.cumsum(m)
    if cum[-1] <= 0:
        return np.inf
    return float(np.interp(0.95, cum / cum[-1], df["R_center"].values))


def ray_group(h5, mode, alpha):
    ag = h5[f"rays/sightline=J122138+043026/mode={mode}/alpha={alpha}"]
    return ag[list(ag.keys())[0]]


def projection(sid, mode, alpha, rc, rho):
    geom = get_geometry(sid, mode, alpha)
    los = geom["los"]; v_sys = geom["v_sys"]
    anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
    e1, e2, nd = rc["e1"], rc["e2"], rc["n_disk"]; center = rc["center_kpc"]
    rel = anchor - center
    x_d = float(rel @ e1); y_d = float(rel @ e2); R = math.hypot(x_d, y_d)
    los_d = np.array([los @ e1, los @ e2, los @ nd])
    proj = float(np.array([-y_d / R, x_d / R, 0.]) @ los_d) if R > 0.2 else np.nan
    Rc = rc["R_center"]; v = rc["v_fid_median"]; fin = np.isfinite(v)
    vfid = float(np.interp(R, Rc[fin], v[fin])) if fin.sum() >= 2 else np.nan
    v_ism = vfid * proj if (np.isfinite(vfid) and np.isfinite(proj)) else np.nan
    return dict(los=los, v_sys=v_sys, anchor=anchor, center=center,
                e1=e1, e2=e2, nd=nd, R_anchor=R, proj=proj, v_ism_model=v_ism)


def spectrum_v(lam, rest, v_sys):
    return -(C_KMS * (lam / rest - 1.0)) - v_sys


def diagnose(sid, mode, alpha):
    rc = dict(np.load(RC_DIR / f"rc_sid{sid}.npz"))
    R_edge = r95_cold_gas(sid)
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = projection(sid, mode, alpha, rc, rho)
    los, v_sys, anchor, center = M["los"], M["v_sys"], M["anchor"], M["center"]
    e1, e2, nd = M["e1"], M["e2"], M["nd"]
    v_ism_model = M["v_ism_model"]

    with h5py.File(combined_path(sid), "r") as h:
        r = ray_group(h, mode, alpha)
        grid = r["original_trident_ray_h5/grid"]
        xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC
        vlos = grid["velocity_los"][()] / 1e5
        dl = grid["dl"][()] / CM_PER_KPC
        dens = grid["density"][()]
        ion_n = {fld: (grid[fld][()] if fld in grid else np.zeros(len(vlos)))
                 for _, _, fld, _, _ in IONS}
        specs = {}
        for key, rest, fld, lab, col in IONS:
            g = r.get(f"spectrum_by_line/{key}/lsf")
            if g is not None:
                specs[key] = (g["lambda_A"][()], g["flux"][()], rest, lab, col)

    v_rest = -vlos - v_sys
    rel = xyz - center
    rel = rel - BOX_KPC * np.round(rel / BOX_KPC)   # periodic wrap (galaxies near box edge)
    x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ nd
    R_disk = np.hypot(x_d, y_d)
    s = (rel - (anchor - center)) @ los

    disk = (np.abs(z_d) < Z0) & (R_disk < R_edge)

    def cw(w):
        ww = w[disk]
        return float(np.average(v_rest[disk], weights=ww)) if (disk.any() and ww.sum() > 0) else np.nan

    w_dens = dens * dl; w_si = ion_n[SIII] * dl; w_hi = ion_n[HI] * dl
    v_dens = cw(w_dens)     # PRIMARY: gas-density (mass) column-weighted
    v_si = cw(w_si); v_hi = cw(w_hi)
    f_disk_si = (w_si[disk].sum() / w_si.sum()) if w_si.sum() > 0 else np.nan
    R_cross = float(R_disk[np.abs(z_d).argmin()])
    in_disk = bool(R_cross < R_edge)

    # R95-boundary velocity: fiducial rotation curve clamped to the disk edge (R95),
    # projected onto the LOS. Used as the ISM reference when the sightline crosses
    # BEYOND the disk (R_cross >= R_edge), where there is no disk gas to weight.
    Rc_a = rc["R_center"]; vf_a = rc["v_fid_median"]; fin2 = np.isfinite(vf_a)
    v_edge = (float(np.interp(R_edge, Rc_a[fin2], vf_a[fin2])) * M["proj"]
              if (fin2.sum() >= 2 and np.isfinite(M["proj"])) else np.nan)
    has_cool = bool(np.isfinite(v_si) and f_disk_si > 0.05)
    v_primary = v_dens if (in_disk and has_cool) else v_edge
    v_mode = "direct" if (in_disk and has_cool) else "R95-edge"

    v_dip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = spectrum_v(lam, rest, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            v_dip = float(vv[w][np.argmin(flux[w])])

    print(f"\n=== SID {sid} {mode} a{alpha} ===")
    print(f"  R_edge(R95)={R_edge:.1f}  R_cross={R_cross:.1f}  in_disk={in_disk}  rho={rho:.1f}  v_sys={v_sys:.1f}")
    print(f"  v_ISM_model={v_ism_model:.1f} | DIRECT density={v_dens:.1f} SiII={v_si:.1f} HI={v_hi:.1f} | R95-edge={v_edge:.1f}  (SiII dip={v_dip:.1f})")
    print(f"  --> PRIMARY v_ISM={v_primary:.1f} [{v_mode}] ; primary-dip={v_primary - v_dip:+.1f} km/s ; f_disk(SiII)={f_disk_si:.2f}")

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"SID {sid} · {mode} · α={alpha}° · rho={rho:.1f} · R_cross={R_cross:.1f} · "
                 f"R_edge={R_edge:.1f} · in_disk={in_disk} · v_sys={v_sys:.0f}", fontsize=12)

    a = ax[0, 0]
    for key, rest, fld, lab, col in IONS:
        if key not in specs:
            continue
        lam, flux, rest, lab, col = specs[key]
        vv = spectrum_v(lam, rest, v_sys); m = (vv > -700) & (vv < 700); o = np.argsort(vv[m])
        a.plot(vv[m][o], flux[m][o], color=col, lw=1.3, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_dens):
        a.axvline(v_dens, color="#06a77d", lw=2.6, label=f"v_ISM direct(ρ)={v_dens:.0f}")
    if np.isfinite(v_ism_model):
        a.axvline(v_ism_model, color="#0077b6", lw=1.5, ls="--", label=f"v_ISM model={v_ism_model:.0f}")
    if np.isfinite(v_edge):
        a.axvline(v_edge, color="#ff8c00", lw=2.2, ls="-.", label=f"v_ISM R95-edge={v_edge:.0f}")
    a.set_xlim(-700, 700); a.set_ylim(-0.05, 1.15)
    a.set_xlabel(r"$v_{\rm rest}$ [km s$^{-1}$] (+=recession)"); a.set_ylabel("flux")
    a.set_title("Multi-ion absorption"); a.legend(fontsize=8, ncol=2, loc="lower left")

    a = ax[0, 1]
    c = np.log10(np.clip(ion_n[SIII], 1e-14, None))
    sz = 6 + 300 * (w_si / (w_si.max() + 1e-30))
    sc = a.scatter(s, v_rest, c=c, s=sz, cmap="viridis", alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=a, label=r"$\log_{10} n_{\rm SiII}$")
    ds = disk & (s > s.min())
    if ds.any():
        a.axvspan(s[ds].min(), s[ds].max(), color="#ffd166", alpha=0.3,
                  label=f"disk |z|<{Z0:g}, R<{R_edge:.0f}")
    if np.isfinite(v_dens):
        a.axhline(v_dens, color="#06a77d", lw=2.0, label=f"direct={v_dens:.0f}")
    if np.isfinite(v_ism_model):
        a.axhline(v_ism_model, color="#0077b6", lw=1.5, ls="--", label=f"model={v_ism_model:.0f}")
    a.axhline(0, color="0.5", ls=":", lw=0.9)
    a.set_xlabel("s from impact point [kpc]"); a.set_ylabel(r"$v_{\rm rest}$")
    a.set_title("Ray velocity profile (size ∝ Si II column)"); a.legend(fontsize=8, loc="upper right")

    a = ax[1, 0]
    bins = np.linspace(-700, 700, 141)
    for key, rest, fld, lab, col in IONS:
        w = ion_n[fld] * dl
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(v_rest, bins=bins, weights=w)
        if hist.max() > 0:
            hist = hist / hist.max()
        a.step(0.5 * (bins[:-1] + bins[1:]), hist, where="mid", color=col, lw=1.4, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_dens):
        a.axvline(v_dens, color="#06a77d", lw=2.6, label=f"direct={v_dens:.0f}")
    if np.isfinite(v_ism_model):
        a.axvline(v_ism_model, color="#0077b6", lw=1.5, ls="--", label=f"model={v_ism_model:.0f}")
    if np.isfinite(v_edge):
        a.axvline(v_edge, color="#ff8c00", lw=2.2, ls="-.", label=f"R95-edge={v_edge:.0f}")
    a.set_xlim(-700, 700); a.set_xlabel(r"$v_{\rm rest}$"); a.set_ylabel("column-wt (norm.)")
    a.set_title("Ion column-weighted velocity"); a.legend(fontsize=8, ncol=2)

    a = ax[1, 1]
    a.plot(s, z_d, color="#e76f51", lw=1.5, label=r"$z_{\rm disk}$")
    a.plot(s, R_disk, color="#264653", lw=1.5, ls="--", label=r"$R_{\rm disk}$")
    a.axhspan(-Z0, Z0, color="#ffd166", alpha=0.3)
    a.axhline(R_edge, color="0.5", ls=":", label=f"R_edge={R_edge:.0f}")
    a.axhline(0, color="0.5", ls=":", lw=0.9)
    a.set_xlabel("s [kpc]"); a.set_ylabel("disk-frame [kpc]")
    a.set_title("Sightline geometry"); a.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"raydiag_sid{sid}_{mode}_alpha{alpha:03d}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")
    return dict(sid=sid, mode=mode, alpha=alpha, rho=rho, R_edge=R_edge, R_cross=R_cross,
                in_disk=in_disk, v_sys=v_sys, v_ism_model=v_ism_model, v_dens=v_dens,
                v_si=v_si, v_hi=v_hi, v_edge=v_edge, v_primary=v_primary, v_mode=v_mode,
                v_dip=v_dip, f_disk_si=f_disk_si)


def main():
    args = sys.argv[1:]
    cases = [(int(args[i]), args[i + 1], int(args[i + 2])) for i in range(0, len(args), 3)]
    rows = [diagnose(sid, mode, alpha) for sid, mode, alpha in cases]
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "ray_diag_summary.csv", index=False)
        print(f"\n[summary] {OUT}/ray_diag_summary.csv")


if __name__ == "__main__":
    main()
