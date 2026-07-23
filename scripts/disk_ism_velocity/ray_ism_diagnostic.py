#!/usr/bin/env python3
"""Ray-level ISM diagnostic — is the disk gas at v_ISM, and is it the main absorber?

For each (sid, mode, alpha) it loads the Trident ray (ion-resolved grid + per-line
spectra) from the combined all_rays_L2Rvir.h5, computes the corrected rest-frame
LOS velocity of every cell, locates the disk-midplane crossing (|z_disk| < Z0),
overlays the disk-model v_ISM, and shows the multi-ion absorption. It prints the
decisive numbers:
  - closest-approach 3D distance to centre  (sanity: should ~= rho)
  - the Si II column-weighted v_rest of the DISK-plane gas vs v_ISM
  - the fraction of the total Si II column that comes from the disk plane
  - the Si II column-weighted v_rest of the WHOLE ray vs the Si II spectrum dip
    (sign self-consistency check between ray grid and spectrum)

Velocity convention (corrected, matches the pipeline):
  v_rest = -velocity_los/1e5 - v_sys ;  positive = recession ; v_sys = SubhaloVel.los.

Usage: python ray_ism_diagnostic.py <sid> <mode> <alpha> [<sid> <mode> <alpha> ...]
Requires rc_sid<sid>.npz from build_sid_rc.py.
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
import h5py  # noqa: E402
from pm_general import (C_KMS, CM_PER_KPC, get_geometry, get_original_rho,  # noqa: E402
                        compute_endpoints)

RC_DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/ray_diagnostics")
Z0 = 3.0   # kpc, disk-plane half-thickness for "disk" selection

def combined_path(sid):
    return Path(f"/scratch/tsingh65/m61-tng/outputs/sid{sid}"
                f"/rays_and_spectra_sid{sid}_snap99_L2Rvir/combined/all_rays_L2Rvir.h5")

# (per-line-spectrum key, rest wave A, grid ion-density field, label, colour)
IONS = [
    ("H_I_1216",   1215.6701, "H_p0_number_density",  "H I 1216",   "#444444"),
    ("Si_II_1260", 1260.4221, "Si_p1_number_density", "Si II 1260", "#1d3557"),
    ("C_II_1335",  1334.5323, "C_p1_number_density",  "C II 1335",  "#2a9d8f"),
    ("Si_IV_1403", 1402.770,  "Si_p3_number_density", "Si IV 1403", "#e63946"),
    ("N_V_1239",   1238.821,  "N_p4_number_density",  "N V 1239",   "#9d4edd"),
]
SIII = "Si_p1_number_density"   # Si II, the fitted line


def ray_group(h5, mode, alpha):
    ag = h5[f"rays/sightline=J122138+043026/mode={mode}/alpha={alpha}"]
    return ag[list(ag.keys())[0]]


def disk_frame_projection(sid, mode, alpha, rc, rho):
    geom = get_geometry(sid, mode, alpha)
    los = geom["los"]; v_sys = geom["v_sys"]
    anchor = compute_endpoints(sid, mode, alpha, rho, 50.)["anchor_kpc"]
    e1, e2, nd = rc["e1"], rc["e2"], rc["n_disk"]
    center = rc["center_kpc"]
    rel = anchor - center
    x_d = float(rel @ e1); y_d = float(rel @ e2); R = math.hypot(x_d, y_d)
    los_d = np.array([los @ e1, los @ e2, los @ nd])
    proj = float(np.array([-y_d / R, x_d / R, 0.]) @ los_d) if R > 0.2 else np.nan
    Rc = rc["R_center"]; v = rc["v_fid_median"]; fin = np.isfinite(v)
    vfid = float(np.interp(R, Rc[fin], v[fin])) if fin.sum() >= 2 else np.nan
    v_ism = vfid * proj if (np.isfinite(vfid) and np.isfinite(proj)) else np.nan
    return dict(los=los, v_sys=v_sys, anchor=anchor, center=center,
                e1=e1, e2=e2, nd=nd, R_anchor=R, proj=proj, v_ism=v_ism)


def spectrum_v(lam, rest, v_sys):
    return -(C_KMS * (lam / rest - 1.0)) - v_sys


def diagnose(sid, mode, alpha):
    rc = dict(np.load(RC_DIR / f"rc_sid{sid}.npz"))
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = disk_frame_projection(sid, mode, alpha, rc, rho)
    los, v_sys, anchor, center = M["los"], M["v_sys"], M["anchor"], M["center"]
    e1, e2, nd = M["e1"], M["e2"], M["nd"]
    v_ism = M["v_ism"]

    with h5py.File(combined_path(sid), "r") as h:
        r = ray_group(h, mode, alpha)
        grid = r["original_trident_ray_h5/grid"]
        xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC  # kpc
        vlos = grid["velocity_los"][()] / 1e5                                          # km/s
        dl = grid["dl"][()] / CM_PER_KPC                                               # kpc
        T = grid["temperature"][()]
        ion_n = {fld: (grid[fld][()] if fld in grid else np.zeros(len(vlos)))
                 for _, _, fld, _, _ in IONS}
        specs = {}
        for key, rest, fld, lab, col in IONS:
            g = r.get(f"spectrum_by_line/{key}/lsf")
            if g is not None:
                specs[key] = (g["lambda_A"][()], g["flux"][()], rest, lab, col)

    v_rest = -vlos - v_sys
    rel = xyz - center
    x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ nd
    R_disk = np.hypot(x_d, y_d)
    r3d = np.sqrt(np.sum(rel ** 2, axis=1))
    s = (xyz - anchor) @ los                     # LOS distance from the impact point

    w_si = ion_n[SIII] * dl                       # Si II column per cell
    disk = (np.abs(z_d) < Z0)
    tot_si = w_si.sum()
    f_disk = w_si[disk].sum() / tot_si if tot_si > 0 else np.nan
    v_si_all = np.average(v_rest, weights=w_si) if tot_si > 0 else np.nan
    v_si_disk = (np.average(v_rest[disk], weights=w_si[disk])
                 if (disk.any() and w_si[disk].sum() > 0) else np.nan)

    # Si II spectrum dip (deepest point) in corrected velocity
    v_dip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = spectrum_v(lam, rest, v_sys)
        w = (vv > -700) & (vv < 700)
        if w.any():
            v_dip = float(vv[w][np.argmin(flux[w])])

    print(f"\n=== SID {sid} {mode} a{alpha} ===")
    print(f"  closest-approach 3D dist to centre = {r3d.min():.1f} kpc  (native rho={rho:.1f})")
    print(f"  ray length s in [{s.min():.0f},{s.max():.0f}] kpc ; n_cells={len(s)}")
    print(f"  v_sys={v_sys:.1f}  proj_phi={M['proj']:.3f}  R_anchor={M['R_anchor']:.1f}  v_ISM={v_ism:.1f}")
    print(f"  Si II: f_disk(|z|<{Z0:g})={f_disk:.2f}  v_SiII(disk)={v_si_disk:.1f}  "
          f"v_SiII(all)={v_si_all:.1f}  spectrum dip={v_dip:.1f}")
    print(f"  --> disk Si II gas vs v_ISM: {v_si_disk - v_ism:+.1f} km/s ; "
          f"sign check (v_SiII_all vs dip): {v_si_all - v_dip:+.1f} km/s")

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    title = (f"SID {sid} · {mode} · α={alpha}°  ·  rho={rho:.1f} kpc · "
             f"R_anchor={M['R_anchor']:.1f} · proj={M['proj']:.2f} · v_sys={v_sys:.0f}")
    fig.suptitle(title, fontsize=12)

    # (0,0) multi-ion spectra vs v_rest
    a = ax[0, 0]
    for key, rest, fld, lab, col in IONS:
        if key not in specs:
            continue
        lam, flux, rest, lab, col = specs[key]
        vv = spectrum_v(lam, rest, v_sys)
        m = (vv > -700) & (vv < 700); o = np.argsort(vv[m])
        a.plot(vv[m][o], flux[m][o], color=col, lw=1.3, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_ism):
        a.axvline(v_ism, color="#0077b6", lw=2.4, label=f"v_ISM model={v_ism:.0f}")
    if np.isfinite(v_si_disk):
        a.axvline(v_si_disk, color="#06a77d", lw=2.4, ls="--", label=f"v_ISM direct={v_si_disk:.0f}")
    a.set_xlim(-700, 700); a.set_ylim(-0.05, 1.15)
    a.set_xlabel(r"$v_{\rm rest}$ [km s$^{-1}$] (positive=recession)"); a.set_ylabel("flux")
    a.set_title("Multi-ion absorption"); a.legend(fontsize=8, ncol=2, loc="lower left")

    # (0,1) ray v_rest vs s, coloured by log Si II density, size by Si II column
    a = ax[0, 1]
    nsi = ion_n[SIII]
    c = np.log10(np.clip(nsi, 1e-14, None))
    sz = 6 + 300 * (w_si / (w_si.max() + 1e-30))
    sc = a.scatter(s, v_rest, c=c, s=sz, cmap="viridis", alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=a, label=r"$\log_{10} n_{\rm Si II}$ [cm$^{-3}$]")
    ds = disk & (s > s.min())
    if ds.any():
        a.axvspan(s[ds].min(), s[ds].max(), color="#ffd166", alpha=0.25,
                  label=f"disk |z|<{Z0:g} kpc")
    if np.isfinite(v_ism):
        a.axhline(v_ism, color="#0077b6", lw=2.0, label=f"v_ISM={v_ism:.0f}")
    a.axhline(0, color="0.5", ls=":", lw=0.9)
    a.set_xlabel(r"LOS distance $s$ from impact point [kpc]")
    a.set_ylabel(r"$v_{\rm rest}$ [km s$^{-1}$]")
    a.set_title("Ray velocity profile (marker size ∝ Si II column)")
    a.legend(fontsize=8, loc="upper right")

    # (1,0) ion column-weighted velocity distribution
    a = ax[1, 0]
    bins = np.linspace(-700, 700, 141)
    for key, rest, fld, lab, col in IONS:
        w = ion_n[fld] * dl
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(v_rest, bins=bins, weights=w)
        hist = hist / hist.max() if hist.max() > 0 else hist
        a.step(0.5 * (bins[:-1] + bins[1:]), hist, where="mid", color=col, lw=1.4, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_ism):
        a.axvline(v_ism, color="#0077b6", lw=2.4, label=f"v_ISM model={v_ism:.0f}")
    if np.isfinite(v_si_disk):
        a.axvline(v_si_disk, color="#06a77d", lw=2.4, ls="--", label=f"v_ISM direct={v_si_disk:.0f}")
    a.set_xlim(-700, 700)
    a.set_xlabel(r"$v_{\rm rest}$ [km s$^{-1}$]"); a.set_ylabel("column-wt (norm.)")
    a.set_title("Ion column-weighted velocity"); a.legend(fontsize=8, ncol=2)

    # (1,1) disk geometry along the ray
    a = ax[1, 1]
    a.plot(s, z_d, color="#e76f51", lw=1.5, label=r"$z_{\rm disk}$")
    a.plot(s, R_disk, color="#264653", lw=1.5, ls="--", label=r"$R_{\rm disk}$")
    a.axhspan(-Z0, Z0, color="#ffd166", alpha=0.25)
    a.axhline(0, color="0.5", ls=":", lw=0.9)
    a.set_xlabel(r"LOS distance $s$ [kpc]"); a.set_ylabel("disk-frame [kpc]")
    a.set_title("Sightline geometry (yellow = disk plane)")
    a.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"raydiag_sid{sid}_{mode}_alpha{alpha:03d}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")
    return dict(sid=sid, mode=mode, alpha=alpha, v_ism=v_ism, v_si_disk=v_si_disk,
                v_si_all=v_si_all, f_disk=f_disk, v_dip=v_dip, r3d_min=float(r3d.min()))


def main():
    args = sys.argv[1:]
    cases = [(int(args[i]), args[i + 1], int(args[i + 2])) for i in range(0, len(args), 3)]
    for sid, mode, alpha in cases:
        diagnose(sid, mode, alpha)


if __name__ == "__main__":
    main()
