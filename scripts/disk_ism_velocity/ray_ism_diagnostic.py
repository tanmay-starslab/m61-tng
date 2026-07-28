#!/usr/bin/env python3
"""Ray-level ISM diagnostic + DIRECT cool-gas-weighted v_ISM (shared core).

For each (sid, mode, alpha): from the combined all_rays_L2Rvir.h5 ray (ion grid +
per-line spectra) compute the corrected v_rest along the sightline, locate the disk
plane (|z_disk| < Z0 kpc AND R_disk < R_edge), and measure v_ISM as the COOL-GAS
(T<1e4 K) density column-weighted v_rest of that gas. All-gas density, Si II and H I
weightings are kept as cross-checks. R_edge = R95 of the cold-gas (T<1e4) mass
profile, per galaxy. When the sightline crosses beyond the disk (R_cross >= R_edge)
or carries no cool ISM, v_ISM falls back to the R95-edge rotation-curve value.

Velocity: v_rest = -velocity_los/1e5 - v_sys ; positive = recession ; v_sys = SubhaloVel.los.

`compute_vism_fields()` is the single source of truth used by BOTH this diagnostic
plotter and the production stage_b_vism.py -- so they can never diverge.

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
Z0 = 2.0        # kpc, disk-plane half-thickness
COOL_T = 1e4    # K, cool-ISM temperature ceiling


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


def compute_vism_fields(grid, M, R_edge, rc):
    """Single source of truth for the disk-ISM velocity from a ray grid group.

    Returns arrays (for plotting) plus a `fields` dict of scalar outputs. PRIMARY v_ISM
    = cool-gas (T<COOL_T) density column-weighted v_rest of disk-plane cells
    (|z_disk|<Z0 AND R_disk<R_edge); falls back to the R95-edge rotation-curve value
    when the sightline crosses beyond the disk (R_cross>=R_edge) or carries no cool ISM.
    """
    los, v_sys = M["los"], M["v_sys"]
    e1, e2, nd = M["e1"], M["e2"], M["nd"]
    center, anchor = M["center"], M["anchor"]

    xyz = np.vstack([grid["x"][()], grid["y"][()], grid["z"][()]]).T / CM_PER_KPC
    vlos = grid["velocity_los"][()] / 1e5
    dl_cm = grid["dl"][()]
    dl = dl_cm / CM_PER_KPC
    dens = grid["density"][()]
    temp = grid["temperature"][()] if "temperature" in grid else np.full(len(vlos), np.nan)
    n_si = grid[SIII][()] if SIII in grid else np.zeros(len(vlos))
    n_hi = grid[HI][()] if HI in grid else np.zeros(len(vlos))

    v_rest = -vlos - v_sys
    rel = xyz - center
    rel = rel - BOX_KPC * np.round(rel / BOX_KPC)   # periodic wrap (galaxies near box edge)
    x_d = rel @ e1; y_d = rel @ e2; z_d = rel @ nd
    R_disk = np.hypot(x_d, y_d)
    s = (rel - (anchor - center)) @ los

    disk = (np.abs(z_d) < Z0) & (R_disk < R_edge)
    cool = np.isfinite(temp) & (temp < COOL_T)
    disk_cool = disk & cool

    w_dens = dens * dl; w_si = n_si * dl; w_hi = n_hi * dl

    def cw(w, sel):
        ww = w[sel]
        return float(np.average(v_rest[sel], weights=ww)) if (sel.any() and ww.sum() > 0) else np.nan

    v_cool = cw(w_dens, disk_cool)   # PRIMARY: cool-gas density-weighted
    v_dens = cw(w_dens, disk)        # cross-check: ALL disk gas (warm/hot included)
    v_si = cw(w_si, disk)            # Si II column-weighted (cool-selective by ionization)
    v_hi = cw(w_hi, disk)            # H I column-weighted

    N_SiII_disk = float((n_si[disk_cool] * dl_cm[disk_cool]).sum())  # cm^-2
    N_HI_disk = float((n_hi[disk_cool] * dl_cm[disk_cool]).sum())    # cm^-2
    n_cool_cells = int(disk_cool.sum())
    f_disk_si = float(w_si[disk].sum() / w_si.sum()) if w_si.sum() > 0 else np.nan
    R_cross = float(R_disk[np.abs(z_d).argmin()])
    in_disk = bool(R_cross < R_edge)

    Rc = rc["R_center"]; vf = rc["v_fid_median"]; fin = np.isfinite(vf)
    v_edge = (float(np.interp(R_edge, Rc[fin], vf[fin])) * M["proj"]
              if (fin.sum() >= 2 and np.isfinite(M["proj"])) else np.nan)

    has_cool = bool(n_cool_cells >= 1 and N_SiII_disk > 0 and np.isfinite(v_cool))
    v_primary = v_cool if (in_disk and has_cool) else v_edge
    v_mode = "direct_cool" if (in_disk and has_cool) else "R95-edge"

    fields = dict(
        R_edge=float(R_edge), R_cross=R_cross, in_disk=in_disk, proj_phi=M["proj"],
        v_ism_model=M["v_ism_model"], v_ism_direct_cool=v_cool,
        v_ism_direct_density=v_dens, v_ism_SiII=v_si, v_ism_HI=v_hi,
        v_ism_R95edge=v_edge, v_ism_primary=v_primary, v_mode=v_mode,
        f_disk_SiII=f_disk_si, n_cool_disk_cells=n_cool_cells,
        N_SiII_disk=N_SiII_disk, N_HI_disk=N_HI_disk, has_cool=has_cool)
    return dict(v_rest=v_rest, s=s, z_d=z_d, R_disk=R_disk, disk=disk, disk_cool=disk_cool,
                w_si=w_si, w_hi=w_hi, dl=dl, temp=temp, fields=fields)


def diagnose(sid, mode, alpha):
    rc = dict(np.load(RC_DIR / f"rc_sid{sid}.npz"))
    R_edge = r95_cold_gas(sid)
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = projection(sid, mode, alpha, rc, rho)
    v_sys = M["v_sys"]

    with h5py.File(combined_path(sid), "r") as h:
        r = ray_group(h, mode, alpha)
        grid = r["original_trident_ray_h5/grid"]
        res = compute_vism_fields(grid, M, R_edge, rc)
        ion_n = {fld: (grid[fld][()] if fld in grid else np.zeros(len(res["v_rest"])))
                 for _, _, fld, _, _ in IONS}
        specs = {}
        for key, rest, fld, lab, col in IONS:
            g = r.get(f"spectrum_by_line/{key}/lsf")
            if g is not None:
                specs[key] = (g["lambda_A"][()], g["flux"][()], rest, lab, col)

    f = res["fields"]
    v_rest, s, z_d, R_disk = res["v_rest"], res["s"], res["z_d"], res["R_disk"]
    disk, w_si = res["disk"], res["w_si"]
    v_cool = f["v_ism_direct_cool"]; v_model = f["v_ism_model"]; v_edge = f["v_ism_R95edge"]
    v_primary, v_mode = f["v_ism_primary"], f["v_mode"]
    R_cross, in_disk = f["R_cross"], f["in_disk"]

    v_dip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = spectrum_v(lam, rest, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            v_dip = float(vv[w][np.argmin(flux[w])])

    print(f"\n=== SID {sid} {mode} a{alpha} ===")
    print(f"  R_edge(R95)={R_edge:.1f}  R_cross={R_cross:.1f}  in_disk={in_disk}  rho={rho:.1f}  v_sys={v_sys:.1f}")
    print(f"  cool disk cells={f['n_cool_disk_cells']}  logN_SiII_disk={np.log10(f['N_SiII_disk']) if f['N_SiII_disk']>0 else -np.inf:.1f}")
    print(f"  v_model={v_model:.1f} | DIRECT cool={v_cool:.1f} allgas={f['v_ism_direct_density']:.1f} "
          f"SiII={f['v_ism_SiII']:.1f} HI={f['v_ism_HI']:.1f} | R95-edge={v_edge:.1f}  (SiII dip={v_dip:.1f})")
    print(f"  --> PRIMARY v_ISM={v_primary:.1f} [{v_mode}] ; primary-dip={v_primary - v_dip:+.1f} km/s ; f_disk(SiII)={f['f_disk_SiII']:.2f}")

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"SID {sid} · {mode} · α={alpha}° · rho={rho:.1f} · R_cross={R_cross:.1f} · "
                 f"R_edge={R_edge:.1f} · in_disk={in_disk} · v_ISM={v_primary:.0f} [{v_mode}] · v_sys={v_sys:.0f}",
                 fontsize=12)

    a = ax[0, 0]
    for key, rest, fld, lab, col in IONS:
        if key not in specs:
            continue
        lam, flux, rest, lab, col = specs[key]
        vv = spectrum_v(lam, rest, v_sys); m = (vv > -700) & (vv < 700); o = np.argsort(vv[m])
        a.plot(vv[m][o], flux[m][o], color=col, lw=1.3, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_cool):
        a.axvline(v_cool, color="#06a77d", lw=2.6, label=f"v_ISM cool-gas={v_cool:.0f}")
    if np.isfinite(v_model):
        a.axvline(v_model, color="#0077b6", lw=1.5, ls="--", label=f"v_ISM model={v_model:.0f}")
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
    if np.isfinite(v_cool):
        a.axhline(v_cool, color="#06a77d", lw=2.0, label=f"cool-gas={v_cool:.0f}")
    if np.isfinite(v_model):
        a.axhline(v_model, color="#0077b6", lw=1.5, ls="--", label=f"model={v_model:.0f}")
    a.axhline(0, color="0.5", ls=":", lw=0.9)
    a.set_xlabel("s from impact point [kpc]"); a.set_ylabel(r"$v_{\rm rest}$")
    a.set_title("Ray velocity profile (size ∝ Si II column)"); a.legend(fontsize=8, loc="upper right")

    a = ax[1, 0]
    bins = np.linspace(-700, 700, 141)
    for key, rest, fld, lab, col in IONS:
        w = ion_n[fld] * res["dl"]
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(v_rest, bins=bins, weights=w)
        if hist.max() > 0:
            hist = hist / hist.max()
        a.step(0.5 * (bins[:-1] + bins[1:]), hist, where="mid", color=col, lw=1.4, label=lab)
    a.axvline(0, color="0.3", ls=":", lw=1.1)
    if np.isfinite(v_cool):
        a.axvline(v_cool, color="#06a77d", lw=2.6, label=f"cool-gas={v_cool:.0f}")
    if np.isfinite(v_model):
        a.axvline(v_model, color="#0077b6", lw=1.5, ls="--", label=f"model={v_model:.0f}")
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
    return dict(sid=sid, mode=mode, alpha=alpha, rho=rho, v_sys=v_sys, SiII_dip=v_dip, **f)


def main():
    args = sys.argv[1:]
    cases = [(int(args[i]), args[i + 1], int(args[i + 2])) for i in range(0, len(args), 3)]
    rows = [diagnose(sid, mode, alpha) for sid, mode, alpha in cases]
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "ray_diag_summary.csv", index=False)
        print(f"\n[summary] {OUT}/ray_diag_summary.csv")


if __name__ == "__main__":
    main()
