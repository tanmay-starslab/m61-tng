#!/usr/bin/env python3
"""Per-sightline ray diagnostics for the v2 (supervisor-model) v_ISM.

Six panels per (sid, mode, alpha), so the model v_ISM can be audited visually:
  (A) multi-ion absorption spectra with v_ISM(model), its per-tracer components
      (cold/SF/young × proj), the Si II dip, and the v1 direct velocity marked;
  (B) DERIVATION: the 3-tracer rotation curve v_φ(R) (cold gas T<1e4, SF gas, young stars) and
      their average, with R_anchor and v_rot(R_anchor) marked and the arithmetic
      v_ISM = v_rot(R_anchor) × proj spelled out;
  (C) ray velocity profile v_rest vs path length s, coloured by n_SiII, disk region shaded;
  (D) ion column-weighted velocity distribution along the ray;
  (E) sightline geometry (z_disk, R_disk vs s);
  (F) an audit panel with every number and a pass/near/fail verdict on landing on the Si II dip.

Usage: python ray_diagnostic_v2.py <sid> [alpha_step]     (default step=20 -> 36 sightlines/SID)
Output: diagnostics_v2/ray_diagnostics_v2/raydiagv2_sid<sid>_<mode>_alpha<aaa>.png
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
from pm_general import get_geometry, get_original_rho  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402  (helpers: projection, ray_group, spectrum_v, IONS...)

RCV2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves_v2")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2/ray_diagnostics_v2")
TCOL = {"cold_gas": "#1d3557", "sf_gas": "#2a9d8f", "young_stars": "#e76f51"}


def interp(rc, key, Rq):
    Rc = rc["R_center"]; v = rc[key]; fin = np.isfinite(v)
    return float(np.interp(Rq, Rc[fin], v[fin])) if fin.sum() >= 2 else np.nan


def diagnose(sid, mode, alpha, rc):
    rho, _, _ = get_original_rho(sid, mode, alpha)
    M = R.projection(sid, mode, alpha, rc, rho)
    v_sys = M["v_sys"]; proj = M["proj"]; Rd = M["R_anchor"]; v_model = M["v_ism_model"]
    R_edge = R.r95_cold_gas(sid)
    # per-tracer projected components
    comp = {t: interp(rc, f"v_{t}", Rd) * proj for t in ("cold_gas", "sf_gas", "young_stars")}
    v_rot = interp(rc, "v_fid_median", Rd)

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

    f = res["fields"]
    v_rest, s, z_d, R_disk = res["v_rest"], res["s"], res["z_d"], res["R_disk"]
    disk, w_si = res["disk"], res["w_si"]
    v_direct = f["v_ism_direct_cool"]

    v_dip = np.nan
    if "Si_II_1260" in specs:
        lam, flux, rest, _, _ = specs["Si_II_1260"]
        vv = R.spectrum_v(lam, rest, v_sys); w = (vv > -700) & (vv < 700)
        if w.any():
            v_dip = float(vv[w][np.argmin(flux[w])])
    off = v_model - v_dip
    verdict = ("LANDS on Si II dip" if abs(off) < 30 else
               "NEAR dip" if abs(off) < 60 else "OFF dip")

    fig, ax = plt.subplots(2, 3, figsize=(21, 11))
    fig.suptitle(f"SID {sid} · {mode} · α={alpha}° · ρ={rho:.1f} · R_anchor={Rd:.1f} · "
                 f"proj={proj:+.2f} · v_ISM(model)={v_model:.0f} · v_sys={v_sys:.0f} · [{verdict}]",
                 fontsize=13)

    # (A) spectra + markers
    a = ax[0, 0]
    for key, rest, fld, lab, col in R.IONS:
        if key in specs:
            lam, flux, rest, lab, col = specs[key]
            vv = R.spectrum_v(lam, rest, v_sys); m = (vv > -700) & (vv < 700); o = np.argsort(vv[m])
            a.plot(vv[m][o], flux[m][o], color=col, lw=1.3, label=lab)
    a.axvline(0, color="0.4", ls=":", lw=1.0)
    a.axvline(v_model, color="#B02418", lw=3.0, label=f"v_ISM model={v_model:.0f}")
    for t, c in TCOL.items():
        if np.isfinite(comp[t]):
            a.axvline(comp[t], color=c, lw=1.2, ls="--", alpha=0.8,
                      label=f"{t.split('_')[0]}×proj={comp[t]:.0f}")
    if np.isfinite(v_direct):
        a.axvline(v_direct, color="#1B9E77", lw=1.6, ls="-.", label=f"v1 direct={v_direct:.0f}")
    if np.isfinite(v_dip):
        a.axvline(v_dip, color="#E19A3C", lw=1.6, ls=":", label=f"Si II dip={v_dip:.0f}")
    a.set_xlim(-700, 700); a.set_ylim(-0.05, 1.15)
    a.set_xlabel(r"$v_{\rm rest}$ [km/s] (+=recession)"); a.set_ylabel("flux")
    a.set_title("(A) multi-ion absorption + v_ISM"); a.legend(fontsize=7, ncol=2, loc="lower left")

    # (B) derivation: rotation curve
    b = ax[0, 1]
    Rc = rc["R_center"]
    for t, c in TCOL.items():
        v = rc[f"v_{t}"]; n = rc[f"n_{t}"]; ok = n >= 20
        b.plot(Rc[ok], v[ok], color=c, lw=1.4, label=t.replace("_", " "))
    b.plot(Rc, rc["v_fid_median"], color="k", lw=2.6, label="3-tracer avg")
    b.fill_between(Rc, rc["v_fid_p16"], rc["v_fid_p84"], color="0.5", alpha=0.15, lw=0)
    b.axvline(Rd, color="#B02418", lw=1.8, ls="--")
    if np.isfinite(v_rot):
        b.axhline(v_rot, color="#B02418", lw=1.2, ls=":")
        b.plot([Rd], [v_rot], "o", color="#B02418", ms=9)
    b.set_xlim(0, 30); b.set_xlabel(r"$R$ [kpc]"); b.set_ylabel(r"$v_\phi$ [km/s]")
    b.set_title("(B) v_ISM derivation from the rotation curve")
    b.legend(fontsize=8, loc="best")
    b.text(0.03, 0.03,
           f"v_rot(R={Rd:.1f}) = {v_rot:.0f}\nproj = {proj:+.3f}\n"
           f"v_ISM = v_rot × proj = {v_model:.0f} km/s",
           transform=b.transAxes, va="bottom", ha="left", fontsize=10,
           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.92))

    # (C) ray velocity profile
    c = ax[0, 2]
    cc = np.log10(np.clip(ion_n[R.SIII], 1e-14, None))
    sz = 6 + 300 * (w_si / (w_si.max() + 1e-30))
    sc = c.scatter(s, v_rest, c=cc, s=sz, cmap="viridis", alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=c, label=r"$\log_{10} n_{\rm SiII}$")
    ds = disk & (s > s.min())
    if ds.any():
        c.axvspan(s[ds].min(), s[ds].max(), color="#ffd166", alpha=0.3,
                  label=f"disk |z|<{R.Z0:g}, R<{R_edge:.0f}")
    c.axhline(v_model, color="#B02418", lw=1.8, ls="--", label=f"v_ISM model={v_model:.0f}")
    c.axhline(0, color="0.5", ls=":", lw=0.9)
    c.set_xlabel("s from impact point [kpc]"); c.set_ylabel(r"$v_{\rm rest}$")
    c.set_title("(C) ray velocity profile"); c.legend(fontsize=8, loc="upper right")

    # (D) ion column-weighted velocity
    d = ax[1, 0]
    bins = np.linspace(-700, 700, 141)
    for key, rest, fld, lab, col in R.IONS:
        w = ion_n[fld] * res["dl"]
        if w.sum() <= 0:
            continue
        hist, _ = np.histogram(v_rest, bins=bins, weights=w)
        if hist.max() > 0:
            hist = hist / hist.max()
        d.step(0.5 * (bins[:-1] + bins[1:]), hist, where="mid", color=col, lw=1.3, label=lab)
    d.axvline(v_model, color="#B02418", lw=2.4, ls="--", label=f"v_ISM model={v_model:.0f}")
    d.axvline(0, color="0.4", ls=":", lw=1.0)
    d.set_xlim(-700, 700); d.set_xlabel(r"$v_{\rm rest}$"); d.set_ylabel("column-wt (norm.)")
    d.set_title("(D) ion column-weighted velocity"); d.legend(fontsize=7, ncol=2)

    # (E) geometry
    e = ax[1, 1]
    e.plot(s, z_d, color="#e76f51", lw=1.4, label=r"$z_{\rm disk}$")
    e.plot(s, R_disk, color="#264653", lw=1.4, ls="--", label=r"$R_{\rm disk}$")
    e.axhspan(-R.Z0, R.Z0, color="#ffd166", alpha=0.3)
    e.axhline(R_edge, color="0.5", ls=":", label=f"R_edge={R_edge:.0f}")
    e.axhline(0, color="0.5", ls=":", lw=0.9)
    e.set_xlabel("s [kpc]"); e.set_ylabel("disk-frame [kpc]")
    e.set_title("(E) sightline geometry"); e.legend(fontsize=9)

    # (F) audit text
    g = ax[1, 2]; g.axis("off")
    lines = [
        f"AUDIT — v_ISM (supervisor model)",
        f"",
        f"impact parameter ρ      = {rho:.1f} kpc",
        f"anchor disk radius R    = {Rd:.1f} kpc",
        f"disk edge (R95 cold)    = {R_edge:.1f} kpc   -> {'IN disk' if Rd<R_edge else 'BEYOND disk'}",
        f"projection proj = φ̂·los = {proj:+.3f}",
        f"",
        f"v_rot,cold ({rc['n_cold_gas'][np.argmin(np.abs(rc['R_center']-Rd))]:.0f} cells) = {interp(rc,'v_cold_gas',Rd):.0f}",
        f"v_rot,SF                = {interp(rc,'v_sf_gas',Rd):.0f}",
        f"v_rot,young             = {interp(rc,'v_young_stars',Rd):.0f}",
        f"v_rot (3-tracer avg)    = {v_rot:.0f} km/s",
        f"",
        f"v_ISM(model)= v_rot×proj= {v_model:.0f} km/s",
        f"per-tracer projected: cold {comp['cold_gas']:.0f} / SF {comp['sf_gas']:.0f} / young {comp['young_stars']:.0f}",
        f"v1 direct cool-gas      = {v_direct:.0f} km/s",
        f"Si II 1260 dip          = {v_dip:.0f} km/s",
        f"",
        f"v_ISM(model) - dip      = {off:+.0f} km/s   -> {verdict}",
    ]
    g.text(0.0, 1.0, "\n".join(lines), transform=g.transAxes, va="top", ha="left",
           fontsize=11, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"raydiagv2_sid{sid}_{mode}_alpha{alpha:03d}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return dict(sid=sid, mode=mode, alpha=alpha, R_anchor=Rd, proj=proj, v_rot=v_rot,
                v_ism_model=v_model, v_direct=v_direct, SiII_dip=v_dip, off_dip=off,
                verdict=verdict, in_disk=bool(Rd < R_edge))


def main():
    sid = int(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rc = dict(np.load(RCV2 / f"rc_sid{sid}.npz"))
    rows = []
    for mode in ("flip", "noflip"):
        for alpha in range(0, 360, step):
            try:
                rows.append(diagnose(sid, mode, alpha, rc))
            except Exception as ex:
                print(f"  {mode} a{alpha} FAIL: {type(ex).__name__}: {ex}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"summary_sid{sid}.csv", index=False)
    if len(df):
        lands = (df.verdict == "LANDS on Si II dip").mean()
        print(f"[SID {sid}] {len(df)} ray diagnostics; lands-on-dip {lands:.2f}; "
              f"median |off_dip| {df.off_dip.abs().median():.0f} km/s -> {OUT}")


if __name__ == "__main__":
    main()
