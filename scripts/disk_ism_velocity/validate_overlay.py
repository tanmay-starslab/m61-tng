#!/usr/bin/env python3
"""Stage-B prototype + VISUAL VALIDATION gate.

For a few (sid, mode, alpha) sightlines, project the fiducial 3-tracer disk-ISM
rotation curve onto the line of sight (-> v_ism_model), and overlay it on the RAW
Si II 1260 absorption spectrum with the x-axis = velocity relative to the galaxy
systemic velocity (0 = v_sys, positive = recession).

Interpretation: the ISM absorption dip should sit ON v_ism_model; additional dips
offset from it are HVC/IVC candidates. The Si II Voigt centre (v0) and non-parametric
centroid are also marked, converted into the same corrected frame:
    v_corr = -(C*(lam/lam0 - 1)) - v_sys        (spectrum axis)
    v0_corr = -(v0_raw) - v_sys                  (Voigt centre; raw is Trident-sign)

Usage:
    python validate_overlay.py <sid> <mode> <alpha> [<sid> <mode> <alpha> ...]
    python validate_overlay.py --cases cases.csv        # csv columns: sid,mode,alpha
Requires rc_sid<sid>.npz from build_sid_rc.py.
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
from pm_general import (C_KMS, REST_WAVE, OUTPUTS, get_geometry, get_original_rho,  # noqa: E402
                        compute_endpoints)

RC_DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
VAL_DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation")
VOIGT_CSV = Path("/scratch/tsingh65/m61-tng/all_subhalos_siII1260_voigt_results_snr10_parallel.csv")
LINE = "Si II 1260"
LAM0 = REST_WAVE[LINE]
_voigt = None


def voigt_row(sid, mode, alpha):
    global _voigt
    if _voigt is None:
        _voigt = pd.read_csv(VOIGT_CSV)
    # NB: use bracket access -- _voigt.mode is the DataFrame.mode() METHOD, not the column.
    d = _voigt[(_voigt["sid"] == sid) & (_voigt["mode"] == mode) & (_voigt["alpha_deg"] == alpha)]
    return d.iloc[0] if len(d) else None


def spec_path(sid, mode, alpha):
    return (OUTPUTS / f"sid{sid}" / f"rays_and_spectra_sid{sid}_snap99_L4Rvir" / "spectra_h5" /
            f"L4Rvir_sid{sid}_J122138+043026_{mode}_alpha{alpha}_spectrum.h5")


def interp_rc(R, Rc, v):
    fin = np.isfinite(v)
    if fin.sum() < 2:
        return np.nan, False
    extrap = (R < Rc[fin].min()) or (R > Rc[fin].max())
    return float(np.interp(R, Rc[fin], v[fin])), bool(extrap)


def v_ism_case(sid, mode, alpha, rc):
    """Project fiducial + per-tracer rotation velocity onto the LOS at this sightline."""
    geom = get_geometry(sid, mode, alpha)
    los = geom["los"]; v_sys = geom["v_sys"]
    rho, _, _ = get_original_rho(sid, mode, alpha)
    eps = compute_endpoints(sid, mode, alpha, rho, 50.)
    center = rc["center_kpc"]; e1 = rc["e1"]; e2 = rc["e2"]; nd = rc["n_disk"]
    rel = eps["anchor_kpc"] - center
    x_d = float(rel @ e1); y_d = float(rel @ e2); z_d = float(rel @ nd)
    R = math.hypot(x_d, y_d)
    los_d = np.array([los @ e1, los @ e2, los @ nd])
    proj_phi = float(np.array([-y_d / R, x_d / R, 0.]) @ los_d) if R > 0.2 else np.nan
    Rc = rc["R_center"]
    vfid, extrap = interp_rc(R, Rc, rc["v_fid_median"])
    sig, _ = interp_rc(R, Rc, rc["sigma_fid"])
    out = dict(sid=sid, mode=mode, alpha=alpha, v_sys=v_sys, rho=rho,
               R_disk=R, z_disk=z_d, proj_phi=proj_phi, extrap=extrap,
               v_ism_model=vfid * proj_phi if (np.isfinite(vfid) and np.isfinite(proj_phi)) else np.nan,
               v_ism_sigma=abs(sig * proj_phi) if (np.isfinite(sig) and np.isfinite(proj_phi)) else np.nan)
    for t in ("cold_gas_1e3", "sf_gas", "young_stars", "cold_gas_1e4"):
        v, _ = interp_rc(R, Rc, rc[f"v_{t}"])
        out[f"v_ism_{t}"] = v * proj_phi if (np.isfinite(v) and np.isfinite(proj_phi)) else np.nan
    return out


def overlay(sid, mode, alpha):
    npz = RC_DIR / f"rc_sid{sid}.npz"
    if not npz.exists():
        print(f"[skip] no rotation curve for sid {sid} ({npz})"); return None
    rc = dict(np.load(npz))
    info = v_ism_case(sid, mode, alpha, rc)
    v_sys = info["v_sys"]
    sp = spec_path(sid, mode, alpha)
    if not sp.exists():
        print(f"[skip] no spectrum {sp}"); return None
    vr = voigt_row(sid, mode, alpha)
    with h5py.File(sp, "r") as h:
        lam = h["spectrum/lsf/lambda_A"][()]
        flux = h["spectrum/lsf/flux"][()]
    v_corr = -(C_KMS * (lam / LAM0 - 1.0)) - v_sys
    win = 700.0
    m = (v_corr > -win) & (v_corr < win)
    order = np.argsort(v_corr[m])
    vv = v_corr[m][order]; ff = flux[m][order]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(vv, ff, color="#2f6f9f", lw=1.4, zorder=3, label="Si II 1260 (LSF)")
    ax.axhline(1, color="0.6", lw=0.6, ls=":")
    ax.axvline(0, color="0.3", lw=1.1, ls=":", label=r"$v_{\rm sys}=0$")

    vm = info["v_ism_model"]; sg = info["v_ism_sigma"]
    if np.isfinite(vm):
        if np.isfinite(sg):
            ax.axvspan(vm - sg, vm + sg, color="#0077b6", alpha=0.15, zorder=1)
        ax.axvline(vm, color="#0077b6", lw=2.4,
                   label=rf"$v_{{\rm ISM}}$ 3-tracer = {vm:.0f}" + (" [extrap]" if info["extrap"] else ""))
    for t, c, ls, lab in [("cold_gas_1e3", "#1d3557", "--", "cold<10$^3$"),
                          ("cold_gas_1e4", "#2a9d8f", (0, (4, 1, 1, 1)), "cold<10$^4$"),
                          ("sf_gas", "#457b9d", ":", "SF gas"),
                          ("young_stars", "#e9c46a", "-.", "young*")]:
        v = info[f"v_ism_{t}"]
        if np.isfinite(v):
            ax.axvline(v, color=c, ls=ls, lw=1.4, label=rf"{lab} = {v:.0f}")

    if vr is not None:
        for col, c, lab in [("v0_kms", "#9d0208", r"Voigt $v_0$"),
                            ("vcentroid_nonparam_kms", "#e76f51", "centroid")]:
            raw = vr.get(col, np.nan)
            if pd.notna(raw):
                vc = -float(raw) - v_sys
                ax.axvline(vc, color=c, lw=1.8, label=rf"{lab} = {vc:.0f}")

    ax.set_xlim(-win, win); ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel(r"$v_{\rm LOS} - v_{\rm sys}$ [km s$^{-1}$]  (positive = recession)")
    ax.set_ylabel("Normalised flux")
    dipflag = "" if not info["extrap"] else "  (R beyond curve — extrapolated)"
    ax.set_title(f"SID {sid} · {mode} · α={alpha}°  ·  R_disk={info['R_disk']:.1f} kpc · "
                 f"proj={info['proj_phi']:.2f} · v_sys={v_sys:.0f}{dipflag}")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out = VAL_DIR / f"overlay_sid{sid}_{mode}_alpha{alpha:03d}_SiII1260.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    extra = ""
    if vr is not None and pd.notna(vr.get("v0_kms")):
        extra = f"  Voigt_v0_corr={-float(vr['v0_kms']) - v_sys:.1f}"
    print(f"[ok] {out.name}: v_ISM={info['v_ism_model']:.1f} (±{info['v_ism_sigma']:.1f}) "
          f"R={info['R_disk']:.1f} proj={info['proj_phi']:.2f}{extra}")
    return info


def main():
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    cases = []
    if args and args[0] == "--cases":
        df = pd.read_csv(args[1])
        cases = [(int(r.sid), str(r.mode), int(r.alpha)) for r in df.itertuples()]
    else:
        for i in range(0, len(args), 3):
            cases.append((int(args[i]), args[i + 1], int(args[i + 2])))
    rows = []
    for sid, mode, alpha in cases:
        r = overlay(sid, mode, alpha)
        if r:
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(VAL_DIR / "validation_summary.csv", index=False)
        print(f"\n[summary] {VAL_DIR}/validation_summary.csv")


if __name__ == "__main__":
    main()
