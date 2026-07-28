#!/usr/bin/env python3
"""Diagnostic: overlay the supervisor-model v_ISM on the mock spectra and test whether it
lands on a MAJOR absorption component.

For every sightline of one SID we take the Si II 1260 (and C II 1335, H I 1216) LSF flux on
the galaxy-rest velocity axis v = -(c*(lam/lam0-1)) - v_sys, detect absorption troughs
(local maxima of 1-flux via find_peaks, ranked by equivalent-width contribution), and record:
  * offset of v_ism_model to the DEEPEST trough and to the NEAREST major trough,
  * whether a major trough (EW-rank>=threshold) sits within TOL km/s of v_ism_model.
A landing fraction is written per SID; a gallery of example overlays is drawn.

Usage: python diag_spectra_overlay.py <sid>
Output: diagnostics_v2/spectra_overlay/{gallery_sid<sid>.png, landing_sid<sid>.csv}
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
from pm_general import C_KMS  # noqa: E402
import ray_ism_diagnostic as R  # noqa: E402
import m61_style as S  # noqa: E402

V2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2/spectra_overlay")
LINES = {"Si_II_1260": (1260.4221, "Si II 1260"), "C_II_1335": (1334.5323, "C II 1335"),
         "H_I_1216": (1215.6701, "H I 1216")}
REF = "Si_II_1260"           # the reference ISM line for the landing test
TOL = 30.0                   # km/s: "lands on" tolerance
DEPTH_MIN = 0.10             # min flux decrement for a trough to count
VWIN = 500.0


def troughs(v, flux):
    """Return (v_center, depth, ew) of absorption troughs, EW-ranked (strongest first)."""
    dec = np.clip(1.0 - flux, 0, None)
    idx, props = find_peaks(dec, height=DEPTH_MIN, prominence=0.04, distance=3)
    out = []
    for i in idx:
        lo = max(0, i - 25); hi = min(len(v), i + 26)
        ew = float(np.trapz(dec[lo:hi], v[lo:hi]))
        out.append((float(v[i]), float(dec[i]), abs(ew)))
    out.sort(key=lambda t: -t[2])
    return out


def process(sid):
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(V2)
    m = m[m.sid == sid].set_index(["mode", "alpha_deg"])
    rows = []
    panels = []   # (mode, alpha, spectra dict, v_ism, v_dir, v_sys)
    want = set(range(0, 360, 60))   # sample alphas for the gallery
    with h5py.File(R.combined_path(sid), "r") as h:
        base = h["rays/sightline=J122138+043026"]
        for mode in ("flip", "noflip"):
            if f"mode={mode}" not in base:
                continue
            for ag in sorted(base[f"mode={mode}"].keys(), key=lambda k: int(k.split("=")[1])):
                alpha = int(ag.split("=")[1])
                if (mode, alpha) not in m.index:
                    continue
                row = m.loc[(mode, alpha)]
                v_ism = float(row.v_ism_model); v_sys = float(row.v_sys)
                v_dir = float(row.v_ism_direct_cool); v_dip = float(row.SiII_dip)
                if not np.isfinite(v_ism):
                    continue
                grp = base[f"mode={mode}"][ag]
                ray = grp[list(grp.keys())[0]]   # descend into the ray_NNNNNN group
                spectra = {}
                for key, (rest, lab) in LINES.items():
                    g = ray.get(f"spectrum_by_line/{key}/lsf")
                    if g is None:
                        continue
                    lam = g["lambda_A"][()]; fl = g["flux"][()]
                    v = -(C_KMS * (lam / rest - 1.0)) - v_sys
                    sel = np.abs(v) < VWIN
                    o = np.argsort(v[sel])
                    spectra[key] = (v[sel][o], fl[sel][o], rest, lab)
                if REF not in spectra:
                    continue
                v_ref, f_ref, _, _ = spectra[REF]
                tr = troughs(v_ref, f_ref)
                if tr:
                    deepest = min(tr, key=lambda t: -t[1])   # largest depth
                    off_deep = v_ism - deepest[0]
                    nearest = min(tr, key=lambda t: abs(t[0] - v_ism))
                    off_near = v_ism - nearest[0]
                    lands = bool(abs(off_near) < TOL)
                    n_major = len(tr)
                else:
                    off_deep = off_near = np.nan; lands = False; n_major = 0
                rows.append(dict(sid=sid, mode=mode, alpha=alpha, v_ism=v_ism,
                                 v_dir=v_dir, v_dip=v_dip, in_disk_model=bool(row.in_disk_model),
                                 off_deepest=off_deep, off_nearest=off_near,
                                 lands_major=lands, n_troughs=n_major))
                if alpha in want and mode == "flip" and len(panels) < 12:
                    panels.append((mode, alpha, spectra, v_ism, v_dir, v_dip, tr))
    land = pd.DataFrame(rows)
    land.to_csv(OUT / f"landing_sid{sid}.csv", index=False)

    # ---- gallery ----
    S.set_style()
    n = len(panels)
    if n:
        ncol = 3; nrow = int(np.ceil(n / ncol))
        fig, ax = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.8 * nrow), squeeze=False)
        colL = {"Si_II_1260": "#1d3557", "C_II_1335": "#2a9d8f", "H_I_1216": "#888888"}
        for k, (mode, alpha, spectra, v_ism, v_dir, v_dip, tr) in enumerate(panels):
            a = ax[k // ncol][k % ncol]
            for key, (v, fl, rest, lab) in spectra.items():
                a.plot(v, fl, color=colL.get(key, "k"), lw=1.1, label=lab)
            for (vt, dep, ew) in tr[:4]:
                a.axvline(vt, color="0.75", lw=0.8, ls=":")
            a.axvline(0, color="0.4", lw=0.8, ls=":")
            a.axvline(v_ism, color="#B02418", lw=2.2, label=r"$v_{\rm ISM}$ model")
            if np.isfinite(v_dir):
                a.axvline(v_dir, color="#1B9E77", lw=1.4, ls="--", label=r"$v_{\rm ISM}$ direct")
            if np.isfinite(v_dip):
                a.axvline(v_dip, color="#E19A3C", lw=1.2, ls="-.", label="Si II dip")
            a.set_xlim(-VWIN, VWIN); a.set_ylim(-0.05, 1.12)
            a.set_title(rf"{mode} $\alpha={alpha}$", fontsize=8)
            if k == 0:
                a.legend(fontsize=6.5, loc="lower left", ncol=1)
        for k in range(n, nrow * ncol):
            ax[k // ncol][k % ncol].axis("off")
        fig.suptitle(rf"SID {sid}: $v_{{\rm ISM}}$(model) on Si II/C II/H I spectra "
                     rf"(rest frame)", fontsize=11)
        fig.supxlabel(r"$v_{\rm rest}$ [km s$^{-1}$]  (+=recession)")
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        fig.savefig(OUT / f"gallery_sid{sid}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    ok = land.dropna(subset=["off_nearest"])
    ind = ok[ok.in_disk_model]
    print(f"[SID {sid}] {len(land)} sightlines; lands-on-major (all) "
          f"{ok.lands_major.mean():.2f}; in_disk {ind.lands_major.mean():.2f}; "
          f"med|off_nearest| all {ok.off_nearest.abs().median():.1f} "
          f"in_disk {ind.off_nearest.abs().median():.1f} km/s")


if __name__ == "__main__":
    process(int(sys.argv[1]))
