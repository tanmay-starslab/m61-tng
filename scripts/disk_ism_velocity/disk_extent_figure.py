#!/usr/bin/env python3
"""'Where does the disk end?' — cold-gas surface-density profile Sigma_cold(R) for every galaxy,
with the operational disk edge R_disk (outermost R past the peak with Sigma > 10% of peak), R95
of cold-gas mass (inflated by diffuse tails), and the QSO impact parameter rho marked. Reads the
slit-profile npz written by build_vism_v3.py.
Output: diagnostics_v3/disk_extent.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

VT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v3")
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]
RHO = 25.6


def main():
    S.set_style()
    m = pd.read_csv(VT / "vism_master_v3.csv")
    fig, ax = plt.subplots(4, 5, figsize=(22, 15), sharex=True)
    rows = []
    for k, sid in enumerate(SIDS):
        a = ax[k // 5][k % 5]
        sp = np.load(VT / f"slitprof_sid{sid}.npz", allow_pickle=True)
        Rc = sp["Rcen"]; sig = sp["sigmaR"]; Rdisk = float(sp["R_disk"])
        peak = np.nanmax(sig) if np.nanmax(sig) > 0 else 1.0
        a.plot(Rc, sig / peak, "-o", color="#1d3557", ms=3, lw=1.5)
        a.axhline(0.1, color="0.6", ls=":", lw=1.0)
        a.axvline(Rdisk, color="#B02418", lw=2.0, label=rf"$R_{{\rm disk}}={Rdisk:.0f}$")
        r95 = float(m[m.sid == sid].R95_cold.iloc[0])
        a.axvline(r95, color="#7B6FB0", lw=1.3, ls="--", label=rf"$R_{{95,\rm mass}}={r95:.0f}$")
        a.axvline(RHO, color="0.2", lw=1.3, ls=":", label=r"$\rho=25.6$")
        a.axvspan(Rdisk, 40, color="0.85", alpha=0.5, lw=0)
        a.set_xlim(0, 40); a.set_ylim(0, 1.08)
        a.set_title(rf"SID {sid} ({'in-disk' if RHO < Rdisk else 'edge/beyond'})", fontsize=9)
        if k == 0:
            a.legend(fontsize=7, loc="upper right")
        if k % 5 == 0:
            a.set_ylabel(r"$\Sigma_{\rm cold}/\Sigma_{\rm peak}$")
        S.grid(a)
        rows.append(dict(sid=sid, R_disk=Rdisk, R95_mass=r95, rho=RHO, rho_in_disk=RHO < Rdisk))
    fig.suptitle(r"Cold-gas ($T<10^4$ K) surface density vs radius — the disk edge $R_{\rm disk}$ "
                 r"(where $\Sigma$ drops below $10\%$ of peak) vs $\rho$", fontsize=15)
    fig.supxlabel(r"$R$ [kpc]")
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(OUT / "disk_extent.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    d = pd.DataFrame(rows); d.to_csv(OUT / "disk_extent.csv", index=False)
    print(f"saved {OUT}/disk_extent.png")
    print(f"galaxies with rho inside the disk: {int(d.rho_in_disk.sum())}/20")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
