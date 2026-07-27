#!/usr/bin/env python3
"""fig10: spectrum-based absorber detection rate vs velocity offset, one curve per LINE.

Aggregates the per-SID accumulations (fig10_accumulate.py) over all 20 galaxies:
detection rate at Δv = (sightlines with >10% absorption in that Δv bin) / (all sightlines).
The three Si II lines (1190/1193/1260) appear as separate curves (same colour, different
style), directly showing the line-strength dependence. Output: paper_figures_v2/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

ACC = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2/detection_rate")
S.FIGDIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures_v2")

# ion colour (turbo, neutral->high), then per-line (ion, linestyle, label)
_SEQ = ["HI", "OI", "CII", "SiII", "SiIII", "SiIV", "NV"]
_TURBO = plt.get_cmap("turbo")
ICOL = {k: mcolors.to_hex(_TURBO(0.06 + 0.88 * i / (len(_SEQ) - 1))) for i, k in enumerate(_SEQ)}
LINEDEF = {  # spectrum key -> (ion, linestyle, lw, legend label)
    "H_I_1216":   ("HI",   "-",  2.0, r"$\mathrm{H\,I}\ \lambda1216$"),
    "O_I_1302":   ("OI",   "-",  2.0, r"$\mathrm{O\,I}\ \lambda1302$"),
    "C_II_1335":  ("CII",  "-",  2.0, r"$\mathrm{C\,II}\ \lambda1335$"),
    "Si_II_1260": ("SiII", "-",  2.0, r"$\mathrm{Si\,II}\ \lambda1260$"),
    "Si_II_1193": ("SiII", "--", 1.5, r"$\mathrm{Si\,II}\ \lambda1193$"),
    "Si_II_1190": ("SiII", ":",  1.5, r"$\mathrm{Si\,II}\ \lambda1190$"),
    "Si_III_1206": ("SiIII", "-", 2.0, r"$\mathrm{Si\,III}\ \lambda1206$"),
    "Si_IV_1403": ("SiIV", "-",  2.0, r"$\mathrm{Si\,IV}\ \lambda1403$"),
    "N_V_1239":   ("NV",   "-",  2.0, r"$\mathrm{N\,V}\ \lambda1239$"),
}


def main():
    S.set_style()
    files = sorted(ACC.glob("acc_sid*.npz"))
    if not files:
        print("no accumulations found"); return
    d0 = np.load(files[0], allow_pickle=True)
    edges = d0["edges"]; keys = [k for k in d0["lines"]]
    xc = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros_like(d0["counts"]); n_los = np.zeros_like(d0["n_los"])
    for f in files:
        d = np.load(f, allow_pickle=True)
        counts += d["counts"]; n_los += d["n_los"]
    rate = counts / np.clip(n_los[:, None], 1, None)   # (line, bin)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    S.shade_classes(ax, xmax=500, alpha=0.05)
    for j, key in enumerate(keys):
        if key not in LINEDEF:
            continue
        ion, ls, lw, lab = LINEDEF[key]
        ax.plot(xc, 100 * rate[j], color=ICOL[ion], ls=ls, lw=lw, label=lab)
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    ax.set_xlim(-500, 500); ax.set_ylim(0, 100)
    ax.set_xlabel(r"$\Delta v = v_{\rm los}-v_{\rm ISM}\ [\mathrm{km\,s^{-1}}]$")
    ax.set_ylabel(r"detection rate [\% of sightlines]")
    ax.text(-40, 96, "ISM", color=S.CLASS["ISM"], fontsize=9, ha="right", va="top")
    ax.text(70, 96, "IVC", color=S.CLASS["IVC"], fontsize=9, ha="left", va="top")
    ax.text(300, 96, "HVC", color=S.CLASS["HVC"], fontsize=9, ha="center", va="top")
    lg = ax.legend(loc="upper right", fontsize=8.5, ncol=2, handlelength=2.4)
    lg.get_frame().set_alpha(0.92)
    S.grid(ax)
    fig.tight_layout()
    p = S.save(fig, "fig10_detection_rate_vs_velocity")
    # report peaks
    print(f"aggregated {len(files)} SIDs; peak detection rate per line:")
    for j, key in enumerate(keys):
        if key in LINEDEF:
            print(f"  {LINEDEF[key][3]:22s}: peak {100*rate[j].max():.0f}% ; "
                  f"HVC (|Δv|>100) rate {100*rate[j][np.abs(xc)>100].mean():.1f}%")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
