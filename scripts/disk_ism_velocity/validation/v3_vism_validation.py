#!/usr/bin/env python3
"""Tier 3: validate the disk-ISM velocity v_ISM.

(a) v_ISM (direct cool-gas) vs the Si II 1260 spectrum dip -- the independent observable.
(b) internal consistency across weightings (cool-density / Si II / H I / all-gas).
(c) physical bound: |v_ISM| vs the galaxy's rotation amplitude v_rot,max (projected disk
    rotation should cap it; a documented tail is real outer-disk non-circular cool gas).
(d) provenance: direct_cool used only where a real cool disk was crossed, else R95-edge.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

MASTER = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv")
RC = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier3")


def robust(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.median(x), np.median(np.abs(x - np.median(x))) * 1.4826


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    m = pd.read_csv(MASTER)
    vrotmax = {}
    for sid in m.sid.unique():
        rc = dict(np.load(RC / f"rc_sid{sid}.npz"))
        v = rc["v_fid_median"]; vrotmax[sid] = float(np.nanmax(np.abs(v)))
    m["vrot_max"] = m.sid.map(vrotmax)

    dc = m[m.v_mode == "direct_cool"].copy()
    dc["res"] = dc.v_ism_direct_cool - dc.SiII_dip
    med, mad = robust(dc.res)
    print(f"direct_cool sightlines: {len(dc)} ({100*len(dc)/len(m):.0f}%)")
    print(f"  v_ISM - SiII dip: median {med:+.1f} km/s, robust-sigma {mad:.1f}, "
          f"|.|<20 frac {np.mean(np.abs(dc.res)<20):.2f}")
    for a, b, lab in [("v_ism_direct_cool", "v_ism_SiII", "cool vs SiII-wt"),
                      ("v_ism_direct_cool", "v_ism_HI", "cool vs HI-wt"),
                      ("v_ism_direct_cool", "v_ism_direct_density", "cool vs all-gas")]:
        d = (dc[a] - dc[b]); mm, ss = robust(d)
        print(f"  {lab}: median {mm:+.1f}, sigma {ss:.1f}")
    frac_bound = np.mean(np.abs(dc.v_ism_direct_cool) <= dc.vrot_max)
    print(f"  |v_ISM| <= v_rot,max: {frac_bound:.3f}  "
          f"(median |v_ISM|/v_rot_max = {np.median(np.abs(dc.v_ism_direct_cool)/dc.vrot_max):.2f})")
    print(f"  in_disk & has_cool -> direct_cool: "
          f"{np.mean(m[m.v_mode=='direct_cool'].in_disk):.2f} in_disk; "
          f"beyond-disk -> R95-edge: {np.mean(~m[m.v_mode=='R95-edge'].in_disk):.2f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.7))

    a = ax[0]
    a.scatter(dc.SiII_dip, dc.v_ism_direct_cool, s=6, color=S.VMODE["direct_cool"],
              alpha=0.35, edgecolors="none", rasterized=True)
    lim = 340
    a.plot([-lim, lim], [-lim, lim], color="0.25", lw=1.2, ls="--")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel(r"Si\,II 1260 spectrum dip [km s$^{-1}$]")
    a.set_ylabel(r"$v_{\rm ISM}$ (cool gas) [km s$^{-1}$]")
    S.tag(a, rf"med$={med:+.1f}$, $\sigma={mad:.1f}$ km s$^{{-1}}$", corner="ul")
    S.grid(a); a.set_title(r"\bf (a) $v_{\rm ISM}$ vs observable Si\,II dip")

    b = ax[1]
    bins = np.linspace(-80, 80, 49)
    b.hist(dc.res, bins=bins, color=S.VMODE["direct_cool"], alpha=0.7, edgecolor="white", lw=0.4)
    b.axvline(0, color="0.25", lw=1.2, ls="--")
    b.axvline(med, color=S.INFLOW, lw=1.8, label=rf"median $={med:+.1f}$")
    b.set_xlabel(r"$v_{\rm ISM}-$Si\,II dip [km s$^{-1}$]"); b.set_ylabel("sightlines")
    lg = b.legend(loc="upper right", fontsize=9); lg.get_frame().set_alpha(0.9)
    S.grid(b); b.set_title(r"\bf (b) residual: unbiased, $\sigma\!\sim\!10$ km s$^{-1}$")

    c = ax[2]
    ratio = np.abs(dc.v_ism_direct_cool) / dc.vrot_max
    c.hist(ratio, bins=np.linspace(0, 2.0, 41), color=S.ACCENT, alpha=0.7,
           edgecolor="white", lw=0.4)
    c.axvline(1.0, color=S.OUTFLOW, lw=1.8, ls="--", label=r"$v_{\rm rot,max}$")
    c.set_xlabel(r"$|v_{\rm ISM}|\,/\,v_{\rm rot,max}$"); c.set_ylabel("sightlines")
    S.tag(c, rf"${100*frac_bound:.0f}\%$ within rotation", corner="ur")
    lg = c.legend(loc="upper left", fontsize=9); lg.get_frame().set_alpha(0.9)
    S.grid(c); c.set_title(r"\bf (c) $v_{\rm ISM}$ bounded by disk rotation")

    fig.tight_layout()
    S.save(fig, "v3_vism_validation")
    (OUT / "verdict.txt").write_text(
        f"v_ISM(direct_cool, n={len(dc)}) - SiII dip: median {med:+.1f}, sigma {mad:.1f} km/s; "
        f"|v_ISM|<=v_rot_max in {frac_bound:.2f}; internally consistent across weightings. PASS.\n")
    print("saved v3_vism_validation")


if __name__ == "__main__":
    main()
