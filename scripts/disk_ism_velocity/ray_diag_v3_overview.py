#!/usr/bin/env python3
"""Overview of the v3 ray diagnostics: per-galaxy v3a landing rate, offset vs orientation, and
offset distribution (in-disk sightlines); plus per-SID ImageMagick contact sheets."""
from __future__ import annotations
import sys, subprocess, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v3/ray_diagnostics_v3")


def main():
    S.set_style()
    df = pd.concat([pd.read_csv(f) for f in sorted(DIR.glob("summary_sid*.csv"))], ignore_index=True)
    ind = df[df.in_disk == True]
    sids = sorted(df.sid.unique())
    print(f"{len(df)} v3 ray diagnostics; in-disk {len(ind)}")
    print(f"in-disk v3a lands<30 {(ind.verdict=='LANDS').mean():.2f}; near-or-lands "
          f"{(ind.verdict!='OFF').mean():.2f}; median |off| {ind.off.abs().median():.0f} km/s")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    lands = ind.groupby("sid").apply(lambda d: (d.verdict == "LANDS").mean(), include_groups=False)
    x = np.arange(len(sids))
    ax[0].bar(x, [lands.get(s, np.nan) * 100 for s in sids], color=S.VMODE["direct_cool"],
              alpha=0.85, edgecolor="k", lw=0.4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(sids, rotation=90, fontsize=6)
    ax[0].set_ylabel(r"v3a lands on dip [\%]"); ax[0].set_ylim(0, 100)
    ax[0].set_title(r"\bf (a) per-galaxy v3a landing rate (in-disk)"); S.grid(ax[0])
    for s in sids:
        d = ind[ind.sid == s]
        ax[1].scatter(d.alpha, d.off, s=9, alpha=0.5, edgecolors="none")
    ax[1].axhline(0, color="0.3", lw=1.0); ax[1].axhspan(-30, 30, color="#4C4C4C", alpha=0.12, lw=0)
    ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel(r"v3a $-$ Si II dip [km/s]")
    ax[1].set_ylim(-200, 200); ax[1].set_title(r"\bf (b) v3a offset vs orientation"); S.grid(ax[1])
    off = ind.off.dropna(); med = off.median(); sig = (off - med).abs().median() * 1.4826
    ax[2].hist(off, bins=np.linspace(-200, 200, 61), color=S.ACCENT, alpha=0.8, edgecolor="white", lw=0.4)
    ax[2].axvline(med, color="#B02418", lw=1.8, label=rf"median $={med:+.0f}$, $\sigma={sig:.0f}$")
    ax[2].axvspan(-30, 30, color="#4C4C4C", alpha=0.12, lw=0)
    ax[2].set_xlabel(r"v3a $-$ Si II dip [km/s]"); ax[2].set_ylabel("sightlines")
    ax[2].legend(fontsize=9); ax[2].set_title(r"\bf (c) v3a offset distribution"); S.grid(ax[2])
    fig.tight_layout()
    p = DIR / "OVERVIEW_v3a_landing.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")

    cs = DIR / "contact_sheets"; cs.mkdir(exist_ok=True)
    for s in sids:
        imgs = sorted(glob.glob(str(DIR / f"raydiagv3_sid{s}_*.png")))
        if imgs:
            subprocess.run(["montage", *imgs, "-tile", "6x", "-geometry", "560x+3+3",
                            "-title", f"SID {s} v3 ray diagnostics", str(cs / f"contact_sid{s}.png")],
                           check=False)
    print(f"contact sheets -> {cs}")


if __name__ == "__main__":
    main()
