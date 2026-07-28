#!/usr/bin/env python3
"""Big-picture overview of the v2 ray diagnostics: aggregates the per-SID summary CSVs from
ray_diagnostic_v2.py into (A) per-galaxy landing rate + median |offset|, (B) the offset
(v_ISM model - Si II dip) vs orientation for every sampled sightline, (C) its distribution.
Also emits per-SID ImageMagick contact sheets so hundreds of panels can be browsed at a glance.
"""
from __future__ import annotations
import sys, subprocess, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

DIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v2/ray_diagnostics_v2")


def main():
    S.set_style()
    fs = sorted(DIR.glob("summary_sid*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    sids = sorted(df.sid.unique())
    lands = df.groupby("sid").apply(lambda d: (d.verdict == "LANDS on Si II dip").mean(), include_groups=False)
    medoff = df.groupby("sid").off_dip.apply(lambda s: s.abs().median())
    print(f"{len(df)} ray diagnostics over {len(sids)} SIDs")
    print(f"overall lands-on-dip {(df.verdict=='LANDS on Si II dip').mean():.2f}; "
          f"near-or-lands {(df.verdict!='OFF dip').mean():.2f}; "
          f"median |offset| {df.off_dip.abs().median():.0f} km/s")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(sids))
    ax[0].bar(x, [lands[s] * 100 for s in sids], color=S.VMODE["direct_cool"], alpha=0.85,
              edgecolor="k", lw=0.4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(sids, rotation=90, fontsize=6)
    ax[0].set_ylabel(r"lands on Si II dip [\%]"); ax[0].set_ylim(0, 100)
    ax[0].set_title(r"\bf (A) per-galaxy landing rate"); S.grid(ax[0])

    for s in sids:
        d = df[df.sid == s]
        ax[1].scatter(d.alpha, d.off_dip, s=8, alpha=0.5, edgecolors="none")
    ax[1].axhline(0, color="0.3", lw=1.0); ax[1].axhspan(-30, 30, color="#4C4C4C", alpha=0.12, lw=0)
    ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel(r"$v_{\rm ISM}-$Si II dip [km/s]")
    ax[1].set_ylim(-250, 250); ax[1].set_title(r"\bf (B) offset vs orientation"); S.grid(ax[1])

    off = df.off_dip.dropna()
    ax[2].hist(off, bins=np.linspace(-250, 250, 61), color=S.ACCENT, alpha=0.8,
               edgecolor="white", lw=0.4)
    med = off.median(); sig = (off - med).abs().median() * 1.4826
    ax[2].axvline(med, color="#B02418", lw=1.8, label=rf"median $={med:+.0f}$, $\sigma={sig:.0f}$")
    ax[2].axvspan(-30, 30, color="#4C4C4C", alpha=0.12, lw=0)
    ax[2].set_xlabel(r"$v_{\rm ISM}-$Si II dip [km/s]"); ax[2].set_ylabel("sightlines")
    ax[2].legend(fontsize=9); ax[2].set_title(r"\bf (C) offset distribution"); S.grid(ax[2])
    fig.tight_layout()
    p = DIR / "OVERVIEW_landing.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")

    # per-SID contact sheets (browse all orientations of one galaxy at once)
    cs = DIR / "contact_sheets"; cs.mkdir(exist_ok=True)
    for s in sids:
        imgs = sorted(glob.glob(str(DIR / f"raydiagv2_sid{s}_*.png")))
        if not imgs:
            continue
        out = cs / f"contact_sid{s}.png"
        subprocess.run(["montage", *imgs, "-tile", "6x", "-geometry", "600x+4+4",
                        "-title", f"SID {s} v2 ray diagnostics", str(out)], check=False)
    print(f"contact sheets -> {cs}")


if __name__ == "__main__":
    main()
