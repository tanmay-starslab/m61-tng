#!/usr/bin/env python3
"""v3 validation: how well each v_ISM lands on the Si II dip. Compares v3a (3-tracer along
the sightline), v3b (center->impact probe-line), v2 (galaxy-averaged rotation curve), and the
v1 direct method, on the in-disk sightlines. Output: diagnostics_v3/v3_compare.{png}."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

M = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3/vism_master_v3.csv")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/diagnostics_v3")
METHODS = [("v_ism_v3a", "v3a: 3-tracer along sightline", "#1B9E77"),
           ("v_ism_v3b", "v3b: centre->impact line", "#B02418"),
           ("v_ism_v2", "v2: galaxy rotation curve", "#7B6FB0"),
           ("v_ism_direct_cool", "v1 direct cool-gas", "#E19A3C")]


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.median(np.abs(x - np.median(x))) * 1.4826) if len(x) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.set_style()
    m = pd.read_csv(M)
    ind = m[m["in_disk_v1"] == True].copy()
    sids = sorted(m.sid.unique())
    print("Robust sigma of (method - Si II dip), in-disk sightlines:")
    for col, lab, _ in METHODS:
        d = (ind[col] - ind.SiII_dip).dropna()
        print(f"  {lab:34s} median {d.median():+6.1f}  sigma {rsig(d):6.1f}  n={len(d)}")

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5))
    # (a) per-SID sigma
    a = ax[0, 0]; x = np.arange(len(sids)); w = 0.2
    for i, (col, lab, c) in enumerate(METHODS):
        sig = [rsig(ind[ind.sid == s][col] - ind[ind.sid == s].SiII_dip) for s in sids]
        a.bar(x + (i - 1.5) * w, sig, w, color=c, label=lab.split(":")[0])
    a.set_xticks(x); a.set_xticklabels(sids, rotation=90, fontsize=6)
    a.set_ylabel(r"$\sigma(v_{\rm ISM}-$Si II dip$)$ [km/s]"); a.set_ylim(0, 140)
    a.legend(fontsize=8, ncol=2); S.grid(a)
    a.set_title(r"\bf (a) per-galaxy scatter vs the Si II dip (lower = better)")

    # (b,c) scatter for v3a and v3b
    for ax_i, col, lab in ((ax[0, 1], "v_ism_v3a", "v3a: along sightline"),
                           (ax[1, 0], "v_ism_v3b", "v3b: centre->impact line")):
        d = ind.dropna(subset=[col, "SiII_dip"])
        ax_i.scatter(d.SiII_dip, d[col], s=5, color="#333", alpha=0.25, edgecolors="none", rasterized=True)
        lim = 340; ax_i.plot([-lim, lim], [-lim, lim], "k--", lw=1.1)
        ax_i.set_xlim(-lim, lim); ax_i.set_ylim(-lim, lim)
        ax_i.set_xlabel("Si II 1260 dip [km/s]"); ax_i.set_ylabel(f"{col} [km/s]")
        S.tag(ax_i, rf"$\sigma={rsig(d[col]-d.SiII_dip):.0f}$ km/s", corner="ul")
        S.grid(ax_i); ax_i.set_title(rf"\bf ({'b' if col.endswith('a') else 'c'}) {lab}")

    # (d) residual histograms
    b = ax[1, 1]
    for col, lab, c in METHODS:
        d = (ind[col] - ind.SiII_dip).dropna()
        b.hist(d, bins=np.linspace(-200, 200, 61), histtype="step", lw=1.8, color=c,
               label=rf"{lab.split(':')[0]} ($\sigma={rsig(d):.0f}$)", density=True)
    b.axvline(0, color="0.4", ls=":", lw=1.0)
    b.set_xlabel(r"$v_{\rm ISM}-$Si II dip [km/s]"); b.set_ylabel("normalized")
    b.legend(fontsize=8); S.grid(b); b.set_title(r"\bf (d) residual distributions")

    fig.tight_layout()
    fig.savefig(OUT / "v3_compare.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT}/v3_compare.png")


if __name__ == "__main__":
    main()
