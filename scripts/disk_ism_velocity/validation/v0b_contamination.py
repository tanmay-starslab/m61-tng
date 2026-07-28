#!/usr/bin/env python3
"""Tier 0b: quantify how much the periodic-wrap / hyper-velocity cells change the
headline O VI/N V/C IV results, then define the clean science mask.

Loads all 20 rebuilt catalogs (which now carry `wrapped` and `hypervel` flags) and
compares, per ion: HVC column fraction (|dv|>100), IVC fraction, inflowing fraction of
the HVC column (v_r<0), and median |dv| -- ALL cells vs the CLEAN sample
(~wrapped & ~hypervel). Writes a CSV + a short verdict.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402

CAT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/absorber_catalog")
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/tier0b")
FLOOR = {"HI": 1e13, "CII": 1e12, "SiII": 1e12, "SiIII": 1e12, "SiIV": 1e12, "NV": 3e12,
         "CIV": 1e12, "OI": 1e12, "OVI": 3e12, "MgII": 1e11, "FeII": 1e11}


def load_all():
    dfs = [pd.read_parquet(p) for p in sorted(CAT.glob("absorbers_sid*.parquet"))]
    return pd.concat(dfs, ignore_index=True)


def frac_table(df, ions):
    rows = []
    for ion in ions:
        Ncol = f"N_{ion}"
        sub = df[df[Ncol] > FLOOR[ion]]
        w = sub[Ncol].values
        adv = np.abs(sub["dv"].values)
        hvc = adv > 100.0
        ivc = (adv >= 40.0) & (adv <= 100.0)
        vr = sub["v_r"].values
        tot = w.sum()
        f_hvc = w[hvc].sum() / tot if tot > 0 else np.nan
        f_ivc = w[ivc].sum() / tot if tot > 0 else np.nan
        f_in = w[hvc & (vr < 0)].sum() / w[hvc].sum() if w[hvc].sum() > 0 else np.nan
        rows.append(dict(ion=ion, n_cells=len(sub), f_HVC=f_hvc, f_IVC=f_ivc,
                         f_inflow_HVC=f_in))
    return pd.DataFrame(rows).set_index("ion")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    ions = S.ION_KEYS  # HI..OVI ionization sequence
    clean = df[~(df.wrapped | df.hypervel)]
    n_flag = int((df.wrapped | df.hypervel).sum())
    print(f"Total cells {len(df)}; flagged {n_flag} ({100*n_flag/len(df):.3f}%) "
          f"[wrapped {int(df.wrapped.sum())}, hypervel {int(df.hypervel.sum())}]")

    A = frac_table(df, ions)
    C = frac_table(clean, ions)
    comp = A.join(C, lsuffix="_all", rsuffix="_clean")
    comp["dHVC_pct"] = 100 * (comp.f_HVC_all - comp.f_HVC_clean)
    comp["dInflow_pct"] = 100 * (comp.f_inflow_HVC_all - comp.f_inflow_HVC_clean)
    comp.to_csv(OUT / "contamination_by_ion.csv")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n         f_HVC(all) f_HVC(clean)  dHVC[pp]   f_inflowHVC(all/clean)  dInflow[pp]")
    for ion in ions:
        r = comp.loc[ion]
        print(f"  {ion:5s}   {r.f_HVC_all:7.4f}   {r.f_HVC_clean:7.4f}   {r.dHVC_pct:+6.2f}   "
              f"{r.f_inflow_HVC_all:6.3f}/{r.f_inflow_HVC_clean:6.3f}      {r.dInflow_pct:+6.2f}")
    worst_hvc = comp.dHVC_pct.abs().max()
    worst_in = comp.dInflow_pct.abs().max()
    print(f"\nMax |ΔHVC| = {worst_hvc:.2f} pp ; Max |Δinflow| = {worst_in:.2f} pp")
    verdict = "STABLE (<2pp) -> mask is cosmetic but applied for provenance" if (
        worst_hvc < 2 and worst_in < 2) else "MATTERS (>2pp) -> mask is required"
    print("VERDICT:", verdict)
    (OUT / "verdict.txt").write_text(
        f"flagged {n_flag}/{len(df)} ({100*n_flag/len(df):.3f}%); "
        f"max dHVC {worst_hvc:.2f}pp, max dInflow {worst_in:.2f}pp -> {verdict}\n")


if __name__ == "__main__":
    main()
