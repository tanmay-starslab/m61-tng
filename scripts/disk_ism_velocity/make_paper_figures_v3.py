#!/usr/bin/env python3
"""Publication figures for a v3 variant. Same in-band ion set / wavelength labels / no-title
style as v2, but dv is recomputed from the chosen per-orientation v3 velocity.

Usage: python make_paper_figures_v3.py <v3a|v3b>
  v3a = 3-tracer along the actual sightline (lands on the absorption)
  v3b = center->impact probe-line (literal request; custom but coarse)
Output: paper_figures_<variant>/
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure as _mfig

variant = sys.argv[1] if len(sys.argv) > 1 else "v3a"
assert variant in ("v3a", "v3b")
VCOL = f"v_ism_{variant}"

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402
S.FIGDIR = Path(f"/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures_{variant}")

_SEQ = [("HI", r"$\mathrm{H\,I}\ \lambda1216$"), ("OI", r"$\mathrm{O\,I}\ \lambda1302$"),
        ("CII", r"$\mathrm{C\,II}\ \lambda1335$"), ("SiII", r"$\mathrm{Si\,II}\ \lambda1260$"),
        ("SiIII", r"$\mathrm{Si\,III}\ \lambda1206$"), ("SiIV", r"$\mathrm{Si\,IV}\ \lambda1403$"),
        ("NV", r"$\mathrm{N\,V}\ \lambda1239$")]
_T = plt.get_cmap("turbo")
S.IONS = [(k, l, mcolors.to_hex(_T(0.06 + 0.88 * i / (len(_SEQ) - 1)))) for i, (k, l) in enumerate(_SEQ)]
S.ION_KEYS = [k for k, _, _ in S.IONS]
S.ION_LAB = {k: l for k, l, _ in S.IONS}
S.ION_COL = {k: c for k, _, c in S.IONS}

import make_paper_figures as MP  # noqa: E402
MP.RATIO_PAIRS = [("SiIV", "SiII", r"$\log_{10}\,N_{\mathrm{Si\,IV}}/N_{\mathrm{Si\,II}}$"),
                  ("SiIII", "SiII", r"$\log_{10}\,N_{\mathrm{Si\,III}}/N_{\mathrm{Si\,II}}$")]
_mfig.Figure.suptitle = lambda self, *a, **k: None

V3 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v3/vism_master_v3.csv")
_orig_load = MP.load


def load_v3():
    cells, n_sl = _orig_load()
    v3 = (pd.read_csv(V3)[["sid", "mode", "alpha_deg", VCOL]]
          .rename(columns={"alpha_deg": "alpha"}))
    cells = cells.merge(v3, on=["sid", "mode", "alpha"], how="left")
    cells["v_ISM"] = cells[VCOL]
    cells["dv"] = cells["v_rest"] - cells[VCOL]
    cells = cells[cells["dv"].notna()].reset_index(drop=True)
    print(f"[{variant}] dv from {VCOL}; ions = {S.ION_KEYS} -> {len(cells)} cells")
    return cells, n_sl


MP.load = load_v3

if __name__ == "__main__":
    MP.main()
