#!/usr/bin/env python3
"""Publication figures v2: identical to make_paper_figures.py but with dv recomputed from the
supervisor-model v_ISM (vism_tables_v2). The absorbing-gas catalog (ion columns, kinematics,
T, Z) is unchanged -- only v_ISM and dv = v_rest - v_ISM change -- so we reuse every figure
function and just remap dv, routing output to paper_figures_v2/.

Usage: python make_paper_figures_v2.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402
S.FIGDIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures_v2")
import make_paper_figures as MP  # noqa: E402

V2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
_orig_load = MP.load


def load_v2():
    cells, n_sl = _orig_load()   # already drops wrapped/hypervel
    v2 = (pd.read_csv(V2)[["sid", "mode", "alpha_deg", "v_ism_model", "in_disk_model"]]
          .rename(columns={"alpha_deg": "alpha"}))
    cells = cells.merge(v2, on=["sid", "mode", "alpha"], how="left")
    cells["v_ISM"] = cells["v_ism_model"]
    cells["dv"] = cells["v_rest"] - cells["v_ism_model"]
    cells = cells[cells["dv"].notna()].reset_index(drop=True)
    print(f"[v2] remapped dv with supervisor-model v_ISM -> {len(cells)} cells")
    return cells, n_sl


MP.load = load_v2

if __name__ == "__main__":
    MP.main()
