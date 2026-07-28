#!/usr/bin/env python3
"""Publication figures v2. Differences from make_paper_figures.py:
  * dv recomputed from the supervisor-model v_ISM (vism_tables_v2);
  * ONLY the ions that actually appear in the COS-G130M mock spectra are shown --
    H I, O I, C II, Si II, Si III, Si IV, N V (O VI / C IV / Mg II / Fe II removed, since
    they have no spectrum in the 1150-1450 A band);
  * legends/tick-labels carry the transition wavelength, not just the ion name;
  * figure titles (suptitles) removed for publication.
Output: paper_figures_v2/  (see FIGURES_V2.md for per-figure descriptions).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure as _mfig

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import m61_style as S  # noqa: E402
S.FIGDIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures_v2")

# ---- in-spectra ion set, ionization/neutral->high order, labelled with the transition ----
_SEQ = [("HI",   r"$\mathrm{H\,I}\ \lambda1216$"),
        ("OI",   r"$\mathrm{O\,I}\ \lambda1302$"),
        ("CII",  r"$\mathrm{C\,II}\ \lambda1335$"),
        ("SiII", r"$\mathrm{Si\,II}\ \lambda1260$"),
        ("SiIII", r"$\mathrm{Si\,III}\ \lambda1206$"),
        ("SiIV", r"$\mathrm{Si\,IV}\ \lambda1403$"),
        ("NV",   r"$\mathrm{N\,V}\ \lambda1239$")]
_TURBO = plt.get_cmap("turbo")
S.IONS = [(k, l, mcolors.to_hex(_TURBO(0.06 + 0.88 * i / (len(_SEQ) - 1))))
          for i, (k, l) in enumerate(_SEQ)]
S.ION_KEYS = [k for k, _, _ in S.IONS]
S.ION_LAB = {k: l for k, l, _ in S.IONS}
S.ION_COL = {k: c for k, _, c in S.IONS}

import make_paper_figures as MP  # noqa: E402
# in-band ionization-ratio panels for fig7 (no C IV in the spectra)
MP.RATIO_PAIRS = [("SiIV", "SiII", r"$\log_{10}\,N_{\mathrm{Si\,IV}}/N_{\mathrm{Si\,II}}$"),
                  ("SiIII", "SiII", r"$\log_{10}\,N_{\mathrm{Si\,III}}/N_{\mathrm{Si\,II}}$")]
# strip publication-figure titles
_mfig.Figure.suptitle = lambda self, *a, **k: None

V2 = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables_v2/vism_master_v2.csv")
_orig_load = MP.load


def load_v2():
    cells, n_sl = _orig_load()   # already drops wrapped/hypervel
    v2 = (pd.read_csv(V2)[["sid", "mode", "alpha_deg", "v_ism_model"]]
          .rename(columns={"alpha_deg": "alpha"}))
    cells = cells.merge(v2, on=["sid", "mode", "alpha"], how="left")
    cells["v_ISM"] = cells["v_ism_model"]
    cells["dv"] = cells["v_rest"] - cells["v_ism_model"]
    cells = cells[cells["dv"].notna()].reset_index(drop=True)
    print(f"[v2] dv from supervisor-model v_ISM; ions = {S.ION_KEYS} -> {len(cells)} cells")
    return cells, n_sl


MP.load = load_v2

if __name__ == "__main__":
    MP.main()
