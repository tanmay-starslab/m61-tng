#!/usr/bin/env python3
"""Flatten the per-spectrum pyGad Voigt fits into one component catalog.

Reads, for every (sid, mode, alpha) and every one of the 9 fitted transitions, the
individual-line fit HDF5 written by the spectral-fitting stage, and emits:

  voigt_catalog/voigt_components_sid<SID>.parquet   one row per fitted COMPONENT
  voigt_catalog/voigt_linestatus_sid<SID>.parquet   one row per (sightline, line) fit

then (with --combine) merges them into voigt_components.parquet / voigt_line_status.parquet.

Velocity convention (VERIFIED against the Si II 1260 flux minimum, see m61_voigt docstring):
    v_rest = -v_kms - v_sys            # same convention as v_ISM and the ray velocities
    dv_v3a = v_rest - v_ism_v3a        # kinematic offset from the disk ISM
    dv_v3b = v_rest - v_ism_v3b
pyGad's own `dv_kms` is the UNCERTAINTY on v_kms and is stored as `v_err_kms`.

Usage:
    python build_voigt_catalog.py <sid>
    python build_voigt_catalog.py --combine
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
import h5py  # noqa: E402
import m61_voigt as V  # noqa: E402

C_KMS = 299792.458

OUT = V.VOIGT_DIR

# fields kept from result_fieldnames (pyGad names on the left)
NUM_FIELDS = ["EW_mA", "dEW_mA", "logN", "dlogN", "b_kms", "db_kms", "v_kms", "dv_kms",
              "lambda_A", "Chisq", "Nregions",
              # fit CONFIGURATION -- differs per line and is essential for fair comparison
              "fit_velocity_window_kms", "fit_b_max_kms", "fit_b_min_kms", "fit_max_lines",
              "fit_snr", "fit_logN_min", "fit_logN_max"]
BOOL_FIELDS = ["UpLim", "Sat"]

# The narrowest velocity window used by any metal line. H I 1216 was fit over +-1200 km/s
# while the metals used +-800, so raw cross-line comparisons at large |dv| are unfair;
# `beyond_common_window` marks components outside the common +-500 km/s region.
COMMON_WINDOW = 500.0


def _dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def _tobool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def _tofloat(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_one(sid, mode, alpha):
    """-> (list of component dicts, list of line-status dicts) for one sightline."""
    p = V.fit_h5_path(sid, mode, alpha)
    if not p.exists():
        return [], [{"sid": sid, "mode": mode, "alpha": alpha, "line": L.key, "ion": L.ion,
                     "fit_status": "missing_file", "non_detection": True,
                     "n_components": 0, "n_regions": 0} for L in V.LINES]
    comps, stats = [], []
    with h5py.File(p, "r") as h:
        fn = _dec(h["result_fieldnames"][()])
        # ---- per-line status (the covering-fraction denominator) ----
        sfn = _dec(h["line_status_fieldnames"][()])
        for row in h["line_status_rows"][()]:
            r = dict(zip(sfn, _dec(row)))
            key = r.get("line_group")
            if key not in V.LINE_BY_KEY:
                continue
            stats.append({"sid": sid, "mode": mode, "alpha": alpha, "line": key,
                          "ion": V.LINE_BY_KEY[key].ion,
                          "fit_status": r.get("fit_status", ""),
                          "non_detection": _tobool(r.get("non_detection", "True")),
                          "n_components": int(_tofloat(r.get("n_components", 0)) or 0),
                          "n_regions": int(_tofloat(r.get("n_regions", 0)) or 0)})
        # ---- components ----
        if "lines" in h:
            for key in h["lines"]:
                if key not in V.LINE_BY_KEY:
                    continue
                g = h[f"lines/{key}"]
                if "fit_rows" not in g:
                    continue
                rows = g["fit_rows"][()]
                for row in rows:
                    r = dict(zip(fn, _dec(row)))
                    d = {"sid": sid, "mode": mode, "alpha": alpha, "line": key,
                         "ion": V.LINE_BY_KEY[key].ion,
                         "fit_status": r.get("fit_status", "")}
                    for f in NUM_FIELDS:
                        d[f] = _tofloat(r.get(f))
                    for f in BOOL_FIELDS:
                        d[f.lower()] = _tobool(r.get(f))
                    comps.append(d)
    return comps, stats


def build_sid(sid):
    m = V.load_master_v3()
    m = m[m.sid == sid].set_index(["mode", "alpha_deg"])
    if m.empty:
        raise SystemExit(f"sid {sid} not in the v3 master")
    comps, stats = [], []
    n_missing = 0
    for mode in ("flip", "noflip"):
        for alpha in range(360):
            if (mode, alpha) not in m.index:
                continue
            c, s = read_one(sid, mode, alpha)
            if not c and s and s[0]["fit_status"] == "missing_file":
                n_missing += 1
            comps.extend(c)
            stats.extend(s)
    C = pd.DataFrame(comps)
    S = pd.DataFrame(stats)
    if C.empty:
        raise SystemExit(f"sid {sid}: no components read")

    # ---- attach v_sys / v_ISM and build the kinematic offsets ----
    vm = (V.load_master_v3().query("sid == @sid")[
        ["mode", "alpha_deg", "v_sys", "v_ism_v3a", "v_ism_v3b",
         "v_ism_direct_cool", "v_ism_v2", "rho_kpc",
         "v3a_edge_fallback", "v3b_edge_fallback", "in_disk_v1"]]
        .rename(columns={"alpha_deg": "alpha"}))
    C = C.merge(vm, on=["mode", "alpha"], how="left")

    # pyGad dv_kms is the UNCERTAINTY on v_kms -- rename so nothing can confuse it with dv
    C = C.rename(columns={"dv_kms": "v_err_kms", "v_kms": "v_obs_kms"})

    # ---- rest-wavelength zero-point correction (REQUIRED) ------------------------
    # pyGad's line list does not always agree with the wavelengths Trident used to
    # synthesise the spectra. Recover the rest wavelength pyGad actually assumed,
    # lam0_pygad = lambda_A / (1 + v_obs/c), and correct the velocity onto Trident's
    # scale. Verified 2026-08-03: eight of the nine transitions agree to 0.0000 A, but
    # Si II 1260 was fit at 1260.5220 A against a spectrum synthesised at 1260.4220 A --
    # a 0.1000 A / +23.76 km/s systematic that reddens EVERY Si II 1260 component.
    # Uncorrected, its median dv is +24.67 km/s vs +0.39..+1.20 for all other lines;
    # corrected it is +0.91. Doing this per row makes the fix self-applying if any other
    # line list ever drifts.
    lam0_trident = C["line"].map({L.key: L.lam0 for L in V.LINES})
    C["lam0_pygad"] = C["lambda_A"] / (1.0 + C["v_obs_kms"] / C_KMS)
    C["v_zeropoint_kms"] = C_KMS * (C["lam0_pygad"] - lam0_trident) / lam0_trident
    # round to the nearest 0.01 km/s so per-row float noise does not leak into velocities
    C["v_zeropoint_kms"] = C["v_zeropoint_kms"].round(2)
    C["v_rest"] = -(C["v_obs_kms"] + C["v_zeropoint_kms"]) - C["v_sys"]
    C["dv_v3a"] = C["v_rest"] - C["v_ism_v3a"]
    C["dv_v3b"] = C["v_rest"] - C["v_ism_v3b"]
    C["dv_v1"] = C["v_rest"] - C["v_ism_direct_cool"]

    # ---- quality flags (see the caveats in m61_voigt's docstring) ----
    # b pinned at the fitter's upper bound: the component is absorbing a broad/damped
    # profile rather than measuring a real cloud. Overwhelmingly an H I 1216 problem
    # (~59% of its components), essentially absent in the metals.
    C["b_at_ceiling"] = C["b_kms"] >= 0.999 * C["fit_b_max_kms"]
    # logN pinned at the fitter's upper bound -- the column is the BOUND, not a measurement.
    # Overwhelmingly H I again (57.5% of its clean components sit within 0.05 dex of the
    # 19.52 bound, median 19.485), so any H I logN distribution or logN_1/2 is the bound.
    # Metals are <=2.8%. Flagged, not dropped: a saturated damped line genuinely has a high
    # column, we simply cannot measure how high.
    C["logN_at_ceiling"] = C["logN"] >= C["fit_logN_max"] - 0.05
    # outside the velocity range common to all lines -- unfair for cross-line comparison.
    # NOTE min() over the two variants, so a component is flagged only if BOTH exceed the
    # window. That makes the clean sample IDENTICAL for v3a and v3b (so the two variants are
    # directly comparable, which is the point of fig21), at the cost of the v3a figures using
    # a cut partly defined by v3b. ~0.06% of components sit at |dv|>500 in one variant only.
    C["beyond_common_window"] = C[["dv_v3a", "dv_v3b"]].abs().min(axis=1) > COMMON_WINDOW

    OUT.mkdir(parents=True, exist_ok=True)
    C.to_parquet(OUT / f"voigt_components_sid{sid}.parquet", index=False)
    S.to_parquet(OUT / f"voigt_linestatus_sid{sid}.parquet", index=False)

    det = C[~C.uplim]
    print(f"[SID {sid}] {len(C)} rows ({len(det)} detected, {int(C.uplim.sum())} upper limits), "
          f"{S.sid.count()} line-fits, {n_missing} missing files")
    for L in V.SIII_LINES:
        d = det[det.line == L.key]
        nsl = S[(S.line == L.key) & (S.fit_status.isin(V.FIT_RAN))][V.SLKEY].drop_duplicates()
        hv = d[d.dv_v3b.abs() >= V.HVC][V.SLKEY].drop_duplicates()
        print(f"    {L.key:12s} n_fit={len(nsl):4d}  HVC f_c(v3b)={len(hv)/max(len(nsl),1):.3f}")
    return C, S


def combine():
    cf = sorted(OUT.glob("voigt_components_sid*.parquet"))
    sf = sorted(OUT.glob("voigt_linestatus_sid*.parquet"))
    if not cf:
        raise SystemExit("nothing to combine")
    C = pd.concat([pd.read_parquet(f) for f in cf], ignore_index=True)
    S = pd.concat([pd.read_parquet(f) for f in sf], ignore_index=True)
    C.to_parquet(OUT / "voigt_components.parquet", index=False)
    S.to_parquet(OUT / "voigt_line_status.parquet", index=False)
    print(f"combined {len(cf)} SIDs -> {len(C)} components, {len(S)} line-fits")
    print(f"  detected components: {int((~C.uplim).sum())}  upper limits: {int(C.uplim.sum())}")
    print(f"  sightlines: {C[V.SLKEY].drop_duplicates().shape[0]}")
    print("\nper-line fit status:")
    print(S.groupby(["line", "fit_status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "--combine":
        combine()
    else:
        build_sid(int(sys.argv[1]))
