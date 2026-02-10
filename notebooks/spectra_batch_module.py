#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spectra_batch_module.py  (SAFE/RUN-FIRST VERSION)

Goal:
- Generate Trident spectra reliably with ZERO uncaught exceptions.
- Do NOT import matplotlib.
- Do NOT touch ray.r / ires / icoords / off-axis ops (this is what was blowing up).
- Save spectra (lambda, flux, tau) to HDF5 for each processed row.
- Everything else (columns, ray field dumps, combined aggregator) is best-effort and never fatal.

Expected directory layout (same as before):
  <ORIENT_OUT_BASE>/sid<SID>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN_LABEL>/rays_sid<SID>.csv

Public API (kept for run_spectra_one_sid.py):
  - JobPaths, JobParams, SpectraConfig
  - run_all_runs_for_sid(paths, params, cfg)

"""

from __future__ import annotations

import os
import json
import time
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, List, Sequence, Dict, Any

import numpy as np
import pandas as pd
import h5py

import yt
import trident


# -------------------------
# Config dataclasses
# -------------------------

@dataclass
class JobPaths:
    cutout_h5: str
    rays_base: str      # <ORIENT_OUT_BASE>/sid<SID>
    output_base: str    # where to write spectra outputs (can be same as rays_base)


@dataclass
class JobParams:
    sid: int
    snap: int = 99
    run_labels: List[str] = None
    filter_mode: Optional[str] = None   # "noflip" | "flip" | None
    alpha_keep: Optional[List[int]] = None
    sightline_ids: Optional[List[str]] = None
    max_rays: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        if self.run_labels is None:
            self.run_labels = ["L3Rvir", "L4Rvir"]


@dataclass
class SpectraConfig:
    lines: List[str] = None
    instrument: str = "COS-G130M"
    make_plots: bool = False  # ignored here; kept for API compatibility

    def __post_init__(self):
        if self.lines is None:
            self.lines = ["H I 1216"]


# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _field_exists(ds, f) -> bool:
    try:
        return (f in ds.field_list) or (f in ds.derived_field_list)
    except Exception:
        try:
            return f in ds.field_list
        except Exception:
            return False

def ensure_metallicity_field(ds) -> None:
    """
    If ("gas","metallicity") missing, alias from known TNG fields if present.
    Never fatal.
    """
    try:
        if _field_exists(ds, ("gas", "metallicity")):
            return
        candidates = [
            ("gas", "GFM_Metallicity"),
            ("gas", "Metallicity"),
        ]
        src = None
        for c in candidates:
            if _field_exists(ds, c):
                src = c
                break
        if src is None:
            return

        def _metallicity_alias(field, data):
            return data[src]

        ds.add_field(
            ("gas", "metallicity"),
            function=_metallicity_alias,
            sampling_type="cell",
            units="dimensionless",
            force_override=True,
        )
    except Exception:
        # never fatal
        return

def ions_from_lines(lines: Sequence[str]) -> List[str]:
    ions = []
    for s in lines:
        toks = str(s).strip().split()
        if len(toks) >= 2:
            ions.append(f"{toks[0]} {toks[1]}")
    out, seen = [], set()
    for x in ions:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def add_ions(ds, ions: Sequence[str], verbose: bool = True) -> None:
    """
    Best-effort: never fatal.
    """
    if not ions:
        return
    try:
        trident.add_ion_fields(ds, ions=list(ions))
    except Exception as e:
        if verbose:
            print(f"[WARN] add_ion_fields failed (non-fatal): {e}")

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _safe_str(x, default="") -> str:
    try:
        s = str(x)
        return s if s else default
    except Exception:
        return default


# -------------------------
# Ray + spectrum helpers
# -------------------------

def make_ray(ds, p0_abs, p1_abs, data_filename: str, solution_filename: str, verbose: bool = True):
    """
    Create ray reliably:
    - Provide a conservative fields list, but only include fields that exist on ds.
    - Never touches ray.r. Only uses the returned ray for SpectrumGenerator.
    """
    p0 = ds.arr(np.asarray(p0_abs, float), "code_length")
    p1 = ds.arr(np.asarray(p1_abs, float), "code_length")

    # Conservative fields that help Trident derive things; filtered by availability.
    candidate_fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity"),
        ("gas", "velocity_los"),
        ("gas", "H_nuclei_density"),
        ("gas", "H_number_density"),
    ]
    fields = [f for f in candidate_fields if _field_exists(ds, f)]

    if verbose:
        print(f"[RAY] fields sampled: {fields if fields else 'DEFAULTS'}")

    try:
        ray = trident.make_simple_ray(
            ds,
            start_position=p0,
            end_position=p1,
            fields=fields if fields else None,
            data_filename=data_filename,
            solution_filename=solution_filename,
        )
        return ray
    except Exception as e:
        # If fields list triggers weirdness on some datasets, retry with defaults.
        if verbose:
            print(f"[WARN] make_simple_ray with fields failed; retrying defaults: {e}")
        ray = trident.make_simple_ray(
            ds,
            start_position=p0,
            end_position=p1,
            data_filename=data_filename,
            solution_filename=solution_filename,
        )
        return ray

def build_spectrum(ray, lines: Sequence[str], instr: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Return arrays (lambda_A, flux, tau).
    Never fatal to caller: caller catches.
    """
    sg = trident.SpectrumGenerator(instr)
    sg.make_spectrum(ray, lines=list(lines))
    # apply LSF when available; safe
    try:
        sg.apply_lsf()
    except Exception:
        pass

    lam = np.asarray(sg.lambda_field, dtype=float)
    flux = np.asarray(sg.flux_field, dtype=float)
    tau = np.asarray(sg.tau_field, dtype=float)

    if verbose:
        print(f"[SPEC] {instr}  Npix={lam.size}  lam=[{lam.min():.2f},{lam.max():.2f}] A")

    return {"lambda_A": lam, "flux": flux, "tau": tau}

def save_spectrum_h5(out_path: str, meta: Dict[str, Any], spec: Dict[str, np.ndarray]) -> None:
    """
    Atomic write: write temp then rename. Never raises to caller if disk hiccup: caller catches.
    """
    ensure_dir(os.path.dirname(out_path))
    tmp = out_path + f".tmp.{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        gmeta = f.create_group("meta")
        for k, v in meta.items():
            try:
                gmeta.attrs[k] = v
            except TypeError:
                gmeta.attrs[k] = json.dumps(v)

        gs = f.create_group("spectrum")
        gs.create_dataset("lambda_A", data=np.asarray(spec["lambda_A"], dtype=float))
        gs.create_dataset("flux", data=np.asarray(spec["flux"], dtype=float))
        gs.create_dataset("tau", data=np.asarray(spec["tau"], dtype=float))

    os.replace(tmp, out_path)


# -------------------------
# Core processing
# -------------------------

def process_run_for_sid(ds, sid: int, snap: int, run_label: str, paths: JobPaths, cfg: SpectraConfig, params: JobParams):
    rays_csv = os.path.join(
        paths.rays_base,
        f"rays_and_recipes_sid{sid}_snap{snap}_{run_label}",
        f"rays_sid{sid}.csv",
    )

    job_root = os.path.join(paths.output_base, f"rays_and_spectra_sid{sid}_snap{snap}_{run_label}")
    out_dir_base = os.path.join(job_root, "spectra_h5")
    logs_dir = os.path.join(job_root, "logs")
    ensure_dir(out_dir_base)
    ensure_dir(logs_dir)

    if not os.path.isfile(rays_csv):
        # hard fail is fine here because it indicates wrong inputs; keep it explicit
        raise FileNotFoundError(f"Missing rays CSV: {rays_csv}")

    df = pd.read_csv(rays_csv)

    if params.filter_mode is not None and "mode" in df.columns:
        df = df[df["mode"].astype(str).str.lower() == params.filter_mode.lower()]

    if params.alpha_keep is not None and "alpha_deg" in df.columns:
        keep = set(int(a) for a in params.alpha_keep)
        # tolerate floats/strings
        df = df[df["alpha_deg"].apply(lambda x: int(round(_safe_float(x, np.nan))) if np.isfinite(_safe_float(x, np.nan)) else -999999).isin(keep)]

    if params.sightline_ids is not None and "sightline_id" in df.columns:
        keep = set(str(s) for s in params.sightline_ids)
        df = df[df["sightline_id"].astype(str).isin(keep)]

    if params.max_rays is not None:
        df = df.head(int(params.max_rays))

    if df.empty:
        if params.verbose:
            print(f"[WARN] {run_label}: no rows after filtering. Skipping.")
        return

    # scratch per process (avoid collisions)
    ray_scratch_dir = os.environ.get("SLURM_TMPDIR") or os.path.join(paths.output_base, "._tmp_trident")
    ensure_dir(ray_scratch_dir)
    rayfile = os.path.join(ray_scratch_dir, f"ray_sid{sid}.h5")
    trajfile = os.path.join(ray_scratch_dir, f"traj_sid{sid}.txt")

    summary = []
    n_ok = 0
    n_fail = 0

    for idx, row in df.iterrows():
        # Everything per-row is fully sandboxed: never let an exception escape.
        try:
            mode = _safe_str(row.get("mode", "unknown"), "unknown")
            alpha = _safe_float(row.get("alpha_deg", np.nan), np.nan)
            alpha_tag = f"{int(round(alpha))}" if np.isfinite(alpha) else "NA"
            sightline_id = _safe_str(row.get("sightline_id", f"row{idx}"), f"row{idx}")

            # endpoints: must exist
            p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], dtype=float)
            p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], dtype=float)

            tag = f"{run_label}_sid{sid}_{sightline_id}_{mode}_alpha{alpha_tag}"

            # clean scratch files
            for p in (rayfile, trajfile):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

            if params.verbose:
                print(f"[{run_label}] ({idx+1}/{len(df)}) {tag}")

            ray = make_ray(ds, p0, p1, data_filename=rayfile, solution_filename=trajfile, verbose=params.verbose)
            spec = build_spectrum(ray, cfg.lines, instr=cfg.instrument, verbose=params.verbose)

            # metadata for spectrum file
            meta = dict(
                RUN_LABEL=str(run_label),
                SubhaloID=int(sid),
                SNAP=int(snap),
                sightline_id=str(sightline_id),
                mode=str(mode),
                alpha_deg=float(alpha) if np.isfinite(alpha) else np.nan,
                lines=list(cfg.lines),
                instrument=str(cfg.instrument),
                start_ckpch=p0.tolist(),
                end_ckpch=p1.tolist(),
            )

            out_path = os.path.join(out_dir_base, f"{tag}_spectrum.h5")
            save_spectrum_h5(out_path, meta, spec)

            summary.append(dict(tag=tag, spectrum_h5=out_path, ok=True))
            n_ok += 1

        except Exception as e:
            n_fail += 1
            # log only; do not raise
            msg = f"[FAIL] {run_label} row={idx} sid={sid}: {type(e).__name__}: {e}"
            print(msg)
            try:
                with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                    f.write(msg + "\n")
                    f.write(traceback.format_exc() + "\n")
            except Exception:
                pass
            summary.append(dict(tag=f"row{idx}", spectrum_h5="", ok=False, error=str(e)))

    # summary csv never fatal
    try:
        pd.DataFrame(summary).to_csv(os.path.join(job_root, "summary_spectra.csv"), index=False)
    except Exception:
        pass

    if params.verbose:
        print(f"[DONE] {run_label}: ok={n_ok} fail={n_fail}")
        print(f"[OUT ] {out_dir_base}")
        if n_fail > 0:
            print(f"[LOG ] {os.path.join(logs_dir, 'errors.txt')}")


def run_all_runs_for_sid(paths: JobPaths, params: JobParams, cfg: SpectraConfig):
    """
    Load dataset once, define metallicity alias if needed, add ions best-effort,
    then process each run label. No uncaught exceptions except missing cutout/CSV.
    """
    if params.verbose:
        print("[PATHS]", asdict(paths))
        print("[PARAM]", asdict(params))
        print("[CFG  ]", asdict(cfg))

    if not os.path.isfile(paths.cutout_h5):
        raise FileNotFoundError(f"cutout_h5 not found: {paths.cutout_h5}")

    ds = yt.load(paths.cutout_h5)

    ensure_metallicity_field(ds)

    ions_needed = ions_from_lines(cfg.lines)
    add_ions(ds, ions_needed, verbose=params.verbose)

    for run_label in params.run_labels:
        process_run_for_sid(ds, int(params.sid), int(params.snap), str(run_label), paths, cfg, params)


# -------------------------
# CLI (optional)
# -------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate spectra for rays from orient_m61 outputs (robust run-first version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cutout_h5", required=True)
    p.add_argument("--rays_base", required=True)
    p.add_argument("--output_base", required=True)

    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--run_labels", default="L3Rvir")
    p.add_argument("--filter_mode", choices=["noflip", "flip"], default=None)

    p.add_argument("--alpha_keep", default="", help="Comma-separated alpha list, e.g. '0,90'")
    p.add_argument("--sightline_ids", default="", help="Comma-separated IDs")
    p.add_argument("--max_rays", type=int, default=0)

    p.add_argument("--lines", default="H I 1216")
    p.add_argument("--instrument", default="COS-G130M")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def _main_cli(argv=None):
    a = _parse_args(argv)

    alpha_keep = None
    if a.alpha_keep.strip():
        alpha_keep = [int(x) for x in a.alpha_keep.split(",") if x.strip()]

    sightline_ids = None
    if a.sightline_ids.strip():
        sightline_ids = [x.strip() for x in a.sightline_ids.split(",") if x.strip()]

    paths = JobPaths(
        cutout_h5=a.cutout_h5,
        rays_base=a.rays_base,
        output_base=a.output_base,
    )

    params = JobParams(
        sid=int(a.sid),
        snap=int(a.snap),
        run_labels=[s.strip() for s in a.run_labels.split(",") if s.strip()],
        filter_mode=a.filter_mode,
        alpha_keep=alpha_keep,
        sightline_ids=sightline_ids,
        max_rays=(int(a.max_rays) if a.max_rays and int(a.max_rays) > 0 else None),
        verbose=bool(a.verbose),
    )

    cfg = SpectraConfig(
        lines=[s.strip() for s in a.lines.split(",") if s.strip()],
        instrument=str(a.instrument),
        make_plots=False,
    )

    run_all_runs_for_sid(paths, params, cfg)

if __name__ == "__main__":
    _main_cli()