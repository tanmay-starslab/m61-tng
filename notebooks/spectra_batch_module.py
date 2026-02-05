#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spectra_batch_module.py

Generate Trident spectra for rays precomputed by orient_m61.py.

Assumes orient_m61 outputs live under:
  <ORIENT_OUT_BASE>/sid<SID>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN_LABEL>/rays_sid<SID>.csv

This module reads those rays CSVs, makes a trident ray, computes column integrals,
generates spectra for selected lines, and writes:
  - per-ray bundle HDF5
  - combined HDF5 (all rays for a run_label)

Key knobs for sanity checks:
  - filter_mode: only "noflip" or only "flip"
  - alpha_keep: only specific alpha_deg values (e.g. [0])
  - sightline_ids: only specific sightlines
  - max_rays: hard cap number of rows processed
  - make_plots: skip matplotlib entirely if False
"""

import os
import json
import time
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, List

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
    rays_base: str      # should be: <ORIENT_OUT_BASE>/sid<SID>
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
    zooms_A: Optional[List[float]] = None
    zoom_half_A: float = 3.0
    make_plots: bool = False

    def __post_init__(self):
        if self.lines is None:
            self.lines = ["H I 1216", "C II 1335", "Si III 1206"]
        if self.zooms_A is None:
            self.zooms_A = [25.0, 10.0, 5.0]


# Backward-compat alias (your wrapper was importing RunConfig)
RunConfig = SpectraConfig


# -------------------------
# Dataset preparation
# -------------------------

def add_ions(ds):
    # Safe to call multiple times; Trident handles duplicates internally in most cases.
    trident.add_ion_fields(ds, ions=["H I", "C II", "Si III"])


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Ray + spectrum helpers
# -------------------------

def make_ray(ds, p0_ckpch, p1_ckpch, data_filename, solution_filename):
    """
    p0/p1 are in code_length (TNG ckpc/h) in absolute box coordinates.
    """
    return trident.make_simple_ray(
        ds,
        start_position=p0_ckpch,
        end_position=p1_ckpch,
        data_filename=data_filename,
        solution_filename=solution_filename,
    )


def build_spectrum(ray, lines, instr="COS-G130M"):
    sg = trident.SpectrumGenerator(instr)
    sg.make_spectrum(ray, lines=lines)
    return sg


def compute_columns(ray, p0, p1):
    """
    Compute simple LOS integrals for sanity checks.
    """
    cols = {}

    # Path length in kpc (approx, uses code_length assuming z~0 scaling; good enough sanity check)
    dl = np.asarray(ray.r["dl"]).astype(float)  # code_length
    cols["_dl_kind"] = "ray.r['dl']"
    cols["_sum_dl_code"] = float(np.nansum(dl))

    # Column densities: integrate number_density * dl
    # Trident provides number_density fields for ions after add_ion_fields.
    # We convert to cm^-2 by using yt units.
    def _col(field):
        try:
            nd = ray.r[field]  # 1/cm^3
            dl_cm = ray.r["dl"].to("cm")
            return float(np.nansum((nd * dl_cm)).to("cm**-2").value)
        except Exception:
            return np.nan

    cols["N_HI_cm2"]    = _col(("gas", "H_p0_number_density"))  # H I
    cols["N_CII_cm2"]   = _col(("gas", "C_p1_number_density"))  # C II
    cols["N_SiIII_cm2"] = _col(("gas", "Si_p2_number_density")) # Si III

    return cols


def save_spectrum_pngs(spec, out_dir, tag, zooms_A, zoom_half_A, make_plots: bool):
    if not make_plots:
        return

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt  # local import to avoid hard dependency when plotting disabled

    # Full spectrum plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spec.lambda_field, spec.flux_field)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Flux")
    ax.set_title(tag)
    fig.savefig(os.path.join(out_dir, f"{tag}_full.png"), dpi=120)
    plt.close(fig)

    # Zoomed plots around each line center (approx: use the line list wavelength)
    for z in zooms_A:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spec.lambda_field, spec.flux_field)
        ax.set_xlim(z - zoom_half_A, z + zoom_half_A)
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Flux")
        ax.set_title(f"{tag} | zoom {z:.1f}A")
        fig.savefig(os.path.join(out_dir, f"{tag}_zoom_{z:.1f}.png"), dpi=120)
        plt.close(fig)


# -------------------------
# Bundle writers
# -------------------------

def atomic_write_h5(path, write_fn):
    tmp = path + f".tmp.{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        write_fn(f)
    os.replace(tmp, path)


def is_valid_h5(path):
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def _write_ray_pack_into_group(g, meta, ray, spec, cols):
    # metadata
    mg = g.create_group("meta")
    for k, v in meta.items():
        try:
            mg.attrs[k] = v
        except TypeError:
            mg.attrs[k] = json.dumps(v)

    # columns
    cg = g.create_group("columns")
    for k, v in cols.items():
        try:
            cg.attrs[k] = v
        except TypeError:
            cg.attrs[k] = json.dumps(v)

    # ray fields (lightweight subset)
    rg = g.create_group("ray")
    rg.create_dataset("dl_code", data=np.asarray(ray.r["dl"]).astype(float))
    rg.create_dataset("x_code",  data=np.asarray(ray.r["x"]).astype(float))
    rg.create_dataset("y_code",  data=np.asarray(ray.r["y"]).astype(float))
    rg.create_dataset("z_code",  data=np.asarray(ray.r["z"]).astype(float))

    # spectrum arrays
    sg = g.create_group("spectrum")
    sg.create_dataset("lambda_A", data=np.asarray(spec.lambda_field).astype(float))
    sg.create_dataset("flux",     data=np.asarray(spec.flux_field).astype(float))


def save_bundle_hdf5(path, meta, ray, spec, cols):
    def _w(f):
        base = f.create_group("bundle")
        _write_ray_pack_into_group(base, meta, ray, spec, cols)
    atomic_write_h5(path, _w)


def append_to_combined(agg_path, group_path, meta, ray, spec, cols, globals_once, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            if (not os.path.exists(agg_path)) or (not is_valid_h5(agg_path)):
                def _init(f):
                    g = f.create_group("globals")
                    for k, v in globals_once.items():
                        try:
                            g.attrs[k] = v
                        except TypeError:
                            g.attrs[k] = json.dumps(v)
                atomic_write_h5(agg_path, _init)

            with h5py.File(agg_path, "a") as f:
                if "globals" not in f:
                    g = f.create_group("globals")
                    for k, v in globals_once.items():
                        try:
                            g.attrs[k] = v
                        except TypeError:
                            g.attrs[k] = json.dumps(v)

                if group_path in f:
                    del f[group_path]

                base = f.create_group(group_path)
                _write_ray_pack_into_group(base, meta, ray, spec, cols)

            return
        except OSError as e:
            time.sleep(0.5)
            if (not is_valid_h5(agg_path)) and attempt < max_retries:
                try:
                    os.remove(agg_path)
                except Exception:
                    pass
            if attempt == max_retries:
                raise RuntimeError(f"append_to_combined failed for {agg_path}: {e}")


# -------------------------
# Core processing
# -------------------------

def process_run_for_sid(ds, sid, snap, run_label, paths: JobPaths, cfg: SpectraConfig, params: JobParams):
    rays_csv = os.path.join(
        paths.rays_base,
        f"rays_and_recipes_sid{sid}_snap{snap}_{run_label}",
        f"rays_sid{sid}.csv",
    )

    job_root = os.path.join(paths.output_base, f"rays_and_spectra_sid{sid}_snap{snap}_{run_label}")
    rays_dir     = os.path.join(job_root, "rays")
    logs_dir     = os.path.join(job_root, "logs")
    combined_dir = os.path.join(job_root, "combined")
    combined_h5  = os.path.join(combined_dir, f"all_rays_{run_label}.h5")

    ensure_dir(job_root); ensure_dir(rays_dir); ensure_dir(logs_dir); ensure_dir(combined_dir)

    if not os.path.isfile(rays_csv):
        raise FileNotFoundError(f"Missing rays CSV: {rays_csv}")

    df = pd.read_csv(rays_csv)

    # filters
    if params.filter_mode is not None:
        df = df[df["mode"].astype(str).str.lower() == params.filter_mode.lower()]

    if params.alpha_keep is not None:
        df = df[df["alpha_deg"].astype(int).isin([int(a) for a in params.alpha_keep])]

    if params.sightline_ids is not None:
        keep = set([str(s) for s in params.sightline_ids])
        df = df[df["sightline_id"].astype(str).isin(keep)]

    if params.max_rays is not None:
        df = df.head(int(params.max_rays))

    if df.empty:
        raise RuntimeError("No rays to process after filtering.")

    globals_once = dict(
        SID=int(sid),
        SNAP=int(snap),
        instrument=str(cfg.instrument),
        lines=json.dumps(list(cfg.lines)),
    )

    ray_scratch_dir = os.environ.get("SLURM_TMPDIR") or os.path.join(paths.output_base, "_tmp_trident")
    ensure_dir(ray_scratch_dir)

    rayfile  = os.path.join(ray_scratch_dir, f"ray_sid{sid}.h5")
    trajfile = os.path.join(ray_scratch_dir, f"traj_sid{sid}.txt")

    for p in (rayfile, trajfile):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    summary_rows, errors = [], 0

    for i, row in df.iterrows():
        try:
            mode  = str(row.get("mode", "unknown"))
            alpha = float(row.get("alpha_deg", np.nan))
            alpha_tag = f"{int(round(alpha))}" if np.isfinite(alpha) else "NA"

            p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], float)
            p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], float)

            rho_kpc = float(row.get("rho_kpc", np.nan))
            phi_deg = float(row.get("phi_deg", np.nan))
            inc_deg = float(row.get("obs_inc_deg", np.nan))  # corrected key from orient_m61
            rvir_kpc = float(row.get("Rvir_kpc", np.nan))
            total_len_Rvir = float(row.get("total_len_Rvir", np.nan))
            sightline_id = str(row.get("sightline_id", "SL"))

            tag = f"{run_label}_sid{sid}_{sightline_id}_{mode}_alpha{alpha_tag}"

            out_dir = os.path.join(rays_dir, f"sightline={sightline_id}", f"mode={mode}", f"alpha={alpha_tag}")
            ensure_dir(out_dir)

            if params.verbose:
                print(f"[{run_label}] ({i+1}/{len(df)}) {tag}")

            for p in (rayfile, trajfile):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

            ray = make_ray(ds, p0, p1, data_filename=rayfile, solution_filename=trajfile)
            cols = compute_columns(ray, p0, p1)
            spec = build_spectrum(ray, cfg.lines, instr=cfg.instrument)

            save_spectrum_pngs(
                spec, out_dir, tag=tag,
                zooms_A=cfg.zooms_A, zoom_half_A=cfg.zoom_half_A,
                make_plots=cfg.make_plots
            )

            meta_save = dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid),
                sightline_id=sightline_id,
                mode=mode,
                alpha_deg=float(alpha),
                rho_kpc=rho_kpc,
                phi_deg=phi_deg,
                inc_deg=inc_deg,
                Rvir_kpc=rvir_kpc,
                total_len_Rvir=total_len_Rvir,
                start_ckpch=p0.tolist(),
                end_ckpch=p1.tolist(),
                lines=list(cfg.lines),
                instrument=str(cfg.instrument),
            )

            bundle_path = os.path.join(out_dir, f"{tag}_bundle.h5")
            save_bundle_hdf5(bundle_path, meta_save, ray, spec, cols)

            grp_path = f"rays/sightline={sightline_id}/mode={mode}/alpha={alpha_tag}/ray_{i:06d}"
            append_to_combined(combined_h5, grp_path, meta_save, ray, spec, cols, globals_once)

            srow = dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid),
                sightline_id=sightline_id,
                mode=mode,
                alpha_deg=float(alpha),
                rho_kpc=rho_kpc,
                phi_deg=phi_deg,
                inc_deg=inc_deg,
                Rvir_kpc=rvir_kpc,
                total_len_Rvir=total_len_Rvir,
                out_dir=out_dir,
                bundle_h5=bundle_path,
                N_HI_cm2=cols.get("N_HI_cm2"),
                N_CII_cm2=cols.get("N_CII_cm2"),
                N_SiIII_cm2=cols.get("N_SiIII_cm2"),
                combined_h5=combined_h5,
                group_path=grp_path,
            )
            summary_rows.append(srow)

        except Exception as e:
            errors += 1
            msg = f"[ERROR] {run_label} row={i} sid={sid}: {e}"
            print(msg)
            with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                f.write(msg + "\n")

    if summary_rows:
        master_csv = os.path.join(job_root, "summary_all_rays.csv")
        pd.DataFrame(summary_rows).to_csv(master_csv, index=False)
        if params.verbose:
            print(f"[OK] {run_label}: wrote {master_csv} rows={len(summary_rows)} errors={errors}")
            print(f"[OK] {run_label}: combined {combined_h5}")
    else:
        print(f"[WARN] {run_label}: no successful rays; errors={errors}")


def run_all_runs_for_sid(paths: JobPaths, params: JobParams, cfg: SpectraConfig):
    if not os.path.isfile(paths.cutout_h5):
        raise FileNotFoundError(f"cutout_h5 not found: {paths.cutout_h5}")

    if params.verbose:
        print("[PATHS]", asdict(paths))
        print("[PARAM]", asdict(params))
        print("[CFG  ]", asdict(cfg))

    ds = yt.load(paths.cutout_h5)
    add_ions(ds)

    for run_label in params.run_labels:
        process_run_for_sid(ds, params.sid, params.snap, run_label, paths, cfg, params)


# -------------------------
# CLI
# -------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate spectra for rays from orient_m61 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cutout_h5", required=True)
    p.add_argument("--rays_base", required=True, help="Must be <ORIENT_OUT_BASE>/sid<SID>")
    p.add_argument("--output_base", required=True)

    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--run_labels", default="L3Rvir,L4Rvir")
    p.add_argument("--filter_mode", choices=["noflip", "flip"], default=None)

    p.add_argument("--alpha_keep", default="", help="Comma-separated alpha list, e.g. '0,90'")
    p.add_argument("--sightline_ids", default="", help="Comma-separated IDs, e.g. 'QSO-A,QSO-B'")
    p.add_argument("--max_rays", type=int, default=0)

    p.add_argument("--lines", default="H I 1216,C II 1335,Si III 1206")
    p.add_argument("--instrument", default="COS-G130M")
    p.add_argument("--zoom_half_A", type=float, default=3.0)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def _main_cli(argv=None):
    a = _parse_args(argv)

    paths = JobPaths(
        cutout_h5=a.cutout_h5,
        rays_base=a.rays_base,
        output_base=a.output_base,
    )

    alpha_keep = None
    if a.alpha_keep.strip():
        alpha_keep = [int(x) for x in a.alpha_keep.split(",") if x.strip()]

    sightline_ids = None
    if a.sightline_ids.strip():
        sightline_ids = [x.strip() for x in a.sightline_ids.split(",") if x.strip()]

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
        instrument=a.instrument,
        zoom_half_A=float(a.zoom_half_A),
        make_plots=bool(a.make_plots),
    )

    try:
        run_all_runs_for_sid(paths, params, cfg)
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    _main_cli()