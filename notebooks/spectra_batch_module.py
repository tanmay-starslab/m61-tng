#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spectra_batch_module.py

Generalized batch builder: read precomputed rays from
  <rays_base>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN_LABEL>/rays_sid<SID>.csv
and generate:
  • per-ray bundles (HDF5) + spectrum PNGs
  • one combined HDF5 per RUN_LABEL collecting every ray

Import and call `run_all_runs_for_sid(...)` from any script or notebook,
or run this file directly (see CLI at bottom).
"""

from __future__ import annotations
import os, json, traceback, argparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
import yt
import trident
import matplotlib.pyplot as plt

# -------------------------
# Utility + configuration
# -------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

@dataclass
class SpectraConfig:
    lines: List[str] = None
    instrument: str = "COS-G130M"
    # zooms for quick-look PNGs: label -> rest λ [Å]
    zooms_A: Dict[str, float] = None
    zoom_half_A: float = 3.0

    def __post_init__(self):
        if self.lines is None:
            self.lines = ["H I 1216", "C II 1335", "Si III 1206"]
        if self.zooms_A is None:
            self.zooms_A = {
                "Lyα 1215.67": 1215.67,
                "Si III 1206.50": 1206.50,
                "C II 1334.53": 1334.532,
            }

@dataclass
class JobPaths:
    cutout_h5: str          # e.g. ".../sub_<SID>/cutout_*sub<SID>*.hdf5"
    rays_base: str          # base folder that contains rays_and_recipes_sid* dirs (often sub_<SID>)
    output_base: str        # where to place spectra outputs (often sub_<SID>)

@dataclass
class JobParams:
    sid: int
    snap: int = 99
    run_labels: Iterable[str] = ("L3Rvir", "L4Rvir")
    filter_mode: Optional[str] = None     # "noflip" | "flip" | None
    verbose: bool = True



# -------------------------
# saving helpers
# -------------------------
import time, tempfile, errno, random

def is_valid_h5(path: str) -> bool:
    try:
        import h5py, os
        if not os.path.exists(path): return False
        if os.path.getsize(path) == 0: return False
        if not h5py.is_hdf5(path): return False
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def atomic_write_h5(target_path: str, write_fn):
    """
    Write to a temp file in the SAME directory, fsync, then atomic rename.
    write_fn(file_handle) should populate the file_handle (already open in 'w').
    """
    d = os.path.dirname(target_path)
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(target_path)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.tmp.", dir=d)
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w", libver="latest") as f:
            write_fn(f)
            f.flush()
            try: os.fsync(f.id.get_vfd_handle())  # best-effort; may not exist on some builds
            except Exception: pass
        os.replace(tmp_path, target_path)  # atomic on same filesystem
    finally:
        try: os.remove(tmp_path)
        except FileNotFoundError: pass
        
# -------------------------
# yt / trident helpers
# -------------------------

def add_ions(ds):
    needed = ["H I", "C II", "Si III"]
    try:
        trident.add_ion_fields(ds, ions=needed)
        print("[INFO] Ion fields added (or already present):", needed)
    except Exception as e:
        print("[WARN] add_ion_fields:", e)

def make_ray(ds, p0_ckpch, p1_ckpch, data_filename=None, solution_filename=None):
    sp = ds.arr(p0_ckpch, "code_length")
    ep = ds.arr(p1_ckpch, "code_length")
    print(f"[RAY] Start (code_length): {sp}")
    print(f"[RAY] End   (code_length): {ep}")

    L = ds.domain_left_edge.to("code_length").value
    R = ds.domain_right_edge.to("code_length").value
    for name, vec in [("start", sp.value), ("end", ep.value)]:
        inside = np.all(vec >= L - 1e-6) and np.all(vec <= R + 1e-6)
        print(f"[RAY] {name} inside domain: {inside}")

    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity"),
        ("gas", "H_p0_number_density"),
        ("gas", "C_p1_number_density"),
        ("gas", "Si_p2_number_density"),
        ("gas", "H_number_density"),
    ]
    ray = trident.make_simple_ray(ds, 
                                  start_position=sp, 
                                  end_position=ep, 
                                  fields=fields, 
                                  ftype="gas", 
                                  data_filename=data_filename, 
                                  solution_filename=solution_filename)
    ad = ray.all_data()
    nseg = ad[('gas','density')].size
    geom_len_ckpch = np.sqrt(((p1_ckpch - p0_ckpch)**2).sum())
    print(f"[RAY] N segments: {nseg} | geometric length ≈ {geom_len_ckpch:.3f} ckpc/h")
    return ray

def _try_get(ad, fld, unit):
    try:
        return ad[fld].to(unit).value
    except Exception:
        return None

def get_dl_cm(ray, p0_ckpch=None, p1_ckpch=None, verbose=True):
    ad = ray.all_data()
    for cand in [( "gas","dl"), ("dl",), ("gas","segment_length"), ("gas","path_length")]:
        try:
            arr = ad[cand].to("cm").value
            if verbose:
                print(f"[DL] Using native field {cand} (min={arr.min():.3e} cm, max={arr.max():.3e} cm)")
            return arr, "native"
        except Exception:
            pass

    if p0_ckpch is None or p1_ckpch is None:
        raise RuntimeError("No dl-like field and no endpoints for approximation.")

    X = np.vstack([
        ad[("gas","x")].to("code_length").value,
        ad[("gas","y")].to("code_length").value,
        ad[("gas","z")].to("code_length").value
    ]).T
    p0 = np.asarray(p0_ckpch, float); p1 = np.asarray(p1_ckpch, float)
    v  = p1 - p0; L = np.linalg.norm(v)
    if L <= 0:
        raise RuntimeError("Zero-length endpoints; cannot approximate dl.")
    u = v / L
    s = (X - p0[None,:]) @ u
    order = np.argsort(s)
    s_sorted = s[order]
    ds_code = np.diff(s_sorted)
    if ds_code.size == 0:
        raise RuntimeError("Ray has <2 segments; cannot approximate dl.")
    dl_sorted = np.empty_like(s_sorted)
    dl_sorted[:-1] = ds_code
    dl_sorted[-1]  = ds_code[-1]
    conv = ray.ds.arr(1.0, "code_length").to("cm").value
    dl_cm_sorted = dl_sorted * conv
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    dl_cm = dl_cm_sorted[inv]
    if verbose:
        print(f"[DL] Approximated from centers. Σdl≈{dl_cm.sum()/3.086e21:.3f} kpc")
    return dl_cm, "approx"

def compute_columns(ray, p0_ckpch, p1_ckpch):
    ad = ray.all_data()
    dl_cm, dl_kind = get_dl_cm(ray, p0_ckpch, p1_ckpch, verbose=True)
    out = {"_dl_kind": dl_kind, "_sum_dl_kpc": float(dl_cm.sum()/3.08567758e21)}

    def add_if_present(fld, key):
        n = _try_get(ad, fld, "cm**-3")
        out[key] = float((n * dl_cm).sum()) if n is not None else np.nan

    add_if_present(("gas","H_p0_number_density"),  "N_HI_cm2")
    add_if_present(("gas","C_p1_number_density"),  "N_CII_cm2")
    add_if_present(("gas","Si_p2_number_density"), "N_SiIII_cm2")

    for k in ["N_HI_cm2","N_CII_cm2","N_SiIII_cm2"]:
        v = out[k]; lg = np.log10(v) if (v is not None and np.isfinite(v) and v > 0) else -np.inf
        print(f"{k:10s}: {v:.6e}   (logN={lg:.3f})")

    print(f"[DL] kind={dl_kind} | Σdl ≈ {out['_sum_dl_kpc']:.3f} kpc")
    return out

PREFERRED_UNITS = {
    "dl": "cm", "l": "cm",
    "density": "g/cm**3",
    "metallicity": "",
    "H_number_density": "cm**-3",
    "H_nuclei_density": "cm**-3",
    "H_p0_number_density": "cm**-3",
    "C_p1_number_density": "cm**-3",
    "Si_p2_number_density": "cm**-3",
    "temperature": "K",
    "velocity_los": "km/s",
    "relative_velocity_x": "km/s",
    "relative_velocity_y": "km/s",
    "relative_velocity_z": "km/s",
    "redshift": "", "redshift_dopp": "", "redshift_eff": "",
    "x": "code_length", "y": "code_length", "z": "code_length",
}

def _save_dataset_with_unit(group, name, yt_array, preferred_units=None):
    u_target = None if preferred_units is None else preferred_units.get(name, None)
    try:
        if u_target:
            arr = yt_array.to(u_target)
            data = arr.value; unit_str = u_target
        else:
            data = yt_array.value
            unit_str = str(getattr(yt_array, "units", ""))
    except Exception:
        data = np.asarray(yt_array)
        unit_str = ""
    dset = group.create_dataset(name, data=np.asarray(data))
    dset.attrs["unit"] = unit_str
    return dset

def build_spectrum(ray, lines, instr="COS-G130M"):
    sg = trident.SpectrumGenerator(instr)
    sg.make_spectrum(ray, lines=lines)
    sg.apply_lsf()  # ensure instrument LSF is applied

    lam = np.array(sg.lambda_field)
    tau = np.array(sg.tau_field)
    flux_lsf = np.array(sg.flux_field)
    flux_raw = np.exp(-tau)
    print(f"[SPEC] λ grid: {lam.min():.2f}–{lam.max():.2f} Å  (N={lam.size})  |  {instr}")
    return {"raw":{"lambda_A":lam,"flux":flux_raw,"tau":tau},
            "lsf":{"lambda_A":lam,"flux":flux_lsf,"tau":tau}}

def save_spectrum_pngs(spec, out_dir, tag, zooms_A: Dict[str, float], zoom_half_A: float):
    ensure_dir(out_dir)
    lam_r, T_r = spec["raw"]["lambda_A"], spec["raw"]["flux"]
    lam_l, T_l = spec["lsf"]["lambda_A"], spec["lsf"]["flux"]

    # full
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(lam_r, T_r, lw=1.1, label="raw")
    ax.plot(lam_l, T_l, lw=1.1, alpha=0.9, label="LSF")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("Transmission")
    ax.set_title("Full spectrum"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tag}_spectrum_full.png")
    plt.savefig(path, dpi=160); plt.close(fig)
    print("[SAVED]", path)

    # zooms
    fig, axes = plt.subplots(1, len(zooms_A), figsize=(4.0*len(zooms_A), 3), sharey=True)
    if len(zooms_A) == 1:
        axes = [axes]
    for ax, (name, lam0) in zip(axes, zooms_A.items()):
        m  = (lam_r >= lam0 - zoom_half_A) & (lam_r <= lam0 + zoom_half_A)
        ml = (lam_l >= lam0 - zoom_half_A) & (lam_l <= lam0 + zoom_half_A)
        ax.plot(lam_r[m], T_r[m], lw=1.0, label="raw")
        ax.plot(lam_l[ml], T_l[ml], lw=1.0, alpha=0.9, label="LSF")
        ax.axvline(lam0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(lam0 - zoom_half_A, lam0 + zoom_half_A)
        ax.set_title(name); ax.set_xlabel("λ [Å]"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Transmission"); axes[0].legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tag}_spectrum_zooms.png")
    plt.savefig(path, dpi=160); plt.close(fig)
    print("[SAVED]", path)

def _write_ray_pack_into_group(groot, meta, ray, spec, cols):
    # meta
    gmeta = groot.create_group("meta")
    for k, v in meta.items():
        try:
            gmeta.attrs[k] = v
        except TypeError:
            gmeta.attrs[k] = json.dumps(v)

    # columns
    gcols = groot.create_group("columns")
    for k, v in cols.items():
        gcols.attrs[k] = v if isinstance(v, (int,float,np.floating)) else json.dumps(v)

    # spectra
    gs = groot.create_group("spectrum")
    for tag in ("raw","lsf"):
        gt = gs.create_group(tag)
        gt.create_dataset("lambda_A", data=np.asarray(spec[tag]["lambda_A"]))
        gt.create_dataset("flux",     data=np.asarray(spec[tag]["flux"]))
        gt.create_dataset("tau",      data=np.asarray(spec[tag]["tau"]))

    # ray + fields
    ad = ray.all_data()
    gray = groot.create_group("ray")

    # dl used for columns
    try:
        dl_cm, kind = get_dl_cm(ray, meta["start_ckpch"], meta["end_ckpch"], verbose=False)
        gray.create_dataset("dl_cm", data=np.asarray(dl_cm))
        gray.attrs["dl_kind"] = kind
    except Exception:
        pass

    # convenience datasets
    convenience = [
        (("gas","H_p0_number_density"), "nHI_cm3",    "cm**-3"),
        (("gas","C_p1_number_density"), "nCII_cm3",   "cm**-3"),
        (("gas","Si_p2_number_density"),"nSiIII_cm3", "cm**-3"),
        (("gas","density"),              "rho_cgs",    "g/cm**3"),
        (("gas","temperature"),          "T_K",        "K"),
        (("gas","x"),                    "x_code",     "code_length"),
        (("gas","y"),                    "y_code",     "code_length"),
        (("gas","z"),                    "z_code",     "code_length"),
        (("gas","velocity_los"),         "vlos_km_s",  "km/s"),
    ]
    for fld, name, unit in convenience:
        try:
            arr = ad[fld].to(unit).value if unit else ad[fld].value
            ds = gray.create_dataset(name, data=np.asarray(arr))
            if unit: ds.attrs["unit"] = unit
        except Exception:
            pass

    # cell index
    try:
        N = int(gray["dl_cm"].size)
    except Exception:
        N = None
        for (ft,fname) in ray.field_list:
            try:
                a = ad[(ft,fname)]
                if a.ndim == 1:
                    N = int(a.size); break
            except Exception:
                continue
        if N is None: N = 0
    gray.create_dataset("cell_index", data=np.arange(N, dtype=np.int32))

    # full fields
    gfields = gray.create_group("fields")
    seen = set()
    for ftype, fname in ray.field_list:
        key = (str(ftype), str(fname))
        if key in seen: continue
        seen.add(key)
        try:
            arr = ad[(ftype, fname)]
        except Exception:
            continue
        g_ft = gfields.require_group(str(ftype))
        _save_dataset_with_unit(g_ft, str(fname), arr, preferred_units=PREFERRED_UNITS)
    
    
def save_bundle_hdf5(out_path, meta, ray, spec, cols):
    def _writer(f):
        _write_ray_pack_into_group(f, meta, ray, spec, cols)
    atomic_write_h5(out_path, _writer)
    print("[SAVED]", out_path)

def append_to_combined(agg_path, group_path, meta, ray, spec, cols,
                       globals_once, max_retries: int = 3):
    import time

    ensure_dir(os.path.dirname(agg_path))

    # If file exists but is corrupt, recreate fresh.
    if os.path.exists(agg_path) and not is_valid_h5(agg_path):
        print(f"[WARN] Combined file corrupt/non-HDF5. Recreating: {agg_path}")
        os.remove(agg_path)

    # Ensure a valid container exists
    if not os.path.exists(agg_path):
        def _init_writer(f):
            g = f.create_group("globals")
            for k, v in globals_once.items():
                try: g.attrs[k] = v
                except TypeError: g.attrs[k] = json.dumps(v)
        atomic_write_h5(agg_path, _init_writer)

    # resume policy via env (default = overwrite)
    policy = os.environ.get("SPECTRA_RESUME", "overwrite").lower()   # overwrite|skip

    for attempt in range(1, max_retries + 1):
        try:
            with h5py.File(agg_path, "a", libver="latest") as f:
                if "globals" not in f:
                    g = f.create_group("globals")
                    for k, v in globals_once.items():
                        try: g.attrs[k] = v
                        except TypeError: g.attrs[k] = json.dumps(v)

                # handle prior writes
                if group_path in f:
                    if policy == "skip":
                        print(f"[SKIP] {group_path} already present in {agg_path}")
                        return
                    print(f"[INFO] Replacing existing group: {group_path}")
                    del f[group_path]   # clear old partial/previous write

                base = f.create_group(group_path)
                _write_ray_pack_into_group(base, meta, ray, spec, cols)

            print("[APPEND]", agg_path, "::", group_path)
            return

        except OSError as e:
            print(f"[WARN] append_to_combined attempt {attempt} failed: {e}")
            time.sleep(0.5)
            if not is_valid_h5(agg_path) and attempt < max_retries:
                try:
                    os.remove(agg_path)
                    print(f"[WARN] Removed corrupt {agg_path}; reinitializing.")
                    def _init_writer(f):
                        g = f.create_group("globals")
                        for k, v in globals_once.items():
                            try: g.attrs[k] = v
                            except TypeError: g.attrs[k] = json.dumps(v)
                    atomic_write_h5(agg_path, _init_writer)
                except Exception as ee:
                    print(f"[WARN] Could not remove/reinit {agg_path}: {ee}")

    raise RuntimeError(f"append_to_combined: failed after {max_retries} attempts for {agg_path}")

# -------------------------
# Core processing
# -------------------------

def process_run_for_sid(
    ds,
    sid: int,
    snap: int,
    run_label: str,
    paths: JobPaths,
    cfg: SpectraConfig,
    filter_mode: Optional[str] = None,
    verbose: bool = True
):
    rays_csv = os.path.join(paths.rays_base,
                            f"rays_and_recipes_sid{sid}_snap{snap}_{run_label}",
                            f"rays_sid{sid}.csv")
    job_root = os.path.join(paths.output_base,
                            f"rays_and_spectra_sid{sid}_snap{snap}_{run_label}")
    rays_dir    = os.path.join(job_root, "rays")
    logs_dir    = os.path.join(job_root, "logs")
    combined_dir= os.path.join(job_root, "combined")
    combined_h5 = os.path.join(combined_dir, f"all_rays_{run_label}.h5")
    ensure_dir(job_root); ensure_dir(rays_dir); ensure_dir(logs_dir); ensure_dir(combined_dir)

    if verbose:
        print(f"\n===== PROCESSING RUN: {run_label} for SID={sid} SNAP={snap} =====")
        print("[INFO] Rays CSV:", rays_csv)

    if not os.path.isfile(rays_csv):
        raise FileNotFoundError(f"Missing rays CSV for run {run_label}: {rays_csv}")

    df = pd.read_csv(rays_csv)
    if verbose:
        print(f"[INFO] Loaded rays: rows={len(df)}")

    if filter_mode is not None:
        df = df[df["mode"].astype(str).str.lower() == filter_mode.lower()]
        print(f"[INFO] Filtered by mode='{filter_mode}'  -> rows={len(df)}")

    if df.empty:
        raise RuntimeError("No rows to process after filtering.")

    summary_rows, errors = [], 0

    # globals for combined file
    globals_once = dict(SID=int(sid),
                        SNAP=int(snap),
                        instrument=str(cfg.instrument),
                        lines=json.dumps(list(cfg.lines)))
    
    
    # one scratch location per subhalo task
    ray_scratch_dir = os.environ.get("SLURM_TMPDIR")
    if not ray_scratch_dir:
        # fall back to a subhalo-local tmp dir so tasks don't collide
        ray_scratch_dir = os.path.join(paths.rays_base, "_tmp_trident")
    os.makedirs(ray_scratch_dir, exist_ok=True)

    # a single (reused) filename for every sightline of this SID
    rayfile  = os.path.join(ray_scratch_dir, f"ray_sid{sid}.h5")
    trajfile = os.path.join(ray_scratch_dir, f"traj_sid{sid}.txt")

    # start clean
    for p in (rayfile, trajfile):
        try: os.remove(p)
        except FileNotFoundError: pass

    for i, row in df.iterrows():
        try:
            mode  = str(row.get("mode", "unknown"))
            alpha = float(row.get("alpha_deg", np.nan))

            p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], float)
            p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], float)

            rho_kpc  = float(row.get("rho_kpc", np.nan))
            phi_deg  = float(row.get("phi_deg", np.nan))
            inc_deg = float(row.get("obs_inc_deg", row.get("inc_deg", np.nan)))
            rvir_kpc = float(row.get("Rvir_kpc", np.nan))
            total_len_Rvir = float(row.get("total_len_Rvir", np.nan))

            alpha_tag = f"{int(round(alpha))}" if np.isfinite(alpha) else "NA"
            tag = f"{run_label}_sid{sid}_{mode}_alpha{alpha_tag}"

            out_dir = os.path.join(rays_dir, f"mode={mode}", f"alpha={alpha_tag}")
            ensure_dir(out_dir)

            if verbose:
                print(f"\n[{i+1}/{len(df)}] {run_label} Processing {tag} ...")

            # AFTER (reuse the same file for this SID):
            # (optionally delete to ensure a fresh write each time)
            for p in (rayfile, trajfile):
                try: os.remove(p)
                except FileNotFoundError: pass

            ray = make_ray(ds, p0, p1, data_filename=rayfile, solution_filename=trajfile)
            cols = compute_columns(ray, p0, p1)
            spec = build_spectrum(ray, cfg.lines, instr=cfg.instrument)

            save_spectrum_pngs(spec, out_dir, tag=tag,
                               zooms_A=cfg.zooms_A, zoom_half_A=cfg.zoom_half_A)

            meta_save = dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid), mode=mode, alpha_deg=float(alpha),
                rho_kpc=rho_kpc, phi_deg=phi_deg, inc_deg=inc_deg, Rvir_kpc=rvir_kpc,
                total_len_Rvir=total_len_Rvir,
                start_ckpch=p0.tolist(), end_ckpch=p1.tolist(),
                lines=list(cfg.lines), instrument=str(cfg.instrument)
            )
            bundle_path = os.path.join(out_dir, f"{tag}_bundle.h5")
            save_bundle_hdf5(bundle_path, meta_save, ray, spec, cols)

            grp_path = f"rays/mode={mode}/alpha={alpha_tag}/ray_{i:04d}"
            append_to_combined(combined_h5, grp_path, meta_save, ray, spec, cols, globals_once)

            srow = dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid), mode=mode, alpha_deg=float(alpha),
                rho_kpc=rho_kpc, phi_deg=phi_deg, inc_deg=inc_deg, Rvir_kpc=rvir_kpc,
                total_len_Rvir=total_len_Rvir,
                out_dir=out_dir, bundle_h5=bundle_path,
                dl_kind=cols.get("_dl_kind"), sum_dl_kpc=cols.get("_sum_dl_kpc"),
                N_HI_cm2=cols.get("N_HI_cm2"), N_CII_cm2=cols.get("N_CII_cm2"), N_SiIII_cm2=cols.get("N_SiIII_cm2"),
                combined_h5=combined_h5, group_path=grp_path
            )
            summary_rows.append(srow)

        except Exception as e:
            errors += 1
            msg = f"[ERROR] {run_label} row {i} (mode={row.get('mode')}, alpha={row.get('alpha_deg')}): {e}"
            print(msg)
            with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                f.write(msg + "\n")

    if summary_rows:
        master_csv = os.path.join(job_root, "summary_all_rays.csv")
        pd.DataFrame(summary_rows).to_csv(master_csv, index=False)
        print(f"\n[OK] Wrote master summary: {master_csv}  (rows={len(summary_rows)}, errors={errors})")
        print(f"[OK] Combined HDF5 for {run_label}: {combined_h5}")
    else:
        print("\n[WARN] No successful rays to summarize.", f"errors={errors}")

    print("\n[DONE] Run finished:", run_label)
    print("      Per-ray outputs under:", job_root)
    print("      Combined file:", combined_h5)

def run_all_runs_for_sid(paths: JobPaths, params: JobParams, cfg: SpectraConfig):
    """Load dataset ONCE, add ions, then process each RUN_LABEL."""
    print("\n========== SPECTRA BATCH ==========")
    print("[PATHS]", asdict(paths))
    print("[PARAM]", asdict(params))
    print("[CFG  ]", asdict(cfg))

    if not os.path.isfile(paths.cutout_h5):
        raise FileNotFoundError(f"CUTOUT_H5 not found: {paths.cutout_h5}")

    print("[INFO] Loading dataset:", paths.cutout_h5)
    ds = yt.load(paths.cutout_h5)
    print(ds)
    add_ions(ds)

    for run_label in params.run_labels:
        process_run_for_sid(
            ds=ds,
            sid=int(params.sid),
            snap=int(params.snap),
            run_label=str(run_label),
            paths=paths,
            cfg=cfg,
            filter_mode=params.filter_mode,
            verbose=params.verbose
        )

# -------------------------
# CLI (optional)
# -------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generalized spectra batch over precomputed rays CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--cutout_h5", required=True, help="Path to subhalo cutout HDF5.")
    p.add_argument("--rays_base", required=True, help="Base dir containing rays_and_recipes_sid* folders.")
    p.add_argument("--output_base", required=True, help="Output base for spectra bundles.")
    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--run_labels", default="L3Rvir,L4Rvir", help="Comma-separated run labels.")
    p.add_argument("--filter_mode", choices=["noflip","flip"], default=None)
    p.add_argument("--lines", default="H I 1216,C II 1335,Si III 1206", help="Comma-separated line list.")
    p.add_argument("--instrument", default="COS-G130M")
    p.add_argument("--zoom_half_A", type=float, default=3.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def _main_cli(argv=None):
    a = _parse_args(argv)
    paths = JobPaths(cutout_h5=a.cutout_h5, rays_base=a.rays_base, output_base=a.output_base)
    params = JobParams(
        sid=int(a.sid),
        snap=int(a.snap),
        run_labels=[s.strip() for s in a.run_labels.split(",") if s.strip()],
        filter_mode=a.filter_mode,
        verbose=bool(a.verbose),
    )
    cfg = SpectraConfig(
        lines=[s.strip() for s in a.lines.split(",") if s.strip()],
        instrument=a.instrument,
        zoom_half_A=float(a.zoom_half_A),
    )
    try:
        run_all_runs_for_sid(paths, params, cfg)
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    _main_cli()