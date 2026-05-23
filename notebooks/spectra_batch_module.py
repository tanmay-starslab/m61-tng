#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spectra_batch_module.py

Hardened Trident spectra generator for rays precomputed by orient_m61.py.

Primary goals:
  1) NEVER crash the whole run because of one ray.
  2) ALWAYS write a spectrum-only HDF5 when spectrum generation succeeds.
  3) Avoid yt operations that trigger "ParticleContainer is an unindexed type" errors:
     - no off-axis plots
     - no ires/icoords/fcoords/fwidth
     - no ambiguous field access ("x" without (ftype,fname))
  4) Make Trident spectrum generation robust for TNG cutouts by:
     - ensuring ("gas","metallicity") exists (aliasing from GFM_Metallicity if needed)
     - forcing required ion number-density fields to be WRITTEN into the ray HDF5
       via trident.make_simple_ray(..., fields=[...])

Added (parity with older project):
  5) Explicitly apply instrument LSF (sg.apply_lsf()).
  6) Save BOTH raw flux (exp(-tau)) and LSF flux (sg.flux_field).
  7) Save tau whenever available (and keep it with both products).
  8) Optional Gaussian noise injection (OFF by default).

Expected rays CSV:
  <ORIENT_OUT_BASE>/sid<SID>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN_LABEL>/rays_sid<SID>.csv

Outputs:
  <OUTPUT_BASE>/rays_and_spectra_sid<SID>_snap<SNAP>_<RUN_LABEL>/
    spectra_h5/                  <-- spectrum-only, written first (always)
    rays/                        <-- optional per-ray bundles
    combined/                    <-- optional combined file (best effort)
    logs/errors.txt
    summary_all_rays.csv
"""

import os
import json
import time
import argparse
import traceback
import inspect
import re
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, List, Sequence, Tuple, Any, Dict, Mapping

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
    output_base: str    # where to write spectra outputs


@dataclass
class JobParams:
    sid: int
    snap: int = 99
    run_labels: Optional[List[str]] = None
    filter_mode: Optional[str] = None   # "noflip" | "flip" | None
    alpha_keep: Optional[List[int]] = None
    sightline_ids: Optional[List[str]] = None
    max_rays: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        if self.run_labels is None:
            self.run_labels = ["L2Rvir", "L3Rvir", "L4Rvir"]


# IMPORTANT:
# - run_spectra_one_sid.py is calling SpectraConfig(..., zoom_half_A=...)
# - accept/ignore unknown kwargs to avoid future mismatches.
@dataclass(init=False)
class SpectraConfig:
    def __init__(
        self,
        lines: Optional[List[str]] = None,
        instrument: str = "COS-G130M",
        zooms_A: Optional[List[float]] = None,
        zoom_half_A: float = 3.0,
        make_plots: bool = False,

        # New: instrument realism toggles
        apply_lsf: bool = True,
        use_doppler_redshift_only: bool = True,

        # New: optional noise model (OFF by default)
        add_noise: bool = False,
        snr: float = 0.0,                 # SNR at continuum; used if add_noise=True
        noise_seed: int = 0,              # 0 => nondeterministic; else deterministic

        **kwargs: Any,
    ):
        self.lines = lines if lines is not None else ["H I 1216", "C II 1335", "Si III 1206"]
        self.instrument = instrument
        self.zooms_A = zooms_A if zooms_A is not None else [1215.67, 1334.532, 1206.50]
        self.zoom_half_A = float(zoom_half_A)
        self.make_plots = bool(make_plots)

        self.apply_lsf = bool(apply_lsf)
        self.use_doppler_redshift_only = bool(use_doppler_redshift_only)
        self.add_noise = bool(add_noise)
        self.snr = float(snr) if snr is not None else 0.0
        self.noise_seed = int(noise_seed) if noise_seed is not None else 0
        # ignore kwargs on purpose


# -------------------------
# Small utilities
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _ray_dataset(ray):
    # make_simple_ray returns a dataset-like object (YTDataLightRayDataset).
    return getattr(ray, "ds", ray)


def _field_exists(ds, f) -> bool:
    try:
        return (f in ds.field_list) or (f in ds.derived_field_list)
    except Exception:
        try:
            return f in ds.field_list
        except Exception:
            return False


def _pick_field(ray, fname: str, preferred_ftypes: Sequence[str]) -> Tuple[str, str]:
    """
    Pick the first existing (ftype,fname) on the ray dataset, to avoid ambiguity.
    """
    rds = _ray_dataset(ray)
    for ft in preferred_ftypes:
        f = (ft, fname)
        if _field_exists(rds, f):
            return f
    raise KeyError(f"Missing field {fname}. Tried: {[(ft, fname) for ft in preferred_ftypes]}")


def ensure_metallicity_field(ds) -> None:
    """
    Trident typically expects ("gas","metallicity") to exist.
    TNG cutouts often store metallicity as ("gas","GFM_Metallicity") OR as particle fields.

    If the cutout has a gas/cell metallicity field with another name, alias it to ("gas","metallicity").
    """
    if _field_exists(ds, ("gas", "metallicity")):
        return

    candidates = [("gas", "GFM_Metallicity"), ("gas", "Metallicity")]
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


def ions_from_lines(lines: Sequence[str]) -> List[str]:
    """
    Convert line strings like "Si II 1260" -> "Si II".
    """
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


def run_label_lengths(run_label: str) -> Tuple[float, float]:
    """
    Return (half_len_Rvir, total_len_Rvir) for supported run labels.
    """
    specs = {
        "L2Rvir": (1.0, 2.0),
        "L3Rvir": (1.5, 3.0),
        "L4Rvir": (2.0, 4.0),
    }
    return specs.get(str(run_label), (np.nan, np.nan))


def ion_to_trident_nd_field(ion: str) -> Optional[Tuple[str, str]]:
    """
    Map an ion like 'Si II' to Trident's number-density field name.
    Extend this mapping as needed for your use-case.
    """
    ion = ion.strip()
    mapping: Dict[str, Tuple[str, str]] = {
        "H I": ("gas", "H_p0_number_density"),
        "C II": ("gas", "C_p1_number_density"),
        "C III": ("gas", "C_p2_number_density"),
        "C IV": ("gas", "C_p3_number_density"),
        "Si II": ("gas", "Si_p1_number_density"),
        "Si III": ("gas", "Si_p2_number_density"),
        "Si IV": ("gas", "Si_p3_number_density"),
        "O VI": ("gas", "O_p5_number_density"),
        "N V": ("gas", "N_p4_number_density"),
        "Mg II": ("gas", "Mg_p1_number_density"),
        "Fe II": ("gas", "Fe_p1_number_density"),
    }
    return mapping.get(ion, None)


def add_ions(ds, ions: Sequence[str]) -> None:
    if not ions:
        return
    trident.add_ion_fields(ds, ions=list(ions))


# -------------------------
# Ray + spectrum helpers
# -------------------------

def make_ray(
    ds,
    p0_ckpch_abs,
    p1_ckpch_abs,
    data_filename,
    solution_filename,
    ions_needed: Optional[Sequence[str]] = None,
):
    """
    Build a simple ray and FORCE writing required fields into the ray HDF5.

    Key point:
      SpectrumGenerator.make_spectrum(ray, ...) will later operate on the ray dataset
      loaded from the ray HDF5. That dataset must contain the required ion number-density
      field(s), or Trident will raise YTFieldNotFound.

    Therefore, we explicitly include any required ion number-density fields in `fields=...`.
    """
    # Trident mutates these arrays while converting units, so force writable copies.
    p0 = ds.arr(np.array(p0_ckpch_abs, dtype=float, copy=True), "code_length")
    p1 = ds.arr(np.array(p1_ckpch_abs, dtype=float, copy=True), "code_length")

    # Ensure metallicity exists on the parent dataset before we sample it.
    ensure_metallicity_field(ds)

    # Some yt frontends expose read-only domain edge arrays, while Trident's
    # LightRay converts those arrays in-place.  Give Trident writable copies.
    for attr in ("domain_left_edge", "domain_right_edge"):
        try:
            edge = getattr(ds, attr)
            writable_edge = ds.arr(np.array(edge.to("code_length").value, dtype=float, copy=True), "code_length")
            setattr(ds, attr, writable_edge)
        except Exception:
            pass

    ray_fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity"),
        ("gas", "H_number_density"),
    ]

    if ions_needed:
        for ion in ions_needed:
            f = ion_to_trident_nd_field(ion)
            if f is not None and f not in ray_fields:
                ray_fields.append(f)

    return trident.make_simple_ray(
        ds,
        start_position=p0,
        end_position=p1,
        fields=ray_fields,            # IMPORTANT: force-write required fields
        ftype="gas",
        data_filename=data_filename,
        solution_filename=solution_filename,
    )


def _get_sg_arrays(sg) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    lam = np.asarray(sg.lambda_field).astype(float)
    flux = np.asarray(sg.flux_field).astype(float)
    tau = getattr(sg, "tau_field", None)
    if tau is not None:
        try:
            tau = np.asarray(tau).astype(float)
        except Exception:
            tau = None
    return lam, flux, tau


def _maybe_add_noise(flux: np.ndarray, snr: float, seed: int = 0) -> np.ndarray:
    """
    Simple Gaussian noise model around transmission.
    OFF unless add_noise=True and snr>0.

    Notes:
      - This is not a full COS noise model. It's a pragmatic SNR knob.
      - Transmission is clipped to [0,1] after noise.
    """
    if snr <= 0:
        return flux

    rng = np.random.default_rng(None if seed == 0 else seed)
    sigma = 1.0 / snr
    noisy = flux + rng.normal(0.0, sigma, size=flux.shape)
    return np.clip(noisy, 0.0, 1.0)


def safe_line_name(line: str) -> str:
    """
    Convert Trident line names like "Si II 1190" into stable HDF5 group names.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", str(line).strip()).strip("_")


def _trident_make_spectrum_kwargs(cfg: SpectraConfig) -> Dict[str, Any]:
    kwargs = {
        "use_peculiar_velocity": True,
        "use_doppler_redshift_only": bool(cfg.use_doppler_redshift_only),
    }

    try:
        sig = inspect.signature(trident.SpectrumGenerator.make_spectrum)
        supports_doppler_only = "use_doppler_redshift_only" in sig.parameters
    except Exception:
        supports_doppler_only = False

    if kwargs["use_doppler_redshift_only"] and not supports_doppler_only:
        raise RuntimeError(
            "The imported Trident does not support use_doppler_redshift_only. "
            f"trident.__file__={getattr(trident, '__file__', 'unknown')}. "
            "Use the patched source tree, e.g. /home/tsingh65/src/trident."
        )

    if not supports_doppler_only:
        kwargs.pop("use_doppler_redshift_only", None)
    return kwargs


def _single_spectrum_product(ray, lines, cfg: SpectraConfig) -> Dict[str, Mapping[str, Any]]:
    """
    Returns:
      {
        "raw": {"lambda_A":..., "flux":..., "tau":...},
        "lsf": {"lambda_A":..., "flux":..., "tau":...},
      }

    raw flux is exp(-tau) when tau exists, else fallback to current sg.flux_field.
    lsf flux is sg.flux_field after apply_lsf() if enabled, else equal to raw/sg output.
    Optional noise can be applied to raw and/or lsf consistently.
    """
    sg = trident.SpectrumGenerator(cfg.instrument)
    sg.make_spectrum(ray, lines=lines, **_trident_make_spectrum_kwargs(cfg))

    # Grab tau (if available) BEFORE LSF; tau is an optical depth grid product.
    lam0, flux0, tau0 = _get_sg_arrays(sg)

    # Define raw as exp(-tau) if possible.
    if tau0 is not None:
        flux_raw = np.exp(-tau0)
        lam_raw = lam0
        tau_raw = tau0
    else:
        # fallback: if tau missing, we at least return sg flux as "raw"
        flux_raw = flux0
        lam_raw = lam0
        tau_raw = None

    # Apply LSF explicitly (your old code did this)
    if cfg.apply_lsf:
        try:
            sg.apply_lsf()
        except Exception:
            # Do not hard-fail the ray if apply_lsf trips on something; keep going.
            pass

    lam_lsf, flux_lsf, tau_lsf = _get_sg_arrays(sg)

    # Tau should be same grid product; prefer tau0 if present.
    tau_use = tau0 if tau0 is not None else (tau_lsf if tau_lsf is not None else None)

    # Optional noise (OFF by default)
    if cfg.add_noise and cfg.snr > 0:
        flux_raw = _maybe_add_noise(flux_raw, cfg.snr, cfg.noise_seed)
        flux_lsf = _maybe_add_noise(flux_lsf, cfg.snr, cfg.noise_seed)

    return {
        "raw": {"lambda_A": lam_raw, "flux": flux_raw, "tau": tau_use},
        "lsf": {"lambda_A": lam_lsf, "flux": flux_lsf, "tau": tau_use},
    }


def build_spectrum_products(ray, lines, cfg: SpectraConfig) -> Dict[str, Mapping[str, Any]]:
    """
    Build combined/full spectrum products plus one separate product per line.

    Existing combined output remains at /spectrum/raw and /spectrum/lsf.  New
    per-line products are written under /spectrum/individual_lines/<safe_name>.
    """
    combined = _single_spectrum_product(ray, lines, cfg)

    individual_lines: Dict[str, Mapping[str, Any]] = {}
    for line in lines:
        key = safe_line_name(line)
        product = _single_spectrum_product(ray, [line], cfg)
        product["meta"] = {
            "line_name": str(line),
            "safe_line_name": key,
        }
        individual_lines[key] = product

    return {
        "raw": combined["raw"],
        "lsf": combined["lsf"],
        "individual_lines": individual_lines,
        "meta": {
            "instrument": str(cfg.instrument),
            "apply_lsf": bool(cfg.apply_lsf),
            "use_peculiar_velocity": True,
            "use_doppler_redshift_only": bool(cfg.use_doppler_redshift_only),
            "add_noise": bool(cfg.add_noise),
            "snr": float(cfg.snr),
            "noise_seed": int(cfg.noise_seed),
            "trident_version": str(getattr(trident, "__version__", "unknown")),
            "trident_file": str(getattr(trident, "__file__", "unknown")),
        },
    }


def compute_columns(ray):
    """
    Best-effort LOS integrals; never raises.
    """
    cols = {}
    pref = ("index", "ray", "all", "gas", "grid")

    dl = None
    try:
        dl_field = _pick_field(ray, "dl", preferred_ftypes=pref)
        dl = ray.r[dl_field]
        cols["_dl_field"] = str(dl_field)
        cols["_sum_dl_code"] = float(np.nansum(np.asarray(dl).astype(float)))
    except Exception:
        cols["_dl_field"] = "NA"
        cols["_sum_dl_code"] = np.nan

    def _col(field_tuple):
        try:
            if dl is None:
                return np.nan
            nd = ray.r[field_tuple]
            dl_cm = dl.to("cm")
            return float(np.nansum((nd * dl_cm)).to("cm**-2").value)
        except Exception:
            return np.nan

    cols["N_HI_cm2"]    = _col(("gas", "H_p0_number_density"))
    cols["N_CII_cm2"]   = _col(("gas", "C_p1_number_density"))
    cols["N_SiIII_cm2"] = _col(("gas", "Si_p2_number_density"))
    cols["N_SiII_cm2"]  = _col(("gas", "Si_p1_number_density"))
    return cols


# -------------------------
# Writers
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


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def _write_attrs(g, attrs: Mapping[str, Any]) -> None:
    for k, v in attrs.items():
        try:
            g.attrs[k] = v
        except TypeError:
            g.attrs[k] = json.dumps(_jsonable(v))


def _write_one_spectrum_group(parent, products: Mapping[str, Any]) -> None:
    for tag in ("raw", "lsf"):
        g = parent.create_group(tag)
        g.create_dataset("lambda_A", data=np.asarray(products[tag]["lambda_A"]).astype(float))
        g.create_dataset("flux", data=np.asarray(products[tag]["flux"]).astype(float))
        tau = products[tag].get("tau", None)
        if tau is not None:
            g.create_dataset("tau", data=np.asarray(tau).astype(float))


def _write_spectrum_products_group(parent, spec_products: Mapping[str, Any]) -> None:
    _write_one_spectrum_group(parent, spec_products)

    individual = spec_products.get("individual_lines", {})
    lines_group = parent.create_group("individual_lines")
    for safe_name, line_products in individual.items():
        lg = lines_group.create_group(str(safe_name))
        if "meta" in line_products:
            _write_attrs(lg.create_group("meta"), line_products["meta"])
        _write_one_spectrum_group(lg, line_products)


def _copy_h5_root_into_group(src_path: str, dest_group) -> None:
    dest_group.attrs["source_file"] = str(src_path)
    if not src_path or not os.path.isfile(src_path):
        dest_group.attrs["copy_status"] = "missing_source"
        return

    try:
        with h5py.File(src_path, "r") as src:
            for k, v in src.attrs.items():
                try:
                    dest_group.attrs[k] = v
                except TypeError:
                    dest_group.attrs[k] = json.dumps(_jsonable(v))
            for key in src.keys():
                src.copy(key, dest_group, name=key)
        dest_group.attrs["copy_status"] = "ok"
    except Exception as exc:
        dest_group.attrs["copy_status"] = f"failed: {type(exc).__name__}: {exc}"


def save_spectrum_only_h5(path, meta: dict, spec_products: dict) -> None:
    """
    Minimal, robust output:
      - raw: lambda + flux + tau (if present)
      - lsf: lambda + flux + tau (if present)
    Written FIRST, before any bundle/combined work.
    """
    def _w(f):
        mg = f.create_group("meta")
        _write_attrs(mg, meta)

        pm = f.create_group("product_meta")
        _write_attrs(pm, spec_products.get("meta", {}))

        gs = f.create_group("spectrum")
        _write_spectrum_products_group(gs, spec_products)

    atomic_write_h5(path, _w)


def _write_bundle_into_group(g, meta, ray, spec_products, cols, dump_all_ray_fields: bool = False):
    mg = g.create_group("meta")
    _write_attrs(mg, meta)

    cg = g.create_group("columns")
    _write_attrs(cg, cols)

    pm = g.create_group("product_meta")
    _write_attrs(pm, spec_products.get("meta", {}))

    # Preserve the complete Trident-written ray file inside bundle/combined
    # outputs, so downstream analysis can recover every field Trident saved.
    original_ray_path = str(meta.get("original_trident_ray_h5", ""))
    _copy_h5_root_into_group(original_ray_path, g.create_group("original_trident_ray"))

    # ---- ray (minimal, safe)
    pref = ("index", "ray", "all", "gas", "grid")
    rg = g.create_group("ray")

    # dl optional
    try:
        dl_f = _pick_field(ray, "dl", pref)
        rg.create_dataset("dl_code", data=np.asarray(ray.r[dl_f]).astype(float))
        rg.attrs["dl_field"] = str(dl_f)
    except Exception:
        rg.attrs["dl_field"] = "NA"

    # Full ray data are preserved above in /original_trident_ray.  Avoid
    # regenerating derived coordinates here; doing so can trigger yt's
    # ParticleContainer indexing errors for TNG cutout rays.
    rg.attrs["full_ray_copy_group"] = "../original_trident_ray"

    # ---- spectra: raw + lsf
    sg = g.create_group("spectrum")
    _write_spectrum_products_group(sg, spec_products)


def save_bundle_hdf5(path, meta, ray, spec_products, cols, dump_all_ray_fields: bool = False):
    def _w(f):
        base = f.create_group("bundle")
        _write_bundle_into_group(base, meta, ray, spec_products, cols, dump_all_ray_fields=dump_all_ray_fields)
    atomic_write_h5(path, _w)


def append_to_combined(
    agg_path,
    group_path,
    meta,
    ray,
    spec_products,
    cols,
    globals_once,
    max_retries=3,
    dump_all_ray_fields: bool = False,
):
    for attempt in range(1, max_retries + 1):
        try:
            if (not os.path.exists(agg_path)) or (not is_valid_h5(agg_path)):
                def _init(f):
                    g = f.create_group("globals")
                    for k, v in globals_once.items():
                        try:
                            g.attrs[k] = v
                        except TypeError:
                            g.attrs[k] = json.dumps(_jsonable(v))
                atomic_write_h5(agg_path, _init)

            with h5py.File(agg_path, "a") as f:
                if "globals" not in f:
                    g = f.create_group("globals")
                    for k, v in globals_once.items():
                        try:
                            g.attrs[k] = v
                        except TypeError:
                            g.attrs[k] = json.dumps(_jsonable(v))

                if group_path in f:
                    del f[group_path]
                base = f.create_group(group_path)
                _write_bundle_into_group(base, meta, ray, spec_products, cols, dump_all_ray_fields=dump_all_ray_fields)
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
    spectra_dir  = os.path.join(job_root, "spectra_h5")      # spectrum-only (always try)
    rays_dir     = os.path.join(job_root, "rays")            # optional bundles
    original_rays_dir = os.path.join(job_root, "original_rays")
    ray_solutions_dir = os.path.join(job_root, "ray_solutions")
    logs_dir     = os.path.join(job_root, "logs")
    combined_dir = os.path.join(job_root, "combined")
    combined_h5  = os.path.join(combined_dir, f"all_rays_{run_label}.h5")

    ensure_dir(job_root)
    ensure_dir(spectra_dir)
    ensure_dir(rays_dir)
    ensure_dir(original_rays_dir)
    ensure_dir(ray_solutions_dir)
    ensure_dir(logs_dir)
    ensure_dir(combined_dir)

    if not os.path.isfile(rays_csv):
        raise FileNotFoundError(f"Missing rays CSV: {rays_csv}")

    df = pd.read_csv(rays_csv)

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
        apply_lsf=bool(cfg.apply_lsf),
        use_peculiar_velocity=True,
        use_doppler_redshift_only=bool(cfg.use_doppler_redshift_only),
        add_noise=bool(cfg.add_noise),
        snr=float(cfg.snr),
        noise_seed=int(cfg.noise_seed),
        trident_version=str(getattr(trident, "__version__", "unknown")),
        trident_file=str(getattr(trident, "__file__", "unknown")),
    )

    # Scratch: include pid to avoid collisions if multiple jobs run same SID.
    ray_scratch_dir = os.environ.get("SLURM_TMPDIR") or os.path.join(paths.output_base, "_tmp_trident")
    ensure_dir(ray_scratch_dir)
    pid = os.getpid()
    rayfile  = os.path.join(ray_scratch_dir, f"ray_sid{sid}.{pid}.h5")
    trajfile = os.path.join(ray_scratch_dir, f"traj_sid{sid}.{pid}.txt")

    summary_rows, errors = [], 0

    # Determine required ions from line list once per run_label
    ions_needed = ions_from_lines(cfg.lines)

    # Optional: dump all ray fields inside bundles/combined
    dump_all_ray_fields = bool(int(os.environ.get("SPECTRA_DUMP_ALL_RAY_FIELDS", "1")))
    half_len_default, total_len_default = run_label_lengths(run_label)

    for j, row in df.iterrows():
        try:
            mode  = str(row.get("mode", "unknown"))
            alpha = float(row.get("alpha_deg", np.nan))
            alpha_tag = f"{int(round(alpha))}" if np.isfinite(alpha) else "NA"

            p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], float)
            p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], float)

            rho_kpc = float(row.get("rho_kpc", np.nan))
            phi_deg = float(row.get("phi_deg", np.nan))
            inc_deg = float(row.get("obs_inc_deg", row.get("inc_deg", np.nan)))
            rvir_kpc = float(row.get("Rvir_kpc", np.nan))
            half_len_Rvir = float(row.get("half_len_Rvir", half_len_default))
            total_len_Rvir = float(row.get("total_len_Rvir", total_len_default))
            sightline_id = str(row.get("sightline_id", "SL"))

            tag = f"{run_label}_sid{sid}_{sightline_id}_{mode}_alpha{alpha_tag}"

            if params.verbose:
                print(f"[{run_label}] ({len(summary_rows)+errors+1}/{len(df)}) {tag}")

            # clean scratch
            for p in (rayfile, trajfile):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

            # Make the ray and FORCE-writing needed ion fields into ray HDF5
            ray = make_ray(ds, p0, p1, data_filename=rayfile, solution_filename=trajfile, ions_needed=ions_needed)

            original_ray_path = os.path.join(original_rays_dir, f"{tag}_original_trident_ray.h5")
            shutil.copy2(rayfile, original_ray_path)
            ray_solution_path = os.path.join(ray_solutions_dir, f"{tag}_ray_solution.txt")
            if os.path.isfile(trajfile):
                shutil.copy2(trajfile, ray_solution_path)
            else:
                ray_solution_path = ""

            # Ensure metallicity exists on the ray dataset too (harmless if already present)
            rds = _ray_dataset(ray)
            ensure_metallicity_field(rds)

            # Best-effort: add ions to ray dataset context (not relied upon, since we force-wrote fields)
            try:
                add_ions(rds, ions_needed)
            except Exception:
                pass

            cols = compute_columns(ray)

            # Generate spectrum products (raw + lsf + tau)
            spec_products = build_spectrum_products(ray, cfg.lines, cfg)

            meta = dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid),
                SNAP=int(snap),
                sightline_id=sightline_id,
                mode=mode,
                alpha_deg=float(alpha),
                rho_kpc=rho_kpc,
                phi_deg=phi_deg,
                inc_deg=inc_deg,
                Rvir_kpc=rvir_kpc,
                half_len_Rvir=half_len_Rvir,
                total_len_Rvir=total_len_Rvir,
                start_ckpch=p0.tolist(),
                end_ckpch=p1.tolist(),
                lines=list(cfg.lines),
                instrument=str(cfg.instrument),
                use_peculiar_velocity=True,
                use_doppler_redshift_only=bool(cfg.use_doppler_redshift_only),
                trident_version=str(getattr(trident, "__version__", "unknown")),
                trident_file=str(getattr(trident, "__file__", "unknown")),
                original_trident_ray_h5=original_ray_path,
                trident_ray_solution_txt=ray_solution_path,
            )

            # 1) ALWAYS write spectrum-only first
            spec_path = os.path.join(spectra_dir, f"{tag}_spectrum.h5")
            save_spectrum_only_h5(spec_path, meta, spec_products)

            # 2) Best-effort bundle + combined (must not break the run)
            out_dir = os.path.join(rays_dir, f"sightline={sightline_id}", f"mode={mode}", f"alpha={alpha_tag}")
            ensure_dir(out_dir)

            bundle_path = os.path.join(out_dir, f"{tag}_bundle.h5")
            try:
                save_bundle_hdf5(bundle_path, meta, ray, spec_products, cols, dump_all_ray_fields=dump_all_ray_fields)
            except Exception as e:
                with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                    f.write(f"[BUNDLE_FAIL] {tag}: {type(e).__name__}: {e}\n")
                    f.write(traceback.format_exc() + "\n")
                bundle_path = ""

            grp_path = f"rays/sightline={sightline_id}/mode={mode}/alpha={alpha_tag}/ray_{j:06d}"
            try:
                append_to_combined(
                    combined_h5,
                    grp_path,
                    meta,
                    ray,
                    spec_products,
                    cols,
                    globals_once,
                    dump_all_ray_fields=dump_all_ray_fields,
                )
            except Exception as e:
                with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                    f.write(f"[COMBINED_FAIL] {tag}: {type(e).__name__}: {e}\n")
                    f.write(traceback.format_exc() + "\n")

            summary_rows.append(dict(
                RUN_LABEL=run_label,
                SubhaloID=int(sid),
                SNAP=int(snap),
                sightline_id=sightline_id,
                mode=mode,
                alpha_deg=float(alpha),
                rho_kpc=rho_kpc,
                phi_deg=phi_deg,
                inc_deg=inc_deg,
                Rvir_kpc=rvir_kpc,
                half_len_Rvir=half_len_Rvir,
                total_len_Rvir=total_len_Rvir,
                spectrum_h5=spec_path,
                original_trident_ray_h5=original_ray_path,
                trident_ray_solution_txt=ray_solution_path,
                bundle_h5=bundle_path,
                combined_h5=combined_h5,
                group_path=grp_path,
                use_peculiar_velocity=True,
                use_doppler_redshift_only=bool(cfg.use_doppler_redshift_only),
                trident_file=str(getattr(trident, "__file__", "unknown")),
                N_HI_cm2=cols.get("N_HI_cm2"),
                N_CII_cm2=cols.get("N_CII_cm2"),
                N_SiIII_cm2=cols.get("N_SiIII_cm2"),
                N_SiII_cm2=cols.get("N_SiII_cm2"),
            ))

        except Exception as e:
            errors += 1
            msg = f"[ERROR] {run_label} row={j} sid={sid}: {type(e).__name__}: {e}"
            print(msg)
            with open(os.path.join(logs_dir, "errors.txt"), "a") as f:
                f.write(msg + "\n")
                f.write(traceback.format_exc() + "\n")

    if summary_rows:
        master_csv = os.path.join(job_root, "summary_all_rays.csv")
        pd.DataFrame(summary_rows).to_csv(master_csv, index=False)
        if params.verbose:
            print(f"[OK] {run_label}: spectra={len(summary_rows)} errors={errors}")
            print(f"[OK] {run_label}: spectra-only dir = {spectra_dir}")
            print(f"[OK] {run_label}: summary CSV = {master_csv}")
            print(f"[OK] apply_lsf={cfg.apply_lsf} add_noise={cfg.add_noise} snr={cfg.snr}")
    else:
        print(f"[WARN] {run_label}: no successful spectra; errors={errors}")


def run_all_runs_for_sid(paths: JobPaths, params: JobParams, cfg: SpectraConfig):
    if not os.path.isfile(paths.cutout_h5):
        raise FileNotFoundError(f"cutout_h5 not found: {paths.cutout_h5}")

    if params.verbose:
        print("[PATHS]", asdict(paths))
        print("[PARAM]", asdict(params))
        print("[CFG  ]", dict(
            lines=cfg.lines,
            instrument=cfg.instrument,
            zoom_half_A=cfg.zoom_half_A,
            make_plots=cfg.make_plots,
            apply_lsf=cfg.apply_lsf,
            use_peculiar_velocity=True,
            use_doppler_redshift_only=cfg.use_doppler_redshift_only,
            add_noise=cfg.add_noise,
            snr=cfg.snr,
            noise_seed=cfg.noise_seed,
            trident_version=str(getattr(trident, "__version__", "unknown")),
            trident_file=str(getattr(trident, "__file__", "unknown")),
        ))
        try:
            sig = inspect.signature(trident.SpectrumGenerator.make_spectrum)
            print("[TRIDENT] SpectrumGenerator.make_spectrum supports use_doppler_redshift_only:",
                  "use_doppler_redshift_only" in sig.parameters)
        except Exception as exc:
            print("[TRIDENT] Could not inspect make_spectrum signature:", repr(exc))

    ds = yt.load(paths.cutout_h5)
    ensure_metallicity_field(ds)

    ions_needed = ions_from_lines(cfg.lines)
    try:
        add_ions(ds, ions_needed)
    except Exception:
        pass

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
    p.add_argument("--run_labels", default="L2Rvir,L3Rvir,L4Rvir")
    p.add_argument("--filter_mode", choices=["noflip", "flip"], default=None)

    p.add_argument("--alpha_keep", default="", help="Comma-separated alpha list, e.g. '0,90'")
    p.add_argument("--sightline_ids", default="", help="Comma-separated IDs")
    p.add_argument("--max_rays", type=int, default=0)

    p.add_argument("--lines", default="H I 1216,C II 1335,Si III 1206")
    p.add_argument("--instrument", default="COS-G130M")
    p.add_argument("--zoom_half_A", type=float, default=3.0)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # New realism toggles
    p.add_argument("--no-lsf", action="store_true", help="Disable sg.apply_lsf() (default is enabled).")
    p.add_argument("--no-doppler-only", action="store_true",
                   help="Disable Doppler-only redshifting and use Trident's default effective redshift.")
    p.add_argument("--add-noise", action="store_true", help="Add Gaussian noise in transmission space (default off).")
    p.add_argument("--snr", type=float, default=0.0, help="Continuum SNR if --add-noise is set.")
    p.add_argument("--noise-seed", type=int, default=0, help="0=random, else deterministic seed.")

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

        apply_lsf=(not a.no_lsf),
        use_doppler_redshift_only=(not a.no_doppler_only),
        add_noise=bool(a.add_noise),
        snr=float(a.snr),
        noise_seed=int(a.noise_seed),
    )

    run_all_runs_for_sid(paths, params, cfg)


if __name__ == "__main__":
    _main_cli()
