#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pygad_fit_saved_spectra.py

Notebook-importable module to fit Trident-generated saved spectra using pygad.

Use case
--------
Your spectra are saved as one HDF5 file per ray / alpha / flip-mode, e.g.

    /scratch/tsingh65/m61-tng/outputs/sid488530/
        rays_and_spectra_sid488530_snap99_L4Rvir/
            spectra_h5/
                L4Rvir_sid488530_J122138+043026_flip_alpha0_spectrum.h5
                L4Rvir_sid488530_J122138+043026_noflip_alpha0_spectrum.h5
                ...

Each HDF5 file contains a full wavelength/flux spectrum. It does not encode
the fitted ion in the filename. Therefore, the correct workflow is:

    for each spectrum file:
        for each line in line list:
            locate the line center in wavelength space
            convert wavelength to rest frame if needed
            bin the noiseless LSF-convolved spectrum
            add SNR noise
            fit that transition independently with pygad.vpfit
            save table row and diagnostic plot

Important
---------
This version assumes that the instrumental LSF is already included in the
saved spectrum. Therefore, this code does NOT apply any additional LSF
convolution. The default processing order is:

    saved LSF-convolved noiseless spectrum
        -> rest-frame conversion if needed
        -> bin by cfg.bin_npix pixels, default 3
        -> add Gaussian noise at cfg.snr
        -> PYGAD region finding and Voigt fitting

This reproduces the intended ordering from the synthetic-spectroscopy
analysis: LSF first, then binning, then noise, then fitting.

This module supports:
    - Jupyter use
    - one-file testing
    - one SID small-batch testing
    - SNR=10 noise
    - configurable binning, default 3 pixels
    - line mapping from Trident-style labels to pygad line keys
    - defensive HDF5 loading
    - output tables and plots

Author: ChatGPT, adapted for Tanmay's M61/TNG spectra pipeline
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.table import Table, vstack

import pygad as pg


# ============================================================
# Constants
# ============================================================

C_KMS = 299792.458


# ============================================================
# Default plotting style
# ============================================================

def set_plot_style() -> None:
    """
    Publication-style Matplotlib defaults.
    Uses text.usetex=False by default for HPC robustness.
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.major.size": 8,
        "ytick.minor.size": 4,
        "xtick.major.width": 1.6,
        "xtick.minor.width": 1.2,
        "ytick.major.width": 1.6,
        "ytick.minor.width": 1.2,
        "legend.frameon": False,
    })


set_plot_style()


# ============================================================
# Line mapping
# ============================================================

DEFAULT_LINE_MAP: Dict[str, str] = {
    "Si II 1190": "SiII1190",
    "Si II 1193": "SiII1193",
    "Si III 1206": "SiIII1206",
    "N V 1239": "NV1238",
    "Si II 1260": "SiII1260",
    "O I 1302": "OI1302",
    "C II 1335": "CII1334",
    "Si IV 1403": "SiIV1402",
    "H I 1216": "H1215",

    # Extra optional lines
    "H I 1026": "H1025",
    "O VI 1032": "OVI1031",
    "O VI 1038": "OVI1037",
    "C II 1036": "CII1036",
    "N V 1243": "NV1242",
    "Si IV 1394": "SiIV1393",
}


DEFAULT_LINES: List[str] = [
    "Si II 1190",
    "Si II 1193",
    "Si III 1206",
    "N V 1239",
    "Si II 1260",
    "O I 1302",
    "C II 1335",
    "Si IV 1403",
    "H I 1216",
]


# ============================================================
# Dataclass config
# ============================================================

@dataclass
class FitConfig:
    sid: int = 488530
    snap: int = 99
    run_label: str = "L4Rvir"
    base_dir: str = "/scratch/tsingh65/m61-tng/outputs"

    z: float = 0.0
    snr: float = 10.0
    seed: int = 42

    # Instrumental/analysis window choices
    velocity_window: float = 800.0

    # This is half-width. upper_limit_window=50 means total 100 km/s.
    upper_limit_window: float = 50.0

    # Binning configuration.
    # Assumes the saved spectrum is already LSF-convolved.
    # Therefore: saved LSF spectrum -> bin -> add noise -> fit.
    bin_before_noise: bool = True
    bin_npix: int = 3

    min_region_width: int = 3
    N_sigma: float = 3.0
    chisq_lim: float = 1.0
    max_lines: int = 6

    # Defaults kept from your current script.
    # For exact O VI paper reproduction, use:
    # logN_bounds=(13.49, 18.0), b_bounds=(6.0, 100.0)
    logN_bounds: Tuple[float, float] = (12.0, 18.0)
    b_bounds: Tuple[float, float] = (6.0, 150.0)

    line_labels: List[str] = field(default_factory=lambda: DEFAULT_LINES.copy())
    line_map: Dict[str, str] = field(default_factory=lambda: DEFAULT_LINE_MAP.copy())

    output_subdir: str = "test_pygad_fits_snr10"

    make_plots: bool = True
    verbose: bool = True


# ============================================================
# Basic helpers
# ============================================================

def sanitize_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace(":", "")
    )


def wave_to_vel(wavelength_rest: np.ndarray | float, center_wave_rest: float) -> np.ndarray | float:
    return C_KMS * ((np.asarray(wavelength_rest) / center_wave_rest) - 1.0)


def vel_to_wave(velocity: np.ndarray | float, center_wave_rest: float) -> np.ndarray | float:
    return (1.0 + np.asarray(velocity) / C_KMS) * center_wave_rest


def get_spectra_dir(cfg: FitConfig) -> str:
    return os.path.join(
        cfg.base_dir,
        f"sid{cfg.sid}",
        f"rays_and_spectra_sid{cfg.sid}_snap{cfg.snap}_{cfg.run_label}",
    )


def get_spectra_h5_dir(cfg: FitConfig) -> str:
    return os.path.join(get_spectra_dir(cfg), "spectra_h5")


def get_output_dir(cfg: FitConfig) -> str:
    return os.path.join(get_spectra_dir(cfg), cfg.output_subdir)


def get_line_rest_wave(pg_ion: str) -> float:
    return float(pg.analysis.absorption_spectra.lines[pg_ion]["l"].split()[0])


def get_line_fosc(pg_ion: str) -> float:
    return float(pg.analysis.absorption_spectra.lines[pg_ion]["f"])


def print_available_pygad_lines(match: Optional[str] = None) -> None:
    keys = list(pg.analysis.absorption_spectra.lines.keys())
    if match is not None:
        keys = [k for k in keys if match.lower() in k.lower()]
    for k in keys:
        print(k)


def validate_line_map(line_map: Dict[str, str] = DEFAULT_LINE_MAP) -> Dict[str, bool]:
    """
    Return which mapped pygad keys exist in the current pygad install.
    """
    available = pg.analysis.absorption_spectra.lines
    out = {}
    for label, key in line_map.items():
        out[label] = key in available
    return out


def parse_alpha_mode_from_filename(path: str) -> Dict[str, Any]:
    """
    Parse SID/rayid/mode/alpha from filenames like:
        L4Rvir_sid488530_J122138+043026_flip_alpha0_spectrum.h5
    """
    base = os.path.basename(path)

    mode = "unknown"
    if "_noflip_" in base:
        mode = "noflip"
    elif "_flip_" in base:
        mode = "flip"

    alpha = None
    m_alpha = re.search(r"alpha([0-9]+)", base)
    if m_alpha:
        alpha = int(m_alpha.group(1))

    sid = None
    m_sid = re.search(r"sid([0-9]+)", base)
    if m_sid:
        sid = int(m_sid.group(1))

    ray_id = "unknown"
    m_ray = re.search(r"sid[0-9]+_(.*?)_(?:no)?flip_alpha", base)
    if m_ray:
        ray_id = m_ray.group(1)

    return {
        "filename": base,
        "sid": sid,
        "ray_id": ray_id,
        "mode": mode,
        "alpha": alpha,
    }


# ============================================================
# HDF5 inspection / loading
# ============================================================

def list_hdf5_datasets(path: str) -> List[Tuple[str, Tuple[int, ...], str]]:
    """
    Return all datasets in an HDF5 file.
    """
    out = []
    with h5py.File(path, "r") as h5:
        def visitor(key, obj):
            if isinstance(obj, h5py.Dataset):
                out.append((key, obj.shape, str(obj.dtype)))
        h5.visititems(visitor)
    return out


def print_hdf5_structure(path: str) -> None:
    """
    Print HDF5 datasets and shapes.
    Use this first if loading fails.
    """
    print(f"\nHDF5 file: {path}")
    print("-" * 100)
    for key, shape, dtype in list_hdf5_datasets(path):
        print(f"{key:60s} shape={str(shape):20s} dtype={dtype}")


def _find_dataset_recursive(h5: h5py.File, possible_short_names: List[str]) -> Optional[str]:
    """
    Search HDF5 recursively for a dataset whose basename matches one of possible_short_names.
    """
    possible = set(possible_short_names)
    found = []

    def visitor(key, obj):
        if isinstance(obj, h5py.Dataset):
            short = key.split("/")[-1]
            if short in possible:
                found.append(key)

    h5.visititems(visitor)
    if found:
        return found[0]
    return None


def _find_first_1d_numeric_dataset_by_keywords(
    h5: h5py.File,
    keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Fallback search for 1D numeric datasets whose full path contains keyword(s).
    """
    if exclude_keywords is None:
        exclude_keywords = []

    candidates = []

    def visitor(key, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        low = key.lower()
        if any(ex.lower() in low for ex in exclude_keywords):
            return
        if not any(kw.lower() in low for kw in keywords):
            return
        if len(obj.shape) != 1:
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        candidates.append(key)

    h5.visititems(visitor)

    if candidates:
        candidates = sorted(candidates, key=lambda x: (len(x), x))
        return candidates[0]
    return None


def load_saved_spectrum_h5(
    path: str,
    wave_key: Optional[str] = None,
    flux_key: Optional[str] = None,
    tau_key: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load one saved Trident spectrum HDF5 file.

    It tries several possible schemas:
        wave, wavelength, lambda, lambda_field
        flux, flux_field, normalized_flux
        tau, optical_depth

    If flux is unavailable but tau exists, flux = exp(-tau).

    Returns
    -------
    wave : 1D array
        Wavelength array.
    flux : 1D array
        Normalized flux array.
    meta : dict
        Metadata including which keys were used.
    """
    with h5py.File(path, "r") as h5:
        used_wave_key = wave_key
        used_flux_key = flux_key
        used_tau_key = tau_key

        if used_wave_key is None:
            used_wave_key = _find_dataset_recursive(
                h5,
                [
                    "wave",
                    "wavelength",
                    "wavelengths",
                    "lambda",
                    "lambda_field",
                    "lambda_obs",
                    "wavelength_obs",
                    "wave_obs",
                ],
            )

        if used_flux_key is None:
            used_flux_key = _find_dataset_recursive(
                h5,
                [
                    "flux",
                    "flux_field",
                    "normalized_flux",
                    "flux_normalized",
                    "flux_obs",
                    "noiseless_flux",
                ],
            )

        if used_tau_key is None:
            used_tau_key = _find_dataset_recursive(
                h5,
                [
                    "tau",
                    "optical_depth",
                    "tau_field",
                ],
            )

        if used_wave_key is None:
            used_wave_key = _find_first_1d_numeric_dataset_by_keywords(
                h5,
                keywords=["wave", "lambda"],
                exclude_keywords=["velocity", "vel"],
            )

        if used_flux_key is None:
            used_flux_key = _find_first_1d_numeric_dataset_by_keywords(
                h5,
                keywords=["flux"],
                exclude_keywords=["err", "error", "sigma"],
            )

        if used_tau_key is None:
            used_tau_key = _find_first_1d_numeric_dataset_by_keywords(
                h5,
                keywords=["tau", "optical"],
                exclude_keywords=["err", "error", "sigma"],
            )

        if used_wave_key is None:
            raise KeyError(
                f"Could not identify wavelength dataset in {path}. "
                f"Use print_hdf5_structure(path) and pass wave_key manually."
            )

        if used_flux_key is None and used_tau_key is None:
            raise KeyError(
                f"Could not identify flux or tau dataset in {path}. "
                f"Use print_hdf5_structure(path) and pass flux_key/tau_key manually."
            )

        wave = np.asarray(h5[used_wave_key][()]).squeeze()

        if used_flux_key is not None:
            flux = np.asarray(h5[used_flux_key][()]).squeeze()
        else:
            tau = np.asarray(h5[used_tau_key][()]).squeeze()
            flux = np.exp(-tau)

    if wave.ndim > 1:
        if verbose:
            print(f"[WARN] wavelength has shape {wave.shape}; using first row.")
        wave = wave[0]

    if flux.ndim > 1:
        if verbose:
            print(f"[WARN] flux has shape {flux.shape}; using first row.")
        flux = flux[0]

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError(f"Loaded wave/flux must be 1D. Got wave={wave.shape}, flux={flux.shape}")

    if len(wave) != len(flux):
        raise ValueError(f"wave and flux length mismatch: wave={len(wave)}, flux={len(flux)}")

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    meta = {
        "wave_key": used_wave_key,
        "flux_key": used_flux_key,
        "tau_key": used_tau_key,
        "n_pixels": len(wave),
        "wave_min": float(np.nanmin(wave)),
        "wave_max": float(np.nanmax(wave)),
        "flux_min": float(np.nanmin(flux)),
        "flux_max": float(np.nanmax(flux)),
    }

    return wave, flux, meta


# ============================================================
# File discovery
# ============================================================

def discover_spectrum_files(
    cfg: FitConfig,
    mode: Optional[str] = None,
    alpha: Optional[int] = None,
    ray_id_contains: Optional[str] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Discover saved per-spectrum HDF5 files.

    It searches:
        spectra_h5/*_spectrum.h5

    It excludes combined files by default.
    """
    spectra_h5_dir = get_spectra_h5_dir(cfg)

    patterns = [
        os.path.join(spectra_h5_dir, "*_spectrum.h5"),
        os.path.join(spectra_h5_dir, "*_spectrum.hdf5"),
        os.path.join(spectra_h5_dir, "*.h5"),
        os.path.join(spectra_h5_dir, "*.hdf5"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))

    files = [
        f for f in files
        if "all_rays" not in os.path.basename(f).lower()
        and "combined" not in f.lower()
        and "summary" not in os.path.basename(f).lower()
    ]

    if mode is not None:
        mode = mode.strip().lower()
        if mode == "flip":
            files = [f for f in files if "_flip_" in os.path.basename(f) and "_noflip_" not in os.path.basename(f)]
        elif mode == "noflip":
            files = [f for f in files if "_noflip_" in os.path.basename(f)]
        else:
            raise ValueError("mode must be None, 'flip', or 'noflip'.")

    if alpha is not None:
        files = [f for f in files if f"alpha{int(alpha)}_" in os.path.basename(f)]

    if ray_id_contains is not None and ray_id_contains.strip():
        files = [f for f in files if ray_id_contains in os.path.basename(f)]

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    return files


# ============================================================
# Noise, binning, and frame handling
# ============================================================

def add_snr_noise(
    flux: np.ndarray,
    snr: float = 10.0,
    seed: int = 42,
    continuum_level: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to continuum-normalized flux.

    For normalized spectra:
        sigma_flux = continuum_level / SNR

    Returns
    -------
    flux_noisy, error
    """
    rng = np.random.default_rng(seed)
    sigma = continuum_level / float(snr)
    error = np.full_like(flux, sigma, dtype=float)
    flux_noisy = flux + rng.normal(0.0, sigma, size=flux.size)
    return flux_noisy, error


def bin_spectrum_npix(
    wave: np.ndarray,
    flux: np.ndarray,
    npix: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin a spectrum by a fixed number of native pixels.

    This is intended for already LSF-convolved noiseless spectra.

    Processing logic:
        input saved LSF-convolved spectrum
            -> average every npix consecutive wavelength pixels
            -> average every npix consecutive flux pixels

    Parameters
    ----------
    wave : array
        Wavelength array.
    flux : array
        Normalized flux array.
    npix : int
        Number of native pixels per output bin. Default is 3.

    Returns
    -------
    wave_binned : array
        Mean wavelength in each bin.
    flux_binned : array
        Mean normalized flux in each bin.

    Notes
    -----
    This assumes the input wavelength grid is already sorted and nearly uniform.
    The loader already sorts the arrays. Any leftover pixels that do not fill
    a complete bin are discarded.
    """
    if npix is None:
        npix = 1

    npix = int(npix)

    if npix < 1:
        raise ValueError(f"npix must be >= 1. Received npix={npix}")

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError(f"wave and flux must be 1D. Got wave={wave.shape}, flux={flux.shape}")

    if len(wave) != len(flux):
        raise ValueError(f"wave and flux length mismatch: wave={len(wave)}, flux={len(flux)}")

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]

    if len(wave) == 0:
        raise ValueError("No finite pixels available for binning.")

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    if npix == 1:
        return wave.copy(), flux.copy()

    n_full = len(wave) // npix

    if n_full < 1:
        raise ValueError(
            f"Not enough pixels to bin spectrum: len(wave)={len(wave)}, npix={npix}"
        )

    n_use = n_full * npix

    wave_trim = wave[:n_use]
    flux_trim = flux[:n_use]

    wave_binned = wave_trim.reshape(n_full, npix).mean(axis=1)
    flux_binned = flux_trim.reshape(n_full, npix).mean(axis=1)

    return wave_binned, flux_binned


def convert_wave_to_rest_if_needed(
    wave: np.ndarray,
    line_rest: float,
    z: float,
) -> Tuple[np.ndarray, str]:
    """
    Decide whether the saved wavelength is rest-frame or observed-frame.

    If line_rest is inside wave range, assume rest-frame.
    If line_rest*(1+z) is inside wave range, convert observed -> rest.
    If neither is true, use a fallback.
    """
    wmin = float(np.nanmin(wave))
    wmax = float(np.nanmax(wave))
    obs_center = line_rest * (1.0 + z)

    contains_rest = (wmin <= line_rest <= wmax)
    contains_obs = (wmin <= obs_center <= wmax)

    if contains_rest:
        return wave.copy(), "rest"

    if contains_obs:
        return wave / (1.0 + z), "observed_to_rest"

    if z > 0 and np.nanmedian(wave) > line_rest * (1.0 + 0.5 * z):
        return wave / (1.0 + z), "observed_to_rest_fallback"

    return wave.copy(), "rest_fallback"


def preprocess_spectrum_for_fitting(
    cfg: FitConfig,
    wave: np.ndarray,
    flux_clean: np.ndarray,
    line_rest: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, int]:
    """
    Apply the analysis preprocessing in the correct scientific order.

    Since the saved spectrum already includes the LSF, the order is:

        1. convert wavelength to rest frame if needed
        2. bin the noiseless LSF-convolved flux by cfg.bin_npix
        3. add Gaussian noise at cfg.snr

    No additional LSF convolution is applied.

    Returns
    -------
    wave_proc : array
        Processed rest-frame wavelength grid.
    flux_noisy : array
        Binned and noisy normalized flux.
    error : array
        Flux uncertainty array.
    wave_frame : str
        Frame-conversion label.
    did_bin : bool
        Whether binning was applied.
    bin_npix_used : int
        Number of native pixels per bin.
    """
    wave_rest, wave_frame = convert_wave_to_rest_if_needed(
        wave=wave,
        line_rest=line_rest,
        z=cfg.z,
    )

    did_bin = bool(cfg.bin_before_noise)
    bin_npix_used = int(cfg.bin_npix) if cfg.bin_before_noise else 1

    if cfg.bin_before_noise:
        wave_proc, flux_proc = bin_spectrum_npix(
            wave_rest,
            flux_clean,
            npix=cfg.bin_npix,
        )
    else:
        wave_proc = wave_rest.copy()
        flux_proc = flux_clean.copy()

    flux_noisy, error = add_snr_noise(
        flux_proc,
        snr=cfg.snr,
        seed=seed,
        continuum_level=1.0,
    )

    return wave_proc, flux_noisy, error, wave_frame, did_bin, bin_npix_used


# ============================================================
# Table and upper-limit helpers
# ============================================================

def make_results_table() -> Table:
    return Table(
        names=(
            "SID",
            "RayID",
            "Mode",
            "Alpha",
            "Filename",
            "SavedLine",
            "PygadIon",
            "EW_mA",
            "dEW_mA",
            "logN",
            "dlogN",
            "b_kms",
            "db_kms",
            "v_kms",
            "dv_kms",
            "lambda_A",
            "dlambda_A",
            "UpLim",
            "Sat",
            "Chisq",
            "Nregions",
            "WaveFrame",
            "Binned",
            "BinNpix",
            "SourceFile",
        ),
        dtype=[
            "i8",
            "U80",
            "U16",
            "i8",
            "U256",
            "U40",
            "U40",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "bool",
            "bool",
            "f8",
            "i8",
            "U40",
            "bool",
            "i8",
            "U512",
        ],
    )


def make_empty_error_row(
    cfg: FitConfig,
    file_meta: Dict[str, Any],
    saved_line: str,
    pg_ion: str,
    source_file: str,
    wave_frame: str,
    did_bin: bool = False,
    bin_npix_used: int = 1,
) -> Table:
    t = make_results_table()
    t.add_row((
        cfg.sid,
        str(file_meta.get("ray_id", "unknown")),
        str(file_meta.get("mode", "unknown")),
        int(file_meta.get("alpha", -1)) if file_meta.get("alpha") is not None else -1,
        str(file_meta.get("filename", os.path.basename(source_file))),
        saved_line,
        pg_ion,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        False,
        False,
        np.nan,
        0,
        wave_frame,
        bool(did_bin),
        int(bin_npix_used),
        source_file,
    ))
    return t


def ew_to_logN_limit(pg_ion: str, ew_mA: float, sigma_factor: float = 3.0) -> float:
    """
    Convert EW uncertainty in mA to logN upper limit using linear curve-of-growth.

    Formula:
        N [cm^-2] = 1.13e20 * W_A / (f * lambda_A^2)

    Here W_A = sigma_factor * dEW_A.
    """
    if not np.isfinite(ew_mA) or ew_mA <= 0:
        return np.nan

    fosc = get_line_fosc(pg_ion)
    lam = get_line_rest_wave(pg_ion)

    W_A = sigma_factor * ew_mA * 1e-3
    N = 1.13e20 * W_A / (fosc * lam**2)

    if not np.isfinite(N) or N <= 0:
        return np.nan

    return float(np.log10(N))


def equivalent_width_mA(wave_A: np.ndarray, flux: np.ndarray) -> float:
    """
    Compute EW in mA.
    Positive EW for absorption.
    """
    if len(wave_A) < 2:
        return np.nan
    return float(np.trapz(1.0 - flux, wave_A) * 1000.0)


def equivalent_width_error_mA(wave_A: np.ndarray, error: np.ndarray) -> float:
    """
    Compute simple EW error in mA:
        dEW = sqrt(sum(sigma_i^2)) * median(delta_lambda)
    """
    if len(wave_A) < 2:
        return np.nan
    dw = float(np.nanmedian(np.abs(np.diff(wave_A))))
    return float(np.sqrt(np.nansum(error**2)) * dw * 1000.0)


# ============================================================
# Fitting
# ============================================================

def fit_line_in_spectrum(
    cfg: FitConfig,
    wave: np.ndarray,
    flux_clean: np.ndarray,
    source_file: str,
    saved_line: str,
    pg_ion: str,
    file_seed_offset: int = 0,
    wave_key: Optional[str] = None,
    flux_key: Optional[str] = None,
) -> Tuple[Table, Optional[Dict[str, Any]]]:
    """
    Fit one line in one full spectrum file.

    The saved file contains full wavelength coverage.

    This function:
        - maps line label -> pygad ion
        - converts wavelength to rest-frame if needed
        - bins the noiseless already-LSF-convolved spectrum
        - adds SNR noise after binning
        - extracts +/- velocity_window around transition
        - runs pygad find_regions and fit_profiles
        - returns an Astropy table and diagnostic info

    No LSF convolution is applied here.
    """
    line_rest = get_line_rest_wave(pg_ion)
    file_meta = parse_alpha_mode_from_filename(source_file)

    wave_proc, flux_noisy, error, wave_frame, did_bin, bin_npix_used = preprocess_spectrum_for_fitting(
        cfg=cfg,
        wave=wave,
        flux_clean=flux_clean,
        line_rest=line_rest,
        seed=cfg.seed + int(file_seed_offset),
    )

    vel = wave_to_vel(wave_proc, line_rest)
    mask = (vel >= -cfg.velocity_window) & (vel <= cfg.velocity_window)

    t = make_results_table()

    if np.count_nonzero(mask) < max(5, cfg.min_region_width + 2):
        if cfg.verbose:
            print(
                f"[SKIP-LINE] {saved_line:12s} {pg_ion:10s}: "
                f"not enough pixels near line. "
                f"processed wave range=({wave_proc.min():.2f}, {wave_proc.max():.2f}); "
                f"binned={did_bin}, bin_npix={bin_npix_used}"
            )
        return make_empty_error_row(
            cfg,
            file_meta,
            saved_line,
            pg_ion,
            source_file,
            wave_frame,
            did_bin=did_bin,
            bin_npix_used=bin_npix_used,
        ), None

    w = wave_proc[mask]
    f = flux_noisy[mask]
    e = error[mask]
    v = vel[mask]

    sat_flag = bool(np.nanmin(f) <= 0.2)

    try:
        regions, _ = pg.analysis.vpfit.find_regions(
            w,
            f,
            e,
            min_region_width=cfg.min_region_width,
            N_sigma=cfg.N_sigma,
            extend=True,
        )
        n_regions = len(regions)
    except Exception as exc:
        warnings.warn(f"find_regions failed for {saved_line} in {source_file}: {exc}")
        regions = []
        n_regions = 0

    fit = None

    if n_regions > 0:
        try:
            fit = pg.analysis.vpfit.fit_profiles(
                pg_ion,
                w,
                f,
                e,
                chisq_lim=cfg.chisq_lim,
                max_lines=cfg.max_lines,
                mode="Voigt",
                logN_bounds=list(cfg.logN_bounds),
                b_bounds=list(cfg.b_bounds),
                min_region_width=cfg.min_region_width,
                N_sigma=cfg.N_sigma,
                extend=True,
            )
        except Exception as exc:
            warnings.warn(f"fit_profiles failed for {saved_line} in {source_file}: {exc}")
            fit = None

    if fit is not None and len(fit["EW"]) > 0:
        for j in range(len(fit["EW"])):
            lam_j = float(fit["l"][j])
            dlam_j = float(fit["dl"][j])

            v_j = float(wave_to_vel(lam_j, line_rest))
            dv_j = float(abs(wave_to_vel(lam_j + dlam_j, line_rest) - wave_to_vel(lam_j, line_rest)))

            t.add_row((
                cfg.sid,
                str(file_meta.get("ray_id", "unknown")),
                str(file_meta.get("mode", "unknown")),
                int(file_meta.get("alpha", -1)) if file_meta.get("alpha") is not None else -1,
                str(file_meta.get("filename", os.path.basename(source_file))),
                saved_line,
                pg_ion,
                float(fit["EW"][j]) * 1000.0,
                np.nan,
                float(fit["N"][j]),
                float(fit["dN"][j]),
                float(fit["b"][j]),
                float(fit["db"][j]),
                v_j,
                dv_j,
                lam_j,
                dlam_j,
                False,
                sat_flag,
                float(fit["chisq"][j]),
                int(n_regions),
                wave_frame,
                bool(did_bin),
                int(bin_npix_used),
                source_file,
            ))

    else:
        ul_mask = (v >= -cfg.upper_limit_window) & (v <= cfg.upper_limit_window)

        if np.count_nonzero(ul_mask) >= 2:
            ew_mA = equivalent_width_mA(w[ul_mask], f[ul_mask])
            dew_mA = equivalent_width_error_mA(w[ul_mask], e[ul_mask])
            logN_lim = ew_to_logN_limit(pg_ion, dew_mA, sigma_factor=3.0)
        else:
            ew_mA = np.nan
            dew_mA = np.nan
            logN_lim = np.nan

        t.add_row((
            cfg.sid,
            str(file_meta.get("ray_id", "unknown")),
            str(file_meta.get("mode", "unknown")),
            int(file_meta.get("alpha", -1)) if file_meta.get("alpha") is not None else -1,
            str(file_meta.get("filename", os.path.basename(source_file))),
            saved_line,
            pg_ion,
            ew_mA,
            dew_mA,
            logN_lim,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            cfg.upper_limit_window,
            np.nan,
            np.nan,
            True,
            sat_flag,
            np.nan,
            int(n_regions),
            wave_frame,
            bool(did_bin),
            int(bin_npix_used),
            source_file,
        ))

    diag = {
        "saved_line": saved_line,
        "pg_ion": pg_ion,
        "line_rest": line_rest,
        "wave_rest": w,
        "velocity": v,
        "flux_noisy": f,
        "error": e,
        "fit": fit,
        "regions": regions,
        "n_regions": n_regions,
        "sat_flag": sat_flag,
        "wave_frame": wave_frame,
        "binned": bool(did_bin),
        "bin_npix": int(bin_npix_used),
        "source_file": source_file,
        "file_meta": file_meta,
    }

    return t, diag


def fit_all_lines_in_file(
    spectrum_file: str,
    cfg: FitConfig,
    wave_key: Optional[str] = None,
    flux_key: Optional[str] = None,
    tau_key: Optional[str] = None,
    make_plots: Optional[bool] = None,
) -> Tuple[Table, List[Dict[str, Any]]]:
    """
    Fit all configured lines in a single saved spectrum HDF5 file.

    This is the main function to call from Jupyter for one test example.
    """
    if make_plots is None:
        make_plots = cfg.make_plots

    os.makedirs(get_output_dir(cfg), exist_ok=True)

    wave, flux_clean, load_meta = load_saved_spectrum_h5(
        spectrum_file,
        wave_key=wave_key,
        flux_key=flux_key,
        tau_key=tau_key,
        verbose=cfg.verbose,
    )

    if cfg.verbose:
        print("\nLoaded spectrum")
        print("-" * 80)
        print(f"file       : {spectrum_file}")
        print(f"wave_key   : {load_meta['wave_key']}")
        print(f"flux_key   : {load_meta['flux_key']}")
        print(f"tau_key    : {load_meta['tau_key']}")
        print(f"N pixels   : {load_meta['n_pixels']}")
        print(f"wave range : {load_meta['wave_min']:.4f} - {load_meta['wave_max']:.4f}")
        print(f"flux range : {load_meta['flux_min']:.4f} - {load_meta['flux_max']:.4f}")
        print(f"LSF status : assumed already included in saved spectrum")
        print(f"Binning    : {cfg.bin_before_noise}, bin_npix={cfg.bin_npix}")
        print(f"Noise      : SNR={cfg.snr:g}, added after binning")
        print("-" * 80)

    all_rows = make_results_table()
    diagnostics = []

    for iline, saved_line in enumerate(cfg.line_labels):
        if saved_line not in cfg.line_map:
            print(f"[SKIP] No line-map entry for {saved_line}")
            continue

        pg_ion = cfg.line_map[saved_line]

        if pg_ion not in pg.analysis.absorption_spectra.lines:
            print(f"[SKIP] pygad key missing: {saved_line} -> {pg_ion}")
            continue

        if cfg.verbose:
            print(f"Fitting {saved_line:12s} -> {pg_ion:10s}")

        row_table, diag = fit_line_in_spectrum(
            cfg=cfg,
            wave=wave,
            flux_clean=flux_clean,
            source_file=spectrum_file,
            saved_line=saved_line,
            pg_ion=pg_ion,
            file_seed_offset=1000 * iline,
        )

        all_rows = vstack([all_rows, row_table])

        if diag is not None:
            diagnostics.append(diag)

            if make_plots:
                plot_fit_diagnostic(cfg, diag)

    return all_rows, diagnostics


def fit_small_batch_for_sid(
    cfg: FitConfig,
    mode: Optional[str] = None,
    alpha: Optional[int] = None,
    ray_id_contains: Optional[str] = None,
    max_files: Optional[int] = 5,
    write_outputs: bool = True,
) -> Tuple[Table, List[Dict[str, Any]]]:
    """
    Fit all configured lines for a small number of saved spectrum files for one SID.

    Good notebook test call:
        cfg = FitConfig(sid=488530, snr=10, z=0.0, bin_npix=3)
        table, diagnostics = fit_small_batch_for_sid(cfg, mode="flip", alpha=0, max_files=1)
    """
    files = discover_spectrum_files(
        cfg,
        mode=mode,
        alpha=alpha,
        ray_id_contains=ray_id_contains,
        max_files=max_files,
    )

    if cfg.verbose:
        print("=" * 100)
        print("Batch file discovery")
        print(f"spectra_h5 dir : {get_spectra_h5_dir(cfg)}")
        print(f"N files        : {len(files)}")
        print(f"Binning        : {cfg.bin_before_noise}, bin_npix={cfg.bin_npix}")
        print(f"Noise order    : added after binning")
        for f in files[:20]:
            print(f"  {f}")
        print("=" * 100)

    all_rows = make_results_table()
    all_diagnostics = []

    if len(files) == 0:
        print("No files found. Check spectra_h5 path.")
        return all_rows, all_diagnostics

    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(f)}")
        rows, diags = fit_all_lines_in_file(f, cfg, make_plots=cfg.make_plots)
        all_rows = vstack([all_rows, rows])
        all_diagnostics.extend(diags)

    if write_outputs:
        write_fit_outputs(cfg, all_rows, tag=batch_tag(mode=mode, alpha=alpha, max_files=max_files))

    return all_rows, all_diagnostics


def batch_tag(mode: Optional[str] = None, alpha: Optional[int] = None, max_files: Optional[int] = None) -> str:
    parts = []
    if mode is not None:
        parts.append(mode)
    if alpha is not None:
        parts.append(f"alpha{alpha}")
    if max_files is not None:
        parts.append(f"n{max_files}")
    if not parts:
        return "batch"
    return "_".join(parts)


# ============================================================
# Model plotting
# ============================================================

def generate_params_from_fit(fit: Any) -> np.ndarray:
    """
    Convert pygad fit result to model_tau parameter vector:
        [N0, b0, l0, N1, b1, l1, ...]
    """
    params = np.empty(3 * len(fit["N"]))
    params[0::3] = fit["N"]
    params[1::3] = fit["b"]
    params[2::3] = fit["l"]
    return params


def plot_fit_diagnostic(cfg: FitConfig, diag: Dict[str, Any]) -> str:
    """
    Save one diagnostic plot for one fitted line.
    """
    outdir = get_output_dir(cfg)
    os.makedirs(outdir, exist_ok=True)

    saved_line = diag["saved_line"]
    pg_ion = diag["pg_ion"]
    line_rest = diag["line_rest"]
    source_file = diag["source_file"]
    file_meta = diag["file_meta"]

    w = diag["wave_rest"]
    v = diag["velocity"]
    f = diag["flux_noisy"]
    e = diag["error"]
    fit = diag["fit"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.step(
        v,
        f,
        where="mid",
        lw=2.0,
        alpha=0.85,
        label="Binned + noisy spectrum",
    )

    ax.fill_between(
        v,
        f - e,
        f + e,
        step="mid",
        alpha=0.25,
        label=r"$1\sigma$ error",
    )

    ax.axhline(1.0, ls="--", lw=1.8, color="0.4")
    ax.axvline(0.0, ls="--", lw=1.8, color="0.2")

    if fit is not None and len(fit["N"]) > 0:
        try:
            line_data = pg.analysis.absorption_spectra.lines[pg_ion]
            params = generate_params_from_fit(fit)
            tau_model = pg.analysis.vpfit.model_tau(line_data, params, w, mode="Voigt")
            model_flux = np.exp(-tau_model)

            ax.step(
                v,
                model_flux,
                where="mid",
                lw=3.0,
                label="Voigt fit",
            )

            for j in range(len(fit["N"])):
                lam_j = float(fit["l"][j])
                b_j = float(fit["b"][j])
                N_j = float(fit["N"][j])
                v_j = float(wave_to_vel(lam_j, line_rest))
                fwhm_v = 2.0 * np.sqrt(np.log(2.0)) * b_j

                ax.axvline(v_j, ls=":", lw=2.0)

                y_arrow = 0.74 - 0.08 * (j % 5)
                ax.annotate(
                    "",
                    xy=(v_j - fwhm_v / 2.0, y_arrow),
                    xytext=(v_j + fwhm_v / 2.0, y_arrow),
                    arrowprops=dict(arrowstyle="<->", lw=1.4),
                )

                ax.text(
                    v_j,
                    y_arrow + 0.035,
                    rf"$\log N={N_j:.2f}$, $b={b_j:.1f}$",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
        except Exception as exc:
            warnings.warn(f"Could not generate model flux for plot: {exc}")

    info = (
        f"{saved_line}\n"
        f"{file_meta.get('mode', 'unknown')}, alpha={file_meta.get('alpha', 'NA')}\n"
        f"SNR={cfg.snr:g}\n"
        f"bin={diag.get('bin_npix', 1)} px"
    )

    ax.text(
        0.97,
        0.08,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=16,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25"),
    )

    ax.set_xlabel(r"Velocity [km s$^{-1}$]")
    ax.set_ylabel("Normalized Flux")
    ax.set_xlim(-cfg.velocity_window, cfg.velocity_window)
    ax.set_ylim(-0.1, 1.5)
    ax.minorticks_on()
    ax.legend(fontsize=13, loc="best")

    filename = (
        f"fit_sid{cfg.sid}_"
        f"{sanitize_name(file_meta.get('ray_id', 'ray'))}_"
        f"{sanitize_name(file_meta.get('mode', 'mode'))}_"
        f"alpha{file_meta.get('alpha', 'NA')}_"
        f"{sanitize_name(saved_line)}_"
        f"bin{diag.get('bin_npix', 1)}.png"
    )

    outpath = os.path.join(outdir, filename)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return outpath


def show_diagnostic_inline(diag: Dict[str, Any], cfg: FitConfig) -> None:
    """
    Jupyter helper: display one diagnostic plot inline.
    This does not save unless you also call plot_fit_diagnostic.
    """
    saved_line = diag["saved_line"]
    pg_ion = diag["pg_ion"]
    line_rest = diag["line_rest"]

    w = diag["wave_rest"]
    v = diag["velocity"]
    f = diag["flux_noisy"]
    e = diag["error"]
    fit = diag["fit"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.step(v, f, where="mid", lw=2.0, alpha=0.85, label="Binned + noisy spectrum")
    ax.fill_between(v, f - e, f + e, step="mid", alpha=0.25, label=r"$1\sigma$ error")
    ax.axhline(1.0, ls="--", lw=1.8, color="0.4")
    ax.axvline(0.0, ls="--", lw=1.8, color="0.2")

    if fit is not None and len(fit["N"]) > 0:
        line_data = pg.analysis.absorption_spectra.lines[pg_ion]
        params = generate_params_from_fit(fit)
        tau_model = pg.analysis.vpfit.model_tau(line_data, params, w, mode="Voigt")
        model_flux = np.exp(-tau_model)
        ax.step(v, model_flux, where="mid", lw=3.0, label="Voigt fit")

        for j in range(len(fit["N"])):
            lam_j = float(fit["l"][j])
            b_j = float(fit["b"][j])
            N_j = float(fit["N"][j])
            v_j = float(wave_to_vel(lam_j, line_rest))
            ax.axvline(v_j, ls=":", lw=2.0)
            ax.text(
                v_j,
                0.15 + 0.08 * (j % 4),
                rf"$\log N={N_j:.2f}$" + "\n" + rf"$b={b_j:.1f}$",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.text(
        0.97,
        0.08,
        f"{saved_line}\nbin={diag.get('bin_npix', 1)} px",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25"),
    )

    ax.set_xlabel(r"Velocity [km s$^{-1}$]")
    ax.set_ylabel("Normalized Flux")
    ax.set_xlim(-cfg.velocity_window, cfg.velocity_window)
    ax.set_ylim(-0.1, 1.5)
    ax.minorticks_on()
    ax.legend(fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================
# Output
# ============================================================

def write_fit_outputs(cfg: FitConfig, table: Table, tag: str = "results") -> Tuple[str, str]:
    """
    Write fit results robustly.

    Important:
    Astropy ascii.fixed_width fails on empty tables.
    This function skips writing if there are zero rows.
    """
    outdir = get_output_dir(cfg)
    os.makedirs(outdir, exist_ok=True)

    if len(table) == 0:
        print("[WARN] Empty table; not writing output files.")
        return "", ""

    fixed_path = os.path.join(outdir, f"fit_results_sid{cfg.sid}_{tag}.txt")
    csv_path = os.path.join(outdir, f"fit_results_sid{cfg.sid}_{tag}.csv")

    table.write(fixed_path, format="ascii.fixed_width", overwrite=True)
    table.write(csv_path, format="ascii.csv", overwrite=True)

    print(f"Wrote fixed-width table: {fixed_path}")
    print(f"Wrote CSV table        : {csv_path}")

    return fixed_path, csv_path


# ============================================================
# Convenience notebook setup
# ============================================================

def notebook_setup() -> None:
    """
    Run this in Jupyter after importing the module.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONNOUSERSITE"] = "1"
    set_plot_style()

    print("Notebook setup complete.")
    print(f"pygad loaded from: {getattr(pg, '__file__', 'unknown')}")
    print(f"N pygad lines: {len(pg.analysis.absorption_spectra.lines.keys())}")


def print_example_usage() -> None:
    print(
        r'''
Example Jupyter usage
---------------------

import sys
sys.path.insert(0, "/home/tsingh65/m61-tng/notebooks")

import pygad_fit_saved_spectra as pfit
pfit.notebook_setup()

cfg = pfit.FitConfig(
    sid=488530,
    snap=99,
    run_label="L4Rvir",
    base_dir="/scratch/tsingh65/m61-tng/outputs",
    z=0.0,
    snr=10.0,
    velocity_window=800,
    upper_limit_window=50,

    # Important:
    # LSF is assumed already included in the saved spectrum.
    # This bins the noiseless LSF-convolved spectrum before adding noise.
    bin_before_noise=True,
    bin_npix=3,

    N_sigma=3,
    min_region_width=3,
    logN_bounds=(12, 18),
    b_bounds=(6, 150),
    chisq_lim=1,
    max_lines=6,
    make_plots=True,
    verbose=True,
)

# For exact O VI paper-style settings:
cfg_ovi = pfit.FitConfig(
    sid=488530,
    snap=99,
    run_label="L4Rvir",
    base_dir="/scratch/tsingh65/m61-tng/outputs",
    z=0.0,
    snr=10.0,
    velocity_window=800,
    upper_limit_window=50,
    bin_before_noise=True,
    bin_npix=3,
    N_sigma=3,
    min_region_width=3,
    logN_bounds=(13.49, 18.0),
    b_bounds=(6.0, 100.0),
    chisq_lim=1,
    max_lines=6,
    line_labels=["O VI 1032"],
    make_plots=True,
    verbose=True,
)

# Check files
files = pfit.discover_spectrum_files(cfg, mode="flip", alpha=0, max_files=5)
files

# Inspect one file if needed
pfit.print_hdf5_structure(files[0])

# Fit all configured lines in one saved spectrum
table, diagnostics = pfit.fit_all_lines_in_file(files[0], cfg)
table

# Show one fitted diagnostic inline
pfit.show_diagnostic_inline(diagnostics[0], cfg)

# Fit one file only and write outputs
pfit.write_fit_outputs(cfg, table, tag="one_file_flip_alpha0")

# Fit small batch: one alpha, one mode, max 2 files
batch_table, batch_diags = pfit.fit_small_batch_for_sid(
    cfg,
    mode="flip",
    alpha=0,
    max_files=2,
    write_outputs=True,
)
batch_table
'''
    )