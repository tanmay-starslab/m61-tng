#!/usr/bin/env python3
"""
Inclination-sweep alpha movies for M61 / TNG50 SID 488530.

This script makes Slurm-friendly gas and stellar projection products for
alpha = 0..180 deg at a set of requested inclinations.  The geometry follows
the corrected convention used by orient_m61.py:

  fixed observer/QSO sky plane, galaxy rotated about its PCA-v3 disk normal.

For each inclination and alpha it computes/caches a map NPZ.  Separate render
commands then make publication-style PNG frames and MP4 videos.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.ticker import AutoMinorLocator
from yt.visualization.particle_plots import OffAxisParticleProjectionPlot

import m61_oriented_HI_vlos_rotation_alpha5 as base
import orient_m61


SID = 488530
SNAP = 99
H_TNG = base.H_TNG

DEFAULT_OUT = (
    "/scratch/tsingh65/m61-tng/outputs/sid488530/"
    "inclination_sweep_alpha000_180_gas_stars_movies"
)
DEFAULT_RECIPE = (
    "/scratch/tsingh65/m61-tng/outputs/sid488530/"
    "rays_and_recipes_sid488530_snap99_L4Rvir/rays_sid488530.csv"
)
DEFAULT_ORIENT_HEADER = (
    "/scratch/tsingh65/m61-tng/outputs/sid488530/"
    "rays_and_recipes_sid488530_snap99_L4Rvir/orient_header_sid488530.json"
)

INCLINATIONS = [0.0, 23.0, 45.0, 75.0, 90.0, 135.0, 170.0, 180.0]
ALPHAS = list(range(0, 181))
LOGNHI_CUT = float(np.log10(1.25e20))
MSUN_R_AB = 4.65
MAG_ARCSEC2_CONST = 21.572


@dataclass
class Config:
    output_dir: str = DEFAULT_OUT
    recipe_csv: str = DEFAULT_RECIPE
    orient_header: str = DEFAULT_ORIENT_HEADER
    sid: int = SID
    snap: int = SNAP
    sightline_id: str = "J122138+043026"
    mode: str = "noflip"
    pa_deg: float = 138.0
    width_kpc: float = 100.0
    npix: int = 1024
    fps: int = 18
    recompute: bool = False
    rerender: bool = False
    subtract_systemic: bool = False

    @property
    def half_width_kpc(self) -> float:
        return 0.5 * self.width_kpc

    @property
    def pixel_scale_kpc(self) -> float:
        return self.width_kpc / self.npix


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 190,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def inc_label(inc: float) -> str:
    return f"inc{int(round(inc)):03d}"


def setup_paths(config: Config, inc: float | None = None, component: str | None = None) -> dict[str, Path]:
    root = Path(config.output_dir)
    paths: dict[str, Path] = {
        "root": root,
        "logs_sbatch": root / "logs_sbatch",
        "logs": root / "logs",
    }
    if inc is not None:
        idir = root / inc_label(inc)
        paths["inc"] = idir
        paths["inc_logs"] = idir / "logs"
        if component is not None:
            cdir = idir / component
            paths.update(
                {
                    "component": cdir,
                    "data": cdir / "data",
                    "frames": cdir / "frames" / "combined",
                    "videos": cdir / "videos",
                    "figures": cdir / "figures",
                    "logs_component": cdir / "logs",
                }
            )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_dir: Path, label: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / f"{label}.log", mode="w")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(base.json_sanitize(payload), f, indent=2, sort_keys=True)


def normalize(vec: np.ndarray, name: str = "vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def load_orient_header(config: Config) -> dict[str, Any]:
    with open(config.orient_header) as f:
        return json.load(f)


def load_disk_normal(config: Config) -> np.ndarray:
    header = load_orient_header(config)
    normal = header.get("normal_used_hat")
    if normal is None:
        logging.warning("normal_used_hat not found in %s; using PCA-v3 value from prior metadata", config.orient_header)
        normal = [0.1489902267951354, -0.8457239371292382, -0.5123991944628276]
    return normalize(np.asarray(normal, dtype=float), "PCA-v3 disk normal")


def load_reference_qso(config: Config) -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(config.recipe_csv)
    rows = df[
        (df["SubhaloID"].astype(int) == config.sid)
        & (df["sightline_id"].astype(str) == config.sightline_id)
        & (df["mode"].astype(str) == config.mode)
        & (df["alpha_deg"].astype(float).round().astype(int) == 5)
    ]
    if rows.empty:
        logging.warning("Could not find alpha=5 recipe row; using rho=26 kpc, phi=0 deg marker")
        return {"rho_kpc": 26.0, "phi_deg": 0.0, "x_qso_kpc": 26.0, "y_qso_kpc": 0.0}
    row = rows.iloc[0]
    rho = float(row.get("rho_kpc", 26.0))
    phi = float(row.get("phi_deg", 0.0))
    return {
        "rho_kpc": rho,
        "phi_deg": phi,
        "x_qso_kpc": float(rho * np.cos(np.deg2rad(phi))),
        "y_qso_kpc": float(rho * np.sin(np.deg2rad(phi))),
    }


def orientation_for(config: Config, inc: float, alpha: int, normal_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    R_base_noflip, R_base_flip, axis_noflip, axis_flip = orient_m61.build_R_bases(
        normal_hat, float(inc), float(config.pa_deg)
    )
    if config.mode == "flip":
        R_base, axis = R_base_flip, axis_flip
    else:
        R_base, axis = R_base_noflip, axis_noflip
    R_cur = orient_m61.fixed_observer_galaxy_alpha_rotation(R_base, axis, float(alpha))
    ez_obs = np.array([0.0, 0.0, 1.0])
    ey_obs = np.array([0.0, 1.0, 0.0])
    ex_obs = np.array([1.0, 0.0, 0.0])
    los = normalize(ez_obs @ R_cur, "LOS")
    north = normalize(ey_obs @ R_cur, "north")
    east = normalize(ex_obs @ R_cur, "east")
    return los, north, {"R_cur": R_cur, "axis": axis, "east_hat": east}


def gas_npz_path(paths: dict[str, Path], inc: float, alpha: int) -> Path:
    return paths["data"] / f"gas_projection_{inc_label(inc)}_alpha{alpha:03d}_inner100kpc.npz"


def star_npz_path(paths: dict[str, Path], inc: float, alpha: int) -> Path:
    return paths["data"] / f"stellar_projection_{inc_label(inc)}_alpha{alpha:03d}_inner100kpc.npz"


def frame_path(paths: dict[str, Path], alpha: int) -> Path:
    return paths["frames"] / f"frame_{alpha:03d}.png"


def grid(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = config.half_width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def robust_percentile(values: list[np.ndarray] | np.ndarray, p: float, default: float) -> float:
    if isinstance(values, list):
        vals = [np.asarray(v).ravel() for v in values if np.asarray(v).size]
        if not vals:
            return default
        arr = np.concatenate(vals)
    else:
        arr = np.asarray(values).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.nanpercentile(arr, p))


def velocity_component_fields(ds):
    candidates = [
        (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        (("PartType0", "velocity_x"), ("PartType0", "velocity_y"), ("PartType0", "velocity_z")),
        (
            ("PartType0", "particle_velocity_x"),
            ("PartType0", "particle_velocity_y"),
            ("PartType0", "particle_velocity_z"),
        ),
    ]
    for cand in candidates:
        if all(base.yt_field_exists(ds, f) for f in cand):
            return cand
    return None


def add_dynamic_gas_fields(ds, los_holder: dict[str, np.ndarray], systemic_kms: np.ndarray) -> None:
    component_fields = velocity_component_fields(ds)
    raw_vector_field = ("PartType0", "Velocities")
    if component_fields is None and not base.yt_field_exists(ds, raw_vector_field):
        raise RuntimeError("Could not find gas velocity fields.")

    def velocity_component(data, index):
        if component_fields is not None:
            return data[component_fields[index]].to("km/s")
        return data[raw_vector_field][:, index].to("km/s")

    def _vlos(field, data):
        los = los_holder["los"]
        vx = velocity_component(data, 0) - data.ds.quan(systemic_kms[0], "km/s")
        vy = velocity_component(data, 1) - data.ds.quan(systemic_kms[1], "km/s")
        vz = velocity_component(data, 2) - data.ds.quan(systemic_kms[2], "km/s")
        return (los[0] * vx + los[1] * vy + los[2] * vz).to("km/s")

    def _hi_vlos(field, data):
        return data[("gas", "H_p0_number_density")] * data[("gas", "velocity_los_incsweep")]

    def _gas_vlos(field, data):
        return data[("gas", "density")] * data[("gas", "velocity_los_incsweep")]

    if not base.yt_field_exists(ds, ("gas", "velocity_los_incsweep")):
        ds.add_field(("gas", "velocity_los_incsweep"), function=_vlos, sampling_type="particle", units="km/s")
    if not base.yt_field_exists(ds, ("gas", "HI_vlos_integrand_incsweep")):
        ds.add_field(
            ("gas", "HI_vlos_integrand_incsweep"),
            function=_hi_vlos,
            sampling_type="particle",
            units="km/(s*cm**3)",
        )
    if not base.yt_field_exists(ds, ("gas", "gas_density_vlos_integrand_incsweep")):
        ds.add_field(
            ("gas", "gas_density_vlos_integrand_incsweep"),
            function=_gas_vlos,
            sampling_type="particle",
            units="g*km/(s*cm**3)",
        )


def add_dynamic_stellar_fields(ds, los_holder: dict[str, np.ndarray], systemic_kms: np.ndarray) -> None:
    def _stellar_vlos(field, data):
        los = los_holder["los"]
        vx = data[("PartType4", "particle_velocity_x")].to("km/s") - data.ds.quan(systemic_kms[0], "km/s")
        vy = data[("PartType4", "particle_velocity_y")].to("km/s") - data.ds.quan(systemic_kms[1], "km/s")
        vz = data[("PartType4", "particle_velocity_z")].to("km/s") - data.ds.quan(systemic_kms[2], "km/s")
        return (los[0] * vx + los[1] * vy + los[2] * vz).to("km/s")

    def _stellar_r_lum(field, data):
        mag = np.asarray(data[("PartType4", "GFM_StellarPhotometrics_05")].d, dtype=float)
        lum = np.power(10.0, -0.4 * (mag - MSUN_R_AB))
        lum[~np.isfinite(lum)] = 0.0
        return data.ds.arr(lum, "Lsun")

    if not base.yt_field_exists(ds, ("PartType4", "stellar_vlos_incsweep")):
        ds.add_field(
            ("PartType4", "stellar_vlos_incsweep"),
            function=_stellar_vlos,
            sampling_type="particle",
            units="km/s",
        )
    if not base.yt_field_exists(ds, ("PartType4", "stellar_r_luminosity")):
        ds.add_field(
            ("PartType4", "stellar_r_luminosity"),
            function=_stellar_r_lum,
            sampling_type="particle",
            units="Lsun",
        )


def load_common_context(config: Config, paths: dict[str, Path], component: str):
    base_config = base.AnalysisConfig(output_dir=str(paths["root"]), width_kpc=config.width_kpc, npix=config.npix)
    center_ckpch, center_meta = base.load_true_galaxy_center(base_config)
    cutout = base.find_cutout_h5(base_config)
    ds = yt.load(str(cutout))
    if component == "gas":
        base.ensure_gas_alias_fields(ds)
    if config.subtract_systemic:
        systemic = base.systemic_velocity_from_hdf5(cutout, center_ckpch)
    else:
        systemic = {
            "velocity_kms": np.array([0.0, 0.0, 0.0]),
            "method": "disabled for inclination-sweep movies; raw simulation velocities",
            "center_subtracted": False,
        }
    los_holder = {"los": np.array([0.0, 0.0, 1.0])}
    if component == "gas":
        add_dynamic_gas_fields(ds, los_holder, np.asarray(systemic["velocity_kms"], dtype=float))
    else:
        add_dynamic_stellar_fields(ds, los_holder, np.asarray(systemic["velocity_kms"], dtype=float))
    return ds, center_ckpch, center_meta, cutout, systemic, los_holder


def project_gas(config: Config, inc: float, alpha: int) -> Path:
    paths = setup_paths(config, inc, "gas")
    setup_logging(paths["logs_component"], f"project_gas_{inc_label(inc)}_alpha{alpha:03d}")
    out = gas_npz_path(paths, inc, alpha)
    if out.exists() and not config.recompute:
        try:
            with np.load(out, allow_pickle=True) as z:
                _ = z.files
            logging.info("cached gas map: %s", out)
            return out
        except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
            logging.warning("Corrupt cached NPZ %s (%r); recomputing", out, exc)
            out.unlink(missing_ok=True)

    normal = load_disk_normal(config)
    qso = load_reference_qso(config)
    los, north, orientation = orientation_for(config, inc, alpha, normal)
    ds, center_ckpch, center_meta, cutout, systemic, los_holder = load_common_context(config, paths, "gas")
    los_holder["los"] = los

    x, y, X, Y = grid(config)
    center = ds.arr(center_ckpch, "code_length")
    width = ds.arr([config.width_kpc * H_TNG, config.width_kpc * H_TNG], "code_length")

    hi_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "H_p0_number_density"), north)
    hi_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "HI_vlos_integrand_incsweep"), north)
    gas_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "density"), north)
    gas_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "gas_density_vlos_integrand_incsweep"), north)

    N_HI = base.yt_array_to_numpy(hi_den, "cm**-2")
    HI_num = base.yt_array_to_numpy(hi_num, "km/(s*cm**2)")
    gas_sigma = base.yt_array_to_numpy(gas_den, "g/cm**2")
    gas_num_np = base.yt_array_to_numpy(gas_num, "g*km/(s*cm**2)")
    with np.errstate(divide="ignore", invalid="ignore"):
        logN = np.log10(np.where(N_HI > 0, N_HI, np.nan))
        vHI = HI_num / N_HI
        vgas = gas_num_np / gas_sigma
    vHI[~np.isfinite(vHI)] = np.nan
    vgas[~np.isfinite(vgas)] = np.nan
    mask = np.isfinite(logN) & (logN >= LOGNHI_CUT)

    np.savez_compressed(
        out,
        x_kpc=x,
        y_kpc=y,
        X_kpc=X,
        Y_kpc=Y,
        N_HI_cm2=N_HI,
        logN_HI=logN,
        gas_surface_density_g_cm2=gas_sigma,
        vlos_gasweighted_kms=vgas,
        vlos_HIweighted_kms=vHI,
        mask_logNHI_cut=mask,
        logNHI_cut=LOGNHI_CUT,
        alpha_deg=float(alpha),
        inc_deg=float(inc),
        pa_deg=float(config.pa_deg),
        los_hat=los,
        north_hat=north,
        disk_normal_hat=normal,
        x_qso_kpc=float(qso["x_qso_kpc"]),
        y_qso_kpc=float(qso["y_qso_kpc"]),
        rho_kpc=float(qso["rho_kpc"]),
        phi_deg=float(qso["phi_deg"]),
        systemic_velocity_kms=np.asarray(systemic["velocity_kms"], dtype=float),
        subtract_systemic=bool(config.subtract_systemic),
    )
    meta = {
        "component": "gas",
        "output_npz": str(out),
        "inc_deg": inc,
        "alpha_deg": alpha,
        "cutout_h5": str(cutout),
        "galaxy_center_ckpch": center_ckpch,
        "galaxy_center_source": center_meta,
        "systemic_velocity": systemic,
        "los_hat": los,
        "north_hat": north,
        "disk_normal_hat": normal,
        "orientation_axis": orientation["axis"],
        "qso_marker": qso,
        "projection_width_kpc": config.width_kpc,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "logNHI_cut": LOGNHI_CUT,
        "method": "yt.off_axis_projection method=integrate; vlos maps are projection numerator/denominator ratios",
    }
    write_json(paths["logs_component"] / f"project_gas_{inc_label(inc)}_alpha{alpha:03d}_metadata.json", meta)
    good_vhi = vHI[mask & np.isfinite(vHI)]
    logging.info(
        "gas %s alpha=%03d done logN[p50,p99]=%s vHI[p2,p98]=%s -> %s",
        inc_label(inc),
        alpha,
        np.nanpercentile(logN[np.isfinite(logN)], [50, 99]).tolist(),
        np.nanpercentile(good_vhi, [2, 98]).tolist() if good_vhi.size else [],
        out,
    )
    return out


def particle_projection_frb(
    config: Config,
    ds,
    center_ckpch: np.ndarray,
    los: np.ndarray,
    north: np.ndarray,
    field,
    *,
    weight_field=None,
    density=False,
):
    plot = OffAxisParticleProjectionPlot(
        ds,
        normal=normalize(los, "projection LOS"),
        fields=field,
        center=ds.arr(center_ckpch, "code_length"),
        width=ds.quan(config.width_kpc * H_TNG, "code_length"),
        depth=ds.quan(config.width_kpc * H_TNG, "code_length"),
        weight_field=weight_field,
        deposition="cic",
        density=density,
        north_vector=normalize(north, "projection north"),
    )
    plot.set_buff_size((int(config.npix), int(config.npix)))
    return plot.frb[field]


def project_stars(config: Config, inc: float, alpha: int) -> Path:
    paths = setup_paths(config, inc, "stars")
    setup_logging(paths["logs_component"], f"project_stars_{inc_label(inc)}_alpha{alpha:03d}")
    out = star_npz_path(paths, inc, alpha)
    if out.exists() and not config.recompute:
        try:
            with np.load(out, allow_pickle=True) as z:
                _ = z.files
            logging.info("cached stellar map: %s", out)
            return out
        except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
            logging.warning("Corrupt cached NPZ %s (%r); recomputing", out, exc)
            out.unlink(missing_ok=True)

    normal = load_disk_normal(config)
    qso = load_reference_qso(config)
    los, north, orientation = orientation_for(config, inc, alpha, normal)
    ds, center_ckpch, center_meta, cutout, systemic, los_holder = load_common_context(config, paths, "stars")
    los_holder["los"] = los
    x, y, X, Y = grid(config)

    lum_arr = particle_projection_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "stellar_r_luminosity"),
        density=True,
    )
    vlos_arr = particle_projection_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "stellar_vlos_incsweep"),
        weight_field=("PartType4", "particle_mass"),
        density=False,
    )
    sigma_lsun_kpc2 = base.yt_array_to_numpy(lum_arr, "Lsun/kpc**2")
    vlos = base.yt_array_to_numpy(vlos_arr, "km/s")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma_l = np.log10(np.where(sigma_lsun_kpc2 > 0, sigma_lsun_kpc2, np.nan))
        sigma_lsun_pc2 = sigma_lsun_kpc2 / 1.0e6
        mu_r = MSUN_R_AB + MAG_ARCSEC2_CONST - 2.5 * np.log10(np.where(sigma_lsun_pc2 > 0, sigma_lsun_pc2, np.nan))
    mu_r[~np.isfinite(mu_r)] = np.nan
    vlos[~np.isfinite(vlos)] = np.nan

    np.savez_compressed(
        out,
        x_kpc=x,
        y_kpc=y,
        X_kpc=X,
        Y_kpc=Y,
        stellar_r_surface_brightness_lsun_kpc2=sigma_lsun_kpc2,
        log_stellar_r_surface_brightness_lsun_kpc2=log_sigma_l,
        stellar_mu_r_mag_arcsec2=mu_r,
        stellar_vlos_mass_weighted_kms=vlos,
        alpha_deg=float(alpha),
        inc_deg=float(inc),
        pa_deg=float(config.pa_deg),
        los_hat=los,
        north_hat=north,
        disk_normal_hat=normal,
        x_qso_kpc=float(qso["x_qso_kpc"]),
        y_qso_kpc=float(qso["y_qso_kpc"]),
        rho_kpc=float(qso["rho_kpc"]),
        phi_deg=float(qso["phi_deg"]),
        systemic_velocity_kms=np.asarray(systemic["velocity_kms"], dtype=float),
        subtract_systemic=bool(config.subtract_systemic),
    )
    meta = {
        "component": "stars",
        "output_npz": str(out),
        "inc_deg": inc,
        "alpha_deg": alpha,
        "cutout_h5": str(cutout),
        "galaxy_center_ckpch": center_ckpch,
        "galaxy_center_source": center_meta,
        "systemic_velocity": systemic,
        "los_hat": los,
        "north_hat": north,
        "disk_normal_hat": normal,
        "orientation_axis": orientation["axis"],
        "qso_marker": qso,
        "projection_width_kpc": config.width_kpc,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "method": (
            "yt OffAxisParticleProjectionPlot with CIC deposition; surface brightness uses "
            "PartType4/GFM_StellarPhotometrics_05 converted to r-band Lsun; vlos is "
            "mass-weighted using PartType4/particle_mass"
        ),
    }
    write_json(paths["logs_component"] / f"project_stars_{inc_label(inc)}_alpha{alpha:03d}_metadata.json", meta)
    logging.info(
        "stars %s alpha=%03d done mu[p1,p99]=%s vstar[p2,p98]=%s -> %s",
        inc_label(inc),
        alpha,
        np.nanpercentile(mu_r[np.isfinite(mu_r)], [1, 99]).tolist(),
        np.nanpercentile(vlos[np.isfinite(vlos)], [2, 98]).tolist(),
        out,
    )
    return out


def load_map(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def all_component_maps(config: Config, inc: float, component: str) -> dict[int, dict[str, Any]]:
    paths = setup_paths(config, inc, component)
    maps = {}
    missing = []
    for alpha in ALPHAS:
        path = gas_npz_path(paths, inc, alpha) if component == "gas" else star_npz_path(paths, inc, alpha)
        if not path.exists():
            missing.append(alpha)
            continue
        try:
            maps[alpha] = load_map(path)
        except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
            raise RuntimeError(f"Corrupt NPZ for {component} {inc_label(inc)} alpha {alpha}: {path} ({exc!r})") from exc
    if missing:
        raise RuntimeError(f"Missing {component} NPZ files for {inc_label(inc)} alphas: {missing[:20]}...")
    return maps


def plot_common_axes(ax, config: Config, qso_xy: tuple[float, float]) -> None:
    half = config.half_width_kpc
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_{\rm sky}\ [{\rm kpc}]$")
    ax.set_ylabel(r"$y_{\rm sky}\ [{\rm kpc}]$")
    ax.plot(0, 0, marker="+", ms=15, mew=2.2, color="#44e0ff", zorder=5)
    ax.scatter([qso_xy[0]], [qso_xy[1]], s=95, facecolors="none", edgecolors="#48ff72", lw=2.0, zorder=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def add_cbar(fig, ax, im, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=15)


def render_gas_frame(config: Config, inc: float, alpha: int, m: dict[str, Any], scales: dict[str, float], out: Path) -> None:
    configure_matplotlib()
    qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
    mask = m["mask_logNHI_cut"].astype(bool)
    logN = np.where(mask, m["logN_HI"], np.nan)
    vgas = np.where(mask, m["vlos_gasweighted_kms"], np.nan)
    vHI = np.where(mask, m["vlos_HIweighted_kms"], np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(25.8, 8.3), constrained_layout=True)
    cm_hi = plt.get_cmap("magma").copy()
    cm_hi.set_bad("white")
    cm_v = plt.get_cmap("RdBu_r").copy()
    cm_v.set_bad("0.93")

    extent = [-config.half_width_kpc, config.half_width_kpc, -config.half_width_kpc, config.half_width_kpc]
    im0 = axes[0].imshow(
        logN,
        origin="lower",
        extent=extent,
        cmap=cm_hi,
        vmin=LOGNHI_CUT,
        vmax=scales["logN_vmax"],
        interpolation="nearest",
    )
    im1 = axes[1].imshow(
        vgas,
        origin="lower",
        extent=extent,
        cmap=cm_v,
        vmin=-scales["vgas_vlim"],
        vmax=scales["vgas_vlim"],
        interpolation="nearest",
    )
    im2 = axes[2].imshow(
        vHI,
        origin="lower",
        extent=extent,
        cmap=cm_v,
        vmin=-scales["vHI_vlim"],
        vmax=scales["vHI_vlim"],
        interpolation="nearest",
    )
    titles = [
        r"H I column, $N_{\rm HI}>1.25\times10^{20}\ {\rm cm}^{-2}$",
        r"gas-density-weighted $v_{\rm los}$",
        r"H I-weighted $v_{\rm los}$",
    ]
    labels = [
        r"$\log_{10}\,N_{\rm HI}\ [{\rm cm}^{-2}]$",
        r"$v_{\rm los,gas}\ [{\rm km\,s}^{-1}]$",
        r"$v_{\rm los,H I}\ [{\rm km\,s}^{-1}]$",
    ]
    for ax, im, title, label in zip(axes, [im0, im1, im2], titles, labels):
        ax.set_title(title, pad=10)
        plot_common_axes(ax, config, qso)
        add_cbar(fig, ax, im, label)

    note = (
        rf"M61 / TNG50-1 SID {config.sid}    "
        rf"$i={inc:.0f}^\circ$    $\alpha={alpha:03d}^\circ$    "
        rf"raw $v_{{\rm los}}$"
    )
    fig.text(0.5, 1.01, note, ha="center", va="bottom", fontsize=24)
    fig.savefig(out)
    plt.close(fig)


def render_star_frame(config: Config, inc: float, alpha: int, m: dict[str, Any], scales: dict[str, float], out: Path) -> None:
    configure_matplotlib()
    qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
    mu = m["stellar_mu_r_mag_arcsec2"]
    vstar = m["stellar_vlos_mass_weighted_kms"]

    fig, axes = plt.subplots(1, 2, figsize=(18.2, 8.5), constrained_layout=True)
    cm_mu = plt.get_cmap("magma_r").copy()
    cm_mu.set_bad("white")
    cm_v = plt.get_cmap("RdBu_r").copy()
    cm_v.set_bad("0.93")
    extent = [-config.half_width_kpc, config.half_width_kpc, -config.half_width_kpc, config.half_width_kpc]
    im0 = axes[0].imshow(
        mu,
        origin="lower",
        extent=extent,
        cmap=cm_mu,
        vmin=scales["mu_vmin"],
        vmax=scales["mu_vmax"],
        interpolation="nearest",
    )
    im1 = axes[1].imshow(
        vstar,
        origin="lower",
        extent=extent,
        cmap=cm_v,
        vmin=-scales["vstar_vlim"],
        vmax=scales["vstar_vlim"],
        interpolation="nearest",
    )
    axes[0].set_title(r"stellar $r$-band surface brightness", pad=10)
    axes[1].set_title(r"stellar mass-weighted $v_{\rm los}$", pad=10)
    plot_common_axes(axes[0], config, qso)
    plot_common_axes(axes[1], config, qso)
    add_cbar(fig, axes[0], im0, r"$\mu_r\ [{\rm mag\,arcsec}^{-2}]$")
    add_cbar(fig, axes[1], im1, r"$v_{\rm los,\star}\ [{\rm km\,s}^{-1}]$")

    note = (
        rf"M61 / TNG50-1 SID {config.sid}    "
        rf"$i={inc:.0f}^\circ$    $\alpha={alpha:03d}^\circ$    "
        rf"raw $v_{{\rm los}}$"
    )
    fig.text(0.5, 1.01, note, ha="center", va="bottom", fontsize=24)
    fig.savefig(out)
    plt.close(fig)


def gas_scales(maps: dict[int, dict[str, Any]]) -> dict[str, float]:
    log_vals = []
    vgas_vals = []
    vhi_vals = []
    for m in maps.values():
        mask = m["mask_logNHI_cut"].astype(bool)
        logN = m["logN_HI"]
        vgas = m["vlos_gasweighted_kms"]
        vhi = m["vlos_HIweighted_kms"]
        log_vals.append(logN[mask & np.isfinite(logN)])
        vgas_vals.append(np.abs(vgas[mask & np.isfinite(vgas)]))
        vhi_vals.append(np.abs(vhi[mask & np.isfinite(vhi)]))
    logN_vmax = max(LOGNHI_CUT + 0.2, robust_percentile(log_vals, 99.7, LOGNHI_CUT + 1.0))
    return {
        "logN_vmax": float(logN_vmax),
        "vgas_vlim": float(np.clip(robust_percentile(vgas_vals, 98.5, 120.0), 50.0, 450.0)),
        "vHI_vlim": float(np.clip(robust_percentile(vhi_vals, 98.5, 120.0), 50.0, 450.0)),
    }


def star_scales(maps: dict[int, dict[str, Any]]) -> dict[str, float]:
    mu_vals = []
    v_vals = []
    for m in maps.values():
        mu = m["stellar_mu_r_mag_arcsec2"]
        v = m["stellar_vlos_mass_weighted_kms"]
        mu_vals.append(mu[np.isfinite(mu)])
        v_vals.append(np.abs(v[np.isfinite(v)]))
    return {
        "mu_vmin": float(robust_percentile(mu_vals, 0.5, 16.0)),
        "mu_vmax": float(robust_percentile(mu_vals, 99.0, 28.0)),
        "vstar_vlim": float(np.clip(robust_percentile(v_vals, 98.5, 180.0), 50.0, 550.0)),
    }


def run_ffmpeg(config: Config, inc: float, component: str, paths: dict[str, Path]) -> Path:
    video_name = (
        f"m61_{inc_label(inc)}_alpha000_180_gas_HI_vgas_vHI.mp4"
        if component == "gas"
        else f"m61_{inc_label(inc)}_alpha000_180_stars_surface_brightness_vlos.mp4"
    )
    out = paths["videos"] / video_name
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(config.fps),
        "-start_number",
        "0",
        "-i",
        str(paths["frames"] / "frame_%03d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    logging.info("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


def render_component(config: Config, inc: float, component: str) -> Path:
    paths = setup_paths(config, inc, component)
    setup_logging(paths["logs_component"], f"render_{component}_{inc_label(inc)}")
    maps = all_component_maps(config, inc, component)
    scales = gas_scales(maps) if component == "gas" else star_scales(maps)
    logging.info("%s %s scales: %s", component, inc_label(inc), scales)
    for alpha in ALPHAS:
        out = frame_path(paths, alpha)
        if out.exists() and not config.rerender:
            continue
        if component == "gas":
            render_gas_frame(config, inc, alpha, maps[alpha], scales, out)
        else:
            render_star_frame(config, inc, alpha, maps[alpha], scales, out)
        if alpha % 15 == 0:
            logging.info("Rendered %s %s through alpha=%03d", component, inc_label(inc), alpha)
    video = run_ffmpeg(config, inc, component, paths)
    metadata = {
        "SID": config.sid,
        "SNAP": config.snap,
        "component": component,
        "inc_deg": inc,
        "alpha_min": min(ALPHAS),
        "alpha_max": max(ALPHAS),
        "N_frames": len(ALPHAS),
        "fps": config.fps,
        "projection_width_kpc": config.width_kpc,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "pa_deg": config.pa_deg,
        "logNHI_cut": LOGNHI_CUT if component == "gas" else None,
        "subtract_systemic": config.subtract_systemic,
        "plot_scales": scales,
        "video": str(video),
        "output_paths": {k: str(v) for k, v in paths.items()},
    }
    write_json(paths["data"] / f"{component}_{inc_label(inc)}_movie_metadata.json", metadata)
    print(f"{component} video complete for {inc_label(inc)}: {video}")
    return video


def task_to_inc_alpha(task_id: int) -> tuple[float, int]:
    nalpha = len(ALPHAS)
    inc_index = int(task_id) // nalpha
    alpha = int(task_id) % nalpha
    if inc_index < 0 or inc_index >= len(INCLINATIONS):
        raise ValueError(f"Task id {task_id} maps to invalid inclination index {inc_index}")
    return INCLINATIONS[inc_index], alpha


def render_task_to_inc(task_id: int) -> float:
    idx = int(task_id)
    if idx < 0 or idx >= len(INCLINATIONS):
        raise ValueError(f"Render task id {task_id} out of range")
    return INCLINATIONS[idx]


def make_index(config: Config) -> Path:
    paths = setup_paths(config)
    normal = load_disk_normal(config)
    qso = load_reference_qso(config)
    payload = {
        "SID": config.sid,
        "SNAP": config.snap,
        "output_dir": config.output_dir,
        "inclinations_deg": INCLINATIONS,
        "alphas_deg": ALPHAS,
        "N_tasks_per_component": len(INCLINATIONS) * len(ALPHAS),
        "projection_width_kpc": config.width_kpc,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "pa_deg": config.pa_deg,
        "mode": config.mode,
        "disk_normal_hat": normal,
        "qso_marker": qso,
        "logNHI_cut": LOGNHI_CUT,
        "subtract_systemic": config.subtract_systemic,
        "gas_quantity_notes": {
            "HI_column": "yt.off_axis_projection integral of H_p0_number_density",
            "gas_density_weighted_vlos": "integral(rho_gas*vlos) / integral(rho_gas)",
            "HI_weighted_vlos": "integral(n_HI*vlos) / integral(n_HI)",
        },
        "stellar_quantity_notes": {
            "surface_brightness": "PartType4/GFM_StellarPhotometrics_05 converted to r-band Lsun and projected with yt OffAxisParticleProjectionPlot",
            "vlos": "stellar LOS velocity weighted by PartType4/particle_mass",
        },
    }
    path = paths["root"] / "inclination_sweep_movie_index.json"
    write_json(path, payload)
    print(path)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=DEFAULT_OUT)
    common.add_argument("--npix", type=int, default=1024)
    common.add_argument("--fps", type=int, default=18)
    common.add_argument("--pa-deg", type=float, default=138.0)
    common.add_argument("--recompute", action="store_true")
    common.add_argument("--rerender", action="store_true")
    common.add_argument("--subtract-systemic", action="store_true")

    p_gas = sub.add_parser("project-gas", parents=[common])
    p_gas.add_argument("--inc", type=float)
    p_gas.add_argument("--alpha", type=int)
    p_gas.add_argument("--task-id", type=int)

    p_star = sub.add_parser("project-stars", parents=[common])
    p_star.add_argument("--inc", type=float)
    p_star.add_argument("--alpha", type=int)
    p_star.add_argument("--task-id", type=int)

    p_rg = sub.add_parser("render-gas", parents=[common])
    p_rg.add_argument("--inc", type=float)
    p_rg.add_argument("--task-id", type=int)

    p_rs = sub.add_parser("render-stars", parents=[common])
    p_rs.add_argument("--inc", type=float)
    p_rs.add_argument("--task-id", type=int)

    sub.add_parser("write-index", parents=[common])

    args = parser.parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        npix=args.npix,
        fps=args.fps,
        pa_deg=args.pa_deg,
        recompute=args.recompute,
        rerender=args.rerender,
        subtract_systemic=args.subtract_systemic,
    )
    return args, config


def resolve_inc_alpha(args) -> tuple[float, int]:
    if args.task_id is not None:
        return task_to_inc_alpha(args.task_id)
    if args.inc is None or args.alpha is None:
        raise ValueError("Provide either --task-id or both --inc and --alpha")
    return float(args.inc), int(args.alpha)


def resolve_inc(args) -> float:
    if args.task_id is not None:
        return render_task_to_inc(args.task_id)
    if args.inc is None:
        raise ValueError("Provide either --task-id or --inc")
    return float(args.inc)


def main(argv=None):
    configure_matplotlib()
    args, config = parse_args(argv)
    try:
        if args.command == "project-gas":
            inc, alpha = resolve_inc_alpha(args)
            return project_gas(config, inc, alpha)
        if args.command == "project-stars":
            inc, alpha = resolve_inc_alpha(args)
            return project_stars(config, inc, alpha)
        if args.command == "render-gas":
            return render_component(config, resolve_inc(args), "gas")
        if args.command == "render-stars":
            return render_component(config, resolve_inc(args), "stars")
        if args.command == "write-index":
            return make_index(config)
        raise RuntimeError(args.command)
    except Exception:
        root = Path(config.output_dir)
        err_dir = root / "logs"
        err_dir.mkdir(parents=True, exist_ok=True)
        err = err_dir / f"errors_{args.command}.txt"
        err.write_text(traceback.format_exc())
        logging.exception("Workflow failed; wrote %s", err)
        raise


if __name__ == "__main__":
    main()
