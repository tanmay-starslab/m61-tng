#!/usr/bin/env python
"""
Inner 100 kpc high-resolution H I/vlos projection and 500-annulus rotation
analysis for M61 / TNG50-1 SID 488530, alpha=5 noflip.

This reuses the true-center geometry and yt field machinery from
m61_oriented_HI_vlos_rotation_alpha5.py, then performs a narrower 100 kpc
projection and fine 0.1 kpc tilted-ring annuli from the saved 2D maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

import m61_oriented_HI_vlos_rotation_alpha5 as base


@dataclass
class InnerConfig:
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "oriented_HI_vlos_projection_alpha5_noflip_inner100kpc_highres_annuli"
    )
    projection_width_kpc: float = 100.0
    npix: int = 1024
    recompute_projections: bool = False
    r_min_kpc: float = 0.0
    r_max_kpc: float = 50.0
    n_annuli: int = 500
    min_pixels: int = 20
    thresholds: tuple[float, ...] = (17.0, 18.0, 19.0)
    main_threshold: float = 17.0
    smoothing_window_bins: int = 11
    sid: int = 488530
    snap: int = 99
    sightline_id: str = "J122138+043026"
    alpha_deg: float = 5.0
    mode: str = "noflip"

    @property
    def dR_kpc(self) -> float:
        return (self.r_max_kpc - self.r_min_kpc) / self.n_annuli

    @property
    def pixel_scale_kpc(self) -> float:
        return self.projection_width_kpc / self.npix


def setup_paths(config: InnerConfig) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {
        "out": out,
        "figures": out / "figures",
        "data": out / "data",
        "logs": out / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: dict[str, Path]) -> None:
    err = paths["logs"] / "errors.txt"
    if err.exists():
        err.unlink()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(paths["logs"] / "run.log", mode="w")
    file_handler.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(file_handler)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(base.json_sanitize(payload), f, indent=2, sort_keys=True)


def grid_1d(config: InnerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = 0.5 * config.projection_width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def projection_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "hi": paths["data"] / "HI_column_density_inner100kpc_alpha5_noflip.npz",
        "vhi": paths["data"] / "vlos_HI_weighted_inner100kpc_alpha5_noflip.npz",
        "vgas": paths["data"] / "vlos_gas_density_weighted_inner100kpc_alpha5_noflip.npz",
    }


def load_base_context(config: InnerConfig, paths: dict[str, Path]):
    base_config = base.AnalysisConfig(
        output_dir=config.output_dir,
        width_kpc=config.projection_width_kpc,
        npix=config.npix,
        logN_HI_min=config.main_threshold,
        annulus_rmax_kpc=config.r_max_kpc,
        annulus_dr_kpc=config.dR_kpc,
    )
    geometry = base.load_recipe_and_geometry(base_config, paths)
    ds, field_info = base.load_dataset_and_define_fields(base_config, geometry, paths)
    return base_config, geometry, ds, field_info


def compute_or_load_maps(config: InnerConfig, paths: dict[str, Path], geometry, ds=None) -> dict[str, Any]:
    p = projection_paths(paths)
    if (
        not config.recompute_projections
        and p["hi"].exists()
        and p["vhi"].exists()
        and p["vgas"].exists()
    ):
        logging.info("Loading cached inner 100 kpc projection NPZ files.")
        hi = np.load(p["hi"], allow_pickle=True)
        vhi = np.load(p["vhi"], allow_pickle=True)
        vgas = np.load(p["vgas"], allow_pickle=True)
        return {
            "x_kpc": hi["x_kpc"],
            "y_kpc": hi["y_kpc"],
            "X_kpc": hi["X_kpc"],
            "Y_kpc": hi["Y_kpc"],
            "N_HI_cm2": hi["N_HI_cm2"],
            "logN_HI": hi["logN_HI"],
            "vlos_HIweighted_kms": vhi["vlos_HIweighted_kms"],
            "vlos_gas_density_weighted_kms": vgas["vlos_gas_density_weighted_kms"],
            "mask": hi["mask"],
        }

    if ds is None:
        raise RuntimeError("Projection cache missing and no yt dataset was supplied.")

    x, y, X, Y = grid_1d(config)
    center = ds.arr(geometry["galaxy_center_ckpch"], "code_length")
    width = ds.arr([config.projection_width_kpc * base.H_TNG] * 2, "code_length")
    normal = geometry["normal_vector"]
    north = geometry["e2_hat"]

    logging.info("Computing inner 100 kpc projections at npix=%d.", config.npix)
    hi_den = base.off_axis_integral(ds, center, normal, width, config.npix, ("gas", "H_p0_number_density"), north)
    hi_num = base.off_axis_integral(ds, center, normal, width, config.npix, ("gas", "HI_vlos_integrand"), north)
    gas_den = base.off_axis_integral(ds, center, normal, width, config.npix, ("gas", "density"), north)
    gas_num = base.off_axis_integral(ds, center, normal, width, config.npix, ("gas", "gas_density_vlos_integrand"), north)

    N_HI = base.yt_array_to_numpy(hi_den, "cm**-2")
    HI_vlos_num = base.yt_array_to_numpy(hi_num, "km/(s*cm**2)")
    gas_sigma = base.yt_array_to_numpy(gas_den, "g/cm**2")
    gas_vlos_num = base.yt_array_to_numpy(gas_num, "g*km/(s*cm**2)")
    with np.errstate(divide="ignore", invalid="ignore"):
        logN = np.log10(np.where(N_HI > 0, N_HI, np.nan))
        vhi = HI_vlos_num / N_HI
        vgas = gas_vlos_num / gas_sigma
    vhi[~np.isfinite(vhi)] = np.nan
    vgas[~np.isfinite(vgas)] = np.nan
    mask = np.isfinite(logN) & np.isfinite(vhi) & (N_HI > 0)

    metadata = json.dumps({"projection_width_kpc": config.projection_width_kpc, "npix": config.npix})
    np.savez_compressed(
        p["hi"], x_kpc=x, y_kpc=y, X_kpc=X, Y_kpc=Y, N_HI_cm2=N_HI,
        logN_HI=logN, mask=mask, metadata=metadata
    )
    np.savez_compressed(
        p["vhi"], x_kpc=x, y_kpc=y, X_kpc=X, Y_kpc=Y, N_HI_cm2=N_HI,
        logN_HI=logN, vlos_HIweighted_kms=vhi, mask=mask, metadata=metadata
    )
    np.savez_compressed(
        p["vgas"], x_kpc=x, y_kpc=y, X_kpc=X, Y_kpc=Y, gas_surface_density_g_cm2=gas_sigma,
        vlos_gas_density_weighted_kms=vgas, mask=np.isfinite(vgas), metadata=metadata
    )
    logging.info("Saved projection NPZ files: %s", p)
    return {
        "x_kpc": x, "y_kpc": y, "X_kpc": X, "Y_kpc": Y, "N_HI_cm2": N_HI,
        "logN_HI": logN, "vlos_HIweighted_kms": vhi,
        "vlos_gas_density_weighted_kms": vgas, "mask": mask,
    }


def symmetric_velocity_limit(v: np.ndarray) -> float:
    good = v[np.isfinite(v)]
    if good.size == 0:
        return 250.0
    lo, hi = np.nanpercentile(good, [2, 98])
    return float(min(max(abs(lo), abs(hi), 40.0), 300.0))


def save_map_figure(data, extent, outbase, cmap, cbar_label, vmin=None, vmax=None, geometry=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar_label)
    ax.plot(0, 0, marker="+", ms=11, mew=1.8, color="cyan", label="galaxy center")
    if geometry is not None:
        ax.scatter([geometry["x_qso_kpc"]], [geometry["y_qso_kpc"]], s=42, facecolors="none", edgecolors="lime", lw=1.4, label="QSO")
    ax.set_xlabel("Projected major-axis x [kpc]")
    ax.set_ylabel("Projected minor-axis y [kpc]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(outbase.with_suffix(f".{suffix}"), dpi=350)
    plt.close(fig)


def plot_projection_maps(config: InnerConfig, paths, maps, geometry):
    extent = [-50, 50, -50, 50]
    logN = maps["logN_HI"]
    hi_good = logN[np.isfinite(logN)]
    hi_vmin = max(12.0, float(np.nanpercentile(hi_good, 1)))
    hi_vmax = float(np.nanpercentile(hi_good, 99.5))
    save_map_figure(
        logN, extent, paths["figures"] / "HI_column_density_inner100kpc_alpha5_noflip",
        "magma", r"$\log_{10} N_{\rm HI}\ [{\rm cm}^{-2}]$", hi_vmin, hi_vmax, geometry
    )
    vhi_lim = symmetric_velocity_limit(maps["vlos_HIweighted_kms"])
    save_map_figure(
        maps["vlos_HIweighted_kms"], extent, paths["figures"] / "vlos_HI_weighted_inner100kpc_alpha5_noflip",
        "RdBu_r", r"$v_{\rm los,HI-weighted}$ [km/s]", -vhi_lim, vhi_lim, geometry
    )
    vgas_lim = symmetric_velocity_limit(maps["vlos_gas_density_weighted_kms"])
    save_map_figure(
        maps["vlos_gas_density_weighted_kms"], extent, paths["figures"] / "vlos_gas_density_weighted_inner100kpc_alpha5_noflip",
        "RdBu_r", r"$v_{\rm los,gas-weighted}$ [km/s]", -vgas_lim, vgas_lim, geometry
    )


def disk_coordinates(maps, geometry):
    X = maps["X_kpc"]
    Y = maps["Y_kpc"]
    inc = math.radians(float(geometry["inc_deg"]))
    q = math.cos(inc)
    y_disk = Y / q
    R = np.sqrt(X**2 + y_disk**2)
    theta = np.arctan2(y_disk, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(R > 0, np.cos(theta), np.nan)
    return R, theta, cos_theta, q


def weighted_fit(v, c, w):
    X = np.vstack([c, np.ones_like(c)]).T
    M = X.T @ (w[:, None] * X)
    rhs = X.T @ (w * v)
    cond = np.linalg.cond(M)
    if not np.isfinite(cond) or cond > 1e12:
        raise np.linalg.LinAlgError(f"ill-conditioned matrix cond={cond}")
    beta = np.linalg.solve(M, rhs)
    A, B = float(beta[0]), float(beta[1])
    residual = v - (A * c + B)
    sigma = float(np.sqrt(np.average(residual**2, weights=w)))
    return A, B, sigma, cond


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, center=True, min_periods=max(3, window // 3)).median().to_numpy()


def rotation_curve_for_threshold(config: InnerConfig, maps, geometry, threshold: float) -> pd.DataFrame:
    R, _, cos_theta, _ = disk_coordinates(maps, geometry)
    vmap = maps["vlos_HIweighted_kms"]
    logN = maps["logN_HI"]
    N_HI = maps["N_HI_cm2"]
    sin_inc = math.sin(math.radians(float(geometry["inc_deg"])))
    bins = np.linspace(config.r_min_kpc, config.r_max_kpc, config.n_annuli + 1)
    rows = []
    for r0, r1 in zip(bins[:-1], bins[1:]):
        valid = (
            (R >= r0) & (R < r1) & np.isfinite(vmap) & np.isfinite(cos_theta)
            & np.isfinite(N_HI) & (N_HI > 0) & np.isfinite(logN) & (logN >= threshold)
        )
        n = int(np.count_nonzero(valid))
        row = {
            "R_inner_kpc": float(r0), "R_outer_kpc": float(r1), "R_mid_kpc": float(0.5 * (r0 + r1)),
            "dR_kpc": config.dR_kpc, "N_pixels": n, "N_eff_weighted": np.nan,
            "v_sys_kms": np.nan, "A_kms": np.nan, "vrot_signed_kms": np.nan,
            "vrot_abs_kms": np.nan, "sigma_residual_kms": np.nan, "sigma_vrot_kms": np.nan,
            "simple_abs_vrot_kms": np.nan, "median_logN_HI": np.nan, "mean_logN_HI": np.nan,
            "logN_HI_threshold": threshold, "fit_success": False, "warning_flag": "",
        }
        if n < config.min_pixels:
            row["warning_flag"] = "too_few_pixels"
            rows.append(row)
            continue
        v = vmap[valid].astype(float)
        c = cos_theta[valid].astype(float)
        w = N_HI[valid].astype(float)
        sumw = float(np.sum(w))
        if not np.isfinite(sumw) or sumw <= 0:
            row["warning_flag"] = "low_weight"
            rows.append(row)
            continue
        row["N_eff_weighted"] = float(sumw**2 / np.sum(w**2))
        row["simple_abs_vrot_kms"] = float(np.average(np.abs(v), weights=w) / sin_inc)
        row["median_logN_HI"] = float(np.nanmedian(logN[valid]))
        row["mean_logN_HI"] = float(np.nanmean(logN[valid]))
        try:
            A, B, sigma, cond = weighted_fit(v, c, w)
            vrot_signed = A / sin_inc
            row.update({
                "v_sys_kms": B, "A_kms": A, "vrot_signed_kms": vrot_signed,
                "vrot_abs_kms": abs(vrot_signed), "sigma_residual_kms": sigma,
                "sigma_vrot_kms": sigma / sin_inc, "fit_success": True,
                "warning_flag": "" if cond < 1e8 else "high_condition_number",
            })
        except np.linalg.LinAlgError:
            row["warning_flag"] = "singular_fit"
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ["vrot_abs_kms", "vrot_signed_kms", "sigma_vrot_kms", "simple_abs_vrot_kms"]:
        table[f"{col}_smooth"] = smooth_series(table[col].to_numpy(float), config.smoothing_window_bins)
    return table


def save_rotation_tables(config: InnerConfig, paths, maps, geometry) -> dict[float, pd.DataFrame]:
    tables = {}
    combined = []
    for threshold in config.thresholds:
        table = rotation_curve_for_threshold(config, maps, geometry, threshold)
        tables[threshold] = table
        label = int(threshold)
        out = paths["data"] / f"rotation_curve_inner100kpc_500annuli_logNHI{label}_alpha5_noflip.csv"
        table.to_csv(out, index=False)
        combined.append(table.assign(threshold_label=label))
        frac = float(table["fit_success"].mean())
        logging.info("Threshold %.1f: successful annuli fraction %.3f", threshold, frac)
    pd.concat(combined, ignore_index=True).to_csv(
        paths["data"] / "rotation_curve_inner100kpc_500annuli_threshold_comparison_alpha5_noflip.csv",
        index=False,
    )
    return tables


def interp_at_rho(table: pd.DataFrame, rho: float, col: str) -> float:
    good = np.isfinite(table[col].to_numpy(float))
    if np.count_nonzero(good) < 2:
        return np.nan
    return float(np.interp(rho, table.loc[good, "R_mid_kpc"], table.loc[good, col]))


def ism_velocity_json(config: InnerConfig, paths, table17, geometry):
    rho = float(geometry["rho_kpc"])
    nearest_idx = int(np.nanargmin(np.abs(table17["R_mid_kpc"].to_numpy(float) - rho)))
    payload = {
        "rho_kpc": rho,
        "nearest_annulus_R_mid_kpc": float(table17.loc[nearest_idx, "R_mid_kpc"]),
        "interpolated_vrot_abs_kms": interp_at_rho(table17, rho, "vrot_abs_kms_smooth"),
        "interpolated_vrot_signed_kms": interp_at_rho(table17, rho, "vrot_signed_kms_smooth"),
        "interpolated_sigma_vrot_kms": interp_at_rho(table17, rho, "sigma_vrot_kms_smooth"),
        "simple_abs_vrot_at_rho_kms": interp_at_rho(table17, rho, "simple_abs_vrot_kms_smooth"),
        "logN_HI_min": config.main_threshold,
        "dR_kpc": config.dR_kpc,
        "N_annuli": config.n_annuli,
        "inc_deg": float(geometry["inc_deg"]),
        "PA_used": float(geometry["pa_deg"]),
        "method_notes": "Interpolated from smoothed 0.1 kpc true-centered tilted-ring curve; raw values are preserved in CSV.",
    }
    write_json(paths["data"] / "ism_velocity_at_rho_inner100kpc_500annuli_logNHI17_alpha5_noflip.json", payload)
    return payload


def plot_publication_rotation(config: InnerConfig, paths, table, ism):
    fig, ax = plt.subplots(figsize=(7.1, 4.7), constrained_layout=True)
    good = table["fit_success"].to_numpy(bool)
    r = table["R_mid_kpc"].to_numpy(float)
    raw = table["vrot_abs_kms"].to_numpy(float)
    smooth = table["vrot_abs_kms_smooth"].to_numpy(float)
    sig = table["sigma_vrot_kms_smooth"].to_numpy(float)
    simple = table["simple_abs_vrot_kms_smooth"].to_numpy(float)
    ax.scatter(r[good], raw[good], s=7, color="0.1", alpha=0.25, label="raw 0.1 kpc bins")
    ax.plot(r, smooth, color="black", lw=2.2, label="rolling-median guide")
    band = np.isfinite(smooth) & np.isfinite(sig)
    ax.fill_between(r[band], smooth[band] - sig[band], smooth[band] + sig[band], color="0.2", alpha=0.15, lw=0)
    ax.plot(r, simple, color="tab:blue", lw=1.2, ls="--", label=r"mean $|v_{\rm los}|/\sin i$")
    ax.axvline(ism["rho_kpc"], color="tab:red", lw=1.1, ls=":")
    ax.annotate(
        f"Vrot(26 kpc) = {ism['interpolated_vrot_abs_kms']:.1f} +/- {ism['interpolated_sigma_vrot_kms']:.1f} km/s",
        xy=(ism["rho_kpc"], ism["interpolated_vrot_abs_kms"]),
        xytext=(27.5, ism["interpolated_vrot_abs_kms"] + 25),
        arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.8),
        fontsize=9,
    )
    text = "SID 488530\nM61 / NGC 4303\nalpha=5 deg, noflip\ninc=23 deg\nlog N_HI >= 17\ndR=0.1 kpc"
    ax.text(0.03, 0.96, text, transform=ax.transAxes, va="top", fontsize=8.8, bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.85"))
    ax.set_xlim(0, 50)
    ax.set_xlabel(r"$R_{\rm disk}$ [kpc]")
    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_logNHI17_publication.{suffix}", dpi=400)
    plt.close(fig)


def plot_signed_curve(paths, table):
    fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
    r = table["R_mid_kpc"].to_numpy(float)
    ax.scatter(r, table["vrot_signed_kms"], s=7, alpha=0.28, color="black")
    ax.plot(r, table["vrot_signed_kms_smooth"], lw=2, color="tab:purple")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(0, 50)
    ax.set_xlabel(r"$R_{\rm disk}$ [kpc]")
    ax.set_ylabel("signed Vrot [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_signed_logNHI17.{suffix}", dpi=350)
    plt.close(fig)


def plot_threshold_comparison(config, paths, tables):
    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    styles = {17.0: ("black", "-"), 18.0: ("tab:orange", "--"), 19.0: ("tab:green", "-.")}
    for threshold, table in tables.items():
        color, ls = styles[float(threshold)]
        ax.plot(table["R_mid_kpc"], table["vrot_abs_kms_smooth"], color=color, ls=ls, lw=2, label=fr"$\log N_{{HI}}\geq {threshold:.0f}$")
    ax.axvline(26, color="tab:red", lw=1.1, ls=":")
    ax.set_xlim(0, 50)
    ax.set_xlabel(r"$R_{\rm disk}$ [kpc]")
    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_threshold_comparison_publication.{suffix}", dpi=400)
    plt.close(fig)


def plot_fit_quality(paths, table):
    r = table["R_mid_kpc"].to_numpy(float)
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(r, table["N_pixels"], color="black", lw=1)
    axes[0].set_ylabel("N pixels")
    axes[1].plot(r, table["sigma_residual_kms"], color="tab:red", lw=1)
    axes[1].set_ylabel("resid sigma [km/s]")
    axes[2].plot(r, table["fit_success"].astype(int), color="tab:green", lw=1)
    axes[2].set_ylabel("fit success")
    axes[2].set_ylim(-0.1, 1.1)
    axes[3].plot(r, table["median_logN_HI"], color="tab:blue", lw=1)
    axes[3].set_ylabel("median log NHI")
    axes[3].set_xlabel(r"$R_{\rm disk}$ [kpc]")
    for ax in axes:
        ax.set_xlim(0, 50)
        ax.minorticks_on()
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_fit_quality.{suffix}", dpi=350)
    plt.close(fig)


def add_ellipse_set(ax, inc_deg, radii, color="white", lw=0.8, alpha=0.75):
    q = math.cos(math.radians(inc_deg))
    for R in radii:
        ax.add_patch(Ellipse((0, 0), 2 * R, 2 * R * q, fill=False, color=color, lw=lw, alpha=alpha))


def plot_annuli_overlays(config, paths, maps, geometry):
    extent = [-50, 50, -50, 50]
    logN = maps["logN_HI"]
    hi_good = logN[np.isfinite(logN)]
    vmin = max(12.0, float(np.nanpercentile(hi_good, 1)))
    vmax = float(np.nanpercentile(hi_good, 99.5))
    for dense in [False, True]:
        fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
        im = ax.imshow(logN, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.92, label=r"$\log_{10} N_{\rm HI}$")
        if dense:
            add_ellipse_set(ax, geometry["inc_deg"], np.arange(20, 30.001, 0.5), color="white", lw=0.55, alpha=0.7)
            ax.set_xlim(-35, 35)
            ax.set_ylim(-32, 32)
            outbase = paths["figures"] / "ellipse_annuli_inner100kpc_dense_zoom_diagnostic"
        else:
            add_ellipse_set(ax, geometry["inc_deg"], np.arange(5, 50.001, 5), color="white", lw=0.9, alpha=0.75)
            add_ellipse_set(ax, geometry["inc_deg"], [26], color="cyan", lw=1.4, alpha=0.95)
            ax.arrow(0, 0, 10, 0, color="cyan", width=0.08, head_width=1.2, length_includes_head=True)
            ax.arrow(0, 0, 0, 10 * math.cos(math.radians(geometry["inc_deg"])), color="lime", width=0.08, head_width=1.2, length_includes_head=True)
            outbase = paths["figures"] / "ellipse_annuli_inner100kpc_highres_overlay"
        ax.plot(0, 0, marker="+", ms=11, mew=1.8, color="cyan")
        ax.scatter([geometry["x_qso_kpc"]], [geometry["y_qso_kpc"]], s=46, facecolors="none", edgecolors="lime", lw=1.4)
        ax.set_xlabel("Projected major-axis x [kpc]")
        ax.set_ylabel("Projected minor-axis y [kpc]")
        ax.minorticks_on()
        for suffix in ("png", "pdf"):
            fig.savefig(outbase.with_suffix(f".{suffix}"), dpi=350)
        plt.close(fig)


def plot_tilted_ring_diagnostics(config, paths, maps, geometry, threshold=17.0):
    R, _, cos_theta, _ = disk_coordinates(maps, geometry)
    vmap = maps["vlos_HIweighted_kms"]
    logN = maps["logN_HI"]
    N_HI = maps["N_HI_cm2"]
    sin_inc = math.sin(math.radians(float(geometry["inc_deg"])))
    selected = [5, 10, 20, 26, 30, 40]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    for ax, r0 in zip(axes.ravel(), selected):
        mask = (
            (R >= r0 - config.dR_kpc / 2) & (R < r0 + config.dR_kpc / 2)
            & np.isfinite(vmap) & np.isfinite(cos_theta) & np.isfinite(N_HI)
            & (N_HI > 0) & np.isfinite(logN) & (logN >= threshold)
        )
        if np.count_nonzero(mask) < config.min_pixels:
            ax.text(0.5, 0.5, "too few pixels", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"R={r0} kpc")
            continue
        c = cos_theta[mask]
        v = vmap[mask]
        w = N_HI[mask]
        if c.size > 8000:
            rng = np.random.default_rng(42 + int(r0))
            idx = rng.choice(c.size, 8000, replace=False, p=w / np.sum(w))
            cp, vp = c[idx], v[idx]
        else:
            cp, vp = c, v
        A, B, sigma, _ = weighted_fit(v, c, w)
        xline = np.linspace(-1, 1, 100)
        ax.scatter(cp, vp, s=3, alpha=0.18, color="black", rasterized=True)
        ax.plot(xline, B + A * xline, color="tab:red", lw=1.6)
        ax.set_title(f"R={r0} kpc, Vrot={A/sin_inc:.1f}, sigma={sigma:.1f}")
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$v_{\rm los}$ [km/s]")
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"tilted_ring_fit_diagnostics_inner100kpc_selected_annuli.{suffix}", dpi=350)
    plt.close(fig)


def save_metadata(config, paths, geometry, field_info, maps, ism, tables):
    payload = {
        "SID": config.sid, "SNAP": config.snap,
        "projection_width_kpc": config.projection_width_kpc,
        "half_width_kpc": config.projection_width_kpc / 2,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "R_min_kpc": config.r_min_kpc, "R_max_kpc": config.r_max_kpc,
        "N_annuli": config.n_annuli, "dR_kpc": config.dR_kpc,
        "logN_HI_thresholds_used": list(config.thresholds),
        "galaxy_center_ckpch": geometry["galaxy_center_ckpch"],
        "ray_anchor_ckpch": geometry["anchor_ckpch"],
        "x_qso_kpc": geometry["x_qso_kpc"], "y_qso_kpc": geometry["y_qso_kpc"],
        "rho_check_kpc": geometry["rho_check_kpc"],
        "los_hat": geometry["los_hat"], "e1": geometry["e1_hat"], "e2": geometry["e2_hat"],
        "inc_deg": geometry["inc_deg"], "PA_used": geometry["pa_deg"],
        "annulus_axis_ratio_q": geometry["axis_ratio_q"],
        "systemic_velocity": field_info["systemic_velocity"],
        "output_paths": {k: str(v) for k, v in paths.items()},
        "ism_velocity_at_rho": ism,
        "successful_annulus_fraction_by_threshold": {
            str(k): float(v["fit_success"].mean()) for k, v in tables.items()
        },
    }
    out = paths["data"] / "geometry_projection_metadata_inner100kpc_highres_annuli_alpha5_noflip.json"
    write_json(out, payload)
    return out


def run_inner_workflow(config: InnerConfig):
    paths = setup_paths(config)
    setup_logging(paths)
    try:
        base_config, geometry, ds, field_info = load_base_context(config, paths)
        maps = compute_or_load_maps(config, paths, geometry, ds)
        plot_projection_maps(config, paths, maps, geometry)
        tables = save_rotation_tables(config, paths, maps, geometry)
        table17 = tables[17.0]
        ism = ism_velocity_json(config, paths, table17, geometry)
        plot_publication_rotation(config, paths, table17, ism)
        plot_signed_curve(paths, table17)
        plot_threshold_comparison(config, paths, tables)
        plot_fit_quality(paths, table17)
        plot_annuli_overlays(config, paths, maps, geometry)
        plot_tilted_ring_diagnostics(config, paths, maps, geometry, config.main_threshold)
        meta_path = save_metadata(config, paths, geometry, field_info, maps, ism, tables)
        logging.info("Output directory: %s", paths["out"])
        logging.info("Npix=%d pixel_scale=%.5f kpc", config.npix, config.pixel_scale_kpc)
        logging.info("N_annuli=%d dR=%.3f kpc", config.n_annuli, config.dR_kpc)
        for th, table in tables.items():
            logging.info(
                "threshold %.0f: success_fraction=%.3f, Vrot@rho=%.3f km/s",
                th, float(table["fit_success"].mean()), interp_at_rho(table, geometry["rho_kpc"], "vrot_abs_kms_smooth")
            )
        logging.info("Main publication figure: %s", paths["figures"] / "rotation_curve_inner100kpc_500annuli_logNHI17_publication.png")
        logging.info("Metadata JSON: %s", meta_path)
        print("\nInner 100 kpc high-resolution run complete")
        print(f"output_dir = {paths['out']}")
        print(f"Npix = {config.npix}, pixel_scale_kpc = {config.pixel_scale_kpc:.5f}")
        print(f"N_annuli = {config.n_annuli}, dR_kpc = {config.dR_kpc:.3f}")
        for th, table in tables.items():
            print(f"logNHI >= {th:.0f}: success_fraction={table['fit_success'].mean():.3f}, Vrot@rho={interp_at_rho(table, geometry['rho_kpc'], 'vrot_abs_kms_smooth'):.2f} km/s")
        print(f"main_figure = {paths['figures'] / 'rotation_curve_inner100kpc_500annuli_logNHI17_publication.png'}")
        print(f"metadata = {meta_path}")
        return {"paths": paths, "geometry": geometry, "field_info": field_info, "maps": maps, "tables": tables, "ism": ism}
    except Exception:
        err = paths["logs"] / "errors.txt"
        err.write_text(traceback.format_exc())
        logging.exception("Inner workflow failed; wrote %s", err)
        raise


def parse_args(argv=None) -> InnerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--recompute-projections", action="store_true")
    parser.add_argument("--output-dir", type=str, default=InnerConfig.output_dir)
    args = parser.parse_args(argv)
    return InnerConfig(npix=args.npix, recompute_projections=args.recompute_projections, output_dir=args.output_dir)


def main(argv=None):
    return run_inner_workflow(parse_args(argv))


if __name__ == "__main__":
    main()
