#!/usr/bin/env python
"""
Run the true-centered inner-100 kpc H I/vlos projection and 500-annulus
tilted-ring workflow for multiple alpha orientations.

This driver deliberately reuses the successful alpha=5 field/geometry machinery
from m61_oriented_HI_vlos_rotation_alpha5.py and the annulus math helpers from
m61_inner100_highres_annuli_alpha5.py.  Each alpha gets its own fresh yt dataset
instance because the LOS velocity derived field depends on that alpha's LOS.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

import m61_inner100_highres_annuli_alpha5 as inner
import m61_oriented_HI_vlos_rotation_alpha5 as base


@dataclass
class MultiAlphaConfig:
    parent_output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "oriented_HI_vlos_projection_multi_alpha_inner100kpc_highres_annuli"
    )
    alphas: tuple[float, ...] = (45.0, 90.0, 180.0, 255.0)
    mode: str = "noflip"
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
    subtract_systemic_velocity: bool = True
    sid: int = 488530
    snap: int = 99
    sightline_id: str = "J122138+043026"
    rho_kpc: float = 26.0
    rvir_kpc: float = 457.0

    @property
    def dR_kpc(self) -> float:
        return (self.r_max_kpc - self.r_min_kpc) / self.n_annuli

    @property
    def pixel_scale_kpc(self) -> float:
        return self.projection_width_kpc / self.npix


def alpha_tag(alpha: float, mode: str = "noflip") -> str:
    return f"alpha{int(round(alpha)):03d}_{mode}"


def setup_parent_paths(config: MultiAlphaConfig) -> dict[str, Path]:
    parent = Path(config.parent_output_dir)
    combined = parent / "combined_alpha_comparison"
    paths = {
        "parent": parent,
        "combined": combined,
        "combined_figures": combined / "figures",
        "combined_data": combined / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def alpha_paths(config: MultiAlphaConfig, alpha: float) -> dict[str, Path]:
    out = Path(config.parent_output_dir) / alpha_tag(alpha, config.mode)
    paths = {
        "out": out,
        "figures": out / "figures",
        "data": out / "data",
        "logs": out / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_alpha_logging(paths: dict[str, Path]) -> None:
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


def projection_paths(paths: dict[str, Path], tag: str) -> dict[str, Path]:
    return {
        "hi": paths["data"] / f"HI_column_density_inner100kpc_{tag}.npz",
        "vhi": paths["data"] / f"vlos_HI_weighted_inner100kpc_{tag}.npz",
        "vgas": paths["data"] / f"vlos_gas_density_weighted_inner100kpc_{tag}.npz",
    }


def base_config_for_alpha(config: MultiAlphaConfig, alpha: float, output_dir: Path) -> base.AnalysisConfig:
    return base.AnalysisConfig(
        subhalo_id=config.sid,
        sightline_id=config.sightline_id,
        alpha_deg=alpha,
        mode=config.mode,
        run_label="L4Rvir",
        rho_kpc=config.rho_kpc,
        rvir_kpc=config.rvir_kpc,
        width_kpc=config.projection_width_kpc,
        npix=config.npix,
        logN_HI_min=config.main_threshold,
        annulus_rmax_kpc=config.r_max_kpc,
        annulus_dr_kpc=config.dR_kpc,
        subtract_systemic_velocity=config.subtract_systemic_velocity,
        output_dir=str(output_dir),
    )


def compute_or_load_maps(config: MultiAlphaConfig, paths: dict[str, Path], tag: str, geometry, ds=None) -> dict[str, Any]:
    p = projection_paths(paths, tag)
    if (
        not config.recompute_projections
        and p["hi"].exists()
        and p["vhi"].exists()
        and p["vgas"].exists()
    ):
        logging.info("Loading cached projection NPZ files for %s.", tag)
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
        raise RuntimeError(f"Projection cache for {tag} is missing and no yt dataset was supplied.")

    x, y, X, Y = inner.grid_1d(
        inner.InnerConfig(
            projection_width_kpc=config.projection_width_kpc,
            npix=config.npix,
            r_max_kpc=config.r_max_kpc,
            n_annuli=config.n_annuli,
        )
    )
    center = ds.arr(geometry["galaxy_center_ckpch"], "code_length")
    width = ds.arr([config.projection_width_kpc * base.H_TNG] * 2, "code_length")
    normal = geometry["normal_vector"]
    north = geometry["e2_hat"]

    logging.info("Computing %s projections at npix=%d.", tag, config.npix)
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

    metadata = json.dumps(
        {
            "alpha_deg": float(geometry["selected_recipe_row"]["alpha_deg"]),
            "mode": str(geometry["selected_recipe_row"]["mode"]),
            "projection_width_kpc": config.projection_width_kpc,
            "npix": config.npix,
        }
    )
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
        N_HI_cm2=N_HI, logN_HI=logN, vlos_gas_density_weighted_kms=vgas,
        mask=np.isfinite(vgas), metadata=metadata
    )
    logging.info("Saved projection NPZ files for %s: %s", tag, p)
    return {
        "x_kpc": x,
        "y_kpc": y,
        "X_kpc": X,
        "Y_kpc": Y,
        "N_HI_cm2": N_HI,
        "logN_HI": logN,
        "vlos_HIweighted_kms": vhi,
        "vlos_gas_density_weighted_kms": vgas,
        "mask": mask,
    }


def map_extent(config: MultiAlphaConfig) -> list[float]:
    half = config.projection_width_kpc / 2
    return [-half, half, -half, half]


def save_map_figure(data, extent, outbase, cmap, cbar_label, geometry, alpha: float, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6.3, 5.6), constrained_layout=True)
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar_label)
    ax.plot(0, 0, marker="+", ms=11, mew=1.8, color="cyan", label="galaxy center")
    ax.scatter([geometry["x_qso_kpc"]], [geometry["y_qso_kpc"]], s=44, facecolors="none", edgecolors="lime", lw=1.4, label="QSO")
    ax.text(
        0.03, 0.04,
        f"alpha={alpha:.0f} deg, noflip\nrho_check={geometry['rho_check_kpc']:.2f} kpc\ninc={geometry['inc_deg']:.1f} deg, PA={geometry['pa_deg']:.1f} deg",
        transform=ax.transAxes, fontsize=8.5, color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3),
    )
    ax.set_xlabel("Projected major-axis x [kpc]")
    ax.set_ylabel("Projected minor-axis y [kpc]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(outbase.with_suffix(f".{suffix}"), dpi=350)
    plt.close(fig)


def plot_projection_maps(config: MultiAlphaConfig, paths, tag: str, maps, geometry, alpha: float):
    extent = map_extent(config)
    logN = maps["logN_HI"]
    hi_good = logN[np.isfinite(logN)]
    hi_vmin = max(12.0, float(np.nanpercentile(hi_good, 1)))
    hi_vmax = float(np.nanpercentile(hi_good, 99.5))
    save_map_figure(
        logN, extent, paths["figures"] / f"HI_column_density_inner100kpc_{tag}",
        "magma", r"$\log_{10} N_{\rm HI}\ [{\rm cm}^{-2}]$", geometry, alpha, hi_vmin, hi_vmax
    )
    vhi_lim = inner.symmetric_velocity_limit(maps["vlos_HIweighted_kms"])
    save_map_figure(
        maps["vlos_HIweighted_kms"], extent, paths["figures"] / f"vlos_HI_weighted_inner100kpc_{tag}",
        "RdBu_r", r"$v_{\rm los,HI-weighted}$ [km/s]", geometry, alpha, -vhi_lim, vhi_lim
    )
    vgas_lim = inner.symmetric_velocity_limit(maps["vlos_gas_density_weighted_kms"])
    save_map_figure(
        maps["vlos_gas_density_weighted_kms"], extent, paths["figures"] / f"vlos_gas_density_weighted_inner100kpc_{tag}",
        "RdBu_r", r"$v_{\rm los,gas-weighted}$ [km/s]", geometry, alpha, -vgas_lim, vgas_lim
    )


def rotation_curve_for_threshold(config: MultiAlphaConfig, maps, geometry, alpha: float, threshold: float) -> pd.DataFrame:
    inner_cfg = inner.InnerConfig(
        projection_width_kpc=config.projection_width_kpc,
        npix=config.npix,
        r_min_kpc=config.r_min_kpc,
        r_max_kpc=config.r_max_kpc,
        n_annuli=config.n_annuli,
        min_pixels=config.min_pixels,
        smoothing_window_bins=config.smoothing_window_bins,
    )
    table = inner.rotation_curve_for_threshold(inner_cfg, maps, geometry, threshold)
    table.insert(0, "mode", config.mode)
    table.insert(0, "alpha_deg", float(alpha))
    warning = table["warning_flag"].fillna("").astype(str)
    table["too_few_pixels"] = warning.eq("too_few_pixels")
    table["low_weight"] = warning.eq("low_weight")
    table["singular_fit"] = warning.eq("singular_fit")
    return table


def save_rotation_tables(config: MultiAlphaConfig, paths, tag: str, maps, geometry, alpha: float) -> dict[float, pd.DataFrame]:
    tables = {}
    combined = []
    for threshold in config.thresholds:
        table = rotation_curve_for_threshold(config, maps, geometry, alpha, threshold)
        tables[threshold] = table
        label = int(threshold)
        out = paths["data"] / f"rotation_curve_inner100kpc_500annuli_logNHI{label}_{tag}.csv"
        table.to_csv(out, index=False)
        combined.append(table.assign(threshold_label=label))
        logging.info(
            "%s threshold %.1f: successful annuli fraction %.3f",
            tag, threshold, float(table["fit_success"].mean())
        )
    pd.concat(combined, ignore_index=True).to_csv(
        paths["data"] / f"rotation_curve_inner100kpc_500annuli_threshold_comparison_{tag}.csv",
        index=False,
    )
    return tables


def interp_at_rho(table: pd.DataFrame, rho: float, col: str) -> float:
    good = np.isfinite(table[col].to_numpy(float))
    if np.count_nonzero(good) < 2:
        return np.nan
    return float(np.interp(rho, table.loc[good, "R_mid_kpc"], table.loc[good, col]))


def ism_velocity_json(config: MultiAlphaConfig, paths, tag: str, table17, geometry, alpha: float):
    rho = float(geometry["rho_kpc"])
    nearest_idx = int(np.nanargmin(np.abs(table17["R_mid_kpc"].to_numpy(float) - rho)))
    payload = {
        "alpha_deg": float(alpha),
        "mode": config.mode,
        "rho_kpc": rho,
        "rho_check_kpc": float(geometry["rho_check_kpc"]),
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
    out = paths["data"] / f"ism_velocity_at_rho_inner100kpc_500annuli_logNHI17_{tag}.json"
    write_json(out, payload)
    return payload


def plot_publication_rotation(config: MultiAlphaConfig, paths, tag: str, table, ism, alpha: float):
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
    text = (
        "SID 488530\nM61 / NGC 4303\n"
        f"alpha={alpha:.0f} deg, noflip\ninc=23 deg\n"
        "log N_HI >= 17\ndR=0.1 kpc"
    )
    ax.text(
        0.03, 0.96, text, transform=ax.transAxes, va="top", fontsize=8.8,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.85")
    )
    ax.set_xlim(0, 50)
    ax.set_xlabel(r"$R_{\rm disk}$ [kpc]")
    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_logNHI17_publication_{tag}.{suffix}", dpi=400)
    plt.close(fig)


def plot_signed_curve(paths, tag: str, table):
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
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_signed_logNHI17_{tag}.{suffix}", dpi=350)
    plt.close(fig)


def plot_threshold_comparison(paths, tag: str, tables):
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
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_threshold_comparison_publication_{tag}.{suffix}", dpi=400)
    plt.close(fig)


def plot_fit_quality(paths, tag: str, table):
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
        fig.savefig(paths["figures"] / f"rotation_curve_inner100kpc_500annuli_fit_quality_{tag}.{suffix}", dpi=350)
    plt.close(fig)


def add_ellipse_set(ax, inc_deg, radii, color="white", lw=0.8, alpha=0.75):
    q = math.cos(math.radians(inc_deg))
    for radius in radii:
        ax.add_patch(Ellipse((0, 0), 2 * radius, 2 * radius * q, fill=False, color=color, lw=lw, alpha=alpha))


def plot_annuli_overlays(config: MultiAlphaConfig, paths, tag: str, maps, geometry):
    extent = map_extent(config)
    logN = maps["logN_HI"]
    hi_good = logN[np.isfinite(logN)]
    vmin = max(12.0, float(np.nanpercentile(hi_good, 1)))
    vmax = float(np.nanpercentile(hi_good, 99.5))
    for dense in (False, True):
        fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
        im = ax.imshow(logN, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.92, label=r"$\log_{10} N_{\rm HI}$")
        if dense:
            add_ellipse_set(ax, geometry["inc_deg"], np.arange(20, 30.001, 0.5), color="white", lw=0.55, alpha=0.7)
            ax.set_xlim(-35, 35)
            ax.set_ylim(-32, 32)
            outbase = paths["figures"] / f"ellipse_annuli_inner100kpc_dense_zoom_diagnostic_{tag}"
        else:
            add_ellipse_set(ax, geometry["inc_deg"], np.arange(5, 50.001, 5), color="white", lw=0.9, alpha=0.75)
            add_ellipse_set(ax, geometry["inc_deg"], [26], color="cyan", lw=1.4, alpha=0.95)
            ax.arrow(0, 0, 10, 0, color="cyan", width=0.08, head_width=1.2, length_includes_head=True)
            ax.arrow(0, 0, 0, 10 * math.cos(math.radians(geometry["inc_deg"])), color="lime", width=0.08, head_width=1.2, length_includes_head=True)
            outbase = paths["figures"] / f"ellipse_annuli_inner100kpc_highres_overlay_{tag}"
        ax.plot(0, 0, marker="+", ms=11, mew=1.8, color="cyan")
        ax.scatter([geometry["x_qso_kpc"]], [geometry["y_qso_kpc"]], s=46, facecolors="none", edgecolors="lime", lw=1.4)
        ax.set_xlabel("Projected major-axis x [kpc]")
        ax.set_ylabel("Projected minor-axis y [kpc]")
        ax.minorticks_on()
        for suffix in ("png", "pdf"):
            fig.savefig(outbase.with_suffix(f".{suffix}"), dpi=350)
        plt.close(fig)


def plot_tilted_ring_diagnostics(config: MultiAlphaConfig, paths, tag: str, maps, geometry, threshold=17.0):
    R, _, cos_theta, _ = inner.disk_coordinates(maps, geometry)
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
            ax.set_title(f"R={r0} kpc, {tag}")
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
        A, B, sigma, _ = inner.weighted_fit(v, c, w)
        xline = np.linspace(-1, 1, 100)
        ax.scatter(cp, vp, s=3, alpha=0.18, color="black", rasterized=True)
        ax.plot(xline, B + A * xline, color="tab:red", lw=1.6)
        ax.set_title(f"R={r0} kpc, Vrot={A/sin_inc:.1f}, sigma={sigma:.1f}")
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$v_{\rm los}$ [km/s]")
        ax.grid(alpha=0.2)
    fig.suptitle(tag, fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"tilted_ring_fit_diagnostics_inner100kpc_selected_annuli_{tag}.{suffix}", dpi=350)
    plt.close(fig)


def save_metadata(config: MultiAlphaConfig, paths, tag: str, geometry, field_info, ism, tables, alpha: float):
    payload = {
        "SID": config.sid,
        "SNAP": config.snap,
        "alpha_deg": float(alpha),
        "mode": config.mode,
        "selected_recipe_row": geometry["selected_recipe_row"],
        "projection_width_kpc": config.projection_width_kpc,
        "half_width_kpc": config.projection_width_kpc / 2,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "R_min_kpc": config.r_min_kpc,
        "R_max_kpc": config.r_max_kpc,
        "N_annuli": config.n_annuli,
        "dR_kpc": config.dR_kpc,
        "logN_HI_thresholds_used": list(config.thresholds),
        "galaxy_center_ckpch": geometry["galaxy_center_ckpch"],
        "ray_anchor_ckpch": geometry["anchor_ckpch"],
        "p0_ckpch": geometry["p0_ckpch"],
        "p1_ckpch": geometry["p1_ckpch"],
        "x_qso_kpc": geometry["x_qso_kpc"],
        "y_qso_kpc": geometry["y_qso_kpc"],
        "rho_kpc_from_recipe": geometry["rho_kpc"],
        "rho_check_kpc": geometry["rho_check_kpc"],
        "saved_los_hat": geometry["saved_los_hat"],
        "los_hat": geometry["los_hat"],
        "dot_saved_los_with_p1_minus_p0": geometry["dot_saved_los_with_p1_minus_p0"],
        "e1": geometry["e1_hat"],
        "e2": geometry["e2_hat"],
        "inc_deg": geometry["inc_deg"],
        "PA_used": geometry["pa_deg"],
        "q_cos_inc": geometry["axis_ratio_q"],
        "systemic_velocity": field_info["systemic_velocity"],
        "subtract_systemic_velocity": config.subtract_systemic_velocity,
        "output_paths": {k: str(v) for k, v in paths.items()},
        "ism_velocity_at_rho": ism,
        "successful_annulus_fraction_by_threshold": {
            str(k): float(v["fit_success"].mean()) for k, v in tables.items()
        },
    }
    out = paths["data"] / f"geometry_projection_metadata_inner100kpc_highres_annuli_{tag}.json"
    write_json(out, payload)
    return out


def run_single_alpha(config: MultiAlphaConfig, alpha: float) -> dict[str, Any]:
    tag = alpha_tag(alpha, config.mode)
    paths = alpha_paths(config, alpha)
    setup_alpha_logging(paths)
    logging.info("Starting %s", tag)
    try:
        bcfg = base_config_for_alpha(config, alpha, paths["out"])
        geometry = base.load_recipe_and_geometry(bcfg, paths)
        ds, field_info = base.load_dataset_and_define_fields(bcfg, geometry, paths)
        maps = compute_or_load_maps(config, paths, tag, geometry, ds)
        plot_projection_maps(config, paths, tag, maps, geometry, alpha)
        tables = save_rotation_tables(config, paths, tag, maps, geometry, alpha)
        table17 = tables[17.0]
        ism = ism_velocity_json(config, paths, tag, table17, geometry, alpha)
        plot_publication_rotation(config, paths, tag, table17, ism, alpha)
        plot_signed_curve(paths, tag, table17)
        plot_threshold_comparison(paths, tag, tables)
        plot_fit_quality(paths, tag, table17)
        plot_annuli_overlays(config, paths, tag, maps, geometry)
        plot_tilted_ring_diagnostics(config, paths, tag, maps, geometry, config.main_threshold)
        meta_path = save_metadata(config, paths, tag, geometry, field_info, ism, tables, alpha)
        logging.info(
            "%s complete: rho_check=%.4f, Vrot@rho=%.3f, sigma=%.3f, success=%.3f",
            tag, geometry["rho_check_kpc"], ism["interpolated_vrot_abs_kms"],
            ism["interpolated_sigma_vrot_kms"], float(table17["fit_success"].mean())
        )
        del ds
        return {
            "alpha": alpha,
            "tag": tag,
            "paths": paths,
            "geometry": geometry,
            "tables": tables,
            "ism": ism,
            "metadata_path": meta_path,
            "rotation_csv": paths["data"] / f"rotation_curve_inner100kpc_500annuli_logNHI17_{tag}.csv",
        }
    except Exception:
        err = paths["logs"] / "errors.txt"
        err.write_text(traceback.format_exc())
        logging.exception("%s failed; wrote %s", tag, err)
        raise


def load_alpha5_if_available() -> dict[str, Any] | None:
    out = Path("/scratch/tsingh65/m61-tng/outputs/sid488530/oriented_HI_vlos_projection_alpha5_noflip_inner100kpc_highres_annuli")
    csv_path = out / "data" / "rotation_curve_inner100kpc_500annuli_logNHI17_alpha5_noflip.csv"
    ism_path = out / "data" / "ism_velocity_at_rho_inner100kpc_500annuli_logNHI17_alpha5_noflip.json"
    meta_path = out / "data" / "geometry_projection_metadata_inner100kpc_highres_annuli_alpha5_noflip.json"
    if not (csv_path.exists() and ism_path.exists() and meta_path.exists()):
        print(f"WARNING: alpha=5 comparison data not found under {out}; skipping alpha=5.")
        return None
    table = pd.read_csv(csv_path)
    with ism_path.open() as f:
        ism = json.load(f)
    with meta_path.open() as f:
        meta = json.load(f)
    return {
        "alpha": 5.0,
        "tag": "alpha005_noflip",
        "table": table,
        "ism": ism,
        "metadata": meta,
        "rotation_csv": csv_path,
        "metadata_path": meta_path,
        "output_dir": out,
    }


def collect_comparison_entries(config: MultiAlphaConfig, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    result_alphas = {float(result["alpha"]) for result in results}
    if config.subtract_systemic_velocity and 5.0 not in result_alphas:
        alpha5 = load_alpha5_if_available()
        if alpha5 is not None:
            entries.append(alpha5)
    for result in results:
        meta_path = result["metadata_path"]
        with Path(meta_path).open() as f:
            meta = json.load(f)
        entries.append({
            "alpha": float(result["alpha"]),
            "tag": result["tag"],
            "table": result["tables"][17.0],
            "ism": result["ism"],
            "metadata": meta,
            "rotation_csv": result["rotation_csv"],
            "metadata_path": meta_path,
            "output_dir": result["paths"]["out"],
        })
    return sorted(entries, key=lambda x: x["alpha"])


def plot_combined_rotation(combined_paths, entries):
    fig, ax = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(entries)))
    for entry, color in zip(entries, colors):
        table = entry["table"]
        ax.plot(
            table["R_mid_kpc"], table["vrot_abs_kms_smooth"],
            color=color, lw=2, label=f"alpha={entry['alpha']:.0f}"
        )
    ax.axvline(26, color="tab:red", lw=1.1, ls=":")
    ax.set_xlim(0, 50)
    ax.set_xlabel(r"$R_{\rm disk}$ [kpc]")
    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(combined_paths["combined_figures"] / f"rotation_curve_alpha_comparison_logNHI17_inner100kpc.{suffix}", dpi=400)
    plt.close(fig)


def plot_vrot_at_rho(combined_paths, entries):
    alphas = np.array([entry["alpha"] for entry in entries], dtype=float)
    vrot = np.array([entry["ism"]["interpolated_vrot_abs_kms"] for entry in entries], dtype=float)
    sigma = np.array([entry["ism"]["interpolated_sigma_vrot_kms"] for entry in entries], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.errorbar(alphas, vrot, yerr=sigma, fmt="o-", color="black", ecolor="0.45", capsize=3)
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel(r"$V_{\rm rot}(26\,{\rm kpc})$ [km/s]")
    ax.minorticks_on()
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(combined_paths["combined_figures"] / f"vrot_at_rho_vs_alpha_logNHI17.{suffix}", dpi=400)
    plt.close(fig)


def load_map_for_entry(entry, kind: str):
    tag = entry["tag"]
    out = Path(entry["output_dir"])
    if entry["alpha"] == 5.0:
        data_dir = out / "data"
        if kind == "hi":
            return np.load(data_dir / "HI_column_density_inner100kpc_alpha5_noflip.npz", allow_pickle=True)
        return np.load(data_dir / "vlos_HI_weighted_inner100kpc_alpha5_noflip.npz", allow_pickle=True)
    data_dir = out / "data"
    if kind == "hi":
        return np.load(data_dir / f"HI_column_density_inner100kpc_{tag}.npz", allow_pickle=True)
    return np.load(data_dir / f"vlos_HI_weighted_inner100kpc_{tag}.npz", allow_pickle=True)


def plot_multi_panel_maps(config: MultiAlphaConfig, combined_paths, entries):
    entries_4 = [entry for entry in entries if entry["alpha"] in {45.0, 90.0, 180.0, 255.0}]
    hi_maps = [load_map_for_entry(entry, "hi") for entry in entries_4]
    v_maps = [load_map_for_entry(entry, "vhi") for entry in entries_4]
    log_arrays = [m["logN_HI"] for m in hi_maps]
    hi_good = np.concatenate([arr[np.isfinite(arr)] for arr in log_arrays])
    hi_vmin = max(12.0, float(np.nanpercentile(hi_good, 1)))
    hi_vmax = float(np.nanpercentile(hi_good, 99.5))
    v_arrays = [m["vlos_HIweighted_kms"] for m in v_maps]
    v_good = np.concatenate([arr[np.isfinite(arr)] for arr in v_arrays])
    vlo, vhi = np.nanpercentile(v_good, [2, 98])
    vlim = float(min(max(abs(vlo), abs(vhi), 40.0), 300.0))
    extent = map_extent(config)

    for kind, arrays, cmap, vmin, vmax, label, stem in [
        ("hi", log_arrays, "magma", hi_vmin, hi_vmax, r"$\log_{10} N_{\rm HI}$", "HI_column_density_multi_alpha_inner100kpc"),
        ("vhi", v_arrays, "RdBu_r", -vlim, vlim, r"$v_{\rm los,HI-weighted}$ [km/s]", "vlos_HI_weighted_multi_alpha_inner100kpc"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
        im = None
        for ax, entry, arr in zip(axes.ravel(), entries_4, arrays):
            meta = entry["metadata"]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.plot(0, 0, marker="+", ms=10, mew=1.6, color="cyan")
            ax.scatter([meta["x_qso_kpc"]], [meta["y_qso_kpc"]], s=38, facecolors="none", edgecolors="lime", lw=1.2)
            ax.set_title(f"alpha={entry['alpha']:.0f} deg")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]")
            ax.minorticks_on()
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label=label)
        for suffix in ("png", "pdf"):
            fig.savefig(combined_paths["combined_figures"] / f"{stem}.{suffix}", dpi=350)
        plt.close(fig)


def save_combined_csv(combined_paths, entries):
    rows = []
    for entry in entries:
        ism = entry["ism"]
        rows.append({
            "alpha_deg": float(entry["alpha"]),
            "mode": ism.get("mode", "noflip"),
            "rho_kpc": ism["rho_kpc"],
            "rho_check_kpc": ism.get("rho_check_kpc", entry["metadata"].get("rho_check_kpc")),
            "interpolated_vrot_abs_kms": ism["interpolated_vrot_abs_kms"],
            "interpolated_vrot_signed_kms": ism["interpolated_vrot_signed_kms"],
            "interpolated_sigma_vrot_kms": ism["interpolated_sigma_vrot_kms"],
            "simple_abs_vrot_at_rho_kms": ism["simple_abs_vrot_at_rho_kms"],
            "path_to_rotation_curve_csv": str(entry["rotation_csv"]),
            "path_to_metadata_json": str(entry["metadata_path"]),
        })
    df = pd.DataFrame(rows).sort_values("alpha_deg")
    out = combined_paths["combined_data"] / "vrot_at_rho_alpha_comparison_logNHI17.csv"
    df.to_csv(out, index=False)
    return df, out


def make_combined_outputs(config: MultiAlphaConfig, parent_paths, results):
    entries = collect_comparison_entries(config, results)
    plot_combined_rotation(parent_paths, entries)
    plot_vrot_at_rho(parent_paths, entries)
    plot_multi_panel_maps(config, parent_paths, entries)
    summary, csv_path = save_combined_csv(parent_paths, entries)
    print(f"Combined alpha comparison CSV: {csv_path}")
    return summary, entries


def run_multi_alpha(config: MultiAlphaConfig):
    parent_paths = setup_parent_paths(config)
    results = []
    for alpha in config.alphas:
        result = run_single_alpha(config, alpha)
        results.append(result)
    summary, _ = make_combined_outputs(config, parent_paths, results)
    print("\nalpha_deg | rho_check_kpc | Vrot_abs(rho) | sigma | successful_annuli_fraction | output_dir")
    for result in results:
        table17 = result["tables"][17.0]
        ism = result["ism"]
        print(
            f"{result['alpha']:8.0f} | {result['geometry']['rho_check_kpc']:13.3f} | "
            f"{ism['interpolated_vrot_abs_kms']:13.3f} | {ism['interpolated_sigma_vrot_kms']:6.3f} | "
            f"{float(table17['fit_success'].mean()):27.3f} | {result['paths']['out']}"
        )
    print(f"\ncombined_summary = {parent_paths['combined_data'] / 'vrot_at_rho_alpha_comparison_logNHI17.csv'}")
    return {"results": results, "combined_summary": summary, "parent_paths": parent_paths}


def parse_args(argv=None) -> MultiAlphaConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--recompute-projections", action="store_true")
    parser.add_argument("--parent-output-dir", type=str, default=MultiAlphaConfig.parent_output_dir)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(MultiAlphaConfig.alphas))
    parser.add_argument("--no-systemic-subtraction", action="store_true")
    args = parser.parse_args(argv)
    return MultiAlphaConfig(
        parent_output_dir=args.parent_output_dir,
        alphas=tuple(args.alphas),
        npix=args.npix,
        recompute_projections=args.recompute_projections,
        subtract_systemic_velocity=not args.no_systemic_subtraction,
    )


def main(argv=None):
    return run_multi_alpha(parse_args(argv))


if __name__ == "__main__":
    main()
