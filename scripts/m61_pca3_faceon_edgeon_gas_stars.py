#!/usr/bin/env python3
"""
PCA-v3 disk-normal face-on/edge-on projection diagnostics for M61/TNG50.

Outputs PNG-only combined panels:
- gas: log N_HI, gas-density-weighted v_los, H I-weighted v_los
- stars: stellar surface density, unweighted stellar-particle mean v_los
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.visualization.particle_plots import OffAxisParticleProjectionPlot

import m61_oriented_HI_vlos_rotation_alpha5 as base
import m61_saved_alpha_projection_movie as movie


@dataclass
class Config:
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "pca3_faceon_edgeon_gas_stars_projection_diagnostics"
    )
    orientation_header: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/orient_header_sid488530.json"
    )
    width_kpc: float = 100.0
    npix: int = 1024
    sid: int = 488530
    snap: int = 99

    @property
    def half_width_kpc(self) -> float:
        return 0.5 * self.width_kpc

    @property
    def pixel_scale_kpc(self) -> float:
        return self.width_kpc / self.npix


def setup_paths(config: Config) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {"out": out, "figures": out / "figures", "data": out / "data", "logs": out / "logs"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: dict[str, Path]) -> None:
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


def normalize(vec, name="vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def load_pca3_normal(config: Config) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(config.orientation_header)
    if path.exists():
        with path.open() as f:
            payload = json.load(f)
        if "normal_used_hat" in payload:
            return normalize(payload["normal_used_hat"], "normal_used_hat"), {
                "source": str(path),
                "orientation_method": payload.get("orientation_method"),
                "alpha_convention": payload.get("alpha_convention"),
                "obs_inc_deg": payload.get("obs_inc_deg"),
                "obs_pa_deg_used": payload.get("obs_pa_deg_used"),
            }
    fallback = np.array([0.1489902267951354, -0.8457239371292382, -0.5123991944628276])
    logging.warning("Could not read normal_used_hat from %s; using PCA-v3 fallback value.", path)
    return normalize(fallback, "fallback normal"), {"source": "hard-coded fallback"}


def deterministic_perpendicular(vec: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    ref = z if abs(np.dot(vec, z)) < 0.9 else x
    return normalize(np.cross(ref, vec), "perpendicular")


def basis_from_los_north(los: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    los = normalize(los, "los")
    north = np.asarray(north, dtype=float)
    north = north - np.dot(north, los) * los
    north = normalize(north, "north projected")
    x_axis = normalize(np.cross(north, los), "image x axis")
    y_axis = north
    return x_axis, y_axis, los


def projection_setups(normal: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    face_north = deterministic_perpendicular(normal)
    edge_los = face_north
    return {
        "face_on": {
            "los": normal,
            "north": normalize(np.cross(normal, face_north), "face-on north"),
            "description": "LOS parallel to PCA-v3 disk normal",
        },
        "edge_on": {
            "los": edge_los,
            "north": normal,
            "description": "LOS in disk plane; PCA-v3 disk normal vertical in image",
        },
    }


def grid(config: Config):
    half = config.half_width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def compute_gas_maps(config: Config, ds, center_ckpch, los, north, los_holder) -> dict[str, np.ndarray]:
    center = ds.arr(center_ckpch, "code_length")
    width = ds.arr([config.width_kpc * base.H_TNG] * 2, "code_length")
    los_holder["los"] = normalize(los, "gas LOS")
    hi_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "H_p0_number_density"), north)
    hi_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "HI_vlos_integrand_movie"), north)
    gas_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "density"), north)
    gas_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "gas_density_vlos_integrand_movie"), north)
    N_HI = base.yt_array_to_numpy(hi_den, "cm**-2")
    HI_num = base.yt_array_to_numpy(hi_num, "km/(s*cm**2)")
    gas_sigma = base.yt_array_to_numpy(gas_den, "g/cm**2")
    gas_num_np = base.yt_array_to_numpy(gas_num, "g*km/(s*cm**2)")
    with np.errstate(divide="ignore", invalid="ignore"):
        logN = np.log10(np.where(N_HI > 0, N_HI, np.nan))
        vhi = HI_num / N_HI
        vgas = gas_num_np / gas_sigma
    vhi[~np.isfinite(vhi)] = np.nan
    vgas[~np.isfinite(vgas)] = np.nan
    return {"N_HI_cm2": N_HI, "logN_HI": logN, "vlos_HIweighted_kms": vhi, "vlos_gasweighted_kms": vgas}


def add_dynamic_stellar_los_field(ds, los_holder: dict[str, np.ndarray], systemic_kms: np.ndarray) -> None:
    if base.yt_field_exists(ds, ("PartType4", "stellar_vlos_pca3_dynamic")):
        return

    def _stellar_vlos(field, data):
        los = los_holder["los"]
        vx = data[("PartType4", "particle_velocity_x")].to("km/s") - data.ds.quan(systemic_kms[0], "km/s")
        vy = data[("PartType4", "particle_velocity_y")].to("km/s") - data.ds.quan(systemic_kms[1], "km/s")
        vz = data[("PartType4", "particle_velocity_z")].to("km/s") - data.ds.quan(systemic_kms[2], "km/s")
        return (los[0] * vx + los[1] * vy + los[2] * vz).to("km/s")

    ds.add_field(
        ("PartType4", "stellar_vlos_pca3_dynamic"),
        function=_stellar_vlos,
        sampling_type="particle",
        units="km/s",
    )


def yt_particle_frb(
    config: Config,
    ds,
    center_ckpch,
    los,
    north,
    field,
    *,
    weight_field=None,
    density=False,
):
    plot = OffAxisParticleProjectionPlot(
        ds,
        normal=normalize(los, "stellar particle projection LOS"),
        fields=field,
        center=ds.arr(center_ckpch, "code_length"),
        width=ds.quan(config.width_kpc * base.H_TNG, "code_length"),
        depth=ds.quan(config.width_kpc * base.H_TNG, "code_length"),
        weight_field=weight_field,
        deposition="cic",
        density=density,
        north_vector=normalize(north, "stellar particle projection north"),
    )
    plot.set_buff_size((int(config.npix), int(config.npix)))
    return plot.frb[field]


def compute_stellar_maps(
    config: Config,
    ds,
    center_ckpch,
    los,
    north,
    star_los_holder: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    star_los_holder["los"] = normalize(los, "stellar LOS")
    sigma_arr = yt_particle_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "particle_mass"),
        density=True,
    )
    vlos_arr = yt_particle_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "stellar_vlos_pca3_dynamic"),
        weight_field=("PartType4", "particle_ones"),
        density=False,
    )
    sigma = base.yt_array_to_numpy(sigma_arr, "Msun/kpc**2")
    vlos_mean = base.yt_array_to_numpy(vlos_arr, "km/s")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma = np.log10(np.where(sigma > 0, sigma, np.nan))
    vlos_mean[~np.isfinite(vlos_mean)] = np.nan
    return {
        "stellar_surface_density_msun_kpc2": sigma,
        "log_stellar_surface_density": log_sigma,
        "stellar_vlos_unweighted_mean_kms": vlos_mean,
        "stellar_projection_method": np.asarray("yt.OffAxisParticleProjectionPlot CIC; vlos weighted by particle_ones"),
    }


def robust_vlim(*arrays, floor=30.0, ceiling=300.0) -> float:
    vals = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrays if np.any(np.isfinite(a))])
    if vals.size == 0:
        return floor
    return float(np.clip(np.nanpercentile(vals, 98), floor, ceiling))


def draw_normal_indicator(ax, normal, x_axis, y_axis, label="PCA-v3 disk normal"):
    nx = float(np.dot(normal, x_axis))
    ny = float(np.dot(normal, y_axis))
    length = np.hypot(nx, ny)
    if length < 0.08:
        ax.scatter([39], [39], s=95, facecolors="none", edgecolors="cyan", lw=1.6)
        ax.scatter([39], [39], s=15, color="cyan")
        ax.text(36, 33.5, "normal\ninto page", color="cyan", fontsize=8, ha="center", va="top")
    else:
        nx /= length
        ny /= length
        ax.arrow(-42, 38, 13 * nx, 13 * ny, color="cyan", width=0.35, head_width=2.2, length_includes_head=True)
        ax.text(-43, 32.5, label, color="cyan", fontsize=8, ha="left", va="top")


def save_gas_figure(config: Config, paths, name, setup, maps, scales):
    x_axis, y_axis, _ = basis_from_los_north(setup["los"], setup["north"])
    extent = [-config.half_width_kpc, config.half_width_kpc, -config.half_width_kpc, config.half_width_kpc]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    panels = [
        (maps["logN_HI"], "magma", scales["logN"][0], scales["logN"][1], r"$\log_{10}N_{\rm HI}\ [{\rm cm}^{-2}]$", "H I column"),
        (maps["vlos_gasweighted_kms"], "RdBu_r", -scales["vgas"], scales["vgas"], r"$v_{\rm los,gas}$ [km/s]", "gas-density weighted"),
        (maps["vlos_HIweighted_kms"], "RdBu_r", -scales["vhi"], scales["vhi"], r"$v_{\rm los,HI}$ [km/s]", "H I weighted"),
    ]
    for ax, (arr, cmap, vmin, vmax, cbar, title) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(0, 0, marker="+", color="lime", ms=10, mew=1.6)
        draw_normal_indicator(ax, setup["los"] if name == "face_on" else setup["north"], x_axis, y_axis)
        ax.set_title(title)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.minorticks_on()
        plt.colorbar(im, ax=ax, shrink=0.86, label=cbar)
    fig.suptitle(f"M61/TNG50 SID 488530 gas projections: PCA-v3 {name.replace('_', '-')}")
    out = paths["figures"] / f"gas_HI_vlos_combined_pca3_{name}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_stellar_figure(config: Config, paths, name, setup, maps, scales):
    x_axis, y_axis, _ = basis_from_los_north(setup["los"], setup["north"])
    extent = [-config.half_width_kpc, config.half_width_kpc, -config.half_width_kpc, config.half_width_kpc]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    panels = [
        (
            maps["log_stellar_surface_density"],
            "inferno",
            scales["logSigma"][0],
            scales["logSigma"][1],
            r"$\log_{10}\Sigma_\star\ [M_\odot\,{\rm kpc}^{-2}]$",
            "stellar surface density",
        ),
        (
            maps["stellar_vlos_unweighted_mean_kms"],
            "RdBu_r",
            -scales["vstar"],
            scales["vstar"],
            r"mean stellar $v_{\rm los}$ [km/s]",
            "unweighted stellar-particle LOS velocity",
        ),
    ]
    for ax, (arr, cmap, vmin, vmax, cbar, title) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(0, 0, marker="+", color="lime", ms=10, mew=1.6)
        draw_normal_indicator(ax, setup["los"] if name == "face_on" else setup["north"], x_axis, y_axis)
        ax.set_title(title)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.minorticks_on()
        plt.colorbar(im, ax=ax, shrink=0.86, label=cbar)
    fig.suptitle(f"M61/TNG50 SID 488530 stellar projections: PCA-v3 {name.replace('_', '-')}")
    out = paths["figures"] / f"stars_surface_density_vlos_pca3_{name}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def run(config: Config):
    paths = setup_paths(config)
    setup_logging(paths)
    base_config = base.AnalysisConfig(output_dir=str(paths["out"]), width_kpc=config.width_kpc, npix=config.npix)
    center_ckpch, center_meta = base.load_true_galaxy_center(base_config)
    cutout = base.find_cutout_h5(base_config)
    ds = yt.load(str(cutout))
    base.ensure_gas_alias_fields(ds)
    systemic = base.systemic_velocity_from_hdf5(cutout, center_ckpch)
    systemic_kms = np.asarray(systemic["velocity_kms"], dtype=float)
    los_holder = {"los": np.array([0.0, 0.0, 1.0])}
    movie.add_dynamic_los_fields(ds, los_holder, systemic_kms)
    star_los_holder = {"los": np.array([0.0, 0.0, 1.0])}
    add_dynamic_stellar_los_field(ds, star_los_holder, systemic_kms)
    normal, normal_meta = load_pca3_normal(config)
    setups = projection_setups(normal)
    logging.info("PCA-v3 normal: %s", normal)
    logging.info("Systemic velocity: %s", systemic)

    gas_maps = {}
    star_maps = {}
    x, y, X, Y = grid(config)
    for name, setup in setups.items():
        logging.info("Computing %s gas maps: %s", name, setup["description"])
        gas_maps[name] = compute_gas_maps(config, ds, center_ckpch, setup["los"], setup["north"], los_holder)
        logging.info("Computing %s stellar maps with yt OffAxisParticleProjectionPlot", name)
        star_maps[name] = compute_stellar_maps(
            config,
            ds,
            center_ckpch,
            setup["los"],
            setup["north"],
            star_los_holder,
        )
        np.savez_compressed(
            paths["data"] / f"pca3_{name}_gas_stars_maps.npz",
            x_kpc=x,
            y_kpc=y,
            X_kpc=X,
            Y_kpc=Y,
            **{f"gas_{k}": v for k, v in gas_maps[name].items()},
            **{f"stars_{k}": v for k, v in star_maps[name].items()},
            los_hat=setup["los"],
            north_hat=setup["north"],
            normal_used_hat=normal,
        )

    logN_vals = np.concatenate([m["logN_HI"][np.isfinite(m["logN_HI"])].ravel() for m in gas_maps.values()])
    logS_vals = np.concatenate(
        [m["log_stellar_surface_density"][np.isfinite(m["log_stellar_surface_density"])].ravel() for m in star_maps.values()]
    )
    gas_scales = {
        "logN": (float(np.nanpercentile(logN_vals, 1)), float(np.nanpercentile(logN_vals, 99.5))),
        "vgas": robust_vlim(*[m["vlos_gasweighted_kms"] for m in gas_maps.values()]),
        "vhi": robust_vlim(*[m["vlos_HIweighted_kms"] for m in gas_maps.values()]),
    }
    star_scales = {
        "logSigma": (float(np.nanpercentile(logS_vals, 1)), float(np.nanpercentile(logS_vals, 99.5))),
        "vstar": robust_vlim(*[m["stellar_vlos_unweighted_mean_kms"] for m in star_maps.values()]),
    }
    outputs = {}
    for name, setup in setups.items():
        outputs[f"gas_{name}"] = str(save_gas_figure(config, paths, name, setup, gas_maps[name], gas_scales))
        outputs[f"stars_{name}"] = str(save_stellar_figure(config, paths, name, setup, star_maps[name], star_scales))

    metadata = {
        "SID": config.sid,
        "SNAP": config.snap,
        "width_kpc": config.width_kpc,
        "Npix": config.npix,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "cutout_h5": str(cutout),
        "galaxy_center_ckpch": center_ckpch,
        "galaxy_center_source": center_meta,
        "normal_used_hat": normal,
        "normal_metadata": normal_meta,
        "projection_setups": setups,
        "systemic_velocity": systemic,
        "gas_plot_scales": gas_scales,
        "stellar_plot_scales": star_scales,
        "outputs": outputs,
    }
    base.write_json(paths["data"] / "pca3_faceon_edgeon_projection_metadata.json", metadata)
    print("PCA-v3 face-on/edge-on projections complete")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return outputs


if __name__ == "__main__":
    run(Config())
