#!/usr/bin/env python
"""
Projection-only diagnostics for the corrected fixed-sky sightline interpretation.

The QSO marker is fixed from the alpha=5 noflip ray anchor.  Alpha-dependent
diagnostics are generated in two LOS conventions:

1. fixed_observer_LOS: all projections use the alpha=5 LOS.  These maps should
   be identical apart from annotation and are a sanity check for the fixed marker.
2. saved_alpha_LOS: projections use each recipe row's LOS, but the plotted QSO
   marker remains the fixed alpha=5 sky-plane point.

No rotation curves are calculated here.
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

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt

import m61_oriented_HI_vlos_rotation_alpha5 as base


@dataclass
class DiagnosticConfig:
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "fixed_sky_sightline_rotated_galaxy_alpha_diagnostics"
    )
    recipe_csv: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/rays_sid488530.csv"
    )
    orientation_header: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/orient_header_sid488530.json"
    )
    sid: int = 488530
    snap: int = 99
    sightline_id: str = "J122138+043026"
    mode: str = "noflip"
    alphas: tuple[float, ...] = (5.0, 45.0, 90.0, 180.0, 255.0)
    reference_alpha: float = 5.0
    width_kpc: float = 100.0
    npix: int = 1024
    logNHI_threshold: float = 19.5
    recompute: bool = False

    @property
    def pixel_scale_kpc(self) -> float:
        return self.width_kpc / self.npix


def setup_paths(config: DiagnosticConfig) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {"out": out, "figures": out / "figures", "data": out / "data", "logs": out / "logs"}
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


def alpha_tag(alpha: float) -> str:
    return f"alpha{int(round(alpha)):03d}"


def convention_field_suffix(convention: str, alpha: float | None = None) -> str:
    if convention == "fixed_observer_LOS":
        return "fixed_observer_los"
    return f"saved_alpha_los_{alpha_tag(float(alpha)).lower()}"


def normalize(vec, name="vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = normalize(axis, "Rodrigues axis")
    vec = np.asarray(vec, dtype=float)
    theta = np.deg2rad(angle_deg)
    return (
        vec * np.cos(theta)
        + np.cross(axis, vec) * np.sin(theta)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(theta))
    )


def plane_basis_from_normal_and_los(n_hat: np.ndarray, los_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    major = np.cross(n_hat, los_hat)
    if np.linalg.norm(major) < 1e-8:
        major, minor = base.deterministic_plane_basis(los_hat)
        return major, minor
    major = normalize(major, "projected disk major axis")
    minor = normalize(np.cross(los_hat, major), "projected disk minor axis")
    return major, minor


def grid(config: DiagnosticConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = 0.5 * config.width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def load_recipe_rows(config: DiagnosticConfig) -> pd.DataFrame:
    df = pd.read_csv(config.recipe_csv)
    mask = (
        (df["SubhaloID"].astype(int) == config.sid)
        & (df["sightline_id"].astype(str) == config.sightline_id)
        & (df["mode"].astype(str) == config.mode)
        & df["alpha_deg"].astype(float).isin(list(config.alphas))
    )
    rows = df.loc[mask].copy()
    if len(rows) != len(config.alphas):
        raise RuntimeError(f"Expected {len(config.alphas)} recipe rows, found {len(rows)}.")
    rows["alpha_deg"] = rows["alpha_deg"].astype(float)
    rows = rows.sort_values("alpha_deg")
    return rows


def row_vector(row: pd.Series, prefix: str) -> np.ndarray:
    if prefix == "p0":
        keys = ["p0_X_ckpch_abs", "p0_Y_ckpch_abs", "p0_Z_ckpch_abs"]
    elif prefix == "p1":
        keys = ["p1_X_ckpch_abs", "p1_Y_ckpch_abs", "p1_Z_ckpch_abs"]
    elif prefix == "anchor":
        keys = ["anchor_X_ckpch_abs", "anchor_Y_ckpch_abs", "anchor_Z_ckpch_abs"]
    elif prefix == "los":
        keys = ["los_x", "los_y", "los_z"]
    else:
        raise ValueError(prefix)
    return np.asarray([row[k] for k in keys], dtype=float)


def load_disk_normal(config: DiagnosticConfig) -> tuple[np.ndarray, dict[str, Any]]:
    header = Path(config.orientation_header)
    if header.exists():
        with header.open() as f:
            payload = json.load(f)
        if "normal_used_hat" in payload:
            return normalize(payload["normal_used_hat"], "normal_used_hat"), {
                "source": str(header),
                "orientation_method": payload.get("orientation_method"),
                "obs_inc_deg": payload.get("obs_inc_deg"),
                "obs_pa_deg_used": payload.get("obs_pa_deg_used"),
            }
    fallback = np.array([0.1489902267951354, -0.8457239371292382, -0.5123991944628276])
    logging.warning("Could not find normal_used_hat metadata; using known fallback value.")
    return normalize(fallback, "fallback normal_used_hat"), {"source": "hard-coded fallback"}


def load_geometry(config: DiagnosticConfig, paths: dict[str, Path]) -> dict[str, Any]:
    base_config = base.AnalysisConfig(output_dir=str(paths["out"]))
    center_ckpch, center_meta = base.load_true_galaxy_center(base_config)
    rows = load_recipe_rows(config)
    ref_row = rows.loc[np.isclose(rows["alpha_deg"], config.reference_alpha)]
    if len(ref_row) != 1:
        raise RuntimeError(f"Reference alpha {config.reference_alpha} not found.")
    ref_row = ref_row.iloc[0]
    p0_ref = row_vector(ref_row, "p0")
    p1_ref = row_vector(ref_row, "p1")
    anchor_ref = row_vector(ref_row, "anchor")
    los_ref = normalize(p1_ref - p0_ref, "reference p1-p0 LOS")
    saved_los_ref = normalize(row_vector(ref_row, "los"), "reference saved LOS")
    normal_used, normal_meta = load_disk_normal(config)
    major_ref, minor_ref = plane_basis_from_normal_and_los(normal_used, los_ref)

    delta_ref = anchor_ref - center_ckpch
    delta_ref_perp = delta_ref - np.dot(delta_ref, los_ref) * los_ref
    rho_check = float(np.linalg.norm(delta_ref_perp) / base.H_TNG)
    x_qso = float(np.dot(delta_ref, major_ref) / base.H_TNG)
    y_qso = float(np.dot(delta_ref, minor_ref) / base.H_TNG)

    alpha_info = {}
    for _, row in rows.iterrows():
        alpha = float(row["alpha_deg"])
        p0 = row_vector(row, "p0")
        p1 = row_vector(row, "p1")
        saved_los = normalize(row_vector(row, "los"), f"saved LOS alpha {alpha:g}")
        los = normalize(p1 - p0, f"p1-p0 LOS alpha {alpha:g}")
        dot_saved = float(np.dot(saved_los, los))
        dot_ref = float(np.dot(saved_los, los_ref))
        d_alpha = alpha - config.reference_alpha
        major_rot = normalize(rodrigues_rotate(major_ref, normal_used, d_alpha), f"major rotated alpha {alpha:g}")
        minor_rot = normalize(rodrigues_rotate(minor_ref, normal_used, d_alpha), f"minor rotated alpha {alpha:g}")
        alpha_info[str(int(round(alpha)))] = {
            "alpha_deg": alpha,
            "selected_recipe_row": row.to_dict(),
            "p0_ckpch": p0,
            "p1_ckpch": p1,
            "anchor_ckpch": row_vector(row, "anchor"),
            "saved_los_hat": saved_los,
            "los_hat_from_p1p0": los,
            "dot_saved_los_with_p1p0": dot_saved,
            "dot_saved_los_with_los_ref": dot_ref,
            "major_rotated_about_disk_normal": major_rot,
            "minor_rotated_about_disk_normal": minor_rot,
        }

    geometry = {
        "galaxy_center_ckpch": center_ckpch,
        "galaxy_center_source": center_meta,
        "reference_alpha": config.reference_alpha,
        "anchor_ref_ckpch": anchor_ref,
        "p0_ref_ckpch": p0_ref,
        "p1_ref_ckpch": p1_ref,
        "los_ref_hat": los_ref,
        "saved_los_ref_hat": saved_los_ref,
        "dot_saved_los_ref_with_p1p0": float(np.dot(saved_los_ref, los_ref)),
        "normal_used_hat": normal_used,
        "normal_metadata": normal_meta,
        "major_ref_hat": major_ref,
        "minor_ref_hat": minor_ref,
        "delta_ref_perp_ckpch": delta_ref_perp,
        "x_qso_fixed_kpc": x_qso,
        "y_qso_fixed_kpc": y_qso,
        "rho_check_kpc": rho_check,
        "alpha_info": alpha_info,
    }
    logging.info("True galaxy center ckpc/h: %s", center_ckpch)
    logging.info("Reference anchor ckpc/h: %s", anchor_ref)
    logging.info("Fixed QSO marker x,y [kpc]: %.6f, %.6f", x_qso, y_qso)
    logging.info("rho_check_kpc = %.6f", rho_check)
    logging.info("normal_used_hat = %s", normal_used)
    logging.info("los_ref_hat = %s", los_ref)
    return geometry


def velocity_component_fields(ds) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]] | None:
    candidates = [
        (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        (("PartType0", "velocity_x"), ("PartType0", "velocity_y"), ("PartType0", "velocity_z")),
        (("PartType0", "particle_velocity_x"), ("PartType0", "particle_velocity_y"), ("PartType0", "particle_velocity_z")),
    ]
    for cand in candidates:
        if all(base.yt_field_exists(ds, field) for field in cand):
            return cand
    return None


def add_los_projection_fields(ds, los_hat: np.ndarray, systemic_kms: np.ndarray, suffix: str) -> dict[str, tuple[str, str]]:
    component_fields = velocity_component_fields(ds)
    raw_vector_field = ("PartType0", "Velocities")
    if component_fields is None and not base.yt_field_exists(ds, raw_vector_field):
        raise RuntimeError("Could not find gas velocity fields.")

    def velocity_component(data, index: int):
        if component_fields is not None:
            return data[component_fields[index]].to("km/s")
        return data[raw_vector_field][:, index].to("km/s")

    vlos_field = ("gas", f"velocity_los_{suffix}")
    hi_vlos_field = ("gas", f"HI_vlos_integrand_{suffix}")

    def _vlos(field, data):
        vx = velocity_component(data, 0) - data.ds.quan(systemic_kms[0], "km/s")
        vy = velocity_component(data, 1) - data.ds.quan(systemic_kms[1], "km/s")
        vz = velocity_component(data, 2) - data.ds.quan(systemic_kms[2], "km/s")
        return (los_hat[0] * vx + los_hat[1] * vy + los_hat[2] * vz).to("km/s")

    def _hi_vlos(field, data):
        return data[("gas", "H_p0_number_density")] * data[vlos_field]

    if not base.yt_field_exists(ds, vlos_field):
        ds.add_field(vlos_field, function=_vlos, sampling_type="particle", units="km/s")
    if not base.yt_field_exists(ds, hi_vlos_field):
        ds.add_field(hi_vlos_field, function=_hi_vlos, sampling_type="particle", units="km/(s*cm**3)")
    return {"vlos_field": vlos_field, "hi_vlos_field": hi_vlos_field}


def load_dataset_and_fields(config: DiagnosticConfig, paths: dict[str, Path], geometry: dict[str, Any]):
    base_config = base.AnalysisConfig(output_dir=str(paths["out"]))
    cutout = base.find_cutout_h5(base_config)
    ds = yt.load(str(cutout))
    field_info = base.ensure_gas_alias_fields(ds)
    systemic = base.systemic_velocity_from_hdf5(cutout, geometry["galaxy_center_ckpch"])
    field_info["systemic_velocity"] = systemic
    field_info["cutout_h5"] = str(cutout)
    logging.info("Systemic velocity used: %s", systemic)
    return ds, field_info


def projection_file(paths: dict[str, Path], alpha: float, convention: str) -> Path:
    return paths["data"] / f"projection_{alpha_tag(alpha)}_{convention}_fixed_marker_inner100kpc_logNHIgt19p5.npz"


def compute_projection(
    config: DiagnosticConfig,
    ds,
    paths: dict[str, Path],
    geometry: dict[str, Any],
    alpha: float,
    convention: str,
    normal: np.ndarray,
    north: np.ndarray,
    hi_vlos_field: tuple[str, str],
) -> dict[str, Any]:
    out = projection_file(paths, alpha, convention)
    if out.exists() and not config.recompute:
        z = np.load(out, allow_pickle=True)
        return {key: z[key] for key in z.files}

    x, y, X, Y = grid(config)
    center = ds.arr(geometry["galaxy_center_ckpch"], "code_length")
    width = ds.arr([config.width_kpc * base.H_TNG] * 2, "code_length")
    hi_den = base.off_axis_integral(ds, center, normal, width, config.npix, ("gas", "H_p0_number_density"), north)
    hi_num = base.off_axis_integral(ds, center, normal, width, config.npix, hi_vlos_field, north)
    N_HI = base.yt_array_to_numpy(hi_den, "cm**-2")
    HI_vlos_num = base.yt_array_to_numpy(hi_num, "km/(s*cm**2)")
    with np.errstate(divide="ignore", invalid="ignore"):
        logN = np.log10(np.where(N_HI > 0, N_HI, np.nan))
        vhi = HI_vlos_num / N_HI
    vhi[~np.isfinite(vhi)] = np.nan
    mask = np.isfinite(logN) & (logN >= config.logNHI_threshold) & np.isfinite(vhi)

    np.savez_compressed(
        out,
        x_kpc=x,
        y_kpc=y,
        X_kpc=X,
        Y_kpc=Y,
        N_HI_cm2=N_HI,
        logN_HI=logN,
        vlos_HIweighted_kms=vhi,
        mask_logNHI_gt19p5=mask,
        x_qso_fixed_kpc=geometry["x_qso_fixed_kpc"],
        y_qso_fixed_kpc=geometry["y_qso_fixed_kpc"],
        rho_check_kpc=geometry["rho_check_kpc"],
        alpha_deg=float(alpha),
        los_hat_used=np.asarray(normal, dtype=float),
        convention=convention,
    )
    valid_log = logN[np.isfinite(logN)]
    valid_v = vhi[mask]
    logging.info(
        "%s %s: logN percentiles=%s; vlos percentiles=%s",
        alpha_tag(alpha), convention,
        np.nanpercentile(valid_log, [1, 50, 99]).tolist() if valid_log.size else [],
        np.nanpercentile(valid_v, [2, 50, 98]).tolist() if valid_v.size else [],
    )
    return {
        "x_kpc": x,
        "y_kpc": y,
        "X_kpc": X,
        "Y_kpc": Y,
        "N_HI_cm2": N_HI,
        "logN_HI": logN,
        "vlos_HIweighted_kms": vhi,
        "mask_logNHI_gt19p5": mask,
        "x_qso_fixed_kpc": np.asarray(geometry["x_qso_fixed_kpc"]),
        "y_qso_fixed_kpc": np.asarray(geometry["y_qso_fixed_kpc"]),
        "rho_check_kpc": np.asarray(geometry["rho_check_kpc"]),
        "alpha_deg": np.asarray(float(alpha)),
        "los_hat_used": np.asarray(normal, dtype=float),
        "convention": np.asarray(convention),
    }


def hi_scale(maps: list[dict[str, Any]], threshold: float) -> tuple[float, float]:
    vals = []
    for m in maps:
        arr = m["logN_HI"]
        vals.append(arr[np.isfinite(arr) & (arr >= threshold)])
    vals = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.array([threshold, threshold + 1])
    return threshold, float(max(threshold + 0.2, np.nanpercentile(vals, 99)))


def velocity_scale(maps: list[dict[str, Any]]) -> float:
    vals = []
    for m in maps:
        v = m["vlos_HIweighted_kms"]
        mask = m["mask_logNHI_gt19p5"].astype(bool)
        vals.append(np.abs(v[mask & np.isfinite(v)]))
    vals = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.array([100.0])
    return float(min(max(np.nanpercentile(vals, 98), 50.0), 300.0))


def save_single_plot(config, paths, alpha, convention, m, kind, vmin, vmax):
    extent = [-50, 50, -50, 50]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    if kind == "HI":
        data = np.where(m["logN_HI"] >= config.logNHI_threshold, m["logN_HI"], np.nan)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("white")
        label = r"$\log_{10} N_{\rm HI}\ [{\rm cm}^{-2}]$"
        stem = f"HI_logNHI_gt19p5_fixed_marker_{alpha_tag(alpha)}_{convention}"
    else:
        data = np.where(m["mask_logNHI_gt19p5"].astype(bool), m["vlos_HIweighted_kms"], np.nan)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        label = r"$v_{\rm los,HI-weighted}$ [km/s]"
        stem = f"vlos_HIweighted_logNHIgt19p5_fixed_marker_{alpha_tag(alpha)}_{convention}"
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.91, label=label)
    ax.plot(0, 0, marker="+", ms=12, mew=1.8, color="cyan")
    ax.scatter([float(m["x_qso_fixed_kpc"])], [float(m["y_qso_fixed_kpc"])], s=52, facecolors="none", edgecolors="lime", lw=1.5)
    ax.text(
        0.03, 0.04,
        f"SID 488530\nalpha={alpha:.0f} deg, noflip\nfixed sky marker\nrho={float(m['rho_check_kpc']):.2f} kpc\nLOS={convention}",
        transform=ax.transAxes,
        fontsize=8.4,
        color="black" if kind == "HI" else "white",
        bbox=dict(facecolor="white" if kind == "HI" else "black", alpha=0.68, edgecolor="none", pad=3),
    )
    ax.set_xlabel("fixed sky x [kpc]")
    ax.set_ylabel("fixed sky y [kpc]")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.minorticks_on()
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"{stem}.{suffix}", dpi=350)
    plt.close(fig)


def save_multipanel(config, paths, convention, maps_by_alpha, kind, vmin, vmax):
    extent = [-50, 50, -50, 50]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8), constrained_layout=True)
    axes = axes.ravel()
    im = None
    for ax, alpha in zip(axes, config.alphas):
        m = maps_by_alpha[float(alpha)]
        if kind == "HI":
            data = np.where(m["logN_HI"] >= config.logNHI_threshold, m["logN_HI"], np.nan)
            cmap = plt.get_cmap("magma").copy()
            cmap.set_bad("white")
            label = r"$\log_{10} N_{\rm HI}$"
            stem = f"HI_logNHI_gt19p5_multi_alpha_fixed_marker_{convention}"
        else:
            data = np.where(m["mask_logNHI_gt19p5"].astype(bool), m["vlos_HIweighted_kms"], np.nan)
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("white")
            label = r"$v_{\rm los,HI-weighted}$ [km/s]"
            stem = f"vlos_HIweighted_logNHIgt19p5_multi_alpha_fixed_marker_{convention}"
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(0, 0, marker="+", ms=10, mew=1.5, color="cyan")
        ax.scatter([float(m["x_qso_fixed_kpc"])], [float(m["y_qso_fixed_kpc"])], s=42, facecolors="none", edgecolors="lime", lw=1.2)
        ax.set_title(f"alpha={alpha:.0f} deg")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.minorticks_on()
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes[:-1].tolist(), shrink=0.9, label=label)
    for suffix in ("png", "pdf"):
        fig.savefig(paths["figures"] / f"{stem}.{suffix}", dpi=350)
    plt.close(fig)


def make_plots(config, paths, all_maps):
    for convention, maps_by_alpha in all_maps.items():
        maps = [maps_by_alpha[float(alpha)] for alpha in config.alphas]
        hi_vmin, hi_vmax = hi_scale(maps, config.logNHI_threshold)
        vlim = velocity_scale(maps)
        for alpha in config.alphas:
            m = maps_by_alpha[float(alpha)]
            save_single_plot(config, paths, alpha, convention, m, "HI", hi_vmin, hi_vmax)
            save_single_plot(config, paths, alpha, convention, m, "vlos", -vlim, vlim)
        save_multipanel(config, paths, convention, maps_by_alpha, "HI", hi_vmin, hi_vmax)
        save_multipanel(config, paths, convention, maps_by_alpha, "vlos", -vlim, vlim)


def run(config: DiagnosticConfig):
    paths = setup_paths(config)
    setup_logging(paths)
    try:
        geometry = load_geometry(config, paths)
        ds, field_info = load_dataset_and_fields(config, paths, geometry)
        systemic = np.asarray(field_info["systemic_velocity"]["velocity_kms"], dtype=float)

        fixed_suffix = convention_field_suffix("fixed_observer_LOS")
        fixed_fields = add_los_projection_fields(ds, geometry["los_ref_hat"], systemic, fixed_suffix)
        all_maps: dict[str, dict[float, dict[str, Any]]] = {
            "fixed_observer_LOS": {},
            "saved_alpha_LOS": {},
        }

        logging.info("Computing fixed-observer LOS projection once, then saving alpha-labeled copies.")
        fixed_map_ref = compute_projection(
            config, ds, paths, geometry, config.reference_alpha, "fixed_observer_LOS",
            geometry["los_ref_hat"], geometry["minor_ref_hat"], fixed_fields["hi_vlos_field"],
        )
        for alpha in config.alphas:
            if float(alpha) == float(config.reference_alpha):
                all_maps["fixed_observer_LOS"][float(alpha)] = fixed_map_ref
                continue
            out = projection_file(paths, alpha, "fixed_observer_LOS")
            if out.exists() and not config.recompute:
                z = np.load(out, allow_pickle=True)
                all_maps["fixed_observer_LOS"][float(alpha)] = {key: z[key] for key in z.files}
            else:
                payload = dict(fixed_map_ref)
                payload["alpha_deg"] = np.asarray(float(alpha))
                np.savez_compressed(out, **payload)
                all_maps["fixed_observer_LOS"][float(alpha)] = payload

        for alpha in config.alphas:
            info = geometry["alpha_info"][str(int(round(alpha)))]
            los = np.asarray(info["saved_los_hat"], dtype=float)
            _, north = plane_basis_from_normal_and_los(geometry["normal_used_hat"], los)
            suffix = convention_field_suffix("saved_alpha_LOS", alpha)
            fields = add_los_projection_fields(ds, los, systemic, suffix)
            logging.info(
                "alpha %.0f saved_alpha_LOS: saved_los=%s dot(saved, ref)=%.6f marker=(%.3f, %.3f)",
                alpha, los, info["dot_saved_los_with_los_ref"], geometry["x_qso_fixed_kpc"], geometry["y_qso_fixed_kpc"]
            )
            all_maps["saved_alpha_LOS"][float(alpha)] = compute_projection(
                config, ds, paths, geometry, alpha, "saved_alpha_LOS", los, north, fields["hi_vlos_field"]
            )

        make_plots(config, paths, all_maps)
        metadata = {
            "SID": config.sid,
            "SNAP": config.snap,
            "alpha_list": list(config.alphas),
            "reference_alpha": config.reference_alpha,
            "galaxy_center_ckpch": geometry["galaxy_center_ckpch"],
            "anchor_ref_ckpch": geometry["anchor_ref_ckpch"],
            "fixed_marker_coordinates": {
                "x_qso_fixed_kpc": geometry["x_qso_fixed_kpc"],
                "y_qso_fixed_kpc": geometry["y_qso_fixed_kpc"],
                "rho_check_kpc": geometry["rho_check_kpc"],
            },
            "normal_used_hat": geometry["normal_used_hat"],
            "normal_metadata": geometry["normal_metadata"],
            "los_ref_hat": geometry["los_ref_hat"],
            "saved_los_hat_by_alpha": {
                key: value["saved_los_hat"] for key, value in geometry["alpha_info"].items()
            },
            "alpha_info": geometry["alpha_info"],
            "chosen_conventions": ["fixed_observer_LOS", "saved_alpha_LOS"],
            "projection_width_kpc": config.width_kpc,
            "Npix": config.npix,
            "pixel_scale_kpc": config.pixel_scale_kpc,
            "HI_display_threshold_logNHI": config.logNHI_threshold,
            "systemic_velocity": field_info["systemic_velocity"],
            "output_paths": {k: str(v) for k, v in paths.items()},
        }
        meta_path = paths["data"] / "fixed_sky_sightline_rotated_galaxy_projection_metadata.json"
        write_json(meta_path, metadata)
        logging.info("Metadata written: %s", meta_path)
        print("\nFixed-sky diagnostic projections complete")
        print(f"output_dir = {paths['out']}")
        print(f"fixed_marker = ({geometry['x_qso_fixed_kpc']:.3f}, {geometry['y_qso_fixed_kpc']:.3f}) kpc")
        print(f"rho_check_kpc = {geometry['rho_check_kpc']:.6f}")
        print(f"metadata = {meta_path}")
        return {"paths": paths, "geometry": geometry, "field_info": field_info, "maps": all_maps}
    except Exception:
        err = paths["logs"] / "errors.txt"
        err.write_text(traceback.format_exc())
        logging.exception("Diagnostic projection workflow failed; wrote %s", err)
        raise


def parse_args(argv=None) -> DiagnosticConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--output-dir", type=str, default=DiagnosticConfig.output_dir)
    args = parser.parse_args(argv)
    return DiagnosticConfig(npix=args.npix, recompute=args.recompute, output_dir=args.output_dir)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
